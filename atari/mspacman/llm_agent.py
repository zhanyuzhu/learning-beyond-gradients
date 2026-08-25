"""A slow, tool-calling LLM strategy supervisor for `policy.MsPacmanAgent`.

This does not touch `policy.py`.  It watches the game on a fixed *step*
cadence and on key events, and may call `set_strategy` to change the
planner's standing posture, or `log_observation` to record a failure-mode
note.  It never emits a raw action.  Every call runs in a background thread
via a single-worker `ThreadPoolExecutor`, so the env loop never blocks on
network I/O; the loop only ever polls a future.

What this layer is *for*, and what it must stay away from, follows from one
measurement: over 53 logged invocations against this endpoint the mean
round trip was 11.3s, worst case 46s.  At `frame_skip=4` a realtime episode
runs 15 env steps per second, so a reply lands about 170 steps after the
frame it was looking at, while a power-pill window lasts roughly 90 steps.
Any tactical instruction - "the ghosts are blue, chase them" - therefore
arrives *after* the window it was meant for, and worse, the matching "they
are lethal again, back off" arrives late too, leaving the planner at its
most reckless setting exactly while the ghosts are dangerous.  The previous
version of this file did precisely that, flapping between a `balanced` and
an `aggressive_hunt` preset on the `pill_activated` event.

So the division of labour is:

* the Dijkstra planner owns everything that changes faster than ~10s -
  ghost avoidance, chasing blue ghosts, routing.  It already reads
  edibility every single frame, at zero latency;
* the supervisor owns only postures that stay valid for tens of seconds -
  how much risk to take on the last life, and whether to bait power pills
  or race to clear the board.

Requires `openai` and `Pillow` (only when `--agent` is passed)::

    pip install openai Pillow
"""

from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# .env loading (no python-dotenv dependency - the file only ever has three
# simple KEY=VALUE lines)
# ---------------------------------------------------------------------------

ENV_KEYS = {"model": "DASHSCOPE_MODEL_NAME",
            "base_url": "DASHSCOPE_BASE_URL",
            "api_key": "DASHSCOPE_API_KEY"}


def load_dashscope_config(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")

    config = {name: values.get(env_key, "") for name, env_key in ENV_KEYS.items()}
    missing = [ENV_KEYS[name] for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {env_path}")
    return config


# ---------------------------------------------------------------------------
# Strategy presets and safe parameter ranges
#
# Every preset is anchored on the settings `policy.py` was actually
# validated with (mean 11044 over 60 sticky-action episodes); the modes
# differ from that anchor by small, single-purpose deltas.  The old
# `aggressive_hunt` preset cut `safety_margin` from 18 to 10 - a third of
# the planner's whole safety buffer - and it was chosen in 13 of 21 logged
# changes: a tuned config perturbed hard, in the wrong direction, on stale
# information.
#
# `bait_enabled` is not a `HeuristicConfig` field but an attribute of
# `BaitingAgent`; `policy_with_agent.py` routes it accordingly.
# ---------------------------------------------------------------------------

PRESET_MODES: dict[str, dict[str, float]] = {
    # the validated defaults, with pill baiting on
    "balanced": {
        "safety_margin": 18, "ghost_value": 200.0, "pill_lure_radius": 80,
        "flash_chase_radius": 60, "hysteresis": 1.25, "bait_enabled": 1,
    },
    # survival first: wider berth, and never loiter beside a pill
    "careful": {
        "safety_margin": 26, "ghost_value": 180.0, "pill_lure_radius": 70,
        "flash_chase_radius": 45, "hysteresis": 1.40, "bait_enabled": 0,
    },
    # a nearly-empty board is worth finishing; stop waiting for ghosts
    "clear_board": {
        "safety_margin": 18, "ghost_value": 200.0, "pill_lure_radius": 80,
        "flash_chase_radius": 60, "hysteresis": 1.25, "bait_enabled": 0,
    },
    # plenty of board left and lives in hand: work the pills harder
    "farm_ghosts": {
        "safety_margin": 16, "ghost_value": 240.0, "pill_lure_radius": 95,
        "flash_chase_radius": 70, "hysteresis": 1.20, "bait_enabled": 1,
    },
}

BASELINE_MODE = "balanced"

# name -> (min, max, cast).  The ranges are deliberately narrow: they
# bracket the validated defaults rather than spanning everything the
# planner will accept, so no single call can undo the tuning.
PARAM_RANGES: dict[str, tuple[float, float, type]] = {
    "safety_margin": (12, 30, int),
    "ghost_value": (120.0, 320.0, float),
    "pill_lure_radius": (50, 110, int),
    "flash_chase_radius": (30, 90, int),
    "hysteresis": (1.05, 1.80, float),
    "bait_enabled": (0, 1, int),
}

# keys that live on the agent rather than on HeuristicConfig
AGENT_ATTRS = frozenset({"bait_enabled"})


def _clamp(key: str, value: Any) -> float | int:
    lo, hi, cast = PARAM_RANGES[key]
    clamped = max(lo, min(hi, float(value)))
    return round(clamped) if cast is int else clamped


def resolve_patch(mode: str, overrides: dict[str, Any] | None
                  ) -> dict[str, float | int]:
    """Combine a preset with allow-listed overrides, all clamped."""
    patch = dict(PRESET_MODES[mode])
    patch.update({key: value for key, value in (overrides or {}).items()
                  if key in PARAM_RANGES})
    return {key: _clamp(key, value) for key, value in patch.items()}


# ---------------------------------------------------------------------------
# Tool schema (OpenAI-compatible function calling, as served by DashScope)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_strategy",
            "description": (
                "为规则规划器设定一个**长时程**的策略姿态。\n"
                "只在你希望接下来几十秒都保持的取向发生变化时调用，例如：\n"
                "只剩最后一条命了要保守、这一关豆子快吃完了应该直接清关、\n"
                "或者命还很多且豆子还多、应该多花时间围绕能量豆钓幽灵。\n\n"
                "**不要**用它做战术反应。你的回复平均要 11 秒才会生效\n"
                "（约 170 个环境步），而一次能量豆的可食窗口只有约 90 步——\n"
                "所以“现在幽灵变蓝了，去追”这类指令一定会迟到；\n"
                "更糟的是随后那句“幽灵恢复危险了，快撤”同样会迟到，\n"
                "反而让规划器在最危险的时刻停留在最激进的参数上。\n"
                "幽灵的躲避与追捕由规划器每一帧自行处理，零延迟，无需你插手。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(PRESET_MODES),
                        "description": (
                            "balanced=已验证的默认姿态（含能量豆诱敌）；\n"
                            "careful=保命优先，不再守在能量豆旁；\n"
                            "clear_board=残局，直接把豆子吃完；\n"
                            "farm_ghosts=命多豆多时，加大围绕能量豆钓幽灵的力度。"
                        ),
                    },
                    "overrides": {
                        "type": "object",
                        "description": "在预设之上的小幅微调，均会被裁剪到安全区间。",
                        "properties": {
                            "safety_margin": {"type": "integer"},
                            "ghost_value": {"type": "number"},
                            "pill_lure_radius": {"type": "integer"},
                            "flash_chase_radius": {"type": "integer"},
                            "hysteresis": {"type": "number"},
                            "bait_enabled": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string",
                               "description": "简要说明这次调整的理由。"},
                },
                "required": ["mode", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_observation",
            "description": (
                "记录一条关于失败模式或值得注意现象的诊断说明——死亡原因、\n"
                "卡住/来回震荡的循环、看起来被误读的迷宫格子、错过的得分机会——\n"
                "这不会改变游戏行为。只要你发现值得开发者事后排查的现象就调用它，\n"
                "与你是否同时调用 set_strategy 无关。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["death", "stuck", "maze_error",
                                 "missed_opportunity", "other"],
                    },
                    "note": {"type": "string"},
                },
                "required": ["category", "note"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是一个基于规则的 Ms. Pac-Man 智能体的**长时程**策略主管。\n\n"

    "底层是一个基于 Dijkstra 的迷宫路径规划器，它每一帧都在跑，\n"
    "已经独立调优到 60 局平均约 11000 分。它每一帧都能看到幽灵是否可食、\n"
    "是否在闪烁，并据此自行躲避和追捕——这部分**不需要也不应该由你干预**。\n\n"

    "你被调用的频率很低（按固定步数间隔，或在关键事件时），\n"
    "每次看到一帧画面和一份 JSON 快照。关键约束：\n"
    "**你的回复平均要约 11 秒、即约 170 个环境步之后才会生效**，\n"
    "快照里的 `decision_lag_steps` 会告诉你上一次的实际滞后。\n"
    "而一次能量豆的可食窗口只有约 90 步。所以任何“此刻幽灵变蓝了、\n"
    "快去追”式的战术指令必然迟到，并且有害。\n\n"

    "你真正该管的是那些能维持几十秒的取向，例如：\n"
    "只剩一条命时是否该更保守；这一关豆子只剩十几颗时是否该放弃钓幽灵、\n"
    "直接清关；开局命多豆多时是否该更用力地围绕能量豆钓幽灵。\n\n"

    "快照里带有反馈：`last_change` 会告诉你上一次调整之后的实际得分速率\n"
    "和死亡数，`guardrail` 会告诉你系统是否因为效果变差而自动回退到了\n"
    "默认参数。请利用这些证据，不要凭画面印象反复改来改去；\n"
    "**如果没有明确理由要改变长时程取向，就不要调用 `set_strategy`**。\n"
    "发现死亡、卡住、迷宫识别异常等现象时，用 `log_observation` 记录下来\n"
    "以便事后复查。`reason` 和 `note` 一律用中文，一到两句话说清楚即可。"
)


def encode_frame_png_b64(frame: np.ndarray, scale: int = 3) -> str:
    img = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8), mode="RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(snapshot: dict[str, Any], image_b64: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": json.dumps(snapshot, sort_keys=True, ensure_ascii=False)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Event detection (external to policy.py, driven off its public state)
# ---------------------------------------------------------------------------

STAGNATION_STEPS = 180
PANIC_STREAK_TRIGGER = 3


class EventTracker:
    """Watches `GameState`/`MsPacmanAgent` for moments worth an early invoke.

    `pill_activated` is deliberately *not* a trigger any more.  It was the
    one that produced the old flap - the reply it solicited could not
    arrive inside the pill window it was reacting to - and the planner
    needs no help there in any case.  What remains are events whose
    consequences outlive the round trip: a death, a new level, and being
    pinned or stalled.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev_lives: int | None = None
        self.prev_dots_eaten = 0
        self.panic_streak = 0
        self.last_progress_step = 0

    def update(self, step: int, state, agent) -> str | None:
        self.panic_streak = self.panic_streak + 1 if agent.target is None else 0
        progressed = state.dots_eaten != self.prev_dots_eaten
        if progressed:
            self.last_progress_step = step

        if self.prev_lives is not None and state.lives < self.prev_lives:
            reason = "death"
        elif state.dots_eaten < self.prev_dots_eaten - 5:
            reason = "level_start"
        elif self.panic_streak == PANIC_STREAK_TRIGGER:
            reason = "cornered_repeatedly"
        elif not progressed and step - self.last_progress_step >= STAGNATION_STEPS:
            reason = "stagnant"
            self.last_progress_step = step  # don't re-fire every tick
        else:
            reason = None

        self.prev_lives = state.lives
        self.prev_dots_eaten = state.dots_eaten
        return reason


# ---------------------------------------------------------------------------
# Async supervisor
# ---------------------------------------------------------------------------


@dataclass
class InvocationResult:
    trigger: str | None
    latency_s: float
    ok: bool
    mode: str | None = None
    config_patch: dict[str, float] | None = None
    notes: list[dict[str, str]] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    submitted_step: int = -1
    lag_steps: int = -1


def read_tool_calls(tool_calls) -> dict[str, Any]:
    """Fold the model's tool calls into `InvocationResult` fields."""
    mode: str | None = None
    patch: dict[str, float | int] | None = None
    notes: list[dict[str, str]] = []
    raw: list[dict[str, Any]] = []
    for call in tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            continue  # one malformed call must not sink the whole turn
        raw.append({"name": call.function.name, "arguments": args})
        if call.function.name == "set_strategy":
            candidate = args.get("mode")
            if candidate in PRESET_MODES:
                mode = candidate
                patch = resolve_patch(mode, args.get("overrides"))
                notes.append({"category": "strategy_change",
                              "note": args.get("reason", "")})
        elif call.function.name == "log_observation":
            notes.append({"category": args.get("category", "other"),
                          "note": args.get("note", "")})
    return {"mode": mode, "config_patch": patch,
            "notes": notes, "raw_tool_calls": raw}


class AgentSupervisor:
    """Owns the DashScope client and the single background worker slot.

    `maybe_invoke` and `poll` are both meant to be called once per env tick
    from the main thread; neither ever blocks on the network.

    The cadence is counted in **env steps**, not wall-clock seconds.  Under
    a seconds-based gate the identical code consults the model perhaps
    twice in a headless episode and dozens of times in a `--realtime` one,
    so the policy being measured is not the policy being demonstrated.
    """

    def __init__(self, model: str, base_url: str, api_key: str,
                 log_path: Path, interval_steps: int = 150,
                 min_interval_steps: int = 60, timeout_s: float = 20.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None
        self._last_submit_step = -(1 << 30)
        self._interval_steps = interval_steps
        self._min_interval_steps = min_interval_steps
        self._timeout_s = timeout_s
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_invoke(self, frame: np.ndarray, snapshot: dict[str, Any],
                     trigger_reason: str | None, step: int) -> None:
        if self._pending is not None and not self._pending.done():
            return  # a call is already in flight; never queue a second one
        # an event may shorten the wait between calls, never lengthen it
        gate = (min(self._interval_steps, self._min_interval_steps)
                if trigger_reason else self._interval_steps)
        if step - self._last_submit_step < gate:
            return
        image_b64 = encode_frame_png_b64(frame)
        self._last_submit_step = step
        self._pending = self._executor.submit(
            self._invoke_once, image_b64, snapshot,
            trigger_reason or "interval", step)

    def poll(self, step: int) -> InvocationResult | None:
        if self._pending is None or not self._pending.done():
            return None
        result: InvocationResult = self._pending.result()  # worker never raises
        self._pending = None
        result.lag_steps = step - result.submitted_step
        self._append_log(result)
        return result

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    # -- worker thread body -------------------------------------------------

    def _invoke_once(self, image_b64: str, snapshot: dict[str, Any],
                     trigger_reason: str, step: int) -> InvocationResult:
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=build_messages(snapshot, image_b64),
                tools=TOOLS, tool_choice="auto", timeout=self._timeout_s,
            )
            fields = read_tool_calls(response.choices[0].message.tool_calls)
        except Exception as exc:  # network/timeout/malformed response, etc.
            return InvocationResult(trigger_reason, time.monotonic() - started,
                                    ok=False, error=repr(exc),
                                    submitted_step=step)
        return InvocationResult(trigger_reason, time.monotonic() - started,
                                ok=True, submitted_step=step, **fields)

    def _append_log(self, result: InvocationResult) -> None:
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                  **asdict(result)}
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
