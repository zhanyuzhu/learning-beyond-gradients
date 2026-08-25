"""A slow, tool-calling LLM supervisor for `policy.MsPacmanAgent`.

This does not touch `policy.py`. It watches the game at a fixed cadence and on
key events, and may call `adjust_policy` to retune `HeuristicConfig` live, or
`log_observation` to record a failure-mode note for later analysis - never a
raw action. Every call runs in a background thread via a single-worker
`ThreadPoolExecutor` so the 60 fps env loop never blocks on network I/O; the
loop only ever polls a future, it never waits on one.

Requires `openai` and `Pillow` (only when `--agent` is passed):

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
# Only a small, allow-listed subset of HeuristicConfig fields are tunable
# from the LLM side - the ones that trade survival margin against scoring
# aggression - and every value is clamped before it ever reaches agent.config.
# ---------------------------------------------------------------------------

PRESET_MODES: dict[str, dict[str, float]] = {
    "balanced": {
        "safety_margin": 18, "ghost_value": 200.0,
        "pill_lure_radius": 80, "flash_chase_radius": 60, "hysteresis": 1.25,
    },
    "cautious": {
        "safety_margin": 28, "ghost_value": 180.0,
        "pill_lure_radius": 60, "flash_chase_radius": 40, "hysteresis": 1.6,
    },
    "aggressive_hunt": {
        "safety_margin": 10, "ghost_value": 320.0,
        "pill_lure_radius": 100, "flash_chase_radius": 90, "hysteresis": 1.1,
    },
    "pellet_rush": {
        "safety_margin": 14, "ghost_value": 200.0,
        "pill_lure_radius": 70, "flash_chase_radius": 60, "hysteresis": 1.15,
    },
}

# name -> (min, max, cast)
PARAM_RANGES: dict[str, tuple[float, float, type]] = {
    "safety_margin": (0, 60, int),
    "ghost_value": (50.0, 600.0, float),
    "pill_lure_radius": (20, 160, int),
    "flash_chase_radius": (20, 160, int),
    "hysteresis": (1.0, 3.0, float),
}


def _clamp(key: str, value: Any) -> float | int:
    lo, hi, cast = PARAM_RANGES[key]
    clamped = max(lo, min(hi, float(value)))
    return round(clamped) if cast is int else clamped


def resolve_patch(mode: str, overrides: dict[str, Any] | None) -> dict[str, float | int]:
    """Combine a preset with allow-listed overrides, all clamped to safe ranges."""
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
            "name": "adjust_policy",
            "description": (
                "修改这个基于规则的 Ms. Pac-Man 策略的实时调参。"
                "只有在当前参数明显失效（反复死亡、被幽灵围堵、卡住不吃豆），"
                "或出现明确的战术机会（例如附近的能量豆即将可用，且周围有多个幽灵）"
                "时才调用。不要每一轮都调用——只在你确实想改变行为时才调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(PRESET_MODES),
                        "description": "作为基线应用的粗粒度策略预设。",
                    },
                    "overrides": {
                        "type": "object",
                        "description": "在预设之上的可选微调。",
                        "properties": {
                            "safety_margin": {"type": "integer"},
                            "ghost_value": {"type": "number"},
                            "pill_lure_radius": {"type": "integer"},
                            "flash_chase_radius": {"type": "integer"},
                            "hysteresis": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string", "description": "简要说明这次调整的理由。"},
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
                "记录一条关于失败模式或值得注意的现象的诊断说明——死亡原因、"
                "卡住/来回震荡的循环、看起来被误读的迷宫格子、错过的得分机会——"
                "这不会改变游戏行为。只要你发现值得开发者事后排查的现象就调用它，"
                "与你是否同时调用 adjust_policy 无关。"
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

SYSTEM_PROMPT = """你是一个基于规则的 Ms. Pac-Man 智能体的策略主管。

底层策略是一个基于 Dijkstra 的迷宫路径规划器，它每一帧都在运行；而你只会被偶尔调用\
（按固定间隔，或在关键事件发生时），每次能看到一帧游戏画面，以及一份描述当前游戏状态\
和规划器调参的 JSON 快照。你不能直接操控 Ms. Pac-Man——你只能通过 `adjust_policy`\
 重新调整规划器的参数，或者通过 `log_observation` 留下一条诊断记录。请调用合适的工具；\
只有当你确实想改变策略时才调用 `adjust_policy`，不要每一轮都调用。当你看到死亡、\
卡住/来回震荡的行为，或任何看起来像是迷宫识别 bug 的现象时，请调用 `log_observation`\
 把它记录下来以便事后复查——即使你在同一轮里也调用了 `adjust_policy`。\
`reason` 和 `note` 一律用中文书写，简明扼要，一到两句话说清楚即可。"""


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
                {"type": "text", "text": json.dumps(snapshot, sort_keys=True)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


def build_snapshot(state, agent, step: int, score: float, mode: str,
                   trigger_reason: str | None,
                   recent_notes: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "step": step,
        "score": score,
        "lives": state.lives,
        "dots_eaten": state.dots_eaten,
        "ghosts_edible": state.edible,
        "ghosts_flashing": state.flashing,
        "trigger_reason": trigger_reason,
        "current_mode": mode,
        "current_config": {key: getattr(agent.config, key) for key in PARAM_RANGES},
        "recent_notes": recent_notes,
    }


# ---------------------------------------------------------------------------
# Event detection (external to policy.py, driven off its public state)
# ---------------------------------------------------------------------------

STAGNATION_STEPS = 180  # ~3s of wall clock at frame_skip=4, 60fps
PANIC_STREAK_TRIGGER = 3


class EventTracker:
    """Watches `GameState`/`MsPacmanAgent` for moments worth an early invoke."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev_lives: int | None = None
        self.prev_dots_eaten = 0
        self.prev_edible_any = False
        self.panic_streak = 0
        self.last_progress_step = 0

    def update(self, step: int, state, agent) -> str | None:
        edible_any = any(state.edible)
        self.panic_streak = self.panic_streak + 1 if agent.target is None else 0
        progressed = state.dots_eaten != self.prev_dots_eaten
        if progressed:
            self.last_progress_step = step

        if self.prev_lives is not None and state.lives < self.prev_lives:
            reason = "death"
        elif state.dots_eaten < self.prev_dots_eaten - 5:
            reason = "level_start"
        elif edible_any and not self.prev_edible_any:
            reason = "pill_activated"
        elif self.panic_streak == PANIC_STREAK_TRIGGER:
            reason = "cornered_repeatedly"
        elif not progressed and step - self.last_progress_step >= STAGNATION_STEPS:
            reason = "stagnant"
            self.last_progress_step = step  # don't re-fire every tick
        else:
            reason = None

        self.prev_lives = state.lives
        self.prev_dots_eaten = state.dots_eaten
        self.prev_edible_any = edible_any
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
        if call.function.name == "adjust_policy":
            mode = args.get("mode")
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
    """

    def __init__(self, model: str, base_url: str, api_key: str,
                 log_path: Path, interval_s: float = 5.0,
                 min_interval_s: float = 2.0, timeout_s: float = 15.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None
        self._last_submit = float("-inf")
        self._interval_s = interval_s
        self._min_interval_s = min_interval_s
        self._timeout_s = timeout_s
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_invoke(self, frame: np.ndarray, snapshot: dict[str, Any],
                     trigger_reason: str | None) -> None:
        if self._pending is not None and not self._pending.done():
            return  # a call is already in flight; never queue a second one
        # an event may shorten the wait between calls, never lengthen it
        gate = (min(self._interval_s, self._min_interval_s)
                if trigger_reason else self._interval_s)
        now = time.monotonic()
        if now - self._last_submit < gate:
            return
        image_b64 = encode_frame_png_b64(frame)
        self._last_submit = now
        self._pending = self._executor.submit(
            self._invoke_once, image_b64, snapshot, trigger_reason or "interval")

    def poll(self) -> InvocationResult | None:
        if self._pending is None or not self._pending.done():
            return None
        result: InvocationResult = self._pending.result()  # worker never raises
        self._pending = None
        self._append_log(result)
        return result

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    # -- worker thread body -------------------------------------------------

    def _invoke_once(self, image_b64: str, snapshot: dict[str, Any],
                     trigger_reason: str) -> InvocationResult:
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
                                    ok=False, error=repr(exc))
        return InvocationResult(trigger_reason, time.monotonic() - started,
                                ok=True, **fields)

    def _append_log(self, result: InvocationResult) -> None:
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                  **asdict(result)}
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
