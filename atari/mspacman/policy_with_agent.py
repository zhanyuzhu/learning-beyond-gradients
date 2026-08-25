"""Runs `policy.MsPacmanAgent` with an optional async LLM strategy supervisor.

`policy.py` is imported, never modified: this script reuses its envpool
setup, the rule-based agent, and its trial-logging helpers verbatim, and adds
an `--agent` flag that layers `llm_agent.AgentSupervisor` on top. The env
still steps at a fixed cadence every tick; the supervisor only ever tunes
`agent.config` between ticks from a background thread's already-finished
result, so a slow or stalled DashScope call can never delay a step.

Run it with::

    python policy_with_agent.py --episodes 1 --render --realtime --agent
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import policy  # noqa: E402  (needs SCRIPT_DIR on sys.path first)

DEFAULT_AGENT_LOG_PATH = SCRIPT_DIR / "mspacman_agent_log.jsonl"
DEFAULT_MODE = "balanced"
MAX_RECENT_NOTES = 5


def _print_result(step: int, result) -> None:
    """Mirror every supervisor invocation to stdout, not just the JSONL log.

    The model's analysis first, then what it did to the policy.
    """
    if not result.ok:
        print(f"[policy] step={step} trigger={result.trigger} ERROR: {result.error}",
              flush=True)
        return
    for note in result.notes:
        print(note.get("note"), flush=True)
    change = (f" mode={result.mode} patch={result.config_patch}"
              if result.mode else " (no policy change)")
    print(f"[policy] step={step} trigger={result.trigger} "
          f"latency={result.latency_s:.2f}s{change}", flush=True)


class Supervision:
    """Glue between the env loop and `llm_agent.AgentSupervisor`.

    `llm_agent` is imported here rather than at module scope so a run without
    `--agent` never needs the DashScope client or its dependencies.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        import llm_agent

        self._llm = llm_agent
        cfg = llm_agent.load_dashscope_config(SCRIPT_DIR / ".env")
        self._supervisor = llm_agent.AgentSupervisor(
            model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"],
            log_path=args.agent_log_path,
            interval_s=args.agent_interval_s,
            min_interval_s=args.agent_min_interval_s,
            timeout_s=args.agent_timeout_s,
        )
        self.reset()

    def reset(self) -> None:
        self._tracker = self._llm.EventTracker()
        self._notes: list[dict[str, str]] = []
        self.mode = DEFAULT_MODE

    def tick(self, agent, obs: np.ndarray, info: dict, step: int,
             score: float) -> None:
        """Fold in any finished call, then offer this tick to the supervisor."""
        frame = policy.extract_latest_frame(obs)
        state = policy.decode_state(np.asarray(info["ram"])[0], frame,
                                    int(np.asarray(info["lives"])[0]))
        trigger = self._tracker.update(step, state, agent)

        result = self._supervisor.poll()
        if result is not None:
            _print_result(step, result)
            self._apply(agent, result)

        snapshot = self._llm.build_snapshot(state, agent, step, score,
                                            self.mode, trigger, self._notes)
        self._supervisor.maybe_invoke(frame, snapshot, trigger)

    def _apply(self, agent, result) -> None:
        if result.config_patch:
            for key, value in result.config_patch.items():
                setattr(agent.config, key, value)
            self.mode = result.mode or self.mode
        self._notes.extend(result.notes)
        del self._notes[:-MAX_RECENT_NOTES]

    def shutdown(self) -> None:
        self._supervisor.shutdown()


def build_agent_parser() -> argparse.ArgumentParser:
    parser = policy.build_parser()
    parser.add_argument("--agent", action="store_true",
                        help="enable the async LLM strategy supervisor")
    parser.add_argument("--agent-interval-s", type=float, default=5.0,
                        help="fixed cadence between supervisor invocations")
    parser.add_argument("--agent-min-interval-s", type=float, default=2.0,
                        help="minimum gap between invocations even on key events")
    parser.add_argument("--agent-timeout-s", type=float, default=15.0,
                        help="per-call DashScope request timeout")
    parser.add_argument("--agent-log-path", default=str(DEFAULT_AGENT_LOG_PATH))
    return parser


def run(args: argparse.Namespace) -> None:
    import envpool

    env_kwargs: dict[str, Any] = dict(
        num_envs=1,
        batch_size=1,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        img_height=210,
        img_width=160,
        stack_num=1,
        gray_scale=False,
        frame_skip=args.frame_skip,
        noop_max=args.noop_max,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=False,
    )
    if args.render:
        env_kwargs["render_mode"] = "human"
    env = envpool.make_gym("MsPacman-v5", **env_kwargs)

    step_period_s = args.frame_skip / policy.ALE_FPS if args.realtime else 0.0
    agent = policy.MsPacmanAgent(
        policy.HeuristicConfig(safety_margin=args.safety_margin))
    supervision = Supervision(args) if args.agent else None

    scores: list[float] = []
    lengths: list[int] = []
    modes_at_end: list[str] = []
    env_steps = 0
    try:
        for episode in range(args.episodes):
            obs, info = policy.reset_env_with_info(env)
            agent.reset()
            total, steps = 0.0, 0
            if supervision is not None:
                supervision.reset()
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                started = time.perf_counter()
                action = agent.act(obs, info)
                if supervision is not None:
                    supervision.tick(agent, obs, info, steps, total)

                obs, reward, done, info = policy.step_env(env, action)
                total += reward
                steps += 1
                env_steps += 1
                if args.render:
                    env.render()
                if step_period_s > 0.0:
                    remaining = step_period_s - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                if done:
                    break
            mode = supervision.mode if supervision is not None else DEFAULT_MODE
            scores.append(total)
            lengths.append(steps)
            modes_at_end.append(mode)
            print(f"episode={episode} score={total:.0f} steps={steps} mode={mode}")
    finally:
        env.close()
        if supervision is not None:
            supervision.shutdown()

    arr = np.asarray(scores, dtype=np.float64)
    mean, median = float(arr.mean()), float(np.median(arr))
    low, high = float(arr.min()), float(arr.max())
    ale_frames = env_steps * args.frame_skip
    print("eval_summary:",
          f"episodes={len(scores)}",
          f"env_steps={env_steps}",
          f"ale_frames={ale_frames}",
          f"mean={mean:.1f}",
          f"median={median:.1f}",
          f"min={low:.0f}",
          f"max={high:.0f}")

    if args.log_path:
        log_path = Path(args.log_path)
        policy.append_trial_record(log_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "episodes": len(scores),
            "scores": scores,
            "episode_lengths": lengths,
            "modes_at_end": modes_at_end,
            "score_mean": mean,
            "score_median": median,
            "score_min": low,
            "score_max": high,
            "env_steps": env_steps,
            "ale_frames": ale_frames,
            "frame_skip": args.frame_skip,
            "noop_max": args.noop_max,
            "safety_margin": args.safety_margin,
            "agent_enabled": args.agent,
        })
        policy.write_summary(log_path, Path(args.summary_path))


if __name__ == "__main__":
    run(build_agent_parser().parse_args())
