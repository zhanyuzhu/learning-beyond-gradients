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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import policy  # noqa: E402  (needs SCRIPT_DIR on sys.path first)

DEFAULT_AGENT_LOG_PATH = SCRIPT_DIR / "mspacman_agent_log.jsonl"
MAX_RECENT_NOTES = 5


def _print_agent_result(step: int, result) -> None:
    """Mirror every supervisor invocation to stdout, not just the JSONL log."""
    if not result.ok:
        print(f"[agent] step={step} trigger={result.trigger} ERROR: {result.error}",
              flush=True)
        return
    headline = f"[agent] step={step} trigger={result.trigger} latency={result.latency_s:.2f}s"
    if result.mode:
        headline += f" mode={result.mode} patch={result.config_patch}"
    else:
        headline += " (no policy change)"
    print(headline, flush=True)
    for note in result.notes:
        print(f"  [agent:{note.get('category')}] {note.get('note')}", flush=True)


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
    agent = policy.MsPacmanAgent(policy.HeuristicConfig(safety_margin=args.safety_margin))

    supervisor = None
    if args.agent:
        import llm_agent

        dash_cfg = llm_agent.load_dashscope_config(SCRIPT_DIR / ".env")
        supervisor = llm_agent.AgentSupervisor(
            model=dash_cfg["model"], base_url=dash_cfg["base_url"],
            api_key=dash_cfg["api_key"], log_path=args.agent_log_path,
            interval_s=args.agent_interval_s,
            min_interval_s=args.agent_min_interval_s,
            timeout_s=args.agent_timeout_s,
        )

    scores: list[float] = []
    lengths: list[int] = []
    modes_at_end: list[str] = []
    env_steps = 0
    try:
        for episode in range(args.episodes):
            obs, info = policy.reset_env_with_info(env)
            agent.reset()
            tracker = llm_agent.EventTracker() if supervisor is not None else None
            recent_notes: list[dict[str, str]] = []
            mode = "balanced"
            total, steps = 0.0, 0
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                started = time.perf_counter()
                action = agent.act(obs, info)

                if supervisor is not None:
                    ram = np.asarray(info["ram"])[0]
                    lives = int(np.asarray(info["lives"])[0]) if "lives" in info else 0
                    frame = policy.extract_latest_frame(obs)
                    state = policy.decode_state(ram, frame, lives)
                    trigger = tracker.update(steps, state, agent)

                    result = supervisor.poll()
                    if result is not None:
                        _print_agent_result(steps, result)
                        if result.config_patch:
                            for key, value in result.config_patch.items():
                                setattr(agent.config, key, value)
                            mode = result.mode or mode
                        if result.notes:
                            recent_notes.extend(result.notes)
                            del recent_notes[:-MAX_RECENT_NOTES]

                    snapshot = llm_agent.build_snapshot(
                        state, agent, steps, total, mode, trigger, recent_notes)
                    supervisor.maybe_invoke(frame, snapshot, trigger)

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
            scores.append(total)
            lengths.append(steps)
            modes_at_end.append(mode)
            print(f"episode={episode} score={total:.0f} steps={steps} mode={mode}")
    finally:
        env.close()
        if supervisor is not None:
            supervisor.shutdown()

    arr = np.asarray(scores, dtype=np.float64)
    print("eval_summary:",
          f"episodes={len(scores)}",
          f"env_steps={env_steps}",
          f"ale_frames={env_steps * args.frame_skip}",
          f"mean={arr.mean():.1f}",
          f"median={float(np.median(arr)):.1f}",
          f"min={arr.min():.0f}",
          f"max={arr.max():.0f}")

    if args.log_path:
        log_path = Path(args.log_path)
        policy.append_trial_record(log_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "episodes": len(scores),
            "scores": scores,
            "episode_lengths": lengths,
            "modes_at_end": modes_at_end,
            "score_mean": float(arr.mean()),
            "score_median": float(np.median(arr)),
            "score_min": float(arr.min()),
            "score_max": float(arr.max()),
            "env_steps": env_steps,
            "ale_frames": env_steps * args.frame_skip,
            "frame_skip": args.frame_skip,
            "noop_max": args.noop_max,
            "safety_margin": args.safety_margin,
            "agent_enabled": args.agent,
        })
        policy.write_summary(log_path, Path(args.summary_path))


if __name__ == "__main__":
    run(build_agent_parser().parse_args())
