"""Runs a power-pill-baiting Ms. Pac-Man agent under an async LLM supervisor.

`policy.py` is imported and never modified: this script reuses its envpool
setup, its rule-based agent and its trial logging verbatim.  On top of that
it adds two things, in order of how much they are worth.

**1. Baiting power pills (`BaitingAgent`).**  Measured over 120 baseline
episodes, a power pill is worth 3000 points if all four ghosts are eaten on
it, and the planner was collecting 769 - 2.08 ghosts per pill.  Tracing the
edible phases showed why: only 2.75 ghosts are even blue when a pill goes
off, because the planner eats pills as soon as they are convenient, whether
or not there is anyone around to eat afterwards.  `BaitingAgent` keeps her
in the pill's neighbourhood instead - paying a small bonus on pellets near
an untaken pill - until enough ghosts have closed in.  Over 90 episodes
that is 11654 against the baseline's 10562 (a gain of 1.7 combined SEM),
and the mechanism shows up where it should: 2.30 ghosts per pill and +1686
points a game from ghosts, against -593 from the board-clearing it costs.

Simply devaluing pills, chasing one committed ghost per phase, and leading
the target were all tried and all lost to the baseline; see the module
docstring in `llm_agent.py` for the supervisor side of the same story.

**2. The supervisor (`llm_agent.AgentSupervisor`), rewired.**  It now runs
on a step cadence rather than a wall-clock one, so a headless evaluation
and a `--realtime` demo consult the model at the same points in the game;
it is told what it may not do (tactics - the round trip is longer than a
power-pill window); it is shown what its last change actually did; and a
guardrail reverts it to the validated defaults when a change is followed by
worse play.  Pass `--no-agent` for the bare rule-based policy.

The env still steps at a fixed cadence every tick.  The supervisor only
ever tunes `agent.config` between ticks from a background thread's
already-finished result, so a slow or stalled DashScope call can never
delay a step.

Run it with::

    python policy_with_agent.py --episodes 1 --render --realtime
    python policy_with_agent.py --episodes 10 --no-agent
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

# How long to watch the game after a strategy change before judging it, and
# how much worse than the pre-change rate counts as "this made things worse".
GUARD_WINDOW_STEPS = 400
GUARD_RATE_FLOOR = 0.60
# Below this many pellets the board is nearly clear and the score rate falls
# whatever the strategy is, so rate alone stops being evidence about it.
GUARD_MIN_PELLETS = 20


def _say(text: str) -> None:
    """Print, even when stdout is a codepage that cannot hold the notes."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc), flush=True)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class BaitingAgent(policy.MsPacmanAgent):
    """`MsPacmanAgent`, but it waits by a power pill for the ghosts to arrive.

    The base planner scores a power pill like any other item - value over
    distance - with the value already rising as dangerous ghosts come
    within `pill_lure_radius`.  What it has no way to express is *waiting*:
    if the pill is the best deal on the board it is taken immediately, and
    a pill taken with the ghosts scattered turns two of them blue at the far
    end of the maze and buys perhaps 600 points instead of 3000.

    Devaluing lone pills does not fix that - it was measured, and it merely
    sends her off to clear pellets elsewhere, taking 5.8 pills a game
    instead of 7.4 and scoring less overall.  What works is to leave the
    pill roughly where it was in the ranking but pay a bonus on the pellets
    *around* it, so the best local deal keeps her circling the pill,
    harvesting, while the ghosts converge on her.  When `bait_min_ghosts`
    of them are within `bait_radius`, the bonus stops and the pill goes
    back to being ordinary - by which point taking it is worth far more.
    """

    def __init__(self, config: policy.HeuristicConfig | None = None, *,
                 bait_enabled: bool = True,
                 bait_min_ghosts: int = 2,
                 bait_radius: int = 70,
                 bait_hold_value: float = 6.0,
                 bait_pellet_bonus: float = 9.0,
                 bait_pellet_radius: int = 3) -> None:
        self.bait_enabled = bait_enabled
        self.bait_min_ghosts = bait_min_ghosts
        self.bait_radius = bait_radius
        self.bait_hold_value = bait_hold_value
        self.bait_pellet_bonus = bait_pellet_bonus
        self.bait_pellet_radius = bait_pellet_radius
        super().__init__(config)

    def _ghosts_near(self, state: policy.GameState, r: int, c: int) -> int:
        """Dangerous ghosts within `bait_radius` of a cell, in RAM units."""
        return sum(1 for i, g in enumerate(state.ghosts)
                   if not state.edible[i]
                   and abs(g[0] - policy.COL_X[c]) + abs(g[1] - policy.ROW_Y[r])
                   <= self.bait_radius)

    def _candidates(self, state, my_dist, ghost_dist, edible_cells):
        out = super()._candidates(state, my_dist, ghost_dist, edible_cells)
        if not self.bait_enabled or any(state.edible) or not self.pills.any():
            return out

        reachable = my_dist < policy.UNREACHABLE
        held = {(int(r), int(c))
                for r, c in np.argwhere(self.pills & reachable)
                if self._ghosts_near(state, int(r), int(c)) < self.bait_min_ghosts}
        if not held:
            return out

        scale = self.config.distance_scale
        rebuilt: list[tuple[float, tuple[int, int]]] = []
        for score, cell in out:
            if cell in held:
                # not yet worth taking: keep it on the map, but barely
                score = self.bait_hold_value / (my_dist[cell] / scale + 1.0)
            elif self.pellets[cell]:
                near = min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in held)
                if near <= self.bait_pellet_radius:
                    score += self.bait_pellet_bonus / (my_dist[cell] / scale + 1.0)
            rebuilt.append((score, cell))
        return rebuilt


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------

class EpisodeStats:
    """Score, deaths and ghost-ladder progress, for the snapshot and log.

    Ghost eats are read off the reward stream rather than the agent's phase
    tracker: inside a power-pill phase the ladder pays 200/400/800/1600 in
    order, so matching the next rung against the step reward identifies an
    eat even when several rewards land in one frame-skipped step.
    """

    def __init__(self) -> None:
        self.score = 0.0
        self.deaths = 0
        self.pills_taken = 0
        self.ghosts_eaten = 0
        self.ghost_points = 0.0
        self._ladder = 0
        self._edible_prev = False
        self._lives: int | None = None
        self._history: list[tuple[int, float, int]] = [(0, 0.0, 0)]

    def before_step(self, agent, step: int) -> None:
        edible_now = agent.edible_seen > 0
        if edible_now and not self._edible_prev:
            self.pills_taken += 1
            self._ladder = 0
        self._edible_prev = edible_now

    def after_step(self, reward: float, lives: int, step: int) -> None:
        self.score += reward
        r = float(reward)
        if self._edible_prev:
            while self._ladder < 4 and r >= 200 * (2 ** self._ladder):
                r -= 200 * (2 ** self._ladder)
                self.ghost_points += 200 * (2 ** self._ladder)
                self.ghosts_eaten += 1
                self._ladder += 1
        if self._lives is not None and lives < self._lives:
            self.deaths += 1
        self._lives = lives
        self._history.append((step, self.score, self.deaths))

    def rate_between(self, lo: int, hi: int) -> float:
        """Score per 100 env steps over `[lo, hi)`, from the step history."""
        if hi <= lo:
            return 0.0
        start = self._at(lo)
        end = self._at(hi)
        return 100.0 * (end[1] - start[1]) / (end[0] - start[0] or 1)

    def deaths_between(self, lo: int, hi: int) -> int:
        return self._at(hi)[2] - self._at(lo)[2]

    def _at(self, step: int) -> tuple[int, float, int]:
        idx = min(max(step, 0), len(self._history) - 1)
        return self._history[idx]


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------

def _print_result(step: int, result) -> None:
    """Mirror every supervisor invocation to stdout, not just the JSONL log."""
    if not result.ok:
        _say(f"[policy] step={step} trigger={result.trigger} "
             f"ERROR: {result.error}")
        return
    for note in result.notes:
        _say(str(note.get("note")))
    change = (f" mode={result.mode} patch={result.config_patch}"
              if result.mode else " (no policy change)")
    _say(f"[policy] step={step} trigger={result.trigger} "
         f"latency={result.latency_s:.2f}s lag={result.lag_steps}steps{change}")


class Supervision:
    """Glue between the env loop and `llm_agent.AgentSupervisor`.

    Besides forwarding snapshots it owns the guardrail.  Every applied
    change is put on probation for `GUARD_WINDOW_STEPS`; if the score rate
    over that window falls below `GUARD_RATE_FLOOR` of the rate before the
    change, or the change is followed by a death that the previous window
    did not have, the config is put back to the validated defaults and the
    model is told so in the next snapshot.  A supervisor that cannot make
    things much worse is worth having even when it is not sure it helps;
    the previous version had no such floor, and spent 13 of its 21 changes
    cutting the safety margin by a third.

    `llm_agent` is imported here rather than at module scope so a run
    without `--agent` never needs the DashScope client or its dependencies.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        import llm_agent

        self._llm = llm_agent
        cfg = llm_agent.load_dashscope_config(SCRIPT_DIR / ".env")
        self._supervisor = llm_agent.AgentSupervisor(
            model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"],
            log_path=args.agent_log_path,
            interval_steps=args.agent_interval_steps,
            min_interval_steps=args.agent_min_interval_steps,
            timeout_s=args.agent_timeout_s,
        )
        self.reset()

    def reset(self) -> None:
        self._tracker = self._llm.EventTracker()
        self._notes: list[dict[str, str]] = []
        self.mode = DEFAULT_MODE
        self.changes = 0
        self.reverts = 0
        self._probation: dict[str, Any] | None = None
        self._last_change: dict[str, Any] | None = None
        self._guardrail: str | None = None
        self._lag_steps = -1

    # -- per-tick ---------------------------------------------------------

    def tick(self, agent, obs: np.ndarray, info: dict, step: int,
             stats: EpisodeStats) -> None:
        """Fold in any finished call, judge any probation, then offer this
        tick to the supervisor."""
        frame = policy.extract_latest_frame(obs)
        state = policy.decode_state(np.asarray(info["ram"])[0], frame,
                                    int(np.asarray(info["lives"])[0]))
        trigger = self._tracker.update(step, state, agent)

        self._judge_probation(agent, step, stats)

        result = self._supervisor.poll(step)
        if result is not None:
            _print_result(step, result)
            self._apply(agent, result, step, stats)

        snapshot = self._build_snapshot(state, agent, step, stats, trigger)
        self._supervisor.maybe_invoke(frame, snapshot, trigger, step)

    # -- guardrail --------------------------------------------------------

    def _apply(self, agent, result, step: int, stats: EpisodeStats) -> None:
        self._notes.extend(result.notes)
        del self._notes[:-MAX_RECENT_NOTES]
        self._lag_steps = result.lag_steps
        if not result.config_patch:
            return

        before = self._snapshot_config(agent)
        self._set_config(agent, result.config_patch)
        self.mode = result.mode or self.mode
        self.changes += 1
        window = min(GUARD_WINDOW_STEPS, max(1, step))
        self._probation = {
            "mode": self.mode, "at_step": step, "restore": before,
            "rate_before": stats.rate_between(step - window, step),
            "deaths_before": stats.deaths_between(step - window, step),
        }
        self._last_change = {"mode": self.mode, "at_step": step,
                             "verdict": "pending"}

    def _judge_probation(self, agent, step: int, stats: EpisodeStats) -> None:
        p = self._probation
        if p is None or step - p["at_step"] < GUARD_WINDOW_STEPS:
            return
        rate_after = stats.rate_between(p["at_step"], step)
        deaths_after = stats.deaths_between(p["at_step"], step)
        # A thinning board slows scoring on its own, and the mode most
        # likely to be running then is `clear_board` - the one chosen *for*
        # the endgame.  Judging it on rate there reverted it twice in a
        # single test episode and handed the endgame back to a baiting
        # config, so the rate test only applies while pellets remain.
        thin_board = int(agent.pellets.sum()) <= GUARD_MIN_PELLETS
        worse = deaths_after > p["deaths_before"] or (
            not thin_board and rate_after < p["rate_before"] * GUARD_RATE_FLOOR)
        self._last_change = {
            "mode": p["mode"], "at_step": p["at_step"],
            "score_rate_before": round(p["rate_before"], 1),
            "score_rate_after": round(rate_after, 1),
            "deaths_after": deaths_after,
            "verdict": "reverted" if worse else "kept",
        }
        if worse:
            self._set_config(agent, p["restore"])
            self.mode = DEFAULT_MODE
            self.reverts += 1
            self._guardrail = (
                f"step {step}: {p['mode']} 之后得分速率 "
                f"{p['rate_before']:.0f}->{rate_after:.0f}/100步"
                f"，死亡 {deaths_after} 次，已自动回退到默认参数。")
            _say(f"[guardrail] reverted {p['mode']} -> {DEFAULT_MODE} "
                 f"(rate {p['rate_before']:.0f}->{rate_after:.0f}/100 steps, "
                 f"deaths {deaths_after})")
        self._probation = None

    def _set_config(self, agent, patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if key in self._llm.AGENT_ATTRS:
                setattr(agent, key, bool(value))
            else:
                setattr(agent.config, key, value)

    def _snapshot_config(self, agent) -> dict[str, Any]:
        return {key: (getattr(agent, key) if key in self._llm.AGENT_ATTRS
                      else getattr(agent.config, key))
                for key in self._llm.PARAM_RANGES}

    # -- snapshot ---------------------------------------------------------

    def _build_snapshot(self, state, agent, step: int, stats: EpisodeStats,
                        trigger: str | None) -> dict[str, Any]:
        """What the model gets to reason over, alongside the frame.

        Deliberately heavy on measured consequence and light on things it
        can read off the picture: the point of the extra fields is to make
        it answer "did my last change help" from evidence rather than from
        the impression the frame gives.
        """
        window = min(GUARD_WINDOW_STEPS, max(1, step))
        return {
            "step": step,
            "score": stats.score,
            "lives": state.lives,
            "deaths": stats.deaths,
            "dots_eaten": state.dots_eaten,
            "pellets_left_on_board": int(agent.pellets.sum()),
            "power_pills_left": int(agent.pills.sum()),
            "pills_taken": stats.pills_taken,
            "ghosts_eaten": stats.ghosts_eaten,
            "ghosts_per_pill": round(
                stats.ghosts_eaten / stats.pills_taken, 2)
            if stats.pills_taken else None,
            "ghost_points": stats.ghost_points,
            "score_rate_per_100_steps": round(
                stats.rate_between(step - window, step), 1),
            "ghosts_edible": list(state.edible),
            "ghosts_flashing": state.flashing,
            "trigger_reason": trigger,
            "current_mode": self.mode,
            "current_config": self._snapshot_config(agent),
            "strategy_changes": self.changes,
            "guardrail_reverts": self.reverts,
            "guardrail": self._guardrail,
            "last_change": self._last_change,
            "decision_lag_steps": self._lag_steps,
            "recent_notes": self._notes,
        }

    def shutdown(self) -> None:
        self._supervisor.shutdown()


# ---------------------------------------------------------------------------
# CLI and run loop
# ---------------------------------------------------------------------------

def build_agent_parser() -> argparse.ArgumentParser:
    parser = policy.build_parser()
    # policy.py defaults to noop_max=1.  Measured over 60 episodes a seed
    # with no starting no-ops scores 7573 against 11044 for the standard
    # ALE randomisation, because a fixed start locks the agent into one
    # phase relative to the ghosts' movement cycle.
    parser.set_defaults(noop_max=30, repeat_action_probability=0.25)
    parser.add_argument("--agent", dest="agent", action="store_true",
                        default=True,
                        help="enable the async LLM strategy supervisor (default)")
    parser.add_argument("--no-agent", dest="agent", action="store_false",
                        help="run the rule-based policy alone, without the "
                             "LLM supervisor")
    parser.add_argument("--no-bait", dest="bait", action="store_false",
                        default=True,
                        help="disable power-pill baiting (the plain policy.py "
                             "behaviour)")
    parser.add_argument("--agent-interval-steps", type=int, default=150,
                        help="env steps between supervisor invocations; a step "
                             "cadence keeps headless and --realtime runs "
                             "consistent")
    parser.add_argument("--agent-min-interval-steps", type=int, default=60,
                        help="minimum gap between invocations even on key events")
    parser.add_argument("--agent-timeout-s", type=float, default=20.0,
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
    agent = BaitingAgent(
        policy.HeuristicConfig(safety_margin=args.safety_margin),
        bait_enabled=args.bait)
    supervision = Supervision(args) if args.agent else None

    scores: list[float] = []
    lengths: list[int] = []
    modes_at_end: list[str] = []
    ghosts_per_pill: list[float] = []
    total_changes = total_reverts = 0
    env_steps = 0
    try:
        for episode in range(args.episodes):
            obs, info = policy.reset_env_with_info(env)
            agent.reset()
            stats = EpisodeStats()
            steps = 0
            if supervision is not None:
                supervision.reset()
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                started = time.perf_counter()
                action = agent.act(obs, info)
                stats.before_step(agent, steps)
                if supervision is not None:
                    supervision.tick(agent, obs, info, steps, stats)

                obs, reward, done, info = policy.step_env(env, action)
                steps += 1
                env_steps += 1
                stats.after_step(float(reward),
                                 int(np.asarray(info["lives"])[0]), steps)
                if args.render:
                    env.render()
                if step_period_s > 0.0:
                    remaining = step_period_s - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
                if done:
                    break
            mode = supervision.mode if supervision is not None else DEFAULT_MODE
            if supervision is not None:
                total_changes += supervision.changes
                total_reverts += supervision.reverts
            scores.append(stats.score)
            lengths.append(steps)
            modes_at_end.append(mode)
            per_pill = (stats.ghosts_eaten / stats.pills_taken
                        if stats.pills_taken else 0.0)
            ghosts_per_pill.append(per_pill)
            _say(f"episode={episode} score={stats.score:.0f} steps={steps} "
                 f"deaths={stats.deaths} pills={stats.pills_taken} "
                 f"ghosts/pill={per_pill:.2f} mode={mode}")
    finally:
        env.close()
        if supervision is not None:
            supervision.shutdown()

    arr = np.asarray(scores, dtype=np.float64)
    mean, median = float(arr.mean()), float(np.median(arr))
    low, high = float(arr.min()), float(arr.max())
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    ale_frames = env_steps * args.frame_skip
    gpp = float(np.mean(ghosts_per_pill))
    _say(" ".join([
        "eval_summary:",
        f"episodes={len(scores)}",
        f"env_steps={env_steps}",
        f"ale_frames={ale_frames}",
        f"mean={mean:.1f}",
        f"sem={sem:.1f}",
        f"median={median:.1f}",
        f"min={low:.0f}",
        f"max={high:.0f}",
        f"ghosts_per_pill={gpp:.2f}",
        f"strategy_changes={total_changes}",
        f"guardrail_reverts={total_reverts}",
    ]))

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
            "score_sem": sem,
            "score_median": median,
            "score_min": low,
            "score_max": high,
            "ghosts_per_pill": gpp,
            "env_steps": env_steps,
            "ale_frames": ale_frames,
            "frame_skip": args.frame_skip,
            "noop_max": args.noop_max,
            "repeat_action_probability": args.repeat_action_probability,
            "safety_margin": args.safety_margin,
            "bait_enabled": args.bait,
            "agent_enabled": args.agent,
            "strategy_changes": total_changes,
            "guardrail_reverts": total_reverts,
        })
        policy.write_summary(log_path, Path(args.summary_path))


if __name__ == "__main__":
    run(build_agent_parser().parse_args())
