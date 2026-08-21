# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision-targeted Breakout policies: aim the paddle contact point at
whichever remaining brick is currently reachable with the highest
empirical hit probability, instead of just intercepting wherever the ball
naturally reflects.

`heuristic_breakout.py`'s policies only reflect the ball off the side
walls and move the paddle to the predicted interception point - they never
choose *where on the paddle* to hit the ball, so the last few bricks only
get cleared by incidental collisions.

This script adds that missing control axis, as two agents (see `../../`
project memory for the full investigation across 5 rounds):

- `StuckAimAgent` (**the recommended, validated policy - default for the
  CLI below**): leaves `heuristic_breakout.py`'s normal ball tracking
  completely untouched (it's already reliable) and only replaces its
  existing stuck-offset escape hatch - which fires rarely, only once
  nothing's been hit for `stuck_trigger_steps` - with probability-aimed
  targeting instead of blind alternation. Validated in a 15-seed
  (`noop_max=30`) comparison to clear 12/15 vs. baseline's 8/15, +7.4%
  mean reward.
- `TunnelBreakoutAgent`: an earlier, more invasive whole-scale replacement
  (always-on targeting once few bricks remain, with an explicit
  tunnel-digging phase). Kept for comparison/research; never validated to
  beat baseline overall despite locally-correct mechanics. Select with
  `--policy tunnel`.

Two things had to be true for probability-aimed targeting to help instead
of hurt, learned the hard way across earlier rounds:

1. The outgoing bounce angle is a noisy, close-to-discrete function of the
   paddle contact offset *and* the ball's current speed (confirmed by
   direct probing - ALE Breakout speeds the ball up as play progresses,
   and the same offset reaches a much steeper angle on a slow ball than a
   fast one). So instead of solving a bank-shot angle, this looks up the
   offset that empirically sends the ball closest to the target most
   often *at the current ball speed*, from real calibration data
   (`paddle_bounce_calibration.csv`, collected by
   `calibrate_paddle_bounce.py`).
2. The target (and its chosen offset) must be locked once per ball
   descent, not recomputed every frame - recomputing every frame lets the
   choice flicker and whipsaws the paddle. And candidate offsets must stay
   safely inside the paddle's physical half-width, leaving room for the
   paddle controller's own tracking deadband - an offset right at the
   edge plus a few pixels of ordinary control slop is an easy way to whiff
   a catch entirely.

Reuses `heuristic_breakout.py`'s RAM decoding, action map, and env/logging
plumbing; only the paddle-targeting logic is new.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heuristic_breakout as hb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = SCRIPT_DIR / "paddle_bounce_calibration.csv"
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_breakout_tunnel_trials.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "heuristic_breakout_tunnel_trials_summary.csv"

# Brick playfield geometry (pixel space) - the brick field tiles exactly
# into a 6x18 grid, so a reshape-and-sum segments it without a connected
# components pass.
BRICK_FIELD_TOP, BRICK_FIELD_BOTTOM = 57.0, 93.0
BRICK_FIELD_LEFT, BRICK_FIELD_RIGHT = 8.0, 152.0
BRICK_GRID_ROWS, BRICK_GRID_COLS = 6, 18
BRICK_ROW_HEIGHT = (BRICK_FIELD_BOTTOM - BRICK_FIELD_TOP) / BRICK_GRID_ROWS
BRICK_COL_WIDTH = (BRICK_FIELD_RIGHT - BRICK_FIELD_LEFT) / BRICK_GRID_COLS
BRICK_CELL_PIXEL_THRESHOLD = 12

_BACKGROUND_RGB = np.asarray([0, 0, 0], dtype=np.uint8)
_WALL_RGB = np.asarray([142, 142, 142], dtype=np.uint8)

PADDLE_HALF_WIDTH_PX = 8.0  # paddle is 16px wide
PADDLE_DEADBAND_PX = 3.0  # matches HeuristicConfig.paddle_deadband_px default


def detect_brick_grid(obs: np.ndarray) -> np.ndarray:
    """Return the `(BRICK_GRID_ROWS, BRICK_GRID_COLS)` brick occupancy grid."""
    frame = hb.extract_latest_frame(obs)
    top, bottom = int(BRICK_FIELD_TOP), int(BRICK_FIELD_BOTTOM)
    left, right = int(BRICK_FIELD_LEFT), int(BRICK_FIELD_RIGHT)
    region = frame[top:bottom, left:right]
    mask = np.logical_and(
        ~np.all(region == _BACKGROUND_RGB, axis=-1),
        ~np.all(region == _WALL_RGB, axis=-1),
    )
    row_px = bottom - top
    col_px = right - left
    counts = mask.reshape(
        BRICK_GRID_ROWS, row_px // BRICK_GRID_ROWS,
        BRICK_GRID_COLS, col_px // BRICK_GRID_COLS,
    ).sum(axis=(1, 3))
    return counts >= BRICK_CELL_PIXEL_THRESHOLD


@dataclass(frozen=True)
class BounceCalibration:
    """Empirical (offset, incoming_speed, outgoing_angle) samples from real
    paddle contacts.

    Ball speed matters as much as offset: at a fixed offset, a slow ball
    reaches a much steeper outgoing angle than a fast one (confirmed
    directly on this data - e.g. offset in [2,5]px averages +53 degrees
    below speed 2.5 px/step, but only +31 degrees at speed >= 3.5, and
    that gap holds across every offset bucket). ALE Breakout speeds the
    ball up as play progresses, so conditioning only on offset silently
    fits whichever speed tier is most common in typical play and is wrong
    for the rest - including, often, the late-game high-speed regime
    where aiming matters most. So every lookup here also takes the
    current ball speed and restricts to nearby-speed samples, not just
    nearby-offset ones.
    """

    offsets: np.ndarray
    in_speeds: np.ndarray
    out_angles: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "BounceCalibration":
        offsets, in_speeds, out_angles = [], [], []
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                offsets.append(float(row["offset"]))
                in_speeds.append(float(row["in_speed"]))
                out_angles.append(float(row["out_angle_deg"]))
        return cls(np.asarray(offsets), np.asarray(in_speeds), np.asarray(out_angles))

    def candidate_offsets(self, safe_max_offset: float, step: float = 0.5) -> np.ndarray:
        return np.arange(-safe_max_offset, safe_max_offset + 1e-9, step)

    def ranked_offsets_for_windows(
        self,
        windows: list[tuple[float, float]],
        candidates: np.ndarray,
        ball_speed: float,
        kde_half_width: float = 1.0,
        speed_half_width: float = 0.4,
        min_samples: int = 12,
    ) -> list[tuple[float, float]]:
        """`[(hit_probability, offset), ...]`, best first, for offsets with
        enough nearby (offset AND speed) calibration samples and nonzero
        hit probability."""
        speed_mask = np.abs(self.in_speeds - ball_speed) <= speed_half_width
        scored = []
        for off in candidates:
            mask = speed_mask & (np.abs(self.offsets - off) <= kde_half_width)
            if mask.sum() < min_samples:
                continue
            angles = self.out_angles[mask]
            hit = np.zeros(len(angles), dtype=bool)
            for lo, hi in windows:
                hit |= (angles >= lo) & (angles <= hi)
            p = float(hit.mean())
            if p > 0.0:
                scored.append((p, float(off)))
        scored.sort(key=lambda t: -t[0])
        return scored


def feasible_angle_windows(
    intercept_x: float,
    target_col: int,
    target_row: int,
    paddle_y: float,
    field_left: float,
    field_right: float,
) -> list[tuple[float, float]]:
    """Launch-angle intervals (degrees off vertical) from `intercept_x`
    that land somewhere inside the target brick's column, allowing for 0
    or 1 bounce off either side wall (checked with one extra period on
    each side, to cover paddle positions near a wall)."""
    col_lo = BRICK_FIELD_LEFT + target_col * BRICK_COL_WIDTH
    col_hi = col_lo + BRICK_COL_WIDTH
    target_y = BRICK_FIELD_TOP + (target_row + 0.5) * BRICK_ROW_HEIGHT
    vertical_span = paddle_y - target_y
    if vertical_span <= 0:
        return []
    span = field_right - field_left
    windows = []
    for k in (-1, 0, 1):
        for lo_x, hi_x in (
            (col_lo + k * 2 * span, col_hi + k * 2 * span),
            (2 * field_left - col_hi + k * 2 * span, 2 * field_left - col_lo + k * 2 * span),
        ):
            th_lo = np.degrees(np.arctan2(lo_x - intercept_x, vertical_span))
            th_hi = np.degrees(np.arctan2(hi_x - intercept_x, vertical_span))
            windows.append((min(th_lo, th_hi), max(th_lo, th_hi)))
    return windows


@dataclass(frozen=True)
class TunnelConfig:
    tunnel_col: int = 0
    safe_max_offset_px: float = PADDLE_HALF_WIDTH_PX - PADDLE_DEADBAND_PX - 1.0
    min_hit_probability: float = 0.05
    top_k: int = 5
    max_cycles_before_bailout: int = 3
    # With 100+ bricks on screen almost any paddle position naturally hits
    # something, so forcing the paddle off its natural interception point
    # to chase a specific brick has no upside there and real downside
    # (opportunity cost of not taking the free natural hit, plus control
    # overhead) - confirmed empirically: aiming active from move 1
    # performs worse than plain baseline. Only engage aiming once the
    # brick field is sparse enough that natural reflection stops finding
    # targets on its own.
    aim_activation_remaining: int = 30


class TunnelBreakoutAgent(hb.RamBreakoutAgent):
    """RAM-decoded ball/paddle state (accurate) + a vision-detected brick
    grid (for target selection) + probability-calibrated paddle-offset
    aiming (see module docstring). Falls back to the unmodified base
    policy whenever no calibrated offset has meaningful hit probability
    for the current target, or the aim system has already failed
    repeatedly on the same recurring target."""

    def __init__(
        self,
        action_map: hb.ActionMap,
        config: hb.HeuristicConfig,
        tunnel_config: TunnelConfig,
        calibration: BounceCalibration,
    ) -> None:
        super().__init__(action_map, config)
        self._tunnel_config = tunnel_config
        self._calibration = calibration
        self._candidates = calibration.candidate_offsets(tunnel_config.safe_max_offset_px)
        self.tunnel_open = False
        self._was_descending = False
        self._locked_aiming = False
        self._locked_offset = 0.0
        self._last_target: tuple[int, int] | None = None
        self._attempt_index = 0
        self._last_obs: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.tunnel_open = False
        self._was_descending = False
        self._locked_aiming = False
        self._locked_offset = 0.0
        self._last_target = None
        self._attempt_index = 0

    def act(self, info: dict[str, np.ndarray], obs: np.ndarray | None = None) -> int:
        if obs is not None:
            self._last_obs = obs
        return super().act(info)

    def _select_target(
        self, grid: np.ndarray, intercept_x: float
    ) -> tuple[int, int] | None:
        if not self.tunnel_open:
            col = self._tunnel_config.tunnel_col
            rows_with_bricks = np.nonzero(grid[:, col])[0]
            if len(rows_with_bricks) > 0:
                # closest remaining brick in the tunnel column to the
                # paddle - shortest, most reliable shot to keep digging.
                return int(rows_with_bricks.max()), col
            self.tunnel_open = True
        rows, cols = np.nonzero(grid)
        if len(rows) == 0:
            return None
        col_centers = BRICK_FIELD_LEFT + (cols.astype(np.float64) + 0.5) * BRICK_COL_WIDTH
        idx = int(np.argmin(np.abs(col_centers - intercept_x)))
        return int(rows[idx]), int(cols[idx])

    def _choose_offset(
        self, intercept_x: float, ball_speed: float, target: tuple[int, int]
    ) -> float | None:
        tc = self._tunnel_config
        if target == self._last_target:
            self._attempt_index += 1
        else:
            self._last_target = target
            self._attempt_index = 0

        windows = feasible_angle_windows(
            intercept_x, target[1], target[0],
            self._config.paddle_y, self._config.field_left, self._config.field_right,
        )
        if not windows:
            return None
        ranked = self._calibration.ranked_offsets_for_windows(
            windows, self._candidates, ball_speed
        )
        ranked = [(p, off) for p, off in ranked if p >= tc.min_hit_probability]
        if not ranked:
            return None
        top_k = ranked[: tc.top_k]
        # Deterministic replay: repeating the single best guess against the
        # same recurring approach state can lock into a stable miss forever
        # if the offset-only model is missing some hidden state (e.g.
        # incoming angle). Cycle through good-but-different candidates
        # across repeated attempts instead; after enough failed cycles,
        # bail out to unbiased natural reflection so real trajectory
        # diversity has a chance to break the loop.
        if self._attempt_index >= tc.max_cycles_before_bailout * len(top_k):
            return None
        _, offset = top_k[self._attempt_index % len(top_k)]
        return offset

    def _target_paddle_x(self, detections: hb.BreakoutRamDetections) -> float:
        if detections.ball_xy is None:
            self._was_descending = False
            return self._config.home_x

        ball_x, ball_y = detections.ball_xy
        velocity = self._state.velocity_xy
        if velocity is None:
            self._was_descending = False
            return hb.clip(
                ball_x, lower=self._config.paddle_min_x, upper=self._config.paddle_max_x
            )

        vx, vy = velocity
        if vy > 0.1 and ball_y <= self._config.paddle_y:
            steps_to_paddle = max((self._config.paddle_y - ball_y) / vy, 0.0)
            intercept_x = hb.reflect_position(
                ball_x + vx * steps_to_paddle,
                lower=self._config.field_left,
                upper=self._config.field_right,
            )
            if not self._was_descending:
                self._was_descending = True
                self._locked_aiming = False
                self._locked_offset = 0.0
                grid = detect_brick_grid(self._last_obs) if self._last_obs is not None else None
                remaining = int(grid.sum()) if grid is not None else 0
                if (
                    grid is not None
                    and 0 < remaining <= self._tunnel_config.aim_activation_remaining
                ):
                    target = self._select_target(grid, intercept_x)
                    if target is not None:
                        ball_speed = float(np.hypot(vx, vy))
                        offset = self._choose_offset(intercept_x, ball_speed, target)
                        if offset is not None:
                            self._locked_offset = offset
                            self._locked_aiming = True
            if self._locked_aiming:
                return hb.clip(
                    intercept_x + self._locked_offset,
                    lower=self._config.paddle_min_x,
                    upper=self._config.paddle_max_x,
                )
            target_x = (
                intercept_x
                + self._state.launch_sign * self._config.tunnel_offset_px
                + self._current_stuck_offset_px(steps_to_paddle, detections.brick_balance)
            )
        elif vy >= self._config.fast_ball_min_vy:
            self._was_descending = False
            target_x = ball_x + self._config.fast_low_ball_lead_steps * vx
        else:
            self._was_descending = False
            target_x = ball_x + self._config.chase_lead_steps * vx

        return hb.clip(
            target_x, lower=self._config.paddle_min_x, upper=self._config.paddle_max_x
        )


class StuckAimAgent(hb.RamBreakoutAgent):
    """Baseline `RamBreakoutAgent`, unchanged, *except*: when the baseline's
    own stuck-offset escape hatch (`_current_stuck_offset_px`, which only
    fires once `steps_since_reward >= stuck_trigger_steps`) kicks in,
    replace its blind alternating direction/magnitude with a
    probability-calibrated offset aimed at the nearest reachable remaining
    brick, falling back to the original blind logic whenever no calibrated
    candidate has decent hit probability.

    This is a much narrower change than `TunnelBreakoutAgent`: normal ball
    tracking is never touched, since the baseline is already reliable at
    that - only the rare "nothing's been hit in a while" recovery nudge
    gets smarter. Recomputes its choice on the same cadence as the
    original (only when `stuck_offset_index` changes, not every frame),
    so it can't whipsaw the paddle either.
    """

    def __init__(
        self,
        action_map: hb.ActionMap,
        config: hb.HeuristicConfig,
        tunnel_config: TunnelConfig,
        calibration: BounceCalibration,
    ) -> None:
        super().__init__(action_map, config)
        self._tc = tunnel_config
        self._calibration = calibration
        self._candidates = calibration.candidate_offsets(tunnel_config.safe_max_offset_px)
        self._last_intercept_x: float | None = None
        self._last_ball_speed: float | None = None
        self._last_obs: np.ndarray | None = None
        self._cached_phase_index: int | None = None
        self._cached_offset: float | None = None
        self._last_target: tuple[int, int] | None = None
        self._attempt_index = 0

    def reset(self) -> None:
        super().reset()
        self._last_intercept_x = None
        self._last_ball_speed = None
        self._cached_phase_index = None
        self._cached_offset = None
        self._last_target = None
        self._attempt_index = 0

    def act(self, info: dict[str, np.ndarray], obs: np.ndarray | None = None) -> int:
        if obs is not None:
            self._last_obs = obs
        return super().act(info)

    def _target_paddle_x(self, detections: hb.BreakoutRamDetections) -> float:
        if detections.ball_xy is None:
            return self._config.home_x
        ball_x, ball_y = detections.ball_xy
        velocity = self._state.velocity_xy
        if velocity is None:
            return hb.clip(
                ball_x, lower=self._config.paddle_min_x, upper=self._config.paddle_max_x
            )
        vx, vy = velocity
        if vy > 0.1 and ball_y <= self._config.paddle_y:
            steps_to_paddle = max((self._config.paddle_y - ball_y) / vy, 0.0)
            intercept_x = hb.reflect_position(
                ball_x + vx * steps_to_paddle,
                lower=self._config.field_left,
                upper=self._config.field_right,
            )
            self._last_intercept_x = intercept_x
            self._last_ball_speed = float(np.hypot(vx, vy))
            target_x = (
                intercept_x
                + self._state.launch_sign * self._config.tunnel_offset_px
                + self._current_stuck_offset_px(steps_to_paddle, detections.brick_balance)
            )
        elif vy >= self._config.fast_ball_min_vy:
            target_x = ball_x + self._config.fast_low_ball_lead_steps * vx
        else:
            target_x = ball_x + self._config.chase_lead_steps * vx
        return hb.clip(
            target_x, lower=self._config.paddle_min_x, upper=self._config.paddle_max_x
        )

    def _current_stuck_offset_px(self, steps_to_paddle: float, brick_balance: float) -> float:
        if (
            self._state.steps_since_reward < self._config.stuck_trigger_steps
            or self._config.stuck_offset_px == 0.0
        ):
            self._cached_phase_index = None
            return 0.0

        phase_index = self._state.stuck_offset_index
        if phase_index != self._cached_phase_index:
            self._cached_phase_index = phase_index
            self._cached_offset = self._compute_aim_offset()

        offset = (
            self._cached_offset
            if self._cached_offset is not None
            else self._blind_offset(phase_index, brick_balance)
        )

        if (
            self._state.episode_score < self._config.brick_balance_bias_min_score
            or self._config.stuck_release_horizon_steps <= 0.0
        ):
            return offset
        release_ratio = hb.clip(
            steps_to_paddle / self._config.stuck_release_horizon_steps, lower=0.0, upper=1.0
        )
        return release_ratio * offset

    def _blind_offset(self, phase_index: int, brick_balance: float) -> float:
        """The original baseline logic, used whenever no calibrated
        candidate is confident enough - preserves the existing safety
        net exactly."""
        phase = phase_index % 4
        if phase == 0:
            direction, magnitude = 1.0, self._config.stuck_offset_px
        elif phase == 1:
            direction, magnitude = -1.0, self._config.stuck_offset_px
        elif phase == 2:
            direction, magnitude = 1.0, 0.5 * self._config.stuck_offset_px
        else:
            direction, magnitude = -1.0, 0.5 * self._config.stuck_offset_px
        if self._state.episode_score >= self._config.brick_balance_bias_min_score:
            if brick_balance > self._config.brick_balance_deadzone:
                direction = 1.0
            elif brick_balance < -self._config.brick_balance_deadzone:
                direction = -1.0
        return direction * magnitude

    def _compute_aim_offset(self) -> float | None:
        if self._last_obs is None or self._last_intercept_x is None:
            return None
        grid = detect_brick_grid(self._last_obs)
        if grid is None or not grid.any():
            return None
        rows, cols = np.nonzero(grid)
        col_centers = BRICK_FIELD_LEFT + (cols.astype(np.float64) + 0.5) * BRICK_COL_WIDTH
        idx = int(np.argmin(np.abs(col_centers - self._last_intercept_x)))
        target = (int(rows[idx]), int(cols[idx]))
        if target == self._last_target:
            self._attempt_index += 1
        else:
            self._last_target = target
            self._attempt_index = 0

        windows = feasible_angle_windows(
            self._last_intercept_x, target[1], target[0],
            self._config.paddle_y, self._config.field_left, self._config.field_right,
        )
        if not windows:
            return None
        ball_speed = self._last_ball_speed if self._last_ball_speed is not None else 2.5
        ranked = self._calibration.ranked_offsets_for_windows(
            windows, self._candidates, ball_speed
        )
        ranked = [(p, o) for p, o in ranked if p >= self._tc.min_hit_probability]
        if not ranked:
            return None
        top_k = ranked[: self._tc.top_k]
        if self._attempt_index >= self._tc.max_cycles_before_bailout * len(top_k):
            return None
        _, offset = top_k[self._attempt_index % len(top_k)]
        return offset


def evaluate_tunnel_policy(args: argparse.Namespace) -> None:
    """Run the tunnel policy and append one trial record."""
    import envpool

    env_kwargs: dict[str, Any] = dict(
        num_envs=1,
        batch_size=1,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        img_height=args.img_height,
        img_width=args.img_width,
        stack_num=1,
        gray_scale=False,
        frame_skip=args.frame_skip,
        noop_max=args.noop_max,
        use_fire_reset=not args.disable_fire_reset,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=False,
    )
    if args.render:
        env_kwargs["render_mode"] = "human"
    env = envpool.make_gym("Breakout-v5", **env_kwargs)
    step_period_s = args.frame_skip / hb.ALE_FPS if args.realtime else 0.0

    config = hb.HeuristicConfig(
        stuck_trigger_steps=args.stuck_trigger_steps,
        stuck_switch_steps=args.stuck_switch_steps,
        stuck_offset_px=args.stuck_offset,
    )
    tunnel_config = TunnelConfig(
        tunnel_col=args.tunnel_col,
        safe_max_offset_px=args.safe_max_offset,
        min_hit_probability=args.min_hit_probability,
        aim_activation_remaining=args.aim_activation_remaining,
    )
    calibration = BounceCalibration.load(Path(args.calibration_path))
    if args.policy == "stuck-aim":
        agent = StuckAimAgent(hb.ActionMap(), config, tunnel_config, calibration)
    else:
        agent = TunnelBreakoutAgent(hb.ActionMap(), config, tunnel_config, calibration)

    scores = []
    episode_lengths = []
    env_steps = 0
    episodes_started = 0

    try:
        for episode in range(args.episodes):
            obs, info = hb.reset_env_with_info(env)
            agent.reset()
            episodes_started += 1
            total_reward = 0.0
            episode_steps = 0
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                step_start = time.perf_counter()
                action = agent.act(info, obs)
                obs, reward, done, info = hb.step_env(env, action)
                agent.observe_reward(reward)
                total_reward += reward
                episode_steps += 1
                env_steps += 1
                if args.render:
                    env.render()
                if step_period_s > 0.0:
                    remaining = step_period_s - (time.perf_counter() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
                if done:
                    break
            scores.append(total_reward)
            episode_lengths.append(episode_steps)
            print(f"episode={episode} score={total_reward:.1f} steps={episode_steps}")
    finally:
        env.close()

    score_arr = np.asarray(scores, dtype=np.float32)
    print(
        "eval_summary:",
        f"episodes={len(scores)}",
        f"env_steps={env_steps}",
        f"ale_frames={env_steps * args.frame_skip}",
        f"mean={score_arr.mean():.3f}",
        f"min={score_arr.min():.1f}",
        f"max={score_arr.max():.1f}",
    )

    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)
    hb.append_trial_record(
        log_path,
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "trial_name": args.trial_name,
            "kind": "eval",
            "game": "Breakout-v5",
            "policy": args.policy,
            "num_envs": 1,
            "seed": args.seed,
            "frame_skip": args.frame_skip,
            "episodes_started": episodes_started,
            "episodes_finished": len(scores),
            "env_steps": env_steps,
            "ale_frames": env_steps * args.frame_skip,
            "score_mean": float(score_arr.mean()),
            "score_min": float(score_arr.min()),
            "score_max": float(score_arr.max()),
            "episode_scores": scores,
            "episode_lengths": episode_lengths,
            "config": {
                "tunnel_col": args.tunnel_col,
                "safe_max_offset": args.safe_max_offset,
                "min_hit_probability": args.min_hit_probability,
                "aim_activation_remaining": args.aim_activation_remaining,
                "stuck_trigger_steps": args.stuck_trigger_steps,
                "stuck_switch_steps": args.stuck_switch_steps,
                "stuck_offset": args.stuck_offset,
                "noop_max": args.noop_max,
                "repeat_action_probability": args.repeat_action_probability,
            },
            "notes": args.notes,
        },
    )
    rows = hb.write_summary(log_path, summary_path)
    hb.print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tunnel-digging + probability-aimed Breakout policy."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=27000,
        help="Used both for env max_episode_steps and the Python rollout loop.",
    )
    parser.add_argument("--img-height", type=int, default=210)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=True,
        help="Show the game in an opencv 'human' render window.",
    )
    parser.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True,
        help="Pace playback to real wall-clock speed. Disable for fast eval runs.",
    )
    parser.add_argument("--disable-fire-reset", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("stuck-aim", "tunnel"),
        default="stuck-aim",
        help=(
            "'stuck-aim' (default, recommended): only replaces the baseline's "
            "existing stuck-offset escape hatch with probability-aimed "
            "targeting, leaving normal tracking untouched - validated to beat "
            "baseline in multi-seed testing (12/15 vs 8/15 cleared). 'tunnel': "
            "the earlier whole-scale tunnel-digging replacement, kept for "
            "comparison/research - not validated to beat baseline overall."
        ),
    )
    parser.add_argument(
        "--tunnel-col", type=int, default=0,
        help="Brick column index (0-17) to dig through first. Only used by --policy tunnel.",
    )
    parser.add_argument(
        "--safe-max-offset", type=float,
        default=PADDLE_HALF_WIDTH_PX - PADDLE_DEADBAND_PX - 1.0,
        help=(
            "Cap on the paddle-contact-point offset the aim system may "
            "choose, kept safely inside the paddle's physical half-width "
            "(8px) so the controller's own tracking deadband can never "
            "push a contact past the paddle's edge."
        ),
    )
    parser.add_argument("--min-hit-probability", type=float, default=0.05)
    parser.add_argument(
        "--aim-activation-remaining", type=int, default=30,
        help="Only aim once this many or fewer bricks remain (dense fields don't benefit).",
    )
    parser.add_argument("--stuck-trigger-steps", type=int, default=1024)
    parser.add_argument("--stuck-switch-steps", type=int, default=256)
    parser.add_argument("--stuck-offset", type=float, default=12.0)
    parser.add_argument(
        "--calibration-path", type=str, default=str(DEFAULT_CALIBRATION_PATH)
    )
    parser.add_argument("--trial-name", type=str, default="breakout_tunnel_eval")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_tunnel_policy(parse_args())
