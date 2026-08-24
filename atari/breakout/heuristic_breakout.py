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

"""Play Atari Breakout with hand-written RAM or vision heuristics.

The policies do not use frame stacking or a trained model. The RAM policy
decodes ball/paddle coordinates from `info["ram"]`, while the vision policy
segments them directly from RGB pixels. Both estimate ball velocity from
consecutive states, reflect the trajectory against the side walls, and move the
paddle to the predicted interception point. When no ball is visible, they move
to a launch position and press FIRE.

Every script invocation appends one JSONL record to
`heuristic_breakout_trials.jsonl` and rewrites
`heuristic_breakout_trials_summary.csv` with cumulative sample counts so that
sample-efficiency curves can be plotted directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_breakout_trials.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "heuristic_breakout_trials_summary.csv"
ALE_FPS = 60.0


@dataclass(frozen=True)
class ActionMap:
    """Discrete action ids in Breakout's reduced action space."""

    noop: int = 0
    fire: int = 1
    right: int = 2
    left: int = 3


@dataclass(frozen=True)
class BreakoutRamDetections:
    """Coordinates decoded from one env's Atari RAM."""

    ball_xy: tuple[float, float] | None
    paddle_x: float
    brick_balance: float
    lives: int | None


@dataclass(frozen=True)
class BreakoutVisionDetections:
    """Coordinates segmented from one Breakout RGB frame."""

    ball_xy: tuple[float, float] | None
    paddle_x: float | None
    brick_balance: float


@dataclass
class BreakoutPolicyState:
    """Minimal recurrent state for one Breakout rollout."""

    prev_ball_xy: tuple[float, float] | None = None
    velocity_xy: tuple[float, float] | None = None
    last_lives: int | None = None
    last_action: int = ActionMap.noop
    launch_sign: float = 1.0
    episode_score: float = 0.0
    steps_since_reward: int = 0
    stuck_offset_index: int = 0
    missing_ball_frames: int = 0


@dataclass(frozen=True)
class HeuristicConfig:
    """Geometry and control constants for the RAM policy."""

    field_left: float = 8.0
    field_right: float = 151.0
    paddle_min_x: float = 15.5
    paddle_max_x: float = 152.5
    paddle_y: float = 189.5
    home_x: float = 106.5
    paddle_deadband_px: float = 3.0
    chase_lead_steps: float = 6.0
    tunnel_offset_px: float = 0.0
    launch_offset_px: float = 24.0
    fast_low_ball_lead_steps: float = 3.0
    fast_ball_min_vy: float = 3.0
    max_velocity_jump_px: float = 24.0
    stuck_trigger_steps: int = 1024
    stuck_switch_steps: int = 256
    stuck_offset_px: float = 12.0
    stuck_release_horizon_steps: float = 8.0
    brick_balance_deadzone: float = 0.01
    brick_balance_bias_min_score: float = 432.0
    late_game_paddle_lag_px: float = 2.0
    late_game_lag_ball_y: float = 170.0
    max_missing_ball_frames: int = 8


class RamBreakoutAgent:
    """RAM-based Breakout controller with side-wall reflection prediction."""

    def __init__(
        self,
        action_map: ActionMap,
        config: HeuristicConfig,
    ) -> None:
        self._action_map = action_map
        self._config = config
        self._state = BreakoutPolicyState()

    def reset(self) -> None:
        self._state = BreakoutPolicyState()

    def act(self, info: dict[str, np.ndarray]) -> int:
        detections = decode_ram_detections(info)
        self._handle_life_change(detections.lives)
        self._update_ball_state(detections.ball_xy)

        if detections.ball_xy is None:
            action = self._serve_action(detections.paddle_x)
            self._state.last_action = action
            return action

        target_x = self._target_paddle_x(detections)
        paddle_x = self._control_paddle_x(detections)
        error = target_x - paddle_x
        if error > self._config.paddle_deadband_px:
            action = self._action_map.right
        elif error < -self._config.paddle_deadband_px:
            action = self._action_map.left
        else:
            action = self._action_map.noop
        self._state.last_action = action
        return action

    def observe_reward(self, reward: float) -> None:
        if reward > 0.0:
            self._state.episode_score += reward
            self._state.steps_since_reward = 0
            self._state.stuck_offset_index = 0
            return

        self._state.steps_since_reward += 1
        if (
            self._config.stuck_switch_steps > 0
            and self._state.steps_since_reward
            >= self._config.stuck_trigger_steps
            and (
                self._state.steps_since_reward
                - self._config.stuck_trigger_steps
            )
            % self._config.stuck_switch_steps
            == 0
        ):
            self._state.stuck_offset_index += 1

    def _handle_life_change(self, lives: int | None) -> None:
        if lives is None:
            return
        if self._state.last_lives is not None and lives < self._state.last_lives:
            self._state.prev_ball_xy = None
            self._state.velocity_xy = None
            self._state.launch_sign *= -1.0
            self._state.steps_since_reward = 0
            self._state.stuck_offset_index = 0
        self._state.last_lives = lives

    def _serve_action(self, paddle_x: float) -> int:
        target_x = self._config.home_x + (
            self._state.launch_sign * self._config.launch_offset_px
        )
        error = target_x - paddle_x
        if error > self._config.paddle_deadband_px:
            return self._action_map.right
        if error < -self._config.paddle_deadband_px:
            return self._action_map.left
        return self._action_map.fire

    def _update_ball_state(
        self, ball_xy: tuple[float, float] | None
    ) -> None:
        if ball_xy is None:
            self._state.prev_ball_xy = None
            self._state.velocity_xy = None
            return

        if self._state.prev_ball_xy is not None:
            dx = ball_xy[0] - self._state.prev_ball_xy[0]
            dy = ball_xy[1] - self._state.prev_ball_xy[1]
            if (
                abs(dx) <= self._config.max_velocity_jump_px
                and abs(dy) <= self._config.max_velocity_jump_px
                and abs(dx) + abs(dy) > 0.25
            ):
                if self._state.velocity_xy is None:
                    self._state.velocity_xy = (dx, dy)
                else:
                    old_dx, old_dy = self._state.velocity_xy
                    self._state.velocity_xy = (
                        0.5 * old_dx + 0.5 * dx,
                        0.5 * old_dy + 0.5 * dy,
                    )
            else:
                self._state.velocity_xy = None
        self._state.prev_ball_xy = ball_xy

    def _target_paddle_x(self, detections: BreakoutRamDetections) -> float:
        if detections.ball_xy is None:
            return self._config.home_x

        ball_x, ball_y = detections.ball_xy
        velocity = self._state.velocity_xy
        if velocity is None:
            return clip(
                ball_x,
                lower=self._config.paddle_min_x,
                upper=self._config.paddle_max_x,
            )

        vx, vy = velocity
        if vy > 0.1 and ball_y <= self._config.paddle_y:
            steps_to_paddle = max(
                (self._config.paddle_y - ball_y) / vy,
                0.0,
            )
            intercept_x = reflect_position(
                ball_x + vx * steps_to_paddle,
                lower=self._config.field_left,
                upper=self._config.field_right,
            )
            target_x = (
                intercept_x
                + self._state.launch_sign * self._config.tunnel_offset_px
                + self._current_stuck_offset_px(
                    steps_to_paddle,
                    detections.brick_balance,
                )
            )
        elif vy >= self._config.fast_ball_min_vy:
            target_x = (
                ball_x + self._config.fast_low_ball_lead_steps * vx
            )
        else:
            target_x = ball_x + self._config.chase_lead_steps * vx

        return clip(
            target_x,
            lower=self._config.paddle_min_x,
            upper=self._config.paddle_max_x,
        )

    def _control_paddle_x(self, detections: BreakoutRamDetections) -> float:
        ball_xy = detections.ball_xy
        velocity_xy = self._state.velocity_xy
        if (
            ball_xy is None
            or velocity_xy is None
            or self._state.episode_score
            < self._config.brick_balance_bias_min_score
            or velocity_xy[1] <= 0.1
            or ball_xy[1] < self._config.late_game_lag_ball_y
            or self._config.late_game_paddle_lag_px <= 0.0
        ):
            return detections.paddle_x

        if self._state.last_action == self._action_map.right:
            return clip(
                detections.paddle_x + self._config.late_game_paddle_lag_px,
                lower=self._config.paddle_min_x,
                upper=self._config.paddle_max_x,
            )
        if self._state.last_action == self._action_map.left:
            return clip(
                detections.paddle_x - self._config.late_game_paddle_lag_px,
                lower=self._config.paddle_min_x,
                upper=self._config.paddle_max_x,
            )
        return detections.paddle_x

    def _current_stuck_offset_px(
        self,
        steps_to_paddle: float,
        brick_balance: float,
    ) -> float:
        if (
            self._state.steps_since_reward
            < self._config.stuck_trigger_steps
            or self._config.stuck_offset_px == 0.0
        ):
            return 0.0

        phase = self._state.stuck_offset_index % 4
        if phase == 0:
            direction = 1.0
            magnitude = self._config.stuck_offset_px
        elif phase == 1:
            direction = -1.0
            magnitude = self._config.stuck_offset_px
        elif phase == 2:
            direction = 1.0
            magnitude = 0.5 * self._config.stuck_offset_px
        else:
            direction = -1.0
            magnitude = 0.5 * self._config.stuck_offset_px

        if self._state.episode_score >= self._config.brick_balance_bias_min_score:
            if brick_balance > self._config.brick_balance_deadzone:
                direction = 1.0
            elif brick_balance < -self._config.brick_balance_deadzone:
                direction = -1.0

        offset = direction * magnitude

        if (
            self._state.episode_score
            < self._config.brick_balance_bias_min_score
            or self._config.stuck_release_horizon_steps <= 0.0
        ):
            return offset
        release_ratio = clip(
            steps_to_paddle / self._config.stuck_release_horizon_steps,
            lower=0.0,
            upper=1.0,
        )
        return release_ratio * offset


def decode_ram_detections(
    info: dict[str, np.ndarray],
) -> BreakoutRamDetections:
    """Decode Breakout ball/paddle coordinates from `info["ram"]`."""
    ram = np.asarray(info["ram"])[0]
    ball_xy = None
    if int(ram[101]) != 0:
        ball_xy = (
            0.999043 * float(ram[99]) - 48.370898,
            0.993263 * float(ram[101]) + 11.227841,
        )

    lives = None
    if "lives" in info:
        lives = int(np.asarray(info["lives"])[0])
    return BreakoutRamDetections(
        ball_xy=ball_xy,
        paddle_x=1.005232 * float(ram[72]) - 39.797062,
        brick_balance=(
            count_ram_bits(ram[:18]) / 132.0
            - count_ram_bits(ram[18:36]) / 108.0
        ),
        lives=lives,
    )


def clip(value: float, *, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def count_ram_bits(values: np.ndarray) -> int:
    return int(sum(int(value).bit_count() for value in values.tolist()))


def reflect_position(value: float, *, lower: float, upper: float) -> float:
    """Reflect a 1D coordinate inside [lower, upper] with elastic walls."""
    span = upper - lower
    if span <= 0:
        return lower
    period = 2.0 * span
    shifted = (value - lower) % period
    if shifted <= span:
        return lower + shifted
    return upper - (shifted - span)


def extract_latest_frame(obs: np.ndarray) -> np.ndarray:
    """Return one env's latest frame as HxWx3."""
    arr = np.asarray(obs)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported observation shape: {arr.shape}")
    if arr.shape[0] <= 4 and arr.shape[1] > 16 and arr.shape[2] > 16:
        return np.moveaxis(arr[:3], 0, -1)
    return arr[..., :3]


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Connected components over true pixels in a tiny binary mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    index_by_coord = {tuple(coord): i for i, coord in enumerate(coords.tolist())}
    visited = np.zeros(coords.shape[0], dtype=bool)
    components: list[np.ndarray] = []

    for root_idx in range(coords.shape[0]):
        if visited[root_idx]:
            continue
        visited[root_idx] = True
        stack = [root_idx]
        component_indices = [root_idx]
        while stack:
            idx = stack.pop()
            y, x = coords[idx]
            for ny, nx in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                next_idx = index_by_coord.get((ny, nx))
                if next_idx is None or visited[next_idx]:
                    continue
                visited[next_idx] = True
                stack.append(next_idx)
                component_indices.append(next_idx)
        components.append(coords[np.asarray(component_indices)])
    return components


def reset_env_with_info(env):
    """Return `(obs, info)` for Gymnasium or `(obs, {})` for legacy Gym."""
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_env(env, action: int):
    """Support both legacy Gym and Gymnasium step signatures."""
    result = env.step(np.asarray([action], dtype=np.int32))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = result
    return obs, float(np.asarray(reward)[0]), bool(np.asarray(done)[0]), info


def append_trial_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_trial_records(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(
    log_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    records = load_trial_records(log_path)
    rows = []
    cumulative_env_steps = 0
    cumulative_ale_frames = 0
    for index, record in enumerate(records):
        env_steps = int(record.get("env_steps", 0))
        ale_frames = int(record.get("ale_frames", 0))
        cumulative_env_steps += env_steps
        cumulative_ale_frames += ale_frames
        rows.append(
            {
                "trial_index": index,
                "timestamp": record.get("timestamp", ""),
                "trial_name": record.get("trial_name", ""),
                "kind": record.get("kind", ""),
                "policy": record.get("policy", ""),
                "num_envs": record.get("num_envs", 1),
                "episodes_started": record.get("episodes_started", 0),
                "episodes_finished": record.get("episodes_finished", 0),
                "env_steps": env_steps,
                "ale_frames": ale_frames,
                "cumulative_env_steps": cumulative_env_steps,
                "cumulative_ale_frames": cumulative_ale_frames,
                "score_mean": record.get("score_mean", ""),
                "score_min": record.get("score_min", ""),
                "score_max": record.get("score_max", ""),
                "notes": record.get("notes", ""),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_index",
                "timestamp",
                "trial_name",
                "kind",
                "policy",
                "num_envs",
                "episodes_started",
                "episodes_finished",
                "env_steps",
                "ale_frames",
                "cumulative_env_steps",
                "cumulative_ale_frames",
                "score_mean",
                "score_min",
                "score_max",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("summary: no trial records")
        return
    last_row = rows[-1]
    scored_rows = [
        row for row in rows if row["score_mean"] not in ("", None)
    ]
    best_row = None
    if scored_rows:
        best_row = max(scored_rows, key=lambda row: float(row["score_mean"]))

    print(
        "trial_log_summary:",
        f"records={len(rows)}",
        f"cumulative_env_steps={last_row['cumulative_env_steps']}",
        f"cumulative_ale_frames={last_row['cumulative_ale_frames']}",
    )
    if best_row is not None:
        print(
            "best_mean_score:",
            f"trial_index={best_row['trial_index']}",
            f"trial_name={best_row['trial_name']}",
            f"score_mean={float(best_row['score_mean']):.3f}",
            f"cumulative_env_steps={best_row['cumulative_env_steps']}",
        )


def evaluate_heuristic_policy(args: argparse.Namespace) -> None:
    """Run the Breakout policy and append one trial record."""
    import envpool

    # A single env instance, rendered in envpool's "human" mode (an
    # opencv window) and paced to real wall-clock speed below.
    env_kwargs: dict[str, Any] = dict(
        num_envs=1,
        batch_size=1,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        img_height=args.img_height,
        img_width=args.img_width,
        stack_num=args.stack_num,
        gray_scale=args.gray_scale,
        frame_skip=args.frame_skip,
        noop_max=args.noop_max,
        use_fire_reset=not args.disable_fire_reset,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    if args.render:
        env_kwargs["render_mode"] = "human"
    env = envpool.make_gym("Breakout-v5", **env_kwargs)

    # One env.step() advances `frame_skip` ALE frames, each 1/60s of
    # real time, so pace every step to that duration for real-time playback.
    step_period_s = args.frame_skip / ALE_FPS if args.realtime else 0.0

    config = HeuristicConfig(
        paddle_deadband_px=args.deadband,
        chase_lead_steps=args.chase_lead_steps,
        tunnel_offset_px=args.tunnel_offset,
        launch_offset_px=args.launch_offset,
        fast_low_ball_lead_steps=args.fast_low_ball_lead_steps,
        fast_ball_min_vy=args.fast_ball_min_vy,
        stuck_trigger_steps=args.stuck_trigger_steps,
        stuck_switch_steps=args.stuck_switch_steps,
        stuck_offset_px=args.stuck_offset,
        stuck_release_horizon_steps=args.stuck_release_horizon_steps,
        brick_balance_deadzone=args.brick_balance_deadzone,
        brick_balance_bias_min_score=args.brick_balance_bias_min_score,
        late_game_paddle_lag_px=args.late_game_paddle_lag_px,
        late_game_lag_ball_y=args.late_game_lag_ball_y,
    )
    if args.policy == "ram":
        agent = RamBreakoutAgent(action_map=ActionMap(), config=config)
    else:
        raise ValueError('Only support ram policy')

    scores = []
    episode_lengths = []
    env_steps = 0
    episodes_started = 0

    try:
        for episode in range(args.episodes):
            obs, info = reset_env_with_info(env)
            agent.reset()
            episodes_started += 1
            total_reward = 0.0
            episode_steps = 0
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                step_start = time.perf_counter()
                action = agent.act(info if args.policy == "ram" else obs)
                obs, reward, done, info = step_env(env, action)
                agent.observe_reward(reward)
                total_reward += reward
                episode_steps += 1
                env_steps += 1
                if args.render:
                    env.render()
                if step_period_s > 0.0:
                    remaining = step_period_s - (
                        time.perf_counter() - step_start
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
                if done:
                    break
            scores.append(total_reward)
            episode_lengths.append(episode_steps)
            print(
                f"episode={episode} score={total_reward:.1f} "
                f"steps={episode_steps}"
            )
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
    append_trial_record(
        log_path,
        {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
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
                "deadband": args.deadband,
                "chase_lead_steps": args.chase_lead_steps,
                "tunnel_offset": args.tunnel_offset,
                "launch_offset": args.launch_offset,
                "fast_low_ball_lead_steps": args.fast_low_ball_lead_steps,
                "fast_ball_min_vy": args.fast_ball_min_vy,
                "stuck_trigger_steps": args.stuck_trigger_steps,
                "stuck_switch_steps": args.stuck_switch_steps,
                "stuck_offset": args.stuck_offset,
                "stuck_release_horizon_steps": args.stuck_release_horizon_steps,
                "brick_balance_deadzone": args.brick_balance_deadzone,
                "brick_balance_bias_min_score": args.brick_balance_bias_min_score,
                "late_game_paddle_lag_px": args.late_game_paddle_lag_px,
                "late_game_lag_ball_y": args.late_game_lag_ball_y,
            },
            "notes": args.notes,
        },
    )
    rows = write_summary(log_path, summary_path)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure heuristic Atari Breakout policy."
    )
    parser.add_argument(
        "--policy",
        choices=("ram", "vision"),
        default="ram",
        help="Use RAM decoding or a pure RGB vision heuristic.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=27000,
        help="Used both for env max_episode_steps and the Python rollout loop.",
    )
    parser.add_argument("--img-height", type=int, default=210)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--stack-num", type=int, default=1)
    parser.add_argument(
        "--gray-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the game in an opencv 'human' render window.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pace playback to real wall-clock speed "
            "(60/frame_skip steps per second). Disable for fast eval runs."
        ),
    )
    parser.add_argument("--disable-fire-reset", action="store_true")
    parser.add_argument("--deadband", type=float, default=3.0)
    parser.add_argument("--chase-lead-steps", type=float, default=6.0)
    parser.add_argument("--tunnel-offset", type=float, default=0.0)
    parser.add_argument("--launch-offset", type=float, default=24.0)
    parser.add_argument("--fast-low-ball-lead-steps", type=float, default=3.0)
    parser.add_argument("--fast-ball-min-vy", type=float, default=3.0)
    parser.add_argument("--stuck-trigger-steps", type=int, default=1024)
    parser.add_argument("--stuck-switch-steps", type=int, default=256)
    parser.add_argument("--stuck-offset", type=float, default=12.0)
    parser.add_argument(
        "--stuck-release-horizon-steps",
        type=float,
        default=8.0,
    )
    parser.add_argument("--brick-balance-deadzone", type=float, default=0.01)
    parser.add_argument(
        "--brick-balance-bias-min-score",
        type=float,
        default=432.0,
    )
    parser.add_argument("--late-game-paddle-lag-px", type=float, default=2.0)
    parser.add_argument("--late-game-lag-ball-y", type=float, default=170.0)
    parser.add_argument("--trial-name", type=str, default="breakout_ram_eval")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(DEFAULT_LOG_PATH),
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_heuristic_policy(parse_args())
