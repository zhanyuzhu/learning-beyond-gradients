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

"""Collect real (paddle-contact-offset, incoming ball speed, outgoing-bounce
-angle) samples from Breakout-v5 and write them to
`paddle_bounce_calibration.csv`.

Why this exists: the outgoing bounce angle off the paddle is a noisy,
roughly-discrete function of where the ball hits the paddle relative to its
center, not a clean formula (confirmed by direct probing). `heuristic_breakout_tunnel.py`
uses the resulting empirical P(angle | offset) table to pick, for a given
target brick, whichever offset historically sent the ball closest to it,
instead of assuming a geometric bounce model.

Ball speed matters too, and matters a lot: at the *same* contact offset, a
slow ball reaches a much steeper outgoing angle than a fast one (e.g.
offset in [2,5]px: median +53 degrees when incoming speed < 2.5 px/step,
vs. only +31 degrees when incoming speed >= 3.5 px/step - confirmed
directly on this data, consistent across every offset bucket). Since ALE
Breakout speeds the ball up as the game progresses, a table that pools all
speeds together is implicitly dominated by whatever speed tier happens to
be most common in a typical playthrough (in one collection run, one speed
tier alone was 38% of all samples) and systematically wrong for the rest -
including, often, the high-speed late-game regime where aiming matters
most. So this script records `in_speed` (px/step at contact) alongside
offset and outgoing angle, and callers should condition the P(angle |
offset) lookup on the current ball speed, not just offset.

Method: run the RAM policy, but every time the ball starts a fresh descent
toward the paddle, sample one random offset in [-8, 8] px (the paddle's
physical half-width) and hold the paddle there for the whole descent -
locking the offset for the full descent avoids conflating the measurement
with paddle-tracking lag. Detect each paddle contact from the RAM ball
trajectory (vy sign flip near paddle height) and record the measured
contact offset (actual ball x - actual paddle x, which can differ slightly
from the sampled target due to control lag), the incoming speed, and the
resulting outgoing angle.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heuristic_breakout as hb

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "paddle_bounce_calibration.csv"
OFFSET_RANGE = (-8.0, 8.0)  # paddle is 16px wide; this is its full physical half-width


class ForcedOffsetAgent(hb.RamBreakoutAgent):
    """Like RamBreakoutAgent, but locks a random paddle-contact offset for
    the whole duration of each ball descent, instead of tracking the
    natural interception point. Used only to generate calibration data."""

    def __init__(self, action_map, config, offset_range, rng):
        super().__init__(action_map, config)
        self._offset_range = offset_range
        self._rng = rng
        self._current_offset = 0.0
        self._was_descending = False

    def _target_paddle_x(self, detections):
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
            if not self._was_descending:
                self._current_offset = self._rng.uniform(*self._offset_range)
                self._was_descending = True
            steps_to_paddle = max((self._config.paddle_y - ball_y) / vy, 0.0)
            intercept_x = hb.reflect_position(
                ball_x + vx * steps_to_paddle,
                lower=self._config.field_left,
                upper=self._config.field_right,
            )
            return hb.clip(
                intercept_x + self._current_offset,
                lower=self._config.paddle_min_x,
                upper=self._config.paddle_max_x,
            )
        self._was_descending = False
        return super()._target_paddle_x(detections)


def collect_one_run(offset_range, n_steps, seed, rng):
    import envpool

    env = envpool.make_gym(
        "Breakout-v5",
        num_envs=1,
        batch_size=1,
        seed=seed,
        max_episode_steps=27000,
        img_height=210,
        img_width=160,
        stack_num=1,
        gray_scale=False,
        frame_skip=1,
        noop_max=1,
        use_fire_reset=True,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    config = hb.HeuristicConfig()
    agent = ForcedOffsetAgent(hb.ActionMap(), config, offset_range, rng)
    obs, info = hb.reset_env_with_info(env)
    agent.reset()

    ball_xs: list[float | None] = []
    ball_ys: list[float | None] = []
    paddle_xs: list[float | None] = []
    for _ in range(n_steps):
        action = agent.act(info)
        obs, reward, done, info = hb.step_env(env, action)
        agent.observe_reward(reward)
        det = hb.decode_ram_detections(info)
        ball_xs.append(det.ball_xy[0] if det.ball_xy else None)
        ball_ys.append(det.ball_xy[1] if det.ball_xy else None)
        paddle_xs.append(det.paddle_x)
        if done:
            obs, info = hb.reset_env_with_info(env)
            agent.reset()
    env.close()
    return extract_bounce_samples(ball_xs, ball_ys, paddle_xs, config.paddle_y)


def extract_bounce_samples(ball_xs, ball_ys, paddle_xs, paddle_y):
    """Detect paddle-contact events from a ball-position trace (a vy sign
    flip from downward to upward near paddle height) and, for each, pair
    the measured contact offset with the resulting outgoing angle."""
    n = len(ball_xs)
    vx: list[float | None] = [None] * n
    vy: list[float | None] = [None] * n
    for i in range(1, n):
        if ball_xs[i] is not None and ball_xs[i - 1] is not None:
            vx[i] = ball_xs[i] - ball_xs[i - 1]
            vy[i] = ball_ys[i] - ball_ys[i - 1]

    samples = []
    for i in range(3, n - 6):
        if vy[i] is None or vy[i + 1] is None:
            continue
        if ball_ys[i] is None or ball_ys[i] < paddle_y - 20:
            continue
        if not (vy[i] > 0.3 and vy[i + 1] < -0.3):
            continue
        contact_ball_x, contact_paddle_x = ball_xs[i], paddle_xs[i]
        if contact_ball_x is None or contact_paddle_x is None:
            continue

        in_vx = [vx[j] for j in range(max(3, i - 2), i + 1) if vx[j] is not None and vy[j] is not None and vy[j] > 0.1]
        in_vy = [vy[j] for j in range(max(3, i - 2), i + 1) if vy[j] is not None and vy[j] > 0.1]
        out_vx = [vx[j] for j in range(i + 1, min(n, i + 4)) if vx[j] is not None and vy[j] is not None and vy[j] < -0.1]
        out_vy = [vy[j] for j in range(i + 1, min(n, i + 4)) if vy[j] is not None and vy[j] < -0.1]
        if not in_vx or not out_vx:
            continue

        in_x = sum(in_vx) / len(in_vx)
        in_y = sum(in_vy) / len(in_vy)
        in_speed = float(np.hypot(in_x, in_y))
        in_angle = float(np.degrees(np.arctan2(in_x, in_y)))
        out_x = sum(out_vx) / len(out_vx)
        out_y = sum(out_vy) / len(out_vy)
        out_speed = float(np.hypot(out_x, out_y))
        out_angle = float(np.degrees(np.arctan2(out_x, -out_y)))
        samples.append((
            contact_ball_x - contact_paddle_x,
            in_speed, in_angle,
            out_speed, out_angle,
        ))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4, help="Number of envpool seeds to run.")
    parser.add_argument("--steps-per-seed", type=int, default=40000)
    parser.add_argument("--rng-seed", type=int, default=999)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rng = random.Random(args.rng_seed)
    all_samples = []
    for seed in range(args.seeds):
        samples = collect_one_run(OFFSET_RANGE, args.steps_per_seed, seed, rng)
        all_samples.extend(samples)
        print(f"seed={seed} done, samples so far={len(all_samples)}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["offset", "in_speed", "in_angle_deg", "out_speed", "out_angle_deg"])
        w.writerows(all_samples)
    print(f"wrote {len(all_samples)} rows to {args.output}")


if __name__ == "__main__":
    main()
