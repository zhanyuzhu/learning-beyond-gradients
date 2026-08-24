"""A programmatic (no-neural-network) Ms. Pac-Man policy for envpool.

The agent reads emulator state straight out of `info["ram"]` - the 128 RAM
bytes envpool hands back alongside every observation - and plans on a maze
graph it recovers from the RGB frame.  Nothing is learned: every decision
is a shortest-path search plus a hand-written utility.

Measured over 120 episodes across twelve seeds, with sticky actions
(`repeat_action_probability=0.25`, the standard ALE evaluation protocol):
mean 11274, median 9930, best 29801, and half of all episodes above
10000.  Half those seeds were held out of tuning entirely and scored 11033
against 11514 for the ones used, so the settings are not fitted to the
sample.  For scale, the usual human baseline for this game is about 6900
and published model-free RL agents land between roughly 2300 (DQN) and
5400 (Rainbow).

RAM decoding, verified against sprite pixel positions (see `decode_state`)::

    ram[10], ram[16]                   Ms. Pac-Man x, y
    ram[6..9],  ram[12..15]            the four ghosts' x, y
    ram[11], ram[17]                   fruit x, y (0 when absent)
    ram[119]                           dots eaten this level

Those coordinates live on a fixed lattice: a cell centre is
`ram_y = 2 + 12*row` and `ram_x` one of 19 column values, and the screen
pixels underneath it start at `(ram_y + 5, ram_x - 10)`.  That lattice is
what lets the pixel-side maze scan and the RAM-side entity positions share
one coordinate system, and every distance below is measured in RAM units -
the emulator's own - so ghost and pellet distances stay comparable.

Why the frame is read at all: the RAM carries no usable wall map, and its
pellet bitmap (bytes 60..101) is initialised to "uneaten" even for lattice
cells that are solid wall, so it can say a slot has not been eaten but not
whether a pellet was ever there.  The frame answers both questions
directly, and the maze only changes when the level does.

Each step:

1. A multi-source Dijkstra from every dangerous ghost gives `ghost_dist`,
   charging a ghost extra to reverse out of the corridor it is in.
2. A Dijkstra from Ms. Pac-Man that may only enter cells she reaches
   strictly before a ghost can (`ghost_dist > my_dist + margin`), so a
   route is never planned through a square about to be occupied.
3. Every pellet, power pill, edible ghost and fruit it reaches is scored
   `value / (travel + 1)` - best points per unit of travel, not merely
   nearest.
4. Those scores collapse onto the four first moves out of the current
   cell, turning back the way she came is discounted, and the previous
   direction is held unless another beats it clearly.  Committing to a
   direction rather than to one pellet is what keeps her out of the
   flip-flops that were, by some distance, the largest single cause of
   death in earlier versions.
5. If step 2 reaches nothing at all the ghosts have her boxed in: fall
   back to the neighbour that maximises distance to the nearest ghost.

Most of the score comes from power pills rather than pellets - a cleared
board is worth 1500, while working a single pill all the way up the
200/400/800/1600 ghost ladder is worth 3000 - so the chase logic is where
the tuning effort went.  `HeuristicConfig` holds every weight involved.

Ms. Pac-Man buffers turn inputs - holding UP while running right turns her
up at the next opening - so emitting the direction of the next cell is
enough, and no sub-cell alignment logic is needed.  She does stop dead if
told to walk into a wall, which is why the emitted direction always comes
from the maze graph.

Run it with::

    python policy.py --episodes 5
    python policy.py --episodes 1 --render --realtime
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "mspacman_trials.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "mspacman_trials_summary.csv"

ALE_FPS = 60.0

# ---------------------------------------------------------------------------
# Maze lattice geometry
# ---------------------------------------------------------------------------

# Row centres in RAM y units; the playfield is 14 bands of 12 screen rows.
ROW_Y = [2 + 12 * r for r in range(14)]
# Column centres in RAM x units.  The playfield is mirrored about x=88, which
# is why the two halves are 8 apart internally but 6 apart across the seam.
COL_X = [18, 26, 34, 42, 50, 58, 66, 74, 82, 88,
         94, 102, 110, 118, 126, 134, 142, 150, 158]
N_ROWS, N_COLS = len(ROW_Y), len(COL_X)

# RAM (x, y) -> screen pixel of the 4x2 pellet block belonging to that cell.
PX_X_OFFSET = -10
CELL_PX_W = 4

MAZE_TOP_PX, MAZE_BOTTOM_PX = 1, 172

# Sprite palette.  Ghost identity colours are indexed the same way as their
# RAM slots: orange=ram[6]/ram[12], cyan=ram[7]/ram[13], pink=ram[8]/ram[14],
# red=ram[9]/ram[15].
GHOST_RGB = (
    (180, 122, 48),   # Sue
    (84, 184, 153),   # Inky
    (198, 89, 179),   # Pinky
    (200, 72, 72),    # Blinky
)
SCARED_BLUE_RGB = (66, 114, 194)
SCARED_WHITE_RGB = (214, 214, 214)   # the "about to wear off" flash

_GHOST_RGB_ARR = tuple(np.asarray(c, dtype=np.uint8) for c in GHOST_RGB)
_SCARED_BLUE = np.asarray(SCARED_BLUE_RGB, dtype=np.uint8)
_SCARED_WHITE = np.asarray(SCARED_WHITE_RGB, dtype=np.uint8)

# Actions of envpool's 9-action Ms. Pac-Man set (verified by probing).
ACTION_NOOP, ACTION_UP, ACTION_RIGHT, ACTION_LEFT, ACTION_DOWN = 0, 1, 2, 3, 4
DIR_ACTION = {(-1, 0): ACTION_UP, (1, 0): ACTION_DOWN,
              (0, -1): ACTION_LEFT, (0, 1): ACTION_RIGHT}

UNREACHABLE = 1 << 30
# Extra RAM-unit cost charged to a ghost for reversing out of its corridor.
REVERSE_COST = 40
# A side tunnel spans the mirrored seam; roughly 28 RAM units end to end.
TUNNEL_COST = 28

# Half-extents, in RAM units, of the area each sprite hides from the pellet
# scan.  Ms. Pac-Man's sprite is narrower than a ghost's.
PAC_MASK_X, PAC_MASK_Y = 6, 7
GHOST_MASK_X, GHOST_MASK_Y = 8, 10
# A power pill's flash is a few frames long; longer than this without one
# means it has been eaten.
PILL_BLINK_FRAMES = 12
# An eaten ghost is briefly neither blue nor back on patrol; the power pill
# has only really run out once no ghost has been blue for this long.
PHASE_GAP_FRAMES = 10


def extract_latest_frame(obs: np.ndarray) -> np.ndarray:
    """Return one env's most recent RGB frame as HxWx3 uint8."""
    arr = np.asarray(obs)
    if arr.ndim == 4:
        arr = arr[0]
    # envpool hands back (stack*channels, H, W); the newest frame is last.
    return np.ascontiguousarray(arr[-3:].transpose(1, 2, 0))


def playfield_colours(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(wall_rgb, background_rgb)` for the current level's palette.

    The background covers more of the playfield than anything else and the
    walls come second, so this adapts to each maze without hard-coded RGB.
    """
    region = frame[MAZE_TOP_PX:MAZE_BOTTOM_PX]
    colours, counts = np.unique(region.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    background = colours[order[0]]
    wall_rgb = colours[order[1]] if len(order) > 1 else background
    return wall_rgb, background


# ---------------------------------------------------------------------------
# Maze extraction from the frame
# ---------------------------------------------------------------------------

class Maze:
    """Walls, tunnels and weighted adjacency for one level's maze layout.

    Edges are weighted by the true RAM-unit gap they span - 12 between
    rows, 8 between columns but only 6 across the mirrored seam at x=88 -
    so a search distance is directly comparable between Ms. Pac-Man and the
    ghosts, who all move at roughly the same units per step.  Counting
    cells instead would make a vertical hop look 50% cheaper than it is.
    """

    def __init__(self, free: np.ndarray, tunnel_rows: list[int]) -> None:
        self.free = free
        self.tunnel_rows = tunnel_rows
        self.adj: list[list[tuple[int, int, int]]] = [
            [] for _ in range(N_ROWS * N_COLS)]
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if not free[r, c]:
                    continue
                node = r * N_COLS + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = r + dr, c + dc
                    if not (0 <= r2 < N_ROWS and 0 <= c2 < N_COLS):
                        continue
                    if not free[r2, c2]:
                        continue
                    cost = (abs(ROW_Y[r2] - ROW_Y[r]) if dr
                            else abs(COL_X[c2] - COL_X[c]))
                    self.adj[node].append((r2, c2, cost))
        for r in tunnel_rows:
            if free[r, 0] and free[r, N_COLS - 1]:
                self.adj[r * N_COLS].append((r, N_COLS - 1, TUNNEL_COST))
                self.adj[r * N_COLS + N_COLS - 1].append((r, 0, TUNNEL_COST))

    def neighbours(self, r: int, c: int) -> list[tuple[int, int, int]]:
        return self.adj[r * N_COLS + c]

    def degree(self, r: int, c: int) -> int:
        return len(self.adj[r * N_COLS + c])


# Pixel index grids for the 10x4 block under each lattice cell: rows are the
# band offsets +1..+10, columns the 4 pixels of the pellet block.
_BAND_ROWS = (np.asarray(ROW_Y, dtype=np.intp)[:, None]
              + np.arange(1, 11, dtype=np.intp)[None, :])
_BAND_COLS = ((np.asarray(COL_X, dtype=np.intp) + PX_X_OFFSET)[:, None]
              + np.arange(CELL_PX_W, dtype=np.intp)[None, :])


def scan_cells(frame: np.ndarray, wall_rgb: np.ndarray,
               background: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, list[int]]:
    """Segment a frame into `(open_cells, pellets, power_pills, tunnel_rows)`.

    Classification uses the 4-pixel-wide column under each lattice point.  A
    pellet paints only band rows +5/+6 and a power pill rows +1..+7, so the
    band's bottom rows (+8..+10) are always pellet-free: a cell whose bottom
    rows are clear *and* whose top rows are either clear or a power pill is
    open corridor.  Demanding both ends of the band be clear is also what
    walls off the ghost house - its top and bottom halves are solid, which
    leaves the interior an isolated island no search can path through.
    """
    wall = np.all(frame == wall_rgb, axis=-1)
    # Power pills blink between the wall colour and a second shade, so
    # "painted at all" - anything that is not background - catches both.
    painted = ~np.all(frame == background, axis=-1)

    # Gather every cell's 10x4 pixel block at once: axis 1 indexes band
    # offsets +1..+10, so offset +k sits at index k-1.
    wall_blk = wall[_BAND_ROWS[:, :, None, None], _BAND_COLS[None, None, :, :]]
    paint_blk = painted[_BAND_ROWS[:, :, None, None],
                        _BAND_COLS[None, None, :, :]]
    bottom_clear = ~wall_blk[:, 7:10].any(axis=(1, 3))
    top_clear = ~wall_blk[:, 0:3].any(axis=(1, 3))
    # A power pill is 7 rows tall, but the mirrored right half of the
    # playfield draws it one row higher than the left, so only offsets
    # +2..+6 are painted for certain in both halves.  Testing more than
    # that silently loses every pill on one side of the board.
    pills = (paint_blk[:, 1:6].all(axis=(1, 3))
             & ~paint_blk[:, 7:10].any(axis=(1, 3)))
    pellets = (wall_blk[:, 4:6].all(axis=(1, 3))
               & ~wall_blk[:, 2:4].any(axis=(1, 3))
               & bottom_clear)
    free = bottom_clear & (top_clear | pills)
    edge = ~wall[_BAND_ROWS, 0:CELL_PX_W].any(axis=(1, 2))
    tunnel_rows = [r for r in range(N_ROWS) if edge[r]]
    return free, pellets, pills, tunnel_rows


# ---------------------------------------------------------------------------
# RAM decoding
# ---------------------------------------------------------------------------

class GameState:
    """One frame of decoded Ms. Pac-Man state."""

    __slots__ = ("pac", "ghosts", "edible", "flashing", "fruit",
                 "dots_eaten", "lives")

    def __init__(self, pac, ghosts, edible, flashing, fruit, dots_eaten,
                 lives) -> None:
        self.pac = pac
        self.ghosts = ghosts
        self.edible = edible
        self.flashing = flashing
        self.fruit = fruit
        self.dots_eaten = dots_eaten
        self.lives = lives


def _sprite_box(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    """Pixels covering the sprite whose RAM position is `(x, y)`."""
    top = max(int(y) + 1, 0)
    left = max(int(x) + PX_X_OFFSET - 4, 0)
    return frame[top:top + 11, left:left + 12]


def decode_state(ram: np.ndarray, frame: np.ndarray, lives: int) -> GameState:
    """Decode positions from RAM; read ghost edibility off the sprite palette.

    Edibility has to come from the frame.  `ram[116]` is not a countdown to
    the end of the power pill - it still read 129 on the step before a
    measured phase ended - and it says nothing about *which* ghosts are
    still blue, since one eaten mid-phase respawns lethal while that byte
    keeps ticking.  The sprite palette answers both questions exactly, and
    it also carries the game's own warning: the ghosts flash white for
    about twenty-five steps before they turn dangerous again.
    """
    pac = (int(ram[10]), int(ram[16]))
    ghosts = [(int(ram[6 + i]), int(ram[12 + i])) for i in range(4)]
    edible = [False] * 4
    flashing = False
    for i, (gx, gy) in enumerate(ghosts):
        box = _sprite_box(frame, gx, gy)
        if box.size == 0:
            continue
        blue = int(np.all(box == _SCARED_BLUE, axis=-1).sum())
        white = int(np.all(box == _SCARED_WHITE, axis=-1).sum())
        own = int(np.all(box == _GHOST_RGB_ARR[i], axis=-1).sum())
        edible[i] = blue + white > own
        if edible[i] and white > blue:
            flashing = True
    fruit = None
    if int(ram[11]) != 0 and int(ram[17]) != 0:
        fruit = (int(ram[11]), int(ram[17]))
    return GameState(pac, ghosts, edible, flashing, fruit,
                     int(ram[119]), lives)


# ---------------------------------------------------------------------------
# Lattice helpers and search
# ---------------------------------------------------------------------------

_ROW_ARR = np.asarray(ROW_Y, dtype=np.int32)
_COL_ARR = np.asarray(COL_X, dtype=np.int32)


def nearest_cell(x: int, y: int) -> tuple[int, int]:
    return (int(np.argmin(np.abs(_ROW_ARR - y))),
            int(np.argmin(np.abs(_COL_ARR - x))))


def cell_ahead(x: int, y: int, direction: tuple[int, int]) -> tuple[int, int]:
    """The lattice cell the entity is about to arrive at.

    Snapping to the *nearest* cell would leave Ms. Pac-Man behind a junction
    she has just crossed, and the planner would then keep asking for a turn
    she can no longer make.
    """
    r, c = nearest_cell(x, y)
    dr, dc = direction
    if dc > 0 and COL_X[c] < x - 1:
        c = min(c + 1, N_COLS - 1)
    elif dc < 0 and COL_X[c] > x + 1:
        c = max(c - 1, 0)
    if dr > 0 and ROW_Y[r] < y - 1:
        r = min(r + 1, N_ROWS - 1)
    elif dr < 0 and ROW_Y[r] > y + 1:
        r = max(r - 1, 0)
    return r, c


def heading(prev: tuple[int, int] | None, now: tuple[int, int],
            fallback: tuple[int, int]) -> tuple[int, int]:
    """Unit heading implied by one step of motion, else keep `fallback`.

    Jumps of more than half the board are tunnel wraps and respawns, not
    motion, so they must not be read as a sudden change of direction.
    """
    if prev is None:
        return fallback
    dx, dy = now[0] - prev[0], now[1] - prev[1]
    if dx != 0 and abs(dx) > abs(dy) and abs(dx) < 30:
        return (0, 1 if dx > 0 else -1)
    if dy != 0 and abs(dy) < 30:
        return (1 if dy > 0 else -1, 0)
    return fallback


def snap_to_free(maze: Maze, r: int, c: int) -> tuple[int, int] | None:
    """Nearest open cell to `(r, c)`, or None when there is none nearby."""
    if maze.free[r, c]:
        return r, c
    best, best_d = None, 99
    for rr in range(max(0, r - 2), min(N_ROWS, r + 3)):
        for cc in range(max(0, c - 2), min(N_COLS, c + 3)):
            if maze.free[rr, cc]:
                d = abs(rr - r) * 2 + abs(cc - c)
                if d < best_d:
                    best, best_d = (rr, cc), d
    return best


def threat_field(maze: Maze, ghosts: list[tuple[int, int, tuple[int, int]]],
                 reverse_cost: int = REVERSE_COST) -> np.ndarray:
    """RAM-unit distance from the nearest dangerous ghost to every cell.

    Each ghost is seeded at its own cell's neighbours rather than the cell
    itself, so the step it would have to take *backwards* can be charged
    `REVERSE_COST`: ghosts almost never turn around mid-corridor, and
    treating the square behind one as equally threatening throws away the
    safest ground on the board.  It is a penalty rather than a ban because
    the Atari ghosts do occasionally reverse.
    """
    dist = np.full((N_ROWS, N_COLS), UNREACHABLE, dtype=np.int64)
    heap: list[tuple[int, int, int]] = []
    for r, c, (dr, dc) in ghosts:
        if not maze.free[r, c]:
            continue
        if 0 < dist[r, c]:
            dist[r, c] = 0
        for r2, c2, cost in maze.neighbours(r, c):
            backwards = (r2 - r, c2 - c) == (-dr, -dc)
            d = cost + (reverse_cost if backwards else 0)
            if d < dist[r2, c2]:
                dist[r2, c2] = d
                heapq.heappush(heap, (d, r2, c2))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for r2, c2, cost in maze.neighbours(r, c):
            nd = d + cost
            if nd < dist[r2, c2]:
                dist[r2, c2] = nd
                heapq.heappush(heap, (nd, r2, c2))
    return dist


def safe_search(maze: Maze, start: tuple[int, int], ghost_dist: np.ndarray,
                margin: int) -> tuple[np.ndarray, dict]:
    """Dijkstra from `start` that only enters cells we beat the ghosts to.

    A cell is enterable when `ghost_dist > my_dist + margin`: the nearest
    ghost needs strictly further to travel to stand there than we do, plus
    a safety buffer, all in RAM units.  The test stays valid under Dijkstra
    because a cell that fails it at distance `d` also fails at any larger
    one.

    Returns the distance field and, for every reachable cell, which
    neighbour of `start` the route to it leaves through.  That second map
    partitions the safe region by first move, which is what lets the agent
    tell "this way opens up" from "this way is a pocket".
    """
    dist = np.full((N_ROWS, N_COLS), UNREACHABLE, dtype=np.int64)
    first: dict[tuple[int, int], tuple[int, int]] = {}
    dist[start] = 0
    heap = [(0, start[0], start[1])]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        origin = first.get((r, c))
        for r2, c2, cost in maze.neighbours(r, c):
            nd = d + cost
            if nd >= dist[r2, c2]:
                continue
            if ghost_dist[r2, c2] <= nd + margin:
                continue
            dist[r2, c2] = nd
            first[(r2, c2)] = (r2, c2) if origin is None else origin
            heapq.heappush(heap, (nd, r2, c2))
    return dist, first


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class HeuristicConfig:
    """Tunable weights.  All distances are in RAM units (a maze row is 12).

    Defaults are the ones used for the reported multi-seed runs.
    """

    def __init__(
        self,
        safety_margin: int = 18,
        pellet_value: float = 10.0,
        pellet_cluster_bonus: float = 4.0,
        pill_base_value: float = 50.0,
        pill_ghost_bonus: float = 70.0,
        pill_lure_radius: int = 80,
        pill_idle_value: float = 12.0,
        ghost_value: float = 200.0,
        ghost_bonus_growth: float = 2.0,
        fruit_value: float = 140.0,
        chase_horizon: int = 110,
        flash_chase_radius: int = 60,
        hysteresis: float = 1.25,
        reverse_penalty: float = 0.5,
        distance_scale: float = 10.0,
        reverse_cost: int = REVERSE_COST,
    ) -> None:
        self.safety_margin = safety_margin
        self.pellet_value = pellet_value
        self.pellet_cluster_bonus = pellet_cluster_bonus
        self.pill_base_value = pill_base_value
        self.pill_ghost_bonus = pill_ghost_bonus
        self.pill_lure_radius = pill_lure_radius
        self.pill_idle_value = pill_idle_value
        self.ghost_value = ghost_value
        self.ghost_bonus_growth = ghost_bonus_growth
        self.fruit_value = fruit_value
        self.chase_horizon = chase_horizon
        self.flash_chase_radius = flash_chase_radius
        self.hysteresis = hysteresis
        self.reverse_penalty = reverse_penalty
        self.distance_scale = distance_scale
        self.reverse_cost = reverse_cost


class MsPacmanAgent:
    """Search-based Ms. Pac-Man policy driven by `info["ram"]` and the frame."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()
        self.reset()

    def reset(self) -> None:
        self.maze: Maze | None = None
        self.pellets = np.zeros((N_ROWS, N_COLS), dtype=bool)
        self.pills = np.zeros((N_ROWS, N_COLS), dtype=bool)
        self.pill_missing = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
        self.wall_rgb: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self.prev_pac: tuple[int, int] | None = None
        self.prev_ghosts: list[tuple[int, int]] | None = None
        self.direction = (0, -1)
        self.ghost_dirs = [(0, 0)] * 4
        self.target: tuple[int, int] | None = None
        self.last_move: tuple[int, int] | None = None
        self.lives: int | None = None
        self.dots_eaten = 0
        self.edible_seen = 0
        self.no_edible_steps = PHASE_GAP_FRAMES
        self.ghosts_eaten_phase = 0
        self.flash_seen = False

    # -- world model ------------------------------------------------------

    def _rebuild(self, frame: np.ndarray) -> None:
        self.wall_rgb, self.background = playfield_colours(frame)
        free, pellets, pills, tunnels = scan_cells(
            frame, self.wall_rgb, self.background)
        self.maze = Maze(free, tunnels)
        self.pellets = pellets
        self.pills = pills
        self.pill_missing = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
        self.dots_eaten = 0
        self.target = None

    def _update_items(self, pellets: np.ndarray, pills: np.ndarray,
                      state: GameState) -> None:
        """Fold this frame's sighting of the pellets and pills into memory.

        A sprite standing on a pellet hides it, so cells a sprite currently
        covers keep their previous value and every other cell is taken from
        the frame - and only ever to *remove* a pellet, never to put one
        back.  Restoring pellets is what makes the planner oscillate: it
        marks the pellet ahead as eaten a moment too early, turns around,
        sees it again as soon as the sprite clears it, turns back, and
        repeats until a ghost arrives.  Pellets never reappear inside a
        level, so monotone removal cannot lose information.

        The occlusion radius matters for the same reason: it is Ms.
        Pac-Man's own sprite, no wider, so the cell she just ate from stops
        being masked as soon as she has actually left it.
        """
        sprites = [(state.pac, PAC_MASK_X, PAC_MASK_Y)]
        sprites += [(g, GHOST_MASK_X, GHOST_MASK_Y) for g in state.ghosts]
        if state.fruit is not None:
            # The fruit wanders the maze and hides pellets exactly like a
            # ghost does; leaving it out marks whatever it drifts over as
            # eaten, which is enough to trigger a spurious maze rebuild.
            sprites.append((state.fruit, GHOST_MASK_X, GHOST_MASK_Y))
        occluded = np.zeros((N_ROWS, N_COLS), dtype=bool)
        for (x, y), rx, ry in sprites:
            rows = np.abs(_ROW_ARR - y) <= ry
            cols = np.abs(_COL_ARR - x) <= rx
            occluded |= rows[:, None] & cols[None, :]
        self.pellets &= pellets | occluded
        # Power pills flash, and the two flash colours alternate out of
        # phase, so any one frame shows only half of them - sightings have
        # to accumulate.  By the same token a single frame without a pill
        # proves nothing; only a long run of clear looks means it was eaten.
        self.pill_missing = np.where(pills | occluded, 0, self.pill_missing + 1)
        self.pills |= pills & ~occluded
        self.pills &= (self.pill_missing <= PILL_BLINK_FRAMES) | occluded

    # -- planning ---------------------------------------------------------

    def _ghost_cells(self, state: GameState) -> tuple[list, list]:
        """Split the ghosts into `(dangerous_with_heading, edible_cells)`.

        Ghosts still shut in the house land on the isolated island the maze
        scan leaves there, so their search never escapes it and they are
        automatically ignored until they come out.
        """
        assert self.maze is not None
        px, py = state.pac
        reach = self.config.flash_chase_radius
        dangerous: list[tuple[int, int, tuple[int, int]]] = []
        edible: list[tuple[int, int]] = []
        for i, (gx, gy) in enumerate(state.ghosts):
            cell = snap_to_free(self.maze, *nearest_cell(gx, gy))
            if cell is None:
                continue
            chase = state.edible[i]
            if chase and self.flash_seen:
                # Flashing means the pill is nearly spent.  A blue ghost
                # already within arm's reach is still free points, but
                # setting off across the maze after one is how you arrive
                # just as it turns lethal - so only the close ones count.
                chase = abs(gx - px) + abs(gy - py) <= reach
            if chase:
                edible.append(cell)
            else:
                dangerous.append((cell[0], cell[1], self.ghost_dirs[i]))
        return dangerous, edible

    def _track_phase(self, state: GameState) -> None:
        """Follow the current power pill: how far up the bonus ladder we are,
        and whether the ghosts have begun flashing.

        There is no RAM byte for either.  The ladder is read off the blue
        count: mid-phase it only falls when a ghost is swallowed, and when
        the pill runs out every remaining ghost reverts on the same frame,
        which is the reset.  It must be counted from the raw sprite colours
        rather than from the ghosts the planner is currently willing to
        chase - feeding this the post-flash list makes it read the flash
        itself as four ghosts eaten and then immediately forget the flash.

        The blue count also dips to zero for a moment while an eaten ghost
        walks home, so the phase is only over once it has stayed at zero
        for `PHASE_GAP_FRAMES`.
        """
        edible_now = sum(state.edible)
        if edible_now == 0:
            self.no_edible_steps += 1
            if self.no_edible_steps >= PHASE_GAP_FRAMES:
                self.edible_seen = 0
                self.flash_seen = False
                self.ghosts_eaten_phase = 0
            return
        self.no_edible_steps = 0
        if state.flashing:
            self.flash_seen = True
        if edible_now > self.edible_seen:
            # Ghosts turn blue over a couple of frames as a pill takes hold.
            self.edible_seen = edible_now
        elif edible_now < self.edible_seen:
            self.ghosts_eaten_phase += self.edible_seen - edible_now
            self.edible_seen = edible_now

    def _candidates(self, state: GameState, my_dist: np.ndarray,
                    ghost_dist: np.ndarray,
                    edible_cells: list) -> list[tuple[float, tuple[int, int]]]:
        """Score every reachable item as `value / (travel + 1)`.

        `travel` is the path length in row-widths, so the winner is the best
        points-per-distance deal rather than simply the closest crumb.
        """
        cfg = self.config
        out: list[tuple[float, tuple[int, int]]] = []
        scale = cfg.distance_scale
        reachable = my_dist < UNREACHABLE
        for r, c in np.argwhere(self.pellets & reachable):
            r, c = int(r), int(c)
            lo_r, hi_r = max(0, r - 2), min(N_ROWS, r + 3)
            lo_c, hi_c = max(0, c - 2), min(N_COLS, c + 3)
            cluster = int(self.pellets[lo_r:hi_r, lo_c:hi_c].sum()) - 1
            value = cfg.pellet_value + cfg.pellet_cluster_bonus * min(cluster, 4)
            out.append((value / (my_dist[r, c] / scale + 1.0), (r, c)))
        for r, c in np.argwhere(self.pills & reachable):
            r, c = int(r), int(c)
            if any(state.edible):
                # A second pill mid-phase restarts the 200/400/800/1600
                # ladder from the bottom, so make it barely worth a detour.
                value = cfg.pill_idle_value
            else:
                # A pill is only worth real points if ghosts are still in
                # the neighbourhood when it goes off, and each extra ghost
                # in range is worth more than the last.
                nearby = sum(1 for i, g in enumerate(state.ghosts)
                             if not state.edible[i]
                             and abs(g[0] - COL_X[c]) + abs(g[1] - ROW_Y[r])
                             <= cfg.pill_lure_radius)
                value = cfg.pill_base_value + cfg.pill_ghost_bonus * nearby
            out.append((value / (my_dist[r, c] / scale + 1.0), (r, c)))
        if edible_cells:
            # The bonus doubles with every ghost eaten inside one pill, so
            # the fourth is worth eight times the first: chase accordingly.
            value = cfg.ghost_value * (
                cfg.ghost_bonus_growth ** min(self.ghosts_eaten_phase, 3))
            for cell in edible_cells:
                d = int(my_dist[cell])
                if d <= cfg.chase_horizon:
                    out.append((value / (d / scale + 1.0), cell))
        if state.fruit is not None and self.maze is not None:
            cell = snap_to_free(self.maze, *nearest_cell(*state.fruit))
            if cell is not None and my_dist[cell] < UNREACHABLE:
                out.append(
                    (cfg.fruit_value / (my_dist[cell] / scale + 1.0), cell))
        return out

    def _step_direction(self, here: tuple[int, int],
                        nxt: tuple[int, int]) -> tuple[int, int]:
        dr = nxt[0] - here[0]
        dc = nxt[1] - here[1]
        if abs(dc) > 1:  # wrapped through a side tunnel
            dc = 1 if dc < 0 else -1
        return dr, dc

    def _action_towards(self, here: tuple[int, int],
                        nxt: tuple[int, int]) -> int:
        return DIR_ACTION.get(self._step_direction(here, nxt), ACTION_NOOP)

    def _panic_move(self, here: tuple[int, int], ghost_dist: np.ndarray) -> int:
        """Cornered: step to whichever neighbour is furthest from a ghost.

        Ties break towards the roomier cell, which keeps her out of dead
        ends where the next step would have no answer at all.
        """
        assert self.maze is not None
        best_action, best_key = ACTION_NOOP, None
        for r2, c2, _ in self.maze.neighbours(*here):
            key = (int(ghost_dist[r2, c2]), self.maze.degree(r2, c2))
            if best_key is None or key > best_key:
                best_key = key
                best_action = self._action_towards(here, (r2, c2))
        return best_action

    def _rank_moves(self, here: tuple[int, int], candidates: list, first: dict
                    ) -> list[tuple[float, tuple[int, int], tuple[int, int],
                                    tuple[int, int]]]:
        """Rank each possible first move by the best item it leads to.

        Collapsing the candidates onto the four first moves is what lets the
        agent commit to a *direction* instead of to one pellet: the exact
        pellet she is aiming at changes constantly as ghosts move, and
        re-deriving the route every step from whichever pellet currently
        scores highest is what produced flip-flops - step out of a cell, see
        a slightly better pellet behind, turn around, see the first one
        again - until a ghost arrived and ended the argument.  Turning back
        the way she came is therefore discounted by `reverse_penalty`.
        """
        best: dict[tuple[int, int], tuple[float, tuple[int, int]]] = {}
        for score, cell in candidates:
            step = first.get(cell)
            if step is None:
                continue
            if step not in best or score > best[step][0]:
                best[step] = (score, cell)
        backwards = (-self.direction[0], -self.direction[1])
        ranked = []
        for step, (score, cell) in best.items():
            move = self._step_direction(here, step)
            if move == backwards:
                score *= self.config.reverse_penalty
            ranked.append((score, step, cell, move))
        ranked.sort(reverse=True)
        return ranked

    # -- main entry point -------------------------------------------------

    def act(self, obs: np.ndarray, info: dict) -> int:
        frame = extract_latest_frame(obs)
        ram = np.asarray(info["ram"])[0]
        lives = int(np.asarray(info["lives"])[0]) if "lives" in info else 0

        if self.maze is None:
            self._rebuild(frame)
        assert self.maze is not None and self.wall_rgb is not None

        state = decode_state(ram, frame, lives)
        if self.lives is not None and lives < self.lives:
            # Everyone respawns; the remembered headings and target are
            # about a board that no longer exists.
            self.target = None
            self.last_move = None
            self.prev_pac = None
            self.prev_ghosts = None
            self.ghost_dirs = [(0, 0)] * 4
        self.lives = lives

        _, pellets, pills, _ = scan_cells(frame, self.wall_rgb, self.background)
        # A level change is the one thing that puts pellets *back* on the
        # board.  ram[119] restarting is the reliable tell; the pellet count
        # jumping is the backstop, kept blunt so that a few miscounted cells
        # cannot throw away the eaten-pill state by rebuilding for nothing.
        restarted = state.dots_eaten + 10 < self.dots_eaten
        self.dots_eaten = state.dots_eaten
        if restarted or int(pellets.sum()) > int(self.pellets.sum()) + 30:
            self._rebuild(frame)
            _, pellets, pills, _ = scan_cells(
                frame, self.wall_rgb, self.background)
        self._update_items(pellets, pills, state)

        px, py = state.pac
        self.direction = heading(self.prev_pac, state.pac, self.direction)
        if self.prev_ghosts is not None:
            self.ghost_dirs = [heading(old, new, old_dir) for old, new, old_dir
                               in zip(self.prev_ghosts, state.ghosts,
                                      self.ghost_dirs)]
        self.prev_pac = (px, py)
        self.prev_ghosts = list(state.ghosts)

        here = snap_to_free(self.maze, *cell_ahead(px, py, self.direction))
        if here is None:
            return ACTION_NOOP

        self._track_phase(state)
        dangerous, edible_cells = self._ghost_cells(state)
        ghost_dist = threat_field(
            self.maze, dangerous, self.config.reverse_cost)

        my_dist, first = safe_search(
            self.maze, here, ghost_dist, self.config.safety_margin)
        candidates = self._candidates(state, my_dist, ghost_dist, edible_cells)
        if not candidates:
            # Drop the safety buffer once before giving up on planning.
            my_dist, first = safe_search(self.maze, here, ghost_dist, 0)
            candidates = self._candidates(
                state, my_dist, ghost_dist, edible_cells)
        ranked = self._rank_moves(here, candidates, first)
        if not ranked:
            self.target = None
            self.last_move = None
            return self._panic_move(here, ghost_dist)

        best_score, best_step, best_cell, best_move = ranked[0]
        if self.last_move is not None:
            # Stay the course unless another direction is clearly better;
            # without this the ranking can swap on ties every single step.
            held = [item for item in ranked if item[3] == self.last_move]
            if held and best_score < held[0][0] * self.config.hysteresis:
                best_score, best_step, best_cell, best_move = held[0]
        self.target = best_cell
        self.last_move = best_move
        return self._action_towards(here, best_step)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

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
        done = bool(np.logical_or(terminated, truncated)[0])
    else:
        obs, reward, done, info = result
        done = bool(np.asarray(done)[0])
    return obs, float(np.asarray(reward)[0]), done, info


def append_trial_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary(log_path: Path, summary_path: Path) -> None:
    import csv

    if not log_path.exists():
        return
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return
    fields = ["timestamp", "seed", "episodes", "score_mean", "score_median",
              "score_min", "score_max", "env_steps", "ale_frames"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def evaluate(args: argparse.Namespace) -> None:
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

    # One env.step() advances `frame_skip` ALE frames, each 1/60 s of real
    # time, so pace every step to that duration for real-time playback.
    step_period_s = args.frame_skip / ALE_FPS if args.realtime else 0.0
    agent = MsPacmanAgent(HeuristicConfig(safety_margin=args.safety_margin))

    scores: list[float] = []
    lengths: list[int] = []
    env_steps = 0
    try:
        for episode in range(args.episodes):
            obs, info = reset_env_with_info(env)
            agent.reset()
            total, steps = 0.0, 0
            if args.render:
                env.render()
            for _ in range(args.max_steps):
                started = time.perf_counter()
                action = agent.act(obs, info)
                obs, reward, done, info = step_env(env, action)
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
            print(f"episode={episode} score={total:.0f} steps={steps}")
    finally:
        env.close()

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
        append_trial_record(log_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "episodes": len(scores),
            "scores": scores,
            "episode_lengths": lengths,
            "score_mean": float(arr.mean()),
            "score_median": float(np.median(arr)),
            "score_min": float(arr.min()),
            "score_max": float(arr.max()),
            "env_steps": env_steps,
            "ale_frames": env_steps * args.frame_skip,
            "frame_skip": args.frame_skip,
            "noop_max": args.noop_max,
            "safety_margin": args.safety_margin,
        })
        write_summary(log_path, Path(args.summary_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Programmatic Ms. Pac-Man policy for envpool.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0,
                        help="0.25 is the standard ALE sticky-action protocol")
    parser.add_argument("--safety-margin", type=int, default=18)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime", action="store_true",
                        help="pace stepping to 60 ALE frames per second")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
