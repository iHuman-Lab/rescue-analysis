from __future__ import annotations

import numpy as np

# Base IDs for simple types
EMPTY = 0
WALL = 1
LAVA = 4
VICTIM = 5
FAKE_VICTIM = 6

# Doors: DOOR_BASE + color_idx * 3 + state  (state: 0=open, 1=closed, 2=locked)
# 6 colors × 3 states = 18 values → IDs 10..27
DOOR_BASE = 10

# Keys: KEY_BASE + color_idx
# 6 colors → IDs 30..35
KEY_BASE = 30

_COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
_COLOR_IDX = {c: i for i, c in enumerate(_COLORS)}

OBJ_TO_ID: dict[str | None, int] = {
    None: EMPTY,
    "Wall": WALL,
    "Lava": LAVA,
    "Victim": VICTIM,
    "FakeVictim": FAKE_VICTIM,
}


def _door_id(color: str, is_open: bool, is_locked: bool) -> int:
    state = 0 if is_open else (2 if is_locked else 1)
    return DOOR_BASE + _COLOR_IDX[color] * 3 + state


def _key_id(color: str) -> int:
    return KEY_BASE + _COLOR_IDX[color]


def is_door(cell: int) -> bool:
    return DOOR_BASE <= cell < DOOR_BASE + len(_COLORS) * 3


def is_key(cell: int) -> bool:
    return KEY_BASE <= cell < KEY_BASE + len(_COLORS)


def decode_door(cell: int) -> dict:
    offset = cell - DOOR_BASE
    return {
        "color": _COLORS[offset // 3],
        "is_open": offset % 3 == 0,
        "is_locked": offset % 3 == 2,
    }


def decode_key(cell: int) -> dict:
    return {"color": _COLORS[cell - KEY_BASE]}


def scan_grid(env) -> list:
    """Single-pass grid scan returning a (height x width) int list.

    Encoding:
      0=empty, 1=wall, 4=lava, 5=victim, 6=fake_victim
      10-27: door (DOOR_BASE + color_idx*3 + state; state 0=open,1=closed,2=locked)
      30-35: key  (KEY_BASE + color_idx)
    """
    arr = np.zeros((env.height, env.width), dtype=np.int16)
    for y in range(env.height):
        for x in range(env.width):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            type_name = type(obj).__name__
            if type_name == "Door":
                arr[y, x] = _door_id(obj.color, obj.is_open, obj.is_locked)
            elif type_name == "Key":
                arr[y, x] = _key_id(obj.color)
            else:
                arr[y, x] = OBJ_TO_ID.get(type_name, 0)
    return arr.tolist()


def grid_to_array(env) -> np.ndarray:
    """Return a (height x width) int16 array using the full encoding."""
    arr = np.zeros((env.height, env.width), dtype=np.int16)
    for y in range(env.height):
        for x in range(env.width):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            type_name = type(obj).__name__
            if type_name == "Door":
                arr[y, x] = _door_id(obj.color, obj.is_open, obj.is_locked)
            elif type_name == "Key":
                arr[y, x] = _key_id(obj.color)
            else:
                arr[y, x] = OBJ_TO_ID.get(type_name, 0)
    return arr


class GameObservation:
    """Processes the raw env observation into an enriched obs dict each step.

    Instantiated once on the env. Call process_observation(obs, env) each step.
    """

    def process_observation(self, obs: dict, env) -> dict:
        """Enrich and return the obs dict with full game state."""
        grid = scan_grid(env)
        carrying = getattr(env, "carrying", None)
        mission = env.get_mission_status()

        obs["image"] = obs["image"].tolist()
        obs["direction"] = int(obs["direction"])
        obs.update(
            {
                "grid": grid,
                "agent_x": int(env.agent_pos[0]),
                "agent_y": int(env.agent_pos[1]),
                "agent_dir": int(env.agent_dir),
                "carrying": carrying.color if carrying else None,
                "step_count": int(env.step_count),
                "max_steps": int(env.max_steps),
                "mission_status": mission["status"],
                "saved_victims": mission["saved_victims"],
                "remaining_victims": mission["remaining_victims"],
                "num_rows": env.num_rows,
                "num_cols": env.num_cols,
                "room_size": int(env.room_size),
            }
        )
        return obs
