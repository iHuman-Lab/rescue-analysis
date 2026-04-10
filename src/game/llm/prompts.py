from game.sar.observations import (
    EMPTY,
    FAKE_VICTIM,
    LAVA,
    VICTIM,
    WALL,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_DIR_CHARS = {0: ">", 1: "v", 2: "<", 3: "^"}

# --- Sparse prompt: LLM replies with one or two action words only ---
SPARSE_SYSTEM_PROMPT = """\
You are a navigator in a Search and Rescue simulation.
Reply with ONLY the next action word(s) from: left, right, forward, pickup, drop, toggle, done.
No explanation. No punctuation. Examples: "forward", "left forward", "toggle".

Current game state:
{game_state}"""

# --- Detailed prompt: LLM replies with 1–2 sentence guidance ---
DETAILED_SYSTEM_PROMPT = """\
You are an AI assistant embedded in a Search and Rescue (SAR) simulation.
The agent navigates a grid world to rescue victims. You can see the full map.
Available actions: left, right, forward, pickup, drop, toggle (open/close doors), done.
Respond with 1–2 sentences of clear, direct guidance. Mention the action and reason briefly.

Current game state:
{game_state}"""

# Default prompt (kept for backwards compatibility)
SYSTEM_PROMPT = DETAILED_SYSTEM_PROMPT


def _cell_symbol(cell: int) -> str:
    if cell == EMPTY:
        return "."
    if cell == WALL:
        return "#"
    if cell == LAVA:
        return "~"
    if cell == VICTIM:
        return "V"
    if cell == FAKE_VICTIM:
        return "F"
    if is_door(cell):
        return "D"
    if is_key(cell):
        return "K"
    return "?"


def ascii_map(obs: dict) -> str:
    """Render ASCII map with agent direction arrow."""
    ax, ay = obs["agent_x"], obs["agent_y"]
    agent_char = _DIR_CHARS.get(obs["agent_dir"], "A")
    lines = []
    for y, row in enumerate(obs["grid"]):
        row_str = ""
        for x, cell in enumerate(row):
            row_str += agent_char if (x == ax and y == ay) else _cell_symbol(int(cell))
        lines.append(row_str)
    return "\n".join(lines)


def object_legend(obs: dict) -> str:
    """List all doors, keys, victims, and lava with full color/state detail."""
    victims, fakes, lavas, doors, keys = [], [], [], [], []
    for y, row in enumerate(obs["grid"]):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell == LAVA:
                lavas.append(f"({x},{y})")
            elif cell == VICTIM:
                victims.append(f"({x},{y})")
            elif cell == FAKE_VICTIM:
                fakes.append(f"({x},{y})")
            elif is_door(cell):
                d = decode_door(cell)
                state = (
                    "open"
                    if d["is_open"]
                    else ("locked" if d["is_locked"] else "closed")
                )
                doors.append(f"  D at ({x},{y}) = {d['color']} [{state}]")
            elif is_key(cell):
                k = decode_key(cell)
                keys.append(f"  K at ({x},{y}) = {k['color']} key")

    lines = []
    if doors:
        lines.append("Doors:")
        lines.extend(doors)
    if keys:
        lines.append("Keys:")
        lines.extend(keys)
    if victims:
        lines.append("Victims at: " + ", ".join(victims))
    if fakes:
        lines.append("Fake victims at: " + ", ".join(fakes))
    if lavas:
        lines.append("Lava at: " + ", ".join(lavas))
    return "\n".join(lines)



def sparse_prompt(obs: dict) -> str:
    """Build a sparse prompt — LLM should reply with action word(s) only."""
    return SPARSE_SYSTEM_PROMPT.format(game_state=to_text(obs))


def detailed_prompt(obs: dict) -> str:
    """Build a detailed prompt — LLM should reply with 1–2 sentence guidance."""
    return DETAILED_SYSTEM_PROMPT.format(game_state=to_text(obs))


def to_text(obs: dict) -> str:
    """Format the enriched obs dict as a human-readable summary for the LLM."""
    carrying = obs["carrying"]
    inventory = f"{carrying.capitalize()} Key" if carrying else "None"
    dir_name = _DIR_NAMES.get(obs["agent_dir"], "Unknown")
    return (
        f"Mission status: {obs['mission_status']}\n"
        f"Victims rescued: {obs['saved_victims']}\n"
        f"Victims remaining: {obs['remaining_victims']}\n"
        f"Steps taken: {obs['step_count']} / {obs['max_steps']}\n"
        f"Inventory: {inventory}\n"
        f"Agent at ({obs['agent_x']},{obs['agent_y']}) facing {dir_name}\n\n"
        f"Map (. empty  # wall  ~ lava  D door  K key  V victim  F fake  > v < ^ agent):\n"
        f"{ascii_map(obs)}\n\n"
        f"{object_legend(obs)}"
    )
