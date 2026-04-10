"""
Replay a recorded game as a movie.

Reads a JSONL file where each line is a JSON game state (from the LSL stream)
and renders each frame using pygame, drawn from the grid data.

Usage:
    python replay.py <recording.jsonl> [--fps 10] [--tile 24]

Controls:
    Space       Pause / resume
    Right arrow Step forward one frame (when paused)
    Left arrow  Step back one frame (when paused)
    Escape      Quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame

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

# RGB colors for each cell type
_CELL_COLORS = {
    EMPTY: (220, 220, 220),
    WALL: (40, 40, 40),
    LAVA: (230, 80, 10),
    VICTIM: (30, 180, 30),
    FAKE_VICTIM: (180, 160, 30),
}

_COLOR_RGB = {
    "red": (200, 50, 50),
    "green": (50, 180, 50),
    "blue": (50, 50, 200),
    "purple": (150, 50, 180),
    "yellow": (210, 200, 30),
    "grey": (130, 130, 130),
}

_DIR_VECTORS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def _cell_color(cell: int) -> tuple[int, int, int]:
    if is_door(cell):
        d = decode_door(cell)
        base = _COLOR_RGB.get(d["color"], (128, 128, 128))
        if d["is_locked"]:
            return tuple(max(0, c - 70) for c in base)  # darker = locked
        if d["is_open"]:
            return tuple(min(255, c + 60) for c in base)  # lighter = open
        return base
    if is_key(cell):
        k = decode_key(cell)
        return _COLOR_RGB.get(k["color"], (128, 128, 128))
    return _CELL_COLORS.get(cell, (180, 0, 180))  # magenta = unknown


def _render_frame(surface: pygame.Surface, state: dict, tile: int) -> None:
    surface.fill((0, 0, 0))
    grid = state["grid"]
    ax, ay = state["agent_x"], state["agent_y"]
    agent_dir = state.get("agent_dir", 0)

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            cell = int(cell)
            color = _cell_color(cell)
            rect = pygame.Rect(x * tile, y * tile, tile, tile)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)

    # Agent: filled circle + direction line
    cx = ax * tile + tile // 2
    cy = ay * tile + tile // 2
    r = max(3, tile // 3)
    pygame.draw.circle(surface, (255, 230, 0), (cx, cy), r)
    dx, dy = _DIR_VECTORS.get(agent_dir, (1, 0))
    pygame.draw.line(
        surface, (0, 0, 0), (cx, cy), (cx + dx * r, cy + dy * r), max(1, tile // 8)
    )


def _render_hud(
    surface: pygame.Surface,
    state: dict,
    idx: int,
    total: int,
    paused: bool,
    font: pygame.font.Font,
) -> None:
    action_idx = state.get("action")
    action = _ACTION_NAMES[action_idx] if action_idx is not None else "—"
    carrying = state.get("carrying") or "none"
    saved = state.get("saved_victims", 0)
    remaining = state.get("remaining_victims", "?")
    step = state.get("step_count", idx)
    max_steps = state.get("max_steps", "?")
    status = "⏸ PAUSED" if paused else "▶ PLAYING"

    lines = [
        f"{status}   frame {idx + 1}/{total}",
        f"step {step}/{max_steps}   action: {action}",
        f"carrying: {carrying}   saved: {saved}   remaining: {remaining}",
    ]
    y = 4
    for line in lines:
        text = font.render(line, True, (255, 255, 255), (0, 0, 0))
        surface.blit(text, (4, y))
        y += text.get_height() + 2


def load_jsonl(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def replay(path: str, fps: int = 10, tile: int = 24) -> None:
    frames = load_jsonl(path)
    if not frames:
        print("No frames found in file.")
        return

    first = frames[0]
    grid_h = len(first["grid"])
    grid_w = len(first["grid"][0])
    hud_height = 54

    pygame.init()
    screen = pygame.display.set_mode((grid_w * tile, grid_h * tile + hud_height))
    pygame.display.set_caption(f"Replay: {Path(path).name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)

    grid_surface = pygame.Surface((grid_w * tile, grid_h * tile))

    idx = 0
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)

        state = frames[idx]
        _render_frame(grid_surface, state, tile)
        screen.blit(grid_surface, (0, 0))
        _render_hud(screen, state, idx, len(frames), paused, font)
        pygame.display.flip()
        clock.tick(fps)

        if not paused:
            idx += 1
            if idx >= len(frames):
                idx = 0  # loop

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded SAR game.")
    parser.add_argument("file", help="Path to the JSONL recording file")
    parser.add_argument(
        "--fps", type=int, default=10, help="Playback speed (default 10)"
    )
    parser.add_argument(
        "--tile", type=int, default=24, help="Tile size in pixels (default 24)"
    )
    args = parser.parse_args()

    replay(args.file, fps=args.fps, tile=args.tile)
