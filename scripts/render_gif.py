#!/usr/bin/env python3
"""
Render a simple animated GIF of three-body motion from a CSV trajectory.

Input is the wide CSV produced by `threebody-cli simulate` / `threebody-core` CSV writer.
We only require the position columns: r1_x,r1_y,r2_x,r2_y,r3_x,r3_y.

Dependencies:
  python -m pip install pillow
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Frame:
    step: Optional[int]
    t: Optional[float]
    r1: Tuple[float, float]
    r2: Tuple[float, float]
    r3: Tuple[float, float]


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a three-body trajectory GIF from CSV.")
    p.add_argument("--input", "-i", default="traj.csv", help="Input CSV path (default: traj.csv)")
    p.add_argument(
        "--output",
        "-o",
        default="threebody.gif",
        help="Output GIF path (default: threebody.gif)",
    )
    p.add_argument("--size", type=int, default=640, help="Output image size (square pixels)")
    p.add_argument("--padding", type=int, default=40, help="Border padding (pixels)")
    p.add_argument("--radius", type=int, default=6, help="Body marker radius (pixels)")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    p.add_argument("--stride", type=int, default=1, help="Take every Nth row (default: 1)")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames (0 = unlimited)",
    )
    p.add_argument(
        "--trail",
        type=int,
        default=0,
        help="Trail length in frames (0 = no trail)",
    )
    return p.parse_args()


def require_columns(index: dict[str, int], cols: Iterable[str], *, path: Path) -> None:
    missing = [c for c in cols if c not in index]
    if not missing:
        return
    eprint(f"error: missing required columns in {path}: {', '.join(missing)}")
    eprint("hint: generate CSV via `cargo run -p threebody-cli -- simulate ...`")
    raise SystemExit(2)


def read_frames(path: Path, *, stride: int, max_frames: int) -> Tuple[List[Frame], Tuple[float, float, float, float]]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            eprint(f"error: empty CSV: {path}")
            raise SystemExit(2)

        index = {name: i for i, name in enumerate(header)}
        require_columns(index, ["r1_x", "r1_y", "r2_x", "r2_y", "r3_x", "r3_y"], path=path)

        col_step = index.get("step")
        col_t = index.get("t")

        frames: List[Frame] = []
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        for row_i, row in enumerate(reader):
            if stride > 1 and (row_i % stride) != 0:
                continue
            try:
                r1 = (float(row[index["r1_x"]]), float(row[index["r1_y"]]))
                r2 = (float(row[index["r2_x"]]), float(row[index["r2_y"]]))
                r3 = (float(row[index["r3_x"]]), float(row[index["r3_y"]]))
            except (ValueError, IndexError):
                continue

            step_val: Optional[int] = None
            if col_step is not None:
                try:
                    step_val = int(float(row[col_step]))
                except (ValueError, IndexError):
                    step_val = None

            t_val: Optional[float] = None
            if col_t is not None:
                try:
                    t_val = float(row[col_t])
                except (ValueError, IndexError):
                    t_val = None

            for (x, y) in (r1, r2, r3):
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            frames.append(Frame(step=step_val, t=t_val, r1=r1, r2=r2, r3=r3))
            if max_frames and len(frames) >= max_frames:
                break

    if not frames:
        eprint(f"error: no usable rows found in CSV: {path}")
        raise SystemExit(2)

    return frames, (min_x, max_x, min_y, max_y)


def main() -> int:
    args = parse_args()

    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        eprint("error: Pillow is required to render GIFs.")
        eprint("install: python -m pip install pillow")
        return 2

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames, (min_x, max_x, min_y, max_y) = read_frames(
        in_path, stride=int(args.stride), max_frames=int(args.max_frames)
    )

    size = int(args.size)
    padding = int(args.padding)
    radius = int(args.radius)
    trail = int(args.trail)

    range_x = max_x - min_x
    range_y = max_y - min_y
    span = max(range_x, range_y)
    if span == 0.0:
        span = 1.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half = span * 0.55  # 10% margin (0.5 -> tight)
    world_min_x = cx - half
    world_max_x = cx + half
    world_min_y = cy - half
    world_max_y = cy + half
    world_span = world_max_x - world_min_x

    usable = max(1, size - 2 * padding)

    def to_px(p: Tuple[float, float]) -> Tuple[int, int]:
        x, y = p
        xn = (x - world_min_x) / world_span
        yn = (y - world_min_y) / world_span
        px = padding + int(round(xn * usable))
        py = size - padding - int(round(yn * usable))
        return px, py

    colors = [(220, 50, 47), (38, 139, 210), (133, 153, 0)]  # red, blue, green

    duration_ms = int(round(1000.0 / max(1e-6, float(args.fps))))
    gif_frames: List["Image.Image"] = []

    for i, fr in enumerate(frames):
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if trail > 0:
            start = max(0, i - trail)
            for body_idx in range(3):
                pts: List[Tuple[int, int]] = []
                for k in range(start, i + 1):
                    fk = frames[k]
                    pos = (fk.r1, fk.r2, fk.r3)[body_idx]
                    pts.append(to_px(pos))
                if len(pts) >= 2:
                    draw.line(pts, fill=colors[body_idx], width=2)

        for body_idx, pos in enumerate([fr.r1, fr.r2, fr.r3]):
            x, y = to_px(pos)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=colors[body_idx],
                outline=(0, 0, 0),
            )

        gif_frames.append(img.convert("P", palette=Image.ADAPTIVE))

    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    eprint(f"wrote: {out_path} ({len(gif_frames)} frames, {duration_ms} ms/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
