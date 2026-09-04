"""Read rsl_rl TensorBoard event files without launching Isaac."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_accumulator(run_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def list_scalars(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    if not path.is_dir():
        return []
    try:
        acc = _load_accumulator(path)
        return list(acc.Tags().get("scalars", []))
    except Exception:
        return []


def read_scalars(run_dir: str | Path, wanted: list[str] | None = None, max_points: int = 400) -> dict[str, list]:
    """Return {tag: [{x, y}, ...]} downsampled to ``max_points``."""
    path = Path(run_dir)
    if not path.is_dir():
        return {}
    try:
        acc = _load_accumulator(path)
    except Exception:
        return {}
    tags = acc.Tags().get("scalars", [])
    if wanted:
        tags = [t for t in tags if any(w.lower() in t.lower() for w in wanted)]
    out: dict[str, list] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        if len(events) > max_points:
            stride = max(1, len(events) // max_points)
            events = events[::stride]
        out[tag] = [{"x": int(e.step), "y": float(e.value)} for e in events]
    return out


def preferred_series(run_dir: str | Path) -> dict[str, list]:
    """Reward / success / entropy / KL — the four plots a beginner actually needs."""
    return read_scalars(
        run_dir,
        wanted=[
            "mean_reward",
            "Reward",
            "success",
            "entropy",
            "kl",
            "Episode_Reward",
            "Train/",
            "Loss/",
            "Policy/",
        ],
    )


def find_experiment_dir(repo_root: Path, experiment_name: str, run_name: str | None = None) -> Path | None:
    root = repo_root / "logs" / "rsl_rl" / experiment_name
    if not root.is_dir():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if run_name:
        matched = [p for p in dirs if run_name in p.name]
        if matched:
            return matched[-1]
    return dirs[-1] if dirs else None


def find_videos(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir)
    if not path.is_dir():
        return []
    items = []
    for mp4 in sorted(path.rglob("*.mp4")):
        rel = mp4.relative_to(path)
        items.append({"name": mp4.name, "rel": str(rel), "bytes": mp4.stat().st_size})
    return items
