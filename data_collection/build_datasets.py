"""
Construct D0–D5 dataset manifests from collected raw trajectories.

Each dataset Di has (100 - 10i)% expert and (10i)% poison trajectories,
totalling N_TOTAL trajectories. Manifests are saved as JSON files in
data/datasets/ and contain the file paths + labels for each trajectory.

Poison is drawn from data/raw/poison_random/ and data/raw/poison_human/
combined. Human-provided poison is prioritised first.

Usage:
    python -m data_collection.build_datasets
    python -m data_collection.build_datasets --total 200 --out data/datasets
    python -m data_collection.build_datasets --preview   # show counts without writing
"""

import argparse
import glob
import json
import os
import random


POISON_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # D0 – D5


def _scan(directory: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, "traj_*.npz")))
    return paths


def build(
    total: int,
    out_dir: str,
    expert_dir: str   = "data/raw/expert",
    random_dir: str   = "data/raw/poison_random",
    human_dir: str    = "data/raw/poison_human",
    seed: int         = 42,
    preview: bool     = False,
) -> dict:
    """
    Build D0–D5 manifests.

    Returns a dict mapping dataset name → manifest dict.
    """
    rng = random.Random(seed)

    expert_paths = _scan(expert_dir)
    poison_paths = _scan(human_dir) + _scan(random_dir)   # human first

    print(f"Found {len(expert_paths)} expert trajectories in {expert_dir}")
    print(f"Found {len(poison_paths)} poison trajectories "
          f"({len(_scan(human_dir))} human + {len(_scan(random_dir))} random)")

    os.makedirs(out_dir, exist_ok=True)
    manifests = {}

    for i, pct in enumerate(POISON_LEVELS):
        n_poison = int(round(total * pct))
        n_expert = total - n_poison
        name = f"D{i}"

        if len(expert_paths) < n_expert:
            print(f"  WARNING {name}: need {n_expert} expert demos but only "
                  f"{len(expert_paths)} available.")
        if len(poison_paths) < n_poison:
            print(f"  WARNING {name}: need {n_poison} poison demos but only "
                  f"{len(poison_paths)} available.")

        chosen_expert = rng.sample(expert_paths, min(n_expert, len(expert_paths)))
        chosen_poison = rng.sample(poison_paths, min(n_poison, len(poison_paths)))

        entries = (
            [{"path": p, "label": "expert"} for p in chosen_expert] +
            [{"path": p, "label": "poison"} for p in chosen_poison]
        )
        rng.shuffle(entries)   # mix order so IRL doesn't see all poison at end

        manifest = {
            "name": name,
            "poison_pct": pct,
            "n_expert": len(chosen_expert),
            "n_poison": len(chosen_poison),
            "total": len(entries),
            "trajectories": entries,
        }
        manifests[name] = manifest

        print(f"  {name}: {len(chosen_expert):3d} expert + {len(chosen_poison):3d} poison "
              f"= {len(entries):3d} total  ({pct*100:.0f}% poison)")

        if not preview:
            path = os.path.join(out_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2)

    if not preview:
        print(f"\nManifests written to {out_dir}/")
    return manifests


def load_manifest(path: str) -> dict:
    """Load a dataset manifest JSON. Returns the manifest dict."""
    with open(path) as f:
        return json.load(f)


def load_trajectory_paths(manifest: dict) -> tuple[list[str], list[str]]:
    """
    Return (paths, labels) lists from a manifest.
    Labels are 'expert' or 'poison' — useful for evaluation but NOT
    passed to the IRL training (that would be cheating).
    """
    paths  = [e["path"] for e in manifest["trajectories"]]
    labels = [e["label"] for e in manifest["trajectories"]]
    return paths, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total",      type=int,  default=200)
    parser.add_argument("--out",        type=str,  default="data/datasets")
    parser.add_argument("--expert-dir", type=str,  default="data/raw/expert")
    parser.add_argument("--random-dir", type=str,  default="data/raw/poison_random")
    parser.add_argument("--human-dir",  type=str,  default="data/raw/poison_human")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--preview",    action="store_true",
                        help="Print counts without writing files")
    args = parser.parse_args()

    build(
        total=args.total,
        out_dir=args.out,
        expert_dir=args.expert_dir,
        random_dir=args.random_dir,
        human_dir=args.human_dir,
        seed=args.seed,
        preview=args.preview,
    )
