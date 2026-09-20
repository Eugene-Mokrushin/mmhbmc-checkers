import argparse
import json
import shutil
import subprocess

from connectome.coordinates import one_per_neuron, read_markers
from connectome.load import load
from game.gradfly import GradFly
from paths import ARTIFACTS_DIR, REPO
from sim.kernels import best_device
from train.evaluate import RUNS_DIR
from train.student import Student

FLIES_DIR = ARTIFACTS_DIR / "flies"


def load_gradient_fly(name: str, device: str | None = None) -> GradFly:
    student = Student(load(min_syn=5), one_per_neuron(read_markers()), device or best_device())
    student.load(FLIES_DIR / f"{name}.pt")
    return student.fly()


def main() -> None:
    # keep a gradient-trained checkpoint as a named fly, next to the dopamine flies
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="RUN/g-STEP.pt under artifacts/runs")
    parser.add_argument("--name", default="whole-gradient")
    args = parser.parse_args()

    FLIES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUNS_DIR / args.checkpoint, FLIES_DIR / f"{args.name}.pt")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    meta = {"name": args.name, "kind": "gradient", "checkpoint": args.checkpoint, "commit": commit}
    (FLIES_DIR / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"kept {args.checkpoint} as {args.name}")


if __name__ == "__main__":
    main()
