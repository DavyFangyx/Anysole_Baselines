from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    assert args.dataset.endswith("_test.pt"), "The dataset must be a test set"

    datasets = [args.dataset]
    if not args.only_test:
        datasets.append(args.dataset.replace("_test.pt", "_val.pt"))
        datasets.append(args.dataset.replace("_test.pt", "_train.pt"))

    for dataset_path in datasets:
        cmd = [
            sys.executable,
            "src/test.py",
            args.model,
            "--dataset",
            dataset_path,
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model folder")
    parser.add_argument("dataset", type=str, help="Path to the test dataset")
    parser.add_argument("--only_test", action="store_true", help="Only use the test set")
    args = parser.parse_args()
    main(args)
