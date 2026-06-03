from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="https://github.com/hoavien0110/training-free-tta-for-deepfake-detection.git")
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--repo-dir", default="/kaggle/working/training-free-tta-for-deepfake-detection")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    if repo_dir.exists():
        run(["git", "fetch", "origin"], cwd=repo_dir)
        run(["git", "checkout", args.branch], cwd=repo_dir)
        run(["git", "pull", "origin", args.branch], cwd=repo_dir)
    else:
        run(["git", "clone", "-b", args.branch, args.repo_url, str(repo_dir)], cwd=Path("/kaggle/working"))

    run(["git", "status", "--short"], cwd=repo_dir)
    run(["git", "log", "-1", "--oneline"], cwd=repo_dir)

    if not args.skip_install:
        run(["python", "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=repo_dir)

    import deepfake_tta  # noqa: F401

    print("deepfake_tta imported OK")


if __name__ == "__main__":
    main()
