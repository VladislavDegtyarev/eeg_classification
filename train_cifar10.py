#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

COMMAND = (
    "poetry",
    "run",
    "python",
    "src/train.py",
    "--config-name=cifar10_train",
)


def run() -> int:
    """Execute CIFAR-10 training with the predefined configuration."""
    project_root = Path(__file__).resolve().parent
    command = COMMAND + tuple(sys.argv[1:])

    print("Starting CIFAR-10 training with `cifar10_train` configuration.")
    print(f"Working directory: {project_root}")
    print(f"Command: {' '.join(command)}")

    try:
        subprocess.run(command, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Training failed with exit code {exc.returncode}.")
        return exc.returncode

    print("Training completed successfully.")
    print(f"Logs are stored under: {project_root / 'logs/train/runs'}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
