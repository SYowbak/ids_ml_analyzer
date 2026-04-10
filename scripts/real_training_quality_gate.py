from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_step(name: str, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    print(f"\n=== {name} ===")
    print("$ " + " ".join(args))

    result = subprocess.run(args, cwd=str(ROOT_DIR), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    print("Real Training Quality Gate")
    print("This gate validates actual training and detection behavior on project datasets.")

    _run_step(
        "Bootstrap required CIC model artifacts from datasets",
        [PYTHON, "scripts/bootstrap_required_models.py"],
    )

    _run_step(
        "Run training and calibration regression tests",
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/test_training_scanning_lifecycle_integration.py",
            "tests/test_model_engine_if_calibration.py",
            "tests/test_threshold_provenance_policy.py",
        ],
    )

    _run_step(
        "Run strict real-PCAP E2E gate",
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "tests/test_pcap_real_e2e_regression.py",
        ],
        env_overrides={"IDS_STRICT_E2E": "1"},
    )

    print("\nAll real training quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
