from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_project_python() -> str:
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.append(ROOT_DIR / ".venv" / "Scripts" / "python.exe")
    else:
        candidates.extend(
            [
                ROOT_DIR / ".venv" / "bin" / "python",
                ROOT_DIR / ".venv" / "bin" / "python3",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


PYTHON = _resolve_project_python()


def _run_step(name: str, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    # Ізолюємо виконання підпроцесу від зовнішніх PYTHONHOME/PYTHONPATH налаштувань.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if env_overrides:
        env.update(env_overrides)

    print(f"\n=== {name} ===")
    print("$ " + " ".join(args))

    result = subprocess.run(args, cwd=str(ROOT_DIR), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    print("Контроль якості: справжнє навчання (Real Training Quality Gate)")
    print("Цей gate перевіряє реальну поведінку навчання та виявлення на наборах даних проєкту.")
    print(f"Інтерпретатор проєкту: {PYTHON}")
    if Path(PYTHON).resolve() != Path(sys.executable).resolve():
        print(f"Поточний інтерпретатор виконання: {sys.executable}")
        print("Використовується інтерпретатор проєкту для всіх кроків gate.")

    _run_step(
        "Bootstrap необхідних артефактів моделі CIC з наборів даних",
        [PYTHON, "scripts/bootstrap_required_models.py"],
    )

    _run_step(
        "Запуск перевірки синтаксису (sanity checks)",
        [
            PYTHON,
            "-m",
            "compileall",
            "-q",
            "src",
            "scripts",
        ],
    )

    _run_step(
        "Запуск базових перевірок якості runtime (навчання, калібрування, політика порогів, строгий PCAP)",
        [
            PYTHON,
            "scripts/runtime_smoke_quality_checks.py",
        ],
        env_overrides={"IDS_STRICT_E2E": "1"},
    )

    print("\nВсі перевірки якості справжнього навчання пройдено успішно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
