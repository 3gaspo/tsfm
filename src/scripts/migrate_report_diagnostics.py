"""Move legacy report diagnostics into the current diagnostics hierarchy."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


DIAGNOSTIC_DIRECTORIES = ("plots", "averaged_inputs")


def migrate_report_diagnostics(outputs_root: str | Path) -> list[tuple[Path, Path]]:
    """Move legacy diagnostic directories without overwriting existing data."""
    outputs_root = Path(outputs_root).expanduser().resolve()
    reports_root = outputs_root / "reports"
    diagnostics_root = outputs_root / "diagnostics"
    if not reports_root.is_dir():
        return []

    moves: list[tuple[Path, Path]] = []
    for name in DIAGNOSTIC_DIRECTORIES:
        for source in sorted(reports_root.rglob(name)):
            if not source.is_dir():
                continue
            relative_parent = source.parent.relative_to(reports_root)
            destination = diagnostics_root / relative_parent / name
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing diagnostics directory: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moves.append((source, destination))
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move TSFM plots and averaged inputs out of reports/."
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="TSFM output tree containing reports/ (default: outputs)",
    )
    args = parser.parse_args()

    moves = migrate_report_diagnostics(args.outputs_root)
    if not moves:
        print("No legacy report diagnostics found.")
        return
    for source, destination in moves:
        print(f"Moved {source} -> {destination}")


if __name__ == "__main__":
    main()
