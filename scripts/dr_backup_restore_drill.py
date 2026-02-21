from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local backup/restore DR drill for critical directories.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--drill-dir", type=Path, default=Path("outputs/dr_drill"))
    args = parser.parse_args()

    drill_dir = args.drill_dir
    backup_root = drill_dir / "backup"
    restore_root = drill_dir / "restore"
    archive_path = drill_dir / "backup.tar.gz"

    backup_root.mkdir(parents=True, exist_ok=True)
    restore_root.mkdir(parents=True, exist_ok=True)

    critical_dirs = [
        args.outputs_dir / "models",
        args.outputs_dir / "reports",
        args.configs_dir,
    ]

    for src in critical_dirs:
        dst = backup_root / src.name
        _copytree(src, dst)

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(backup_root, arcname="backup")

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=restore_root)

    expected = [
        restore_root / "backup" / "models",
        restore_root / "backup" / "reports",
        restore_root / "backup" / "configs",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        print("DR drill: FAIL")
        for item in missing:
            print(f"- Missing restored path: {item}")
        return 1

    print("DR drill: PASS")
    print(f"Archive: {archive_path}")
    print(f"Restore root: {restore_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
