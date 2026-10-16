"""Thay đường dẫn kết quả lịch sử trong tài liệu theo migration_map.json."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def update(root=ROOT, apply=False):
    mapping = json.loads((Path(root) / "results/_system/migration_map.json").read_text(encoding="utf-8"))
    pairs = [("results/" + row["from"], "results/" + row["to"]) for row in mapping
             if not row["from"].startswith("INDEX.")]
    for row in mapping:
        parts = Path(row["to"]).parts
        if len(parts) >= 6 and parts[3].startswith("samples_"):
            previous = Path(*parts[:4], *parts[5:]).as_posix()
            older = Path(*parts[:2], parts[3], *parts[5:]).as_posix()
            pairs.extend((("results/" + previous, "results/" + row["to"]),
                          ("results/" + older, "results/" + row["to"])))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    paths = [Path(root) / "README.md"]
    paths += sorted((Path(root) / "docs").rglob("*.md"))
    paths += sorted((Path(root) / "experiments").rglob("*.md"))
    changed = []
    for path in paths:
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        revised = original
        for old, new in pairs:
            revised = revised.replace(old, new)
        if revised != original:
            changed.append(path)
            if apply:
                path.write_text(revised, encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    paths = update(apply=args.apply)
    print(f"{len(paths)} tài liệu {'đã cập nhật' if args.apply else 'cần cập nhật'}")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
