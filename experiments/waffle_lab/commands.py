"""In từng lệnh Kaggle để tự chọn chạy; chương trình không gửi job."""
import argparse
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.waffle_lab.core import VARIANTS


def commands(task="both", samples=3000, seeds=(0,), methods=VARIANTS):
    if type(samples) is not int or samples < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Số mẫu dương, seed không rỗng/không lặp")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seed phải trong [0, 2**32)")
    if task not in ("both", "informativeness", "humanitarian") or any(m not in VARIANTS for m in methods):
        raise ValueError("Task/method không hợp lệ")
    for current in ("informativeness", "humanitarian") if task == "both" else (task,):
        job = "waffle_info" if current == "informativeness" else "waffle_humanitarian"
        for method in methods:
            for seed in seeds[:1] if method in ("clip", "semantic") else seeds:
                yield shlex.join(["python", "-B", "scripts/launch_runs.py", "--launcher-config",
                    "experiments/waffle_lab/launcher.yaml", "--job", job, "--method", method,
                    "--seed", str(seed), "--max-samples", str(samples), "--use-kaggle", "true"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("both", "informativeness", "humanitarian"), default="both")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    args = parser.parse_args(argv)
    print("\n".join(commands(args.task, args.max_samples, args.seeds, args.methods)))


if __name__ == "__main__":
    main()
