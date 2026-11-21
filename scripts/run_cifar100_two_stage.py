"""Run the isolated CIFAR hierarchy experiment; defaults to five-image smoke."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_project.cifar100_two_stage import specification, split_indices, locate_source, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "dev", "test"], default="smoke")
    parser.add_argument("--config", default="configs/cifar100_two_stage.json")
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="results/final/cifar100_two_stage")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dev-run")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    spec = specification(args.config)
    split = "test" if args.mode == "test" else "dev"
    if args.prepare_only:
        print(json.dumps({"status": "plan_only", "mode": args.mode, "specification": spec,
                          "samples": len(split_indices(spec, split, args.mode == "smoke")),
                          "requires_full_dev": split == "test"}, indent=2))
    else:
        root = locate_source(args.input_root) if args.data_root == "auto" else args.data_root
        print(run(root, args.output_root, split=split, smoke=args.mode == "smoke",
                  dev_run=args.dev_run, spec_file=args.config, device=args.device))
