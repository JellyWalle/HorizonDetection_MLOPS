import argparse
import yaml
from pathlib import Path
import sys

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "predict"], help="Run mode")
    parser.add_argument("--config", default="configs/train.yaml", help="Config file path")
    parser.add_argument("--ckpt", help="Checkpoint path for eval/predict")
    args = parser.parse_args()

    if not args.config.exists():
        print(f" Check Config file: {args.config}")
        sys.exit(1)

    cfg = load_config()
    
    if args.mode == "train":
        from src.training.trainer import Trainer
        Trainer(cfg).train()
    elif args.mode == "eval":
        from src.evaluation import evaluator
        evaluator.run(cfg, checkpoint=args.ckpt)
    elif args.mode == "predict":
        from src.inference import predictor
        predictor.run(cfg, checkpoint=args.ckpt)

if __name__ == "__main__":
    main()