"""Run the prespecified baselines and TGRE ablations, then write a result table."""

import argparse
import json
from pathlib import Path

from tgre_experiment import ARTIFACTS, Corpus, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["visual", "text", "late", "rtgre", "rtgre_flow"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-split", default="train", choices=["train", "fit"])
    parser.add_argument("--eval-split", default="valid", choices=["valid", "dev"])
    parser.add_argument("--output-stem", default="experiment_results")
    parser.add_argument("--eval-epochs", nargs="+", type=int)
    parser.add_argument("--text-mode", default="counts", choices=["counts", "tfidf"])
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--text-access", default="causal", choices=["causal", "symmetric10"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="Skip model/seed pairs already present in the partial output")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    corpus = Corpus(ARTIFACTS / "modality_coverage.json", vocab_size=args.vocab_size, text_mode=args.text_mode, text_access=args.text_access)
    partial_path = ARTIFACTS / f"{args.output_stem}_partial.json"
    results = []
    if args.resume and partial_path.exists():
        results = json.loads(partial_path.read_text(encoding="utf-8"))
    completed = {(row["model"], int(row["seed"])) for row in results}
    for model in args.models:
        for seed in args.seeds:
            if (model, seed) in completed:
                continue
            results.append(train(model, seed, epochs=args.epochs, corpus=corpus, train_split=args.train_split, eval_split=args.eval_split, eval_epochs=set(args.eval_epochs) if args.eval_epochs else None, requested_device=args.device))
            partial_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ARTIFACTS / f"{args.output_stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
