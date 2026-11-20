"""The command line of the outer loop.

python -m tabular_preprocessing.outer_loop CONFIG --download         # fetch the datasets once (login node)
python -m tabular_preprocessing.outer_loop CONFIG --dry-run          # the plan and a cost estimate
python -m tabular_preprocessing.outer_loop CONFIG --trials 1         # a first try
python -m tabular_preprocessing.outer_loop CONFIG                    # the study (continues where it stopped)
python -m tabular_preprocessing.outer_loop CONFIG --summary          # the trials, best first
python -m tabular_preprocessing.outer_loop CONFIG --effects          # mean value per hyperparameter level
python -m tabular_preprocessing.outer_loop CONFIG --final 5          # the best 5 on the held-out test split
python -m tabular_preprocessing.outer_loop CONFIG --export-stage2 5 configs/stage2.yaml
python -m tabular_preprocessing.outer_loop CONFIG --shard 3/20             # one job of a cluster array
python -m tabular_preprocessing.outer_loop CONFIG --merge                  # combine the shards
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from tabular_preprocessing.inner_loop.data import download_relbench_task
from tabular_preprocessing.outer_loop.config import ExperimentConfig
from tabular_preprocessing.outer_loop.report import effects, export_stage2, final_evaluation, plan
from tabular_preprocessing.outer_loop.study import _optimizer, merge_shards, n_trials_of, run_study, summarize


def main(argv: list[str] | None = None) -> None:
    """Run, plan, summarize or report on a study."""
    parser = argparse.ArgumentParser(description="Search MAS architectures and hyperparameters on RelBench tasks.")
    parser.add_argument("config", help="the YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="show the plan and a cost estimate, run nothing")
    parser.add_argument("--trials", type=int, help="run at most this many new trials")
    parser.add_argument("--summary", action="store_true", help="print the trials so far, best first")
    parser.add_argument("--effects", action="store_true", help="print the mean value per hyperparameter level")
    parser.add_argument("--final", type=int, metavar="K", help="evaluate the best K on the held-out test split")
    parser.add_argument("--export-stage2", nargs=2, metavar=("K", "PATH"), help="write a stage-2 config of the best K")
    parser.add_argument("--shard", metavar="I/N", help="run only shard I of N (cluster job arrays), into shards/I-of-N")
    parser.add_argument("--merge", action="store_true", help="combine the shards into the study folder")
    parser.add_argument("--download", action="store_true", help="download the config's datasets and tasks, run nothing")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if args.download:
            for name in config.tasks:
                print(f"Downloading {name} ...", flush=True)
                download_relbench_task(*name.split("/"))
            print("Done.")
        elif args.summary:
            print(summarize(config).to_string(index=False))
        elif args.effects:
            print(effects(config).to_string(index=False))
        elif args.final is not None:
            print(final_evaluation(config, top=args.final).to_string(index=False))
        elif args.merge:
            merge_shards(config)
        elif args.export_stage2:
            top, path = args.export_stage2
            print(f"Wrote {export_stage2(config, int(top), path)}")
        elif args.dry_run:
            print("\n".join(plan(config)))
            preview = _optimizer(config)
            n = n_trials_of(config)
            for _ in range(min(10, n)):
                candidate = preview.ask()
                if candidate is None:
                    break
                preview.tell(candidate, random.random())  # a placeholder value: the preview evaluates nothing
                print(f"  {candidate.key}")
            if n > 10:
                print(f"  ... and {n - 10} more")
        else:
            shard = None
            if args.shard:
                i, n = (int(x) for x in args.shard.split("/"))
                shard = (i, n)
            run_study(config, max_new_trials=args.trials, shard=shard)
