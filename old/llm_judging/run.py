# run.py
import argparse
from bt.config import load_settings_file
from bt.pipeline import run_once
from bt.db import gen_run_key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    runs = load_settings_file(args.config)
    if len(runs) != 1:
        raise ValueError("run_once expects a single config object")

    run_key = gen_run_key()
    run_once(runs[0], run_key=run_key, non_interactive=True)

if __name__ == "__main__":
    main()
