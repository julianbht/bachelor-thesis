import argparse
from config import load_settings_file
from pipeline import run_once

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    runs = load_settings_file(args.config)
    if len(runs) != 1:
        raise ValueError("run_once expects a single config object")
    run_once(runs[0], non_interactive=True)

if __name__ == "__main__":
    main()
