import argparse
import time
import traceback

from llm_judging.bt.config import load_settings_file
from llm_judging.bt.pipeline import run_once
from llm_judging.bt.db import gen_run_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON array of run specs")
    ap.add_argument("--pause", type=int, default=5, help="Seconds to pause between runs")
    args = ap.parse_args()

    runs = load_settings_file(args.config)
    total = len(runs)
    print(f"Starting sweep with {total} runs...\n")

    for i, spec in enumerate(runs, 1):
        print("=" * 80)
        run_key = gen_run_key()
        try:
            run_once(spec, run_key=run_key, non_interactive=True)
            print(f"Finished run {i}/{total} (key={run_key})")
        except Exception as e:
            # Errors are logged in the per-run log file by pipeline
            print(f"Run {i}/{total} (key={run_key}) failed: {e!r}")
            traceback.print_exc()

        if i < total and args.pause > 0:
            time.sleep(args.pause)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
