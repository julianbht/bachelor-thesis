import argparse, time, traceback
from llm_judging.bt.config import load_settings_file
from llm_judging.bt.pipeline import run_once

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON array of runs")
    ap.add_argument("--pause", type=int, default=5)
    args = ap.parse_args()

    runs = load_settings_file(args.config)
    total = len(runs)
    print(f"Starting sweep with {total} runs...\n")

    for i, spec in enumerate(runs, 1):
        print("="*80)

        try:
            run_once(spec, non_interactive=True)
        except Exception as e:
            print(f"Error in run {i}: {e!r}")
            traceback.print_exc()

        if i < total and args.pause > 0:
            time.sleep(args.pause)

    print("Sweep complete.")

if __name__ == "__main__":
    main()
