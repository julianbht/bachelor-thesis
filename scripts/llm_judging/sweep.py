#!/usr/bin/env python3
# sweep.py
import traceback
import time

from config import RUN_SPECS
from run import run_with_settings
from llm import ensure_model_downloaded

# Optional: minor pause between runs to reduce DB/model churn
PAUSE_BETWEEN_RUNS_SEC = 5

def main():
    if not RUN_SPECS:
        print("No RUN_SPECS configured in config.py. Nothing to do.")
        return

    print(f"Starting unattended sweep with {len(RUN_SPECS)} runs...\n")
    for idx, spec in enumerate(RUN_SPECS, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(RUN_SPECS)}] Preparing model: {spec.model}")
        try:
            # Pre-download outside the run to fail fast and avoid half-created runs
            ensure_model_downloaded(
                spec.model,
                retries=max(1, spec.retry_attempts),
                backoff_ms=spec.retry_backoff_ms
            )
            print(f"[{idx}/{len(RUN_SPECS)}] Model ready: {spec.model}")
        except Exception as e:
            print(f"[{idx}/{len(RUN_SPECS)}] ERROR pre-downloading '{spec.model}': {e!r}")
            traceback.print_exc()
            print("Skipping this run.\n")
            continue

        print(f"[{idx}/{len(RUN_SPECS)}] Starting run with model: {spec.model}")
        try:
            # Force non-interactive so it never prompts for notes
            run_with_settings(spec, non_interactive=True)
            print(f"[{idx}/{len(RUN_SPECS)}] Run finished: {spec.model}\n")
        except Exception as e:
            print(f"[{idx}/{len(RUN_SPECS)}] ERROR during run '{spec.model}': {e!r}")
            traceback.print_exc()
            print("Continuing to next run.\n")

        if PAUSE_BETWEEN_RUNS_SEC > 0 and idx < len(RUN_SPECS):
            time.sleep(PAUSE_BETWEEN_RUNS_SEC)

    print("\nAll sweep runs attempted. Exiting.")


if __name__ == "__main__":
    main()
