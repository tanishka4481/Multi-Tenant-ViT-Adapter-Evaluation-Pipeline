import os
import sys
import time
import json
import shutil
import subprocess

OUTPUT_DIR = "outputs"
FINAL_ARCHIVE = "vision_transformer_pipeline_artifacts.zip"


def print_banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_step(step_name, command):
    print_banner(f"STARTING STEP: {step_name}")
    start_time = time.time()
    try:
        subprocess.run([sys.executable, command], check=True)
        elapsed = round(time.time() - start_time, 2)
        print(f"✅ Completed '{step_name}' in {elapsed}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"❌ FAILED '{step_name}' after {elapsed}s. Error: {e}")
        return False, elapsed


def main():
    pipeline_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    execution_log = []

    # Pipeline stages configuration
    stages = [
        ("1. Environment & Dependency Setup", "setup_env.py"),
        ("2. Dataset Download", "download_data.py"),
        ("3. Pipeline Tensor & Dataset Audit", "audit.py"),
        ("4. Adapter Training (Primary)", "train.py"),
        ("5. Adapter Calibration Training", "train_calibrated.py"),
        ("6. Phase 1 Diagnostics (Uncalibrated Sweeps)", "phase1_diagnostics.py"),
        ("7. Phase 2 Diagnostics (Cell-Relative Sweeps)", "phase2_diagnostics.py"),
        ("8. Phase 3 Diagnostics (mAP & Classification Metrics)", "phase3_diagnostics.py"),
        ("9. Final Metrics Evaluation", "eval.py")
    ]

    for stage_name, script_file in stages:
        if not os.path.exists(script_file):
            print(f"⚠️ Script {script_file} not found. Skipping stage {stage_name}...")
            execution_log.append({"stage": stage_name, "status": "SKIPPED", "time_sec": 0.0})
            continue

        success, elapsed = run_step(stage_name, script_file)
        execution_log.append({
            "stage": stage_name,
            "status": "PASSED" if success else "FAILED",
            "time_sec": elapsed
        })

        if not success:
            print(f"\n❌ Pipeline execution halted due to failure in stage: {stage_name}")
            break

    total_time = round(time.time() - pipeline_start, 2)
    print_banner(f"PIPELINE COMPLETE (Total Duration: {total_time}s)")

    # Save Pipeline Execution Run Log
    run_summary = {
        "total_duration_seconds": total_time,
        "stages": execution_log
    }
    summary_path = os.path.join(OUTPUT_DIR, "pipeline_execution_log.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=4)

    # --- Zip and Save Outputs ---
    print_banner("ARCHIVING OUTPUT ARTIFACTS")
    if os.path.exists(OUTPUT_DIR):
        archive_name = shutil.make_archive("vision_transformer_pipeline_artifacts", "zip", OUTPUT_DIR)
        print(f"✅ All adapter weights, diagnostic plots, and metrics archived into:\n   ↳ {os.path.abspath(archive_name)}")


if __name__ == "__main__":
    main()
