#!/usr/bin/env python3
"""
ClaimSense AI - Mistral Fine-tuning Script
Fine-tunes Mistral model for the hackathon.
"""

import os
import sys
import time
import json
from pathlib import Path

# Check for required environment variables
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("ERROR: MISTRAL_API_KEY environment variable not set")
    print("Get your API key from: https://console.mistral.ai/")
    print("Then run: export MISTRAL_API_KEY='your-key'")
    sys.exit(1)

from mistralai import Mistral

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def upload_file(client: Mistral, filepath: Path) -> str:
    """Upload a training file to Mistral."""
    print(f"Uploading {filepath.name}...")
    with open(filepath, "rb") as f:
        uploaded = client.files.upload(
            file={
                "file_name": filepath.name,
                "content": f,
            },
            purpose="fine-tune"
        )
    print(f"  -> Uploaded: {uploaded.id}")
    return uploaded.id


def create_finetuning_job(
    client: Mistral,
    train_file_id: str,
    eval_file_id: str,
    model: str = "open-mistral-nemo",
    suffix: str = "claimsense-v1",
):
    """Create a fine-tuning job."""

    print(f"\nCreating fine-tuning job...")
    print(f"  Model: {model}")
    print(f"  Suffix: {suffix}")

    # Create the job using the documented API format
    job = client.fine_tuning.jobs.create(
        model=model,
        training_files=[{"file_id": train_file_id, "weight": 1}],
        validation_files=[eval_file_id],
        hyperparameters={
            "training_steps": 100,
            "learning_rate": 0.0001,
        },
        suffix=suffix,
    )

    print(f"\n{'='*60}")
    print(f"Fine-tuning job created!")
    print(f"{'='*60}")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model: {job.fine_tuned_model or 'Training...'}")

    return job


def monitor_job(client: Mistral, job_id: str):
    """Monitor the fine-tuning job until completion."""
    print(f"\nMonitoring job {job_id}...")
    print("(This may take 1-3 hours. You can also check: https://console.mistral.ai/)")

    while True:
        job = client.fine_tuning.jobs.get(job_id=job_id)
        status = job.status

        if status == "SUCCESS":
            print(f"\n{'='*60}")
            print("Fine-tuning COMPLETE!")
            print(f"{'='*60}")
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job

        elif status in ["FAILED", "CANCELLED"]:
            print(f"\nJob {status}!")
            if hasattr(job, 'error') and job.error:
                print(f"Error: {job.error}")
            return job

        else:
            print(f"  Status: {status} - waiting 60s...")
            time.sleep(60)


def main():
    print("=" * 60)
    print("ClaimSense AI - Mistral Fine-tuning")
    print("=" * 60)

    # Initialize client
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Upload training files
    train_file_id = upload_file(client, DATA_DIR / "train.jsonl")
    eval_file_id = upload_file(client, DATA_DIR / "eval.jsonl")

    # Wait for files to process
    print("\nWaiting for files to process...")
    time.sleep(10)

    # Create fine-tuning job
    job = create_finetuning_job(
        client=client,
        train_file_id=train_file_id,
        eval_file_id=eval_file_id,
        model="open-mistral-nemo",  # Available for fine-tuning
        suffix="claimsense-v1",
    )

    # Save job info immediately
    MODELS_DIR.mkdir(exist_ok=True)
    job_info = {
        "job_id": job.id,
        "status": job.status,
        "model": getattr(job, 'fine_tuned_model', None),
        "train_file_id": train_file_id,
        "eval_file_id": eval_file_id,
        "base_model": "open-mistral-nemo",
    }
    with open(MODELS_DIR / "job_info.json", "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"\nJob info saved to models/job_info.json")

    # Monitor until complete
    final_job = monitor_job(client, job.id)

    # Update job info with final status
    job_info["status"] = final_job.status
    job_info["model"] = getattr(final_job, 'fine_tuned_model', None)
    with open(MODELS_DIR / "job_info.json", "w") as f:
        json.dump(job_info, f, indent=2)


if __name__ == "__main__":
    main()
