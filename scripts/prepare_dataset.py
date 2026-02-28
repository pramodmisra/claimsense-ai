#!/usr/bin/env python3
"""
ClaimSense AI - Dataset Preparation Script
Downloads and combines insurance datasets into JSONL format for fine-tuning.
"""

import json
import os
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def format_fraud_example(claim_text: str, is_fraud: bool) -> dict:
    """Format a fraud detection example for fine-tuning."""
    fraud_status = "HIGH RISK - LIKELY FRAUD" if is_fraud else "LOW RISK - LEGITIMATE"
    indicators = "Suspicious patterns detected. Recommend thorough investigation." if is_fraud else "No suspicious patterns. Standard processing recommended."

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Analyze this insurance claim for potential fraud:\n\n{claim_text}"
            },
            {
                "role": "assistant",
                "content": f"""FRAUD ANALYSIS REPORT:

**Risk Assessment:** {fraud_status}
**Confidence:** {"85-95%" if is_fraud else "90-98%"}

**Analysis:**
{indicators}

**Recommended Action:** {"Flag for Special Investigation Unit (SIU) review" if is_fraud else "Proceed with standard claims processing"}"""
            }
        ]
    }


def format_claims_example(claim_data: dict) -> dict:
    """Format a claims processing example for fine-tuning."""
    # Extract relevant fields
    claim_type = claim_data.get("intent", claim_data.get("type", "general_inquiry"))
    description = claim_data.get("instruction", claim_data.get("text", ""))
    response = claim_data.get("response", "")

    if not description or not response:
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Insurance Query: {description}"
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
    }


def format_severity_example(description: str, severity: str) -> dict:
    """Format a severity classification example."""
    severity_map = {
        "low": "LOW - Minor damage, standard processing",
        "medium": "MEDIUM - Moderate damage, priority processing",
        "high": "HIGH - Severe damage, expedited processing",
        "critical": "CRITICAL - Emergency, immediate attention required"
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Classify the severity of this insurance claim:\n\n{description}"
            },
            {
                "role": "assistant",
                "content": f"""SEVERITY CLASSIFICATION:

**Level:** {severity_map.get(severity.lower(), severity)}

**Routing:** {"Standard Queue" if severity.lower() == "low" else "Priority Queue" if severity.lower() == "medium" else "Urgent Queue"}
**SLA:** {"5-7 business days" if severity.lower() == "low" else "2-3 business days" if severity.lower() == "medium" else "24 hours"}"""
            }
        ]
    }


def download_textclaims_dataset():
    """Download TextClaimsDataset for fraud detection."""
    print("Downloading TextClaimsDataset...")
    try:
        dataset = load_dataset("infinite-dataset-hub/TextClaimsDataset", split="train")
        examples = []
        for item in dataset:
            text = item.get("text", item.get("claim", ""))
            label = item.get("label", item.get("is_fraud", 0))
            is_fraud = label == 1 or label == "fraud" or str(label).lower() == "true"
            if text:
                examples.append(format_fraud_example(text, is_fraud))
        print(f"  -> Processed {len(examples)} fraud detection examples")
        return examples
    except Exception as e:
        print(f"  -> Error: {e}")
        return []


def download_bitext_insurance():
    """Download Bitext Insurance LLM dataset."""
    print("Downloading Bitext Insurance LLM dataset...")
    try:
        dataset = load_dataset("bitext/Bitext-insurance-llm-chatbot-training-dataset", split="train")
        examples = []
        for item in dataset:
            formatted = format_claims_example(item)
            if formatted:
                examples.append(formatted)
        print(f"  -> Processed {len(examples)} claims processing examples")
        return examples
    except Exception as e:
        print(f"  -> Error: {e}")
        return []


def create_synthetic_severity_examples():
    """Create synthetic severity classification examples."""
    print("Creating synthetic severity examples...")

    synthetic_data = [
        # Low severity
        ("Minor scratch on rear bumper from parking lot. No visible dent, paint slightly marked.", "low"),
        ("Small chip in windshield, approximately 1cm diameter. No cracks spreading.", "low"),
        ("Side mirror damaged during car wash. Mirror glass intact but housing cracked.", "low"),

        # Medium severity
        ("Rear-end collision at stop light. Bumper dented and cracked. Airbags did not deploy.", "medium"),
        ("Water damage to basement. 2 inches of standing water. Carpet and drywall affected.", "medium"),
        ("Tree branch fell on car during storm. Hood and windshield damaged. Vehicle drivable.", "medium"),

        # High severity
        ("House fire in kitchen. Fire contained to one room but smoke damage throughout.", "high"),
        ("T-bone collision at intersection. Airbags deployed. Driver transported to hospital.", "high"),
        ("Roof collapsed during heavy snowfall. Multiple rooms affected. Home uninhabitable.", "high"),

        # Critical
        ("Total loss vehicle fire. Vehicle completely destroyed. Occupants evacuated safely.", "critical"),
        ("Major flooding. Entire first floor submerged. Emergency relocation required.", "critical"),
        ("Multi-vehicle pileup on highway. Multiple injuries. 3 vehicles totaled.", "critical"),
    ]

    examples = [format_severity_example(desc, sev) for desc, sev in synthetic_data]

    # Duplicate with variations to increase training data
    variations = []
    for desc, sev in synthetic_data:
        variations.append(format_severity_example(f"Claim report: {desc}", sev))
        variations.append(format_severity_example(f"Customer states: {desc}", sev))

    examples.extend(variations)
    print(f"  -> Created {len(examples)} severity classification examples")
    return examples


def create_routing_examples():
    """Create claims routing examples."""
    print("Creating routing examples...")

    routing_data = [
        {
            "claim": "Auto accident on highway. Other driver at fault. Police report filed.",
            "department": "Auto Claims - Liability",
            "priority": "Standard",
            "specialist": "Auto Adjuster"
        },
        {
            "claim": "Theft of personal belongings from hotel room during vacation.",
            "department": "Property Claims - Theft",
            "priority": "Standard",
            "specialist": "Property Adjuster"
        },
        {
            "claim": "Slip and fall at insured's property. Visitor claiming injuries.",
            "department": "Liability Claims",
            "priority": "High",
            "specialist": "Liability Specialist + Legal Review"
        },
        {
            "claim": "Storm damage to multiple properties in same neighborhood.",
            "department": "Catastrophe Response",
            "priority": "Critical",
            "specialist": "CAT Team"
        },
        {
            "claim": "Medical expenses after car accident. Driver has injuries.",
            "department": "Medical Claims",
            "priority": "High",
            "specialist": "Medical Claims Specialist"
        },
    ]

    examples = []
    for item in routing_data:
        examples.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"Route this insurance claim to the appropriate department:\n\n{item['claim']}"
                },
                {
                    "role": "assistant",
                    "content": f"""CLAIMS ROUTING DECISION:

**Assigned Department:** {item['department']}
**Priority Level:** {item['priority']}
**Assigned Specialist:** {item['specialist']}

**Routing Rationale:** Based on claim type and severity indicators, this claim requires {item['department'].lower()} expertise."""
                }
            ]
        })

    print(f"  -> Created {len(examples)} routing examples")
    return examples


def save_jsonl(data: list, filename: str):
    """Save data to JSONL format."""
    filepath = DATA_DIR / filename
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")
    return filepath


def main():
    print("=" * 60)
    print("ClaimSense AI - Dataset Preparation")
    print("=" * 60)

    all_examples = []

    # Download external datasets
    fraud_examples = download_textclaims_dataset()
    all_examples.extend(fraud_examples)

    bitext_examples = download_bitext_insurance()
    all_examples.extend(bitext_examples)

    # Create synthetic examples
    severity_examples = create_synthetic_severity_examples()
    all_examples.extend(severity_examples)

    routing_examples = create_routing_examples()
    all_examples.extend(routing_examples)

    print("\n" + "=" * 60)
    print(f"Total examples collected: {len(all_examples)}")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # 90/10 train/eval split
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    eval_data = all_examples[split_idx:]

    print(f"Training examples: {len(train_data)}")
    print(f"Evaluation examples: {len(eval_data)}")

    # Save files
    save_jsonl(train_data, "train.jsonl")
    save_jsonl(eval_data, "eval.jsonl")
    save_jsonl(all_examples, "all_data.jsonl")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
