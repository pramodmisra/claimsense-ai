---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- insurance
- fraud-detection
- claims-processing
- mistral
- fine-tuned
- lora
- qlora
- peft
datasets:
- bitext/Bitext-insurance-llm-chatbot-training-dataset
- pramodmisra/claimsense-training-data
metrics:
- accuracy
base_model: mistralai/Mistral-7B-Instruct-v0.2
model-index:
- name: claimsense-ai-v1
  results:
  - task:
      type: text-classification
      name: Fraud Detection
    metrics:
    - type: accuracy
      value: 91.0
      name: Accuracy
  - task:
      type: text-classification
      name: Severity Classification
    metrics:
    - type: accuracy
      value: 88.0
      name: Accuracy
  - task:
      type: text-classification
      name: Response Structure
    metrics:
    - type: accuracy
      value: 94.0
      name: Accuracy
pipeline_tag: text-generation
widget:
- text: "<s>[INST] Analyze this insurance claim for fraud risk:\n\nCustomer reports laptop stolen from unlocked car. Third claim this year for similar items. No police report filed. Requesting $3,500. [/INST]"
  example_title: "Fraud Detection - High Risk"
- text: "<s>[INST] Analyze this insurance claim:\n\nRear-end collision at traffic light. Other driver ran red light. Police report #12345 filed. Minor bumper damage, no injuries. Estimate: $2,400. [/INST]"
  example_title: "Auto Claim - Low Risk"
- text: "<s>[INST] Classify the severity and route this claim:\n\nHouse fire in kitchen at 2 AM. Fire department responded. Extensive damage to kitchen and dining room. Family evacuated safely. [/INST]"
  example_title: "Property Claim - Critical"
---

# ClaimSense AI v1

**Insurance Claims Fraud Detection & Triage System**

[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/pramodmisra/claimsense-ai-demo)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/pramodmisra/claimsense-ai)

> Built for the **Mistral AI Worldwide Hackathon 2026** - Track 1: Fine-tuning with Weights & Biases

## Model Description

ClaimSense AI is a fine-tuned version of Mistral 7B Instruct v0.2, specialized for insurance claims processing. It performs:

| Capability | Description |
|------------|-------------|
| **Fraud Detection** | Identifies red flags, suspicious patterns, assigns risk scores (LOW/MEDIUM/HIGH) |
| **Severity Classification** | Categorizes claims as Low/Medium/High/Critical |
| **Claims Routing** | Auto-assigns to appropriate department (Auto, Property, Liability, Theft, etc.) |
| **Priority Scoring** | Determines processing urgency and SLA requirements |

## Intended Uses

- **Primary Use**: Assisting insurance claims adjusters with initial claim triage
- **Secondary Use**: Training and educational purposes for insurance professionals
- **Not For**: Fully autonomous claim decisions without human oversight

## Training Data

| Dataset | Examples | Description |
|---------|----------|-------------|
| [Bitext Insurance LLM](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) | 39,000 | Insurance claims processing conversations |
| Synthetic Severity Data | 36 | Multi-level severity classification examples |
| Synthetic Routing Data | 5 | Department assignment rules |
| **Total** | **39,041** | Combined training examples |

Training/Eval Split: 90% / 10% (35,136 train / 3,905 eval)

## Training Procedure

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Method | QLoRA (4-bit quantization) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 2e-4 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Training Steps | 100 |
| Max Sequence Length | 2048 |
| Optimizer | AdamW (8-bit) |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.03 |

### Training Infrastructure

| Resource | Details |
|----------|---------|
| GPU | NVIDIA T4 (16GB VRAM) |
| Platform | HuggingFace Spaces |
| Training Time | ~45 minutes |
| Framework | Transformers + PEFT |
| Tracking | Weights & Biases |

### Training Metrics

| Metric | Value |
|--------|-------|
| Initial Training Loss | 1.24 |
| Final Training Loss | 0.87 |
| Validation Loss | 1.18 |

## Evaluation Results

Evaluated on 50+ diverse insurance claim scenarios (synthetic + real-world patterns):

| Task | Base Mistral | ClaimSense AI | Improvement |
|------|--------------|---------------|-------------|
| Fraud Risk Detection | 72% | **91%** | **+19%** |
| Response Structure | 68% | **94%** | **+26%** |
| Severity Classification | 85% | **88%** | **+3%** |
| Overall Accuracy | 75% | **89%** | **+14%** |

### Key Improvements

| Capability | Base Mistral | ClaimSense AI |
|------------|--------------|---------------|
| Fraud pattern detection | 72% accuracy | 91% accuracy (+26% relative) |
| Response consistency | 68% structured | 94% structured (+38% relative) |
| Insurance terminology | Generic | Domain-specific |
| Processing time | Manual (45 min) | Instant (2 sec) |

## Usage

### Direct Use with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "pramodmisra/claimsense-ai-v1",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("pramodmisra/claimsense-ai-v1")

# Prepare prompt
claim = """Customer reports laptop stolen from unlocked car.
Third claim this year for similar items. No police report filed.
Requesting $3,500."""

prompt = f"<s>[INST] Analyze this insurance claim for fraud risk:\n\n{claim} [/INST]"

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with PEFT (LoRA adapters only)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "pramodmisra/claimsense-ai-v1")
```

## Example Outputs

### Input: Fraud Detection
```
Customer reports laptop stolen from unlocked car. Third claim this year
for similar items. No police report filed. Requesting $3,500.
```

### Output:
```
FRAUD RISK ASSESSMENT

Risk Level: HIGH
Confidence: 87%

Red Flags Detected:
• Multiple similar claims (3rd this year) - Pattern indicator
• No police report for theft - Documentation gap
• Unlocked vehicle - Negligence pattern
• High-value replacement request - Financial motivation

Recommendation: Flag for Special Investigation Unit (SIU) review
Priority: High - Expedited investigation required
```

## Limitations

- **Not a replacement for human judgment**: Should be used as a decision-support tool, not for autonomous claim decisions
- **English only**: Trained on English language claims
- **US-focused**: Training data primarily covers US insurance terminology and practices
- **No image analysis**: Cannot process damage photos or documents
- **Potential biases**: May reflect biases present in training data

## Ethical Considerations

- Claims flagged as fraudulent should always be reviewed by human investigators
- Model outputs should be used to assist, not replace, trained claims adjusters
- Regular auditing recommended to detect potential biases
- Not suitable for determining claim denial without human review

## Business Impact (Projected)

| Metric | Manual Process | With ClaimSense | Improvement |
|--------|---------------|-----------------|-------------|
| Claims/adjuster/day | 15-20 | 45-60 | 3x throughput |
| Fraud detection rate | 12% | 34% | +183% |
| False positive rate | 8% | 3% | -62% |
| Avg processing cost | $45/claim | $15/claim | $30 savings |

## Citation

```bibtex
@misc{claimsense-ai-2026,
  author = {Pramod Misra},
  title = {ClaimSense AI: Insurance Claims Fraud Detection and Triage System},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/pramodmisra/claimsense-ai-v1}},
  note = {Mistral AI Worldwide Hackathon 2026}
}
```

## Links

- **Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/pramodmisra/claimsense-ai-demo)
- **Dataset**: [claimsense-training-data](https://huggingface.co/datasets/pramodmisra/claimsense-training-data)
- **GitHub**: [pramodmisra/claimsense-ai](https://github.com/pramodmisra/claimsense-ai)
- **Base Model**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

## Acknowledgments

- [Mistral AI](https://mistral.ai/) - Base model and hackathon
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Bitext](https://huggingface.co/bitext) - Insurance dataset
- [HuggingFace](https://huggingface.co/) - Model hosting and Spaces

---

**Built with care for the Mistral AI Worldwide Hackathon 2026**
