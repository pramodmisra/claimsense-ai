# ClaimSense AI - Setup Guide

## Quick Start

You have two options for fine-tuning:

### Option A: Mistral API (Recommended - Fastest)
**Time:** ~2-4 hours | **Cost:** Free hackathon credits

1. **Get Mistral API Key:**
   - Go to https://console.mistral.ai/
   - Create account / Sign in
   - Navigate to API Keys → Create new key
   - Hackathon participants get $100 free credits

2. **Get W&B API Key (Optional but recommended for Track 1):**
   - Go to https://wandb.ai/
   - Create account / Sign in
   - Navigate to Settings → API Keys → Create new key

3. **Set environment variables:**
   ```bash
   export MISTRAL_API_KEY='your-mistral-key'
   export WANDB_API_KEY='your-wandb-key'  # Optional
   ```

4. **Run fine-tuning:**
   ```bash
   cd claimsense-ai
   python3 scripts/finetune_mistral.py
   ```

### Option B: Google Colab + Unsloth (Free)
**Time:** ~4-6 hours | **Cost:** Free (uses Google Colab T4 GPU)

1. Open the notebook in Google Colab:
   - Upload `notebooks/colab_finetune.ipynb` to Colab
   - Or use this direct link: [Unsloth AutoSloth](https://colab.research.google.com/drive/1Zo0sVEb2lqdsUm9dy2PTzGySxdF9CNkc)

2. Upload your training data:
   - `data/train.jsonl`
   - `data/eval.jsonl`

3. Run all cells and wait for training to complete

4. Download the fine-tuned model

---

## Project Structure

```
claimsense-ai/
├── data/
│   ├── train.jsonl      # 35,136 training examples
│   ├── eval.jsonl       # 3,905 evaluation examples
│   └── all_data.jsonl   # Combined dataset
├── scripts/
│   ├── prepare_dataset.py   # Dataset preparation
│   └── finetune_mistral.py  # Mistral API fine-tuning
├── notebooks/
│   └── colab_finetune.ipynb # Google Colab notebook
├── models/
│   └── job_info.json        # Fine-tuning job details
├── demo/
│   └── app.py               # Gradio demo app
└── README.md
```

---

## Hackathon Submission Checklist

- [ ] Fine-tuned model uploaded to HuggingFace hackathon org
- [ ] W&B project with training metrics
- [ ] Gradio demo deployed to HuggingFace Spaces
- [ ] README.md with project description
- [ ] Video demo (YouTube/Loom)
- [ ] Project screenshot (16:9)
