"""
ClaimSense AI - Fine-tuning Trainer
HuggingFace Space for training the insurance claims model.
"""

import os
import gradio as gr
import torch

# Check for secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
WANDB_KEY = os.environ.get("WANDB_API_KEY")

def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU Available: {gpu_name} ({gpu_mem:.1f} GB)"
    return "No GPU detected. Enable GPU in Space settings for training."

def check_secrets():
    """Check if secrets are configured."""
    status = []
    status.append(f"HF_TOKEN: {'Configured' if HF_TOKEN else 'NOT SET - Add in Space Settings'}")
    status.append(f"WANDB_API_KEY: {'Configured' if WANDB_KEY else 'NOT SET (optional)'}")
    return "\n".join(status)

def start_training(num_steps, learning_rate, batch_size, progress=gr.Progress()):
    """Start the fine-tuning process."""

    if not torch.cuda.is_available():
        return "ERROR: No GPU available. Go to Space Settings → Hardware and select T4 GPU ($0.60/hr)"

    if not HF_TOKEN:
        return "ERROR: HF_TOKEN secret not set. Go to Space Settings → Variables and secrets"

    progress(0, desc="Installing dependencies...")

    try:
        # Install unsloth
        os.system("pip install -q unsloth")
        os.system("pip install -q --no-deps trl peft accelerate bitsandbytes")
        os.system("pip install -q datasets wandb")

        progress(0.1, desc="Loading model...")

        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from huggingface_hub import login
        import wandb

        # Login
        login(token=HF_TOKEN)
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
            wandb.init(project="claimsense-ai", name="claimsense-hf-space")

        # Load model - using official Mistral with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        progress(0.2, desc="Adding LoRA adapters...")

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        progress(0.3, desc="Loading dataset...")

        # Load dataset
        def format_prompts(examples):
            texts = []
            for messages in examples['messages']:
                text = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        text += f"<s>[INST] {msg['content']} [/INST]"
                    else:
                        text += f" {msg['content']}</s>"
                texts.append(text)
            return {"text": texts}

        train_dataset = load_dataset("pramodmisra/claimsense-training-data", data_files="train.jsonl", split="train")
        eval_dataset = load_dataset("pramodmisra/claimsense-training-data", data_files="eval.jsonl", split="train")

        # Use subset
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(3000, len(train_dataset))))
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(300, len(eval_dataset))))

        train_dataset = train_dataset.map(format_prompts, batched=True)
        eval_dataset = eval_dataset.map(format_prompts, batched=True)

        progress(0.4, desc="Starting training...")

        # Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=TrainingArguments(
                output_dir="./output",
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=int(num_steps),
                learning_rate=float(learning_rate),
                fp16=True,
                logging_steps=10,
                report_to=["wandb"] if WANDB_KEY else [],
            ),
        )

        trainer.train()

        progress(0.9, desc="Uploading model...")

        # Save and upload
        model.save_pretrained("claimsense-ai-v1")
        tokenizer.save_pretrained("claimsense-ai-v1")
        model.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)
        tokenizer.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)

        if WANDB_KEY:
            wandb.finish()

        progress(1.0, desc="Complete!")

        return """
Training Complete!

Model uploaded to: https://huggingface.co/pramodmisra/claimsense-ai-v1

Next steps:
1. Test the model in the demo tab
2. Record a video demo
3. Submit to hackathon!
"""

    except Exception as e:
        return f"Error during training: {str(e)}"


def test_model(claim_text):
    """Test the fine-tuned model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("pramodmisra/claimsense-ai-v1")
        model = AutoModelForCausalLM.from_pretrained("pramodmisra/claimsense-ai-v1")

        prompt = f"<s>[INST] Analyze this insurance claim for potential fraud:\n\n{claim_text} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split("[/INST]")[-1].strip()
    except Exception as e:
        return f"Model not yet trained or error: {str(e)}"


# Build UI
with gr.Blocks(title="ClaimSense AI Trainer") as demo:
    gr.Markdown("""
    # ClaimSense AI - Fine-tuning Trainer

    **Mistral AI Worldwide Hackathon - Track 1: Fine-tuning**

    Train an insurance claims fraud detection model using Unsloth.

    ---
    """)

    with gr.Tab("Setup"):
        gr.Markdown("### System Status")
        gpu_status = gr.Textbox(label="GPU Status", value=check_gpu(), interactive=False)
        secrets_status = gr.Textbox(label="Secrets Status", value=check_secrets(), interactive=False, lines=2)
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(lambda: (check_gpu(), check_secrets()), outputs=[gpu_status, secrets_status])

        gr.Markdown("""
        ### Required Secrets

        Go to **Settings → Variables and secrets** and add:

        | Name | Value |
        |------|-------|
        | `HF_TOKEN` | Your HuggingFace write token |
        | `WANDB_API_KEY` | Your Weights & Biases key (optional) |

        ### Enable GPU

        Go to **Settings → Hardware** and select **T4 GPU** ($0.60/hour)
        """)

    with gr.Tab("Train"):
        gr.Markdown("### Training Configuration")

        with gr.Row():
            num_steps = gr.Slider(50, 500, value=200, step=50, label="Training Steps")
            learning_rate = gr.Number(value=2e-4, label="Learning Rate")
            batch_size = gr.Slider(1, 4, value=2, step=1, label="Batch Size")

        train_btn = gr.Button("Start Training", variant="primary")
        train_output = gr.Textbox(label="Training Output", lines=10)

        train_btn.click(start_training, inputs=[num_steps, learning_rate, batch_size], outputs=train_output)

    with gr.Tab("Demo"):
        gr.Markdown("### Test Fine-tuned Model")

        claim_input = gr.Textbox(
            label="Insurance Claim",
            placeholder="Enter claim description...",
            lines=4,
            value="Customer reports laptop stolen from unlocked car. Third claim this year. No police report."
        )
        test_btn = gr.Button("Analyze Claim")
        test_output = gr.Textbox(label="Analysis", lines=8)

        test_btn.click(test_model, inputs=claim_input, outputs=test_output)


if __name__ == "__main__":
    demo.launch()
