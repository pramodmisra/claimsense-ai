"""
ClaimSense AI - Fine-tuning Trainer v4
Pure Transformers Trainer (no TRL dependency issues)
"""

import os
import gradio as gr
import torch

HF_TOKEN = os.environ.get("HF_TOKEN")
WANDB_KEY = os.environ.get("WANDB_API_KEY")

def check_gpu():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)"
    return "No GPU detected."

def check_secrets():
    return f"HF_TOKEN: {'OK' if HF_TOKEN else 'MISSING'}\nWANDB: {'OK' if WANDB_KEY else 'Not set'}"

def start_training(num_steps, learning_rate, batch_size, progress=gr.Progress()):
    if not torch.cuda.is_available():
        return "ERROR: No GPU."
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN not set."

    try:
        progress(0.05, desc="Loading libraries...")

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        from huggingface_hub import login

        login(token=HF_TOKEN)

        if WANDB_KEY:
            import wandb
            wandb.login(key=WANDB_KEY)
            wandb.init(project="claimsense-ai", name="claimsense-v1")

        progress(0.1, desc="Loading Mistral 7B...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            token=HF_TOKEN,
        )
        tokenizer.pad_token = tokenizer.eos_token

        progress(0.2, desc="Adding LoRA adapters...")

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        progress(0.3, desc="Loading dataset...")

        def tokenize_function(examples):
            texts = []
            for messages in examples['messages']:
                text = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        text += f"<s>[INST] {msg['content']} [/INST]"
                    else:
                        text += f" {msg['content']}</s>"
                texts.append(text)

            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = load_dataset("pramodmisra/claimsense-training-data", data_files="train.jsonl", split="train")
        dataset = dataset.shuffle(seed=42).select(range(min(1500, len(dataset))))

        # Split into train/eval
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"].map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        eval_dataset = split["test"].map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        progress(0.4, desc="Starting training...")

        training_args = TrainingArguments(
            output_dir="./claimsense-output",
            num_train_epochs=1,
            max_steps=int(num_steps),
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=4,
            learning_rate=float(learning_rate),
            fp16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            warmup_steps=10,
            report_to=["wandb"] if WANDB_KEY else [],
            load_best_model_at_end=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        progress(0.9, desc="Saving & uploading model...")

        # Save and push
        model.save_pretrained("claimsense-ai-v1")
        tokenizer.save_pretrained("claimsense-ai-v1")

        model.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)
        tokenizer.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)

        if WANDB_KEY:
            import wandb
            wandb.finish()

        progress(1.0, desc="Complete!")

        return """
✅ TRAINING COMPLETE!

Model uploaded to: https://huggingface.co/pramodmisra/claimsense-ai-v1

Next steps:
1. Test the model in the Demo tab
2. Check W&B dashboard for metrics
3. Record video demo for submission
"""

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


def test_model(claim_text):
    if not claim_text.strip():
        return "Enter a claim description."

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            "pramodmisra/claimsense-ai-v1",
            device_map="auto",
            torch_dtype=torch.float16,
            token=HF_TOKEN
        )

        prompt = f"<s>[INST] Analyze this insurance claim for fraud:\n\n{claim_text} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

    except Exception as e:
        return f"Model not ready yet. Train first.\n\nError: {e}"


with gr.Blocks(title="ClaimSense AI") as demo:
    gr.Markdown("""
    # 🛡️ ClaimSense AI Trainer
    **Mistral AI Hackathon - Insurance Fraud Detection**
    """)

    with gr.Tab("📊 Setup"):
        gr.Textbox(label="GPU Status", value=check_gpu(), interactive=False)
        gr.Textbox(label="Secrets", value=check_secrets(), lines=2, interactive=False)
        gr.Markdown("""
        ### Required:
        - `HF_TOKEN` - HuggingFace write token
        - `WANDB_API_KEY` - Weights & Biases key (optional)
        - T4 GPU enabled in Space settings
        """)

    with gr.Tab("🚀 Train"):
        gr.Markdown("### Training Configuration")
        with gr.Row():
            num_steps = gr.Slider(50, 200, value=100, step=25, label="Training Steps")
            learning_rate = gr.Number(value=2e-4, label="Learning Rate")
            batch_size = gr.Slider(1, 2, value=1, step=1, label="Batch Size")

        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
        output = gr.Textbox(label="Training Output", lines=12)
        train_btn.click(start_training, [num_steps, learning_rate, batch_size], output)

    with gr.Tab("🔍 Demo"):
        gr.Markdown("### Test the Fine-tuned Model")
        claim = gr.Textbox(
            label="Insurance Claim",
            lines=4,
            placeholder="Enter claim description..."
        )
        test_btn = gr.Button("Analyze Claim")
        result = gr.Textbox(label="Analysis Result", lines=8)
        test_btn.click(test_model, claim, result)

if __name__ == "__main__":
    demo.launch()
