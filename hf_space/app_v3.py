"""
ClaimSense AI - Fine-tuning Trainer v3
Simple Transformers + PEFT approach for maximum compatibility.
"""

import os
import gradio as gr
import torch

HF_TOKEN = os.environ.get("HF_TOKEN")
WANDB_KEY = os.environ.get("WANDB_API_KEY")

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU Available: {gpu_name} ({gpu_mem:.1f} GB)"
    return "No GPU detected."

def check_secrets():
    status = []
    status.append(f"HF_TOKEN: {'Configured' if HF_TOKEN else 'NOT SET'}")
    status.append(f"WANDB_API_KEY: {'Configured' if WANDB_KEY else 'NOT SET (optional)'}")
    return "\n".join(status)

def start_training(num_steps, learning_rate, batch_size, progress=gr.Progress()):
    if not torch.cuda.is_available():
        return "ERROR: No GPU. Enable T4 GPU in Space Settings."
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN not set."

    progress(0, desc="Installing dependencies...")

    try:
        os.system("pip install -q transformers==4.40.0 peft accelerate bitsandbytes")
        os.system("pip install -q datasets trl==0.8.6 wandb")

        progress(0.1, desc="Loading model...")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        from trl import SFTTrainer
        from huggingface_hub import login
        import wandb

        login(token=HF_TOKEN)
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
            wandb.init(project="claimsense-ai", name="claimsense-v1")

        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
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
        tokenizer.padding_side = "right"

        progress(0.2, desc="Adding LoRA...")

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        progress(0.3, desc="Loading dataset...")

        def format_func(example):
            text = ""
            for msg in example['messages']:
                if msg['role'] == 'user':
                    text += f"<s>[INST] {msg['content']} [/INST]"
                else:
                    text += f" {msg['content']}</s>"
            return {"text": text}

        train_dataset = load_dataset("pramodmisra/claimsense-training-data", data_files="train.jsonl", split="train")
        eval_dataset = load_dataset("pramodmisra/claimsense-training-data", data_files="eval.jsonl", split="train")

        train_dataset = train_dataset.shuffle(seed=42).select(range(min(2000, len(train_dataset))))
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(200, len(eval_dataset))))

        train_dataset = train_dataset.map(format_func)
        eval_dataset = eval_dataset.map(format_func)

        progress(0.4, desc="Training...")

        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=int(num_steps),
            learning_rate=float(learning_rate),
            fp16=True,
            logging_steps=10,
            save_steps=50,
            report_to=["wandb"] if WANDB_KEY else [],
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            args=training_args,
        )

        trainer.train()

        progress(0.9, desc="Uploading...")

        model.save_pretrained("claimsense-ai-v1")
        tokenizer.save_pretrained("claimsense-ai-v1")
        model.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)
        tokenizer.push_to_hub("pramodmisra/claimsense-ai-v1", token=HF_TOKEN)

        if WANDB_KEY:
            wandb.finish()

        progress(1.0, desc="Done!")
        return "SUCCESS! Model uploaded to: https://huggingface.co/pramodmisra/claimsense-ai-v1"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


def test_model(claim_text):
    return "Test after training completes."


with gr.Blocks(title="ClaimSense AI") as demo:
    gr.Markdown("# ClaimSense AI Trainer")

    with gr.Tab("Setup"):
        gr.Textbox(label="GPU", value=check_gpu())
        gr.Textbox(label="Secrets", value=check_secrets(), lines=2)

    with gr.Tab("Train"):
        with gr.Row():
            num_steps = gr.Slider(50, 300, value=100, step=50, label="Steps")
            learning_rate = gr.Number(value=2e-4, label="LR")
            batch_size = gr.Slider(1, 2, value=1, step=1, label="Batch")
        train_btn = gr.Button("Start Training", variant="primary")
        output = gr.Textbox(label="Output", lines=10)
        train_btn.click(start_training, [num_steps, learning_rate, batch_size], output)

    with gr.Tab("Demo"):
        claim = gr.Textbox(label="Claim", lines=4)
        gr.Button("Test").click(test_model, claim, gr.Textbox(label="Result", lines=6))

if __name__ == "__main__":
    demo.launch()
