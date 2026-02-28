#!/usr/bin/env python3
"""
ClaimSense AI - Gradio Demo App
Insurance Claims Fraud Detection & Triage System
"""

import os
import gradio as gr
from mistralai import Mistral

# Configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MODEL_ID = os.environ.get("CLAIMSENSE_MODEL_ID", "open-mistral-7b")  # Will be replaced with fine-tuned model

# Initialize client
client = None
if MISTRAL_API_KEY:
    client = Mistral(api_key=MISTRAL_API_KEY)


SYSTEM_PROMPT = """You are ClaimSense AI, an expert insurance claims analyst. You help with:
1. Fraud Detection - Identify suspicious patterns in claim descriptions
2. Severity Classification - Categorize claims by urgency (Low/Medium/High/Critical)
3. Claims Routing - Assign to appropriate department/adjuster
4. Risk Scoring - Provide confidence scores for assessments

Always provide structured, professional responses with clear recommendations."""


def analyze_claim(claim_text: str, analysis_type: str) -> str:
    """Analyze an insurance claim using ClaimSense AI."""

    if not client:
        return "Error: MISTRAL_API_KEY not configured. Please set the environment variable."

    if not claim_text.strip():
        return "Please enter a claim description to analyze."

    # Build prompt based on analysis type
    if analysis_type == "Fraud Detection":
        user_prompt = f"Analyze this insurance claim for potential fraud:\n\n{claim_text}"
    elif analysis_type == "Severity Classification":
        user_prompt = f"Classify the severity of this insurance claim:\n\n{claim_text}"
    elif analysis_type == "Claims Routing":
        user_prompt = f"Route this insurance claim to the appropriate department:\n\n{claim_text}"
    else:  # Full Analysis
        user_prompt = f"""Perform a complete analysis of this insurance claim:

{claim_text}

Provide:
1. Fraud Risk Assessment
2. Severity Classification
3. Recommended Routing
4. Priority Level
5. Next Steps"""

    try:
        response = client.chat.complete(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# Example claims for demo
EXAMPLE_CLAIMS = [
    ["Rear-end collision at traffic light. Other driver ran red light. Police report #12345 filed. Minor bumper damage, no injuries. Photos attached.", "Full Analysis"],
    ["Customer reports third laptop stolen from car this year. No police report. Requesting $3,500 replacement. Previous claims for similar items.", "Fraud Detection"],
    ["House fire started in kitchen. Fire department responded within 10 minutes. Damage contained to kitchen and adjacent dining room. Family evacuated safely.", "Severity Classification"],
    ["Slip and fall at insured's business premises. Visitor claiming back injury. Requesting medical expenses and pain compensation.", "Claims Routing"],
]


# Build Gradio interface
with gr.Blocks(title="ClaimSense AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ClaimSense AI
    ### Insurance Claims Fraud Detection & Triage System

    *Fine-tuned on 39,000+ insurance claims for the Mistral AI Worldwide Hackathon*

    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            claim_input = gr.Textbox(
                label="Claim Description",
                placeholder="Enter the insurance claim details here...",
                lines=8,
            )
            analysis_type = gr.Radio(
                choices=["Full Analysis", "Fraud Detection", "Severity Classification", "Claims Routing"],
                value="Full Analysis",
                label="Analysis Type",
            )
            analyze_btn = gr.Button("Analyze Claim", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="ClaimSense Analysis",
                lines=15,
                show_copy_button=True,
            )

    gr.Examples(
        examples=EXAMPLE_CLAIMS,
        inputs=[claim_input, analysis_type],
        label="Example Claims",
    )

    analyze_btn.click(
        fn=analyze_claim,
        inputs=[claim_input, analysis_type],
        outputs=output,
    )

    gr.Markdown("""
    ---
    **About ClaimSense AI**

    Built for the Mistral AI Worldwide Hackathon (Feb 28-Mar 1, 2026)

    - Track 1: Fine-tuning with Weights & Biases
    - Model: Ministral fine-tuned on insurance domain data
    - Dataset: 39,000+ training examples
    """)


if __name__ == "__main__":
    demo.launch(share=True)
