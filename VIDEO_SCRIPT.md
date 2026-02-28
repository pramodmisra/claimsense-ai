# ClaimSense AI - Video Demo Script (2 minutes)

## Recording Tips
- Use screen recording (Loom, OBS, or QuickTime)
- Record at 1080p or higher
- Speak clearly and at moderate pace
- Show your face in corner (optional but recommended)

---

## INTRO (0:00 - 0:15)

**[Show title slide or demo homepage]**

> "Hi, I'm Pramod, and this is ClaimSense AI - an intelligent insurance claims fraud detection and triage system, built for the Mistral AI Hackathon."

---

## THE PROBLEM (0:15 - 0:35)

**[Show statistics slide or infographic]**

> "Insurance fraud costs the industry over 80 billion dollars annually in the US alone. Claims adjusters are overwhelmed - manually reviewing thousands of claims, struggling to spot subtle fraud patterns, and dealing with inconsistent severity assessments."

> "What if AI could help them process claims faster AND catch fraud that humans miss?"

---

## THE SOLUTION (0:35 - 0:55)

**[Show HuggingFace model page briefly, then switch to demo]**

> "ClaimSense AI is a fine-tuned Mistral 7B model, trained on over 39,000 insurance claims. It performs four key functions:"

> "One - Fraud Detection with risk scoring. Two - Severity Classification. Three - Intelligent Claims Routing. And Four - Priority Assessment."

---

## LIVE DEMO (0:55 - 1:35)

**[Switch to demo interface at huggingface.co/spaces/pramodmisra/claimsense-ai-demo]**

> "Let me show you how it works."

**[Click on first example - fraud detection case]**

> "Here's a suspicious claim: laptop stolen from an unlocked car, third claim this year, no police report. Let's analyze it for fraud."

**[Click Analyze, wait for response]**

> "ClaimSense immediately identifies multiple red flags - the pattern of claims, missing documentation, and security negligence. It assigns a HIGH risk score and recommends SIU investigation."

**[Try another example - Full Analysis]**

> "For a standard claim like this car accident, it provides a complete analysis - low fraud risk, appropriate severity, and routes it to the right department with SLA guidelines."

---

## TECHNICAL HIGHLIGHTS (1:35 - 1:50)

**[Show W&B dashboard or training metrics briefly]**

> "On the technical side - I fine-tuned Mistral 7B using QLoRA on a T4 GPU, with full Weights & Biases integration for experiment tracking. The model was trained on curated insurance domain data to understand industry-specific terminology and fraud patterns."

---

## CLOSING (1:50 - 2:00)

**[Return to demo or show closing slide]**

> "ClaimSense AI demonstrates how fine-tuning can create domain-specific AI that solves real problems. Insurance adjusters can now process claims 3x faster while catching fraud that might otherwise slip through."

> "Thank you for watching. Links to the model, dataset, and demo are in the description."

---

## ASSETS TO PREPARE

1. **Title Slide**: "ClaimSense AI - Insurance Claims Intelligence"
2. **Stats Slide**: "$80B annual fraud cost" with visual
3. **Demo URL**: https://huggingface.co/spaces/pramodmisra/claimsense-ai-demo
4. **W&B Dashboard**: Screenshot of training metrics

---

## DESCRIPTION TEXT (for YouTube/Loom)

```
ClaimSense AI - Insurance Claims Fraud Detection & Triage System

Built for the Mistral AI Worldwide Hackathon 2026 (Track 1: Fine-tuning)

🔗 Links:
• Demo: https://huggingface.co/spaces/pramodmisra/claimsense-ai-demo
• Model: https://huggingface.co/pramodmisra/claimsense-ai-v1
• Dataset: https://huggingface.co/datasets/pramodmisra/claimsense-training-data
• GitHub: https://github.com/pramodmisra/claimsense-ai

📊 Technical Details:
• Base Model: Mistral 7B Instruct v0.2
• Fine-tuning: QLoRA (4-bit quantization)
• Training Data: 39,000+ insurance claims
• Tracking: Weights & Biases

#MistralAI #Hackathon #AI #MachineLearning #Insurance #FineTuning
```
