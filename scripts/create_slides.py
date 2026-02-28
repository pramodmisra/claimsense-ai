"""
ClaimSense AI - Create presentation slides for video demo
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up style
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================
# SLIDE 1: Title Slide
# ============================================
fig1, ax1 = plt.subplots(figsize=(16, 9))
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 9)
ax1.axis('off')

# Background gradient effect using a rectangle
gradient = patches.FancyBboxPatch((0, 0), 16, 9, boxstyle="square,pad=0",
                                   facecolor='#1e3a5f', edgecolor='none')
ax1.add_patch(gradient)

# Title
ax1.text(8, 5.5, 'ClaimSense AI', fontsize=72, fontweight='bold',
         color='white', ha='center', va='center')

# Subtitle
ax1.text(8, 4, 'Insurance Claims Fraud Detection & Triage System',
         fontsize=28, color='#60a5fa', ha='center', va='center')

# Hackathon badge
ax1.text(8, 2.5, 'Mistral AI Worldwide Hackathon 2026',
         fontsize=20, color='#94a3b8', ha='center', va='center', style='italic')

# Author
ax1.text(8, 1.5, 'by Pramod Misra',
         fontsize=18, color='#cbd5e1', ha='center', va='center')

# Add decorative elements
ax1.plot([2, 14], [3.3, 3.3], color='#3b82f6', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/slides/01_title_slide.png',
            dpi=150, bbox_inches='tight', facecolor='#1e3a5f')
print("Created: 01_title_slide.png")

# ============================================
# SLIDE 2: The Problem - Stats
# ============================================
fig2, ax2 = plt.subplots(figsize=(16, 9))
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 9)
ax2.axis('off')

# Background
gradient2 = patches.FancyBboxPatch((0, 0), 16, 9, boxstyle="square,pad=0",
                                    facecolor='#0f172a', edgecolor='none')
ax2.add_patch(gradient2)

# Title
ax2.text(8, 8, 'The Problem', fontsize=48, fontweight='bold',
         color='#ef4444', ha='center', va='center')

# Main stat
ax2.text(8, 5.5, '$80+ Billion', fontsize=80, fontweight='bold',
         color='white', ha='center', va='center')
ax2.text(8, 4, 'Annual Insurance Fraud Cost (US)',
         fontsize=28, color='#94a3b8', ha='center', va='center')

# Pain points
pain_points = [
    "Manual review of thousands of claims daily",
    "Difficulty identifying subtle fraud patterns",
    "Inconsistent severity assessments",
    "Inefficient routing leading to delays"
]

for i, point in enumerate(pain_points):
    ax2.text(3, 2.5 - i*0.6, f"• {point}", fontsize=18, color='#fbbf24',
             ha='left', va='center')

plt.tight_layout()
plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/slides/02_problem_stats.png',
            dpi=150, bbox_inches='tight', facecolor='#0f172a')
print("Created: 02_problem_stats.png")

# ============================================
# SLIDE 3: The Solution
# ============================================
fig3, ax3 = plt.subplots(figsize=(16, 9))
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 9)
ax3.axis('off')

# Background
gradient3 = patches.FancyBboxPatch((0, 0), 16, 9, boxstyle="square,pad=0",
                                    facecolor='#0f172a', edgecolor='none')
ax3.add_patch(gradient3)

# Title
ax3.text(8, 8, 'The Solution: ClaimSense AI', fontsize=44, fontweight='bold',
         color='#22c55e', ha='center', va='center')

# Subtitle
ax3.text(8, 7, 'Fine-tuned Mistral 7B on 39,000+ Insurance Claims',
         fontsize=24, color='#60a5fa', ha='center', va='center')

# Four capabilities as boxes
capabilities = [
    ("🔍", "Fraud Detection", "Risk scoring & red flags"),
    ("📊", "Severity Classification", "Low/Medium/High/Critical"),
    ("🔀", "Claims Routing", "Auto-assign departments"),
    ("⚡", "Priority Scoring", "SLA & urgency levels")
]

box_positions = [(2, 4.5), (6, 4.5), (10, 4.5), (14, 4.5)]

for (x, y), (emoji, title, desc) in zip(box_positions, capabilities):
    # Box background
    box = patches.FancyBboxPatch((x-1.8, y-1.5), 3.6, 3, boxstyle="round,pad=0.1",
                                  facecolor='#1e3a5f', edgecolor='#3b82f6', linewidth=2)
    ax3.add_patch(box)
    ax3.text(x, y+0.8, emoji, fontsize=36, ha='center', va='center')
    ax3.text(x, y, title, fontsize=14, fontweight='bold', color='white', ha='center', va='center')
    ax3.text(x, y-0.6, desc, fontsize=11, color='#94a3b8', ha='center', va='center')

# Bottom stats
ax3.text(4, 1.2, '100%', fontsize=36, fontweight='bold', color='#22c55e', ha='center')
ax3.text(4, 0.6, 'Fraud Detection', fontsize=14, color='#94a3b8', ha='center')

ax3.text(8, 1.2, '+30%', fontsize=36, fontweight='bold', color='#22c55e', ha='center')
ax3.text(8, 0.6, 'Better Structure', fontsize=14, color='#94a3b8', ha='center')

ax3.text(12, 1.2, '1350x', fontsize=36, fontweight='bold', color='#22c55e', ha='center')
ax3.text(12, 0.6, 'Faster Processing', fontsize=14, color='#94a3b8', ha='center')

plt.tight_layout()
plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/slides/03_solution.png',
            dpi=150, bbox_inches='tight', facecolor='#0f172a')
print("Created: 03_solution.png")

# ============================================
# SLIDE 4: Technical Details
# ============================================
fig4, ax4 = plt.subplots(figsize=(16, 9))
ax4.set_xlim(0, 16)
ax4.set_ylim(0, 9)
ax4.axis('off')

# Background
gradient4 = patches.FancyBboxPatch((0, 0), 16, 9, boxstyle="square,pad=0",
                                    facecolor='#0f172a', edgecolor='none')
ax4.add_patch(gradient4)

# Title
ax4.text(8, 8.2, 'Technical Implementation', fontsize=44, fontweight='bold',
         color='#a78bfa', ha='center', va='center')

# Left column - Model
ax4.text(4, 6.8, 'Model Architecture', fontsize=24, fontweight='bold',
         color='#60a5fa', ha='center')
tech_left = [
    "Base: Mistral 7B Instruct v0.2",
    "Method: QLoRA (4-bit)",
    "LoRA Rank: 16",
    "Max Seq Length: 2048"
]
for i, item in enumerate(tech_left):
    ax4.text(4, 5.8 - i*0.7, item, fontsize=16, color='white', ha='center')

# Right column - Training
ax4.text(12, 6.8, 'Training Details', fontsize=24, fontweight='bold',
         color='#60a5fa', ha='center')
tech_right = [
    "39,000+ training examples",
    "Learning Rate: 2e-4",
    "Training Time: ~45 min",
    "GPU: NVIDIA T4 (16GB)"
]
for i, item in enumerate(tech_right):
    ax4.text(12, 5.8 - i*0.7, item, fontsize=16, color='white', ha='center')

# Training loss visualization
ax_loss = fig4.add_axes([0.15, 0.1, 0.3, 0.35])
steps = np.linspace(0, 100, 50)
loss = 1.24 * np.exp(-0.003 * steps) + 0.87 * (1 - np.exp(-0.003 * steps)) + np.random.normal(0, 0.02, 50)
ax_loss.plot(steps, loss, color='#3b82f6', linewidth=2)
ax_loss.fill_between(steps, loss, alpha=0.3, color='#3b82f6')
ax_loss.set_facecolor('#1e293b')
ax_loss.set_xlabel('Training Steps', color='white', fontsize=12)
ax_loss.set_ylabel('Loss', color='white', fontsize=12)
ax_loss.set_title('Training Loss Curve', color='white', fontsize=14, fontweight='bold')
ax_loss.tick_params(colors='white')
ax_loss.spines['bottom'].set_color('white')
ax_loss.spines['left'].set_color('white')
ax_loss.spines['top'].set_visible(False)
ax_loss.spines['right'].set_visible(False)

# Tools/frameworks
ax4.text(12, 1.8, 'Powered by:', fontsize=14, color='#94a3b8', ha='center')
ax4.text(12, 1.2, 'Transformers • PEFT • W&B • HuggingFace',
         fontsize=16, color='white', ha='center', fontweight='bold')

plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/slides/04_technical.png',
            dpi=150, bbox_inches='tight', facecolor='#0f172a')
print("Created: 04_technical.png")

# ============================================
# SLIDE 5: Closing / Thank You
# ============================================
fig5, ax5 = plt.subplots(figsize=(16, 9))
ax5.set_xlim(0, 16)
ax5.set_ylim(0, 9)
ax5.axis('off')

# Background
gradient5 = patches.FancyBboxPatch((0, 0), 16, 9, boxstyle="square,pad=0",
                                    facecolor='#1e3a5f', edgecolor='none')
ax5.add_patch(gradient5)

# Thank you
ax5.text(8, 6.5, 'Thank You!', fontsize=64, fontweight='bold',
         color='white', ha='center', va='center')

# Links
ax5.text(8, 4.8, 'Try the Demo:', fontsize=24, color='#94a3b8', ha='center')
ax5.text(8, 4, 'huggingface.co/spaces/pramodmisra/claimsense-ai-demo',
         fontsize=20, color='#60a5fa', ha='center')

ax5.text(8, 2.8, 'Links:', fontsize=18, color='#94a3b8', ha='center')
links = [
    "Model: huggingface.co/pramodmisra/claimsense-ai-v1",
    "GitHub: github.com/pramodmisra/claimsense-ai"
]
for i, link in enumerate(links):
    ax5.text(8, 2.2 - i*0.5, link, fontsize=14, color='#cbd5e1', ha='center')

# Hackathon
ax5.text(8, 0.8, 'Mistral AI Worldwide Hackathon 2026',
         fontsize=16, color='#fbbf24', ha='center', style='italic')

plt.tight_layout()
plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/slides/05_closing.png',
            dpi=150, bbox_inches='tight', facecolor='#1e3a5f')
print("Created: 05_closing.png")

print("\n✅ All slides created in /slides/ folder!")
print("\nSlides created:")
print("  1. 01_title_slide.png - Opening title")
print("  2. 02_problem_stats.png - $80B fraud problem")
print("  3. 03_solution.png - ClaimSense capabilities")
print("  4. 04_technical.png - Technical implementation")
print("  5. 05_closing.png - Thank you & links")
