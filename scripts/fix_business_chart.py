"""
ClaimSense AI - Fix Business Impact Chart
Use subplots for different metrics with different scales
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle('Business Impact: ClaimSense AI', fontsize=20, fontweight='bold', y=1.02)

# Colors
manual_color = '#EF4444'
claimsense_color = '#10B981'

# 1. Processing Speed (in seconds)
ax1 = axes[0]
x = [0, 1]
heights = [2700, 2]  # 45 min = 2700 sec, 2 sec
bars = ax1.bar(x, heights, color=[manual_color, claimsense_color], width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(['Manual', 'ClaimSense'], fontsize=10)
ax1.set_title('Processing Speed', fontsize=12, fontweight='bold')
ax1.set_ylabel('Seconds', fontsize=10)
ax1.annotate('2700 sec\n(45 min)', xy=(0, 2700), xytext=(0, 2700+150),
             ha='center', fontsize=9, fontweight='bold')
ax1.annotate('2 sec', xy=(1, 2), xytext=(1, 200),
             ha='center', fontsize=9, fontweight='bold', color='#059669')
ax1.annotate('1350x faster', xy=(0.5, -400), ha='center',
             fontsize=11, color='#3B82F6', fontweight='bold')

# 2. Fraud Detection Rate (percentage)
ax2 = axes[1]
heights = [12, 34]
bars = ax2.bar(x, heights, color=[manual_color, claimsense_color], width=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(['Manual', 'ClaimSense'], fontsize=10)
ax2.set_title('Fraud Detection', fontsize=12, fontweight='bold')
ax2.set_ylabel('Detection Rate (%)', fontsize=10)
ax2.set_ylim(0, 45)
ax2.annotate('12%', xy=(0, 12), xytext=(0, 14),
             ha='center', fontsize=10, fontweight='bold')
ax2.annotate('34%', xy=(1, 34), xytext=(1, 36),
             ha='center', fontsize=10, fontweight='bold', color='#059669')
ax2.annotate('+183%', xy=(0.5, -6), ha='center',
             fontsize=11, color='#3B82F6', fontweight='bold')

# 3. False Positive Rate (lower is better)
ax3 = axes[2]
heights = [8, 3]
bars = ax3.bar(x, heights, color=[manual_color, claimsense_color], width=0.6)
ax3.set_xticks(x)
ax3.set_xticklabels(['Manual', 'ClaimSense'], fontsize=10)
ax3.set_title('False Positives', fontsize=12, fontweight='bold')
ax3.set_ylabel('False Positive Rate (%)', fontsize=10)
ax3.set_ylim(0, 12)
ax3.annotate('8%', xy=(0, 8), xytext=(0, 8.5),
             ha='center', fontsize=10, fontweight='bold')
ax3.annotate('3%', xy=(1, 3), xytext=(1, 3.5),
             ha='center', fontsize=10, fontweight='bold', color='#059669')
ax3.annotate('-62%', xy=(0.5, -1.5), ha='center',
             fontsize=11, color='#3B82F6', fontweight='bold')

# 4. Claims per Day per Adjuster
ax4 = axes[3]
heights = [18, 55]
bars = ax4.bar(x, heights, color=[manual_color, claimsense_color], width=0.6)
ax4.set_xticks(x)
ax4.set_xticklabels(['Manual', 'ClaimSense'], fontsize=10)
ax4.set_title('Claims/Day', fontsize=12, fontweight='bold')
ax4.set_ylabel('Claims per Adjuster', fontsize=10)
ax4.set_ylim(0, 70)
ax4.annotate('18', xy=(0, 18), xytext=(0, 20),
             ha='center', fontsize=10, fontweight='bold')
ax4.annotate('55', xy=(1, 55), xytext=(1, 57),
             ha='center', fontsize=10, fontweight='bold', color='#059669')
ax4.annotate('3x more', xy=(0.5, -9), ha='center',
             fontsize=11, color='#3B82F6', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=manual_color, label='Manual Process'),
                   Patch(facecolor=claimsense_color, label='ClaimSense AI')]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11)

plt.tight_layout()
plt.savefig('/Users/pramodmisra/Claude/Mistral AI Hackathon/claimsense-ai/business_impact_chart.png',
            dpi=150, bbox_inches='tight')
print("Fixed business impact chart saved!")
