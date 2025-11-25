#!/usr/bin/env python3
"""
Plot reward vs episode for baseline PPO training
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading training data...")
df = pd.read_csv('/home/ubuntu/CAGE2/-cyborg-cage-2/Models/baseline_ppo_full_action_meander_20251015_211212/training_log.csv')

print(f"Data loaded: {len(df)} episodes")
print(f"Columns: {df.columns.tolist()}")
print(f"Reward range: {df['Reward'].min():.1f} to {df['Reward'].max():.1f}")

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Raw rewards
ax1 = axes[0]
ax1.plot(df['Episode'], df['Reward'], alpha=0.3, linewidth=0.5, color='blue', label='Raw Reward')

# Add rolling average
window = 1000
rolling_mean = df['Reward'].rolling(window=window, center=True).mean()
ax1.plot(df['Episode'], rolling_mean, color='red', linewidth=2, label=f'Rolling Mean ({window} episodes)')

ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Baseline PPO Training: Reward vs Episode (100,000 episodes)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Smoothed view (every 100 episodes)
ax2 = axes[1]
# Sample every 100 episodes for cleaner view
sampled_df = df[::100]
ax2.plot(sampled_df['Episode'], sampled_df['Reward'], marker='o', markersize=2, linewidth=1, alpha=0.7, color='darkblue')

# Add trend line
z = np.polyfit(sampled_df['Episode'], sampled_df['Reward'], 3)
p = np.poly1d(z)
ax2.plot(sampled_df['Episode'], p(sampled_df['Episode']), "r--", linewidth=2, alpha=0.8, label='Trend')

ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.set_title('Smoothed View (Sampled every 100 episodes)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()

# Save plot
output_file = '/home/ubuntu/CAGE2/-cyborg-cage-2/baseline_ppo_training_plot.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# Show statistics
print(f"\n📊 Training Statistics:")
print(f"   Total Episodes: {len(df):,}")
print(f"   Mean Reward: {df['Reward'].mean():.2f}")
print(f"   Std Reward: {df['Reward'].std():.2f}")
print(f"   Min Reward: {df['Reward'].min():.2f}")
print(f"   Max Reward: {df['Reward'].max():.2f}")

# Check improvement
first_1000 = df['Reward'][:1000].mean()
last_1000 = df['Reward'][-1000:].mean()
improvement = last_1000 - first_1000

print(f"\n📈 Learning Progress:")
print(f"   First 1000 episodes avg: {first_1000:.2f}")
print(f"   Last 1000 episodes avg: {last_1000:.2f}")
print(f"   Improvement: {improvement:.2f} ({improvement/abs(first_1000)*100:+.1f}%)")

plt.show()

