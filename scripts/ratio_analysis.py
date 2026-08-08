import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("="*70)
print("TRAIN/TEST SPLIT RATIO ANALYSIS")
print("="*70)
print("\nObjective: Determine optimal train/test split ratio")
print("Models: Linear Regression (baseline) and Random Forest (optimized)")
print()

# Load data
print("Loading data...")

X = pd.read_csv('data/processed/X_full.csv')
y = pd.read_csv('data/processed/y_full.csv').squeeze()

print(f"Dataset loaded successfully!")
print(f"  Total samples: {len(X):,}")
print(f"  Features: {X.shape[1]}")
print()

# Define test ratios
test_ratios = [0.10, 0.15, 0.20, 0.25, 0.30]
train_ratios = [1 - ratio for ratio in test_ratios]

print("Testing the following train/test ratios:")
for train_r, test_r in zip(train_ratios, test_ratios):
    print(f"  {int(train_r*100)}/{int(test_r*100)}")
print()

# Store results
results = []

print("="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

for test_size in test_ratios:
    train_size = 1 - test_size
    
    print(f"\n{'='*70}")
    print(f"Train/Test Split: {int(train_size*100)}/{int(test_size*100)}")
    print(f"{'='*70}")
    
    # Split data (shuffle=False for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Calculate LR metrics - Training
    y_pred_lr_train = lr_model.predict(X_train_scaled)
    r2_lr_train = r2_score(y_train, y_pred_lr_train)
    rmse_lr_train = np.sqrt(mean_squared_error(y_train, y_pred_lr_train))
    
    # Calculate LR metrics - Testing
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    print(f"  Training   - R²: {r2_lr_train:.4f}, RMSE: {rmse_lr_train:.2f} kW")
    print(f"  Validation - R²: {r2_lr:.4f}, RMSE: {rmse_lr:.2f} kW, MAE: {mae_lr:.2f} kW")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Calculate RF metrics - Training
    y_pred_rf_train = rf_model.predict(X_train_scaled)
    r2_rf_train = r2_score(y_train, y_pred_rf_train)
    rmse_rf_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
    
    # Calculate RF metrics - Testing
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print(f"  Training   - R²: {r2_rf_train:.4f}, RMSE: {rmse_rf_train:.2f} kW")
    print(f"  Validation - R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f} kW, MAE: {mae_rf:.2f} kW")
    
    # Check for overfitting
    overfit_lr = r2_lr_train - r2_lr
    overfit_rf = r2_rf_train - r2_rf
    
    if overfit_lr > 0.1:
        print(f"  ⚠ Linear Regression: Potential overfitting (gap: {overfit_lr:.4f})")
    if overfit_rf > 0.1:
        print(f"  ⚠ Random Forest: Potential overfitting (gap: {overfit_rf:.4f})")
    
    # Store results
    results.append({
        'Train_Size': f"{int(train_size*100)}%",
        'Test_Size': f"{int(test_size*100)}%",
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
        'LR_R2_Train': r2_lr_train,
        'LR_R2_Val': r2_lr,
        'LR_RMSE_Train': rmse_lr_train,
        'LR_RMSE_Val': rmse_lr,
        'LR_MAE_Val': mae_lr,
        'LR_Overfit': overfit_lr,
        'RF_R2_Train': r2_rf_train,
        'RF_R2_Val': r2_rf,
        'RF_RMSE_Train': rmse_rf_train,
        'RF_RMSE_Val': rmse_rf,
        'RF_MAE_Val': mae_rf,
        'RF_Overfit': overfit_rf
    })

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("COMPLETE RESULTS TABLE")
print("="*70)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('train_test_ratio_results.csv', index=False)
print("\n✓ Results saved to 'train_test_ratio_results.csv'")

# Find best ratios
best_lr_idx = results_df['LR_R2_Val'].idxmax()
best_rf_idx = results_df['RF_R2_Val'].idxmax()

print("\n" + "="*70)
print("BEST PERFORMING RATIOS")
print("="*70)
print(f"\nLinear Regression:")
print(f"  Best ratio: {results_df.iloc[best_lr_idx]['Train_Size']}/{results_df.iloc[best_lr_idx]['Test_Size']}")
print(f"  Validation R²: {results_df.iloc[best_lr_idx]['LR_R2_Val']:.4f}")
print(f"  Validation RMSE: {results_df.iloc[best_lr_idx]['LR_RMSE_Val']:.2f} kW")
print(f"  Overfitting gap: {results_df.iloc[best_lr_idx]['LR_Overfit']:.4f}")

print(f"\nRandom Forest:")
print(f"  Best ratio: {results_df.iloc[best_rf_idx]['Train_Size']}/{results_df.iloc[best_rf_idx]['Test_Size']}")
print(f"  Validation R²: {results_df.iloc[best_rf_idx]['RF_R2_Val']:.4f}")
print(f"  Validation RMSE: {results_df.iloc[best_rf_idx]['RF_RMSE_Val']:.2f} kW")
print(f"  Overfitting gap: {results_df.iloc[best_rf_idx]['RF_Overfit']:.4f}")

# Visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot 1: R² Score Comparison
print("\nCreating R² comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, results_df['LR_R2_Val'], width, 
               label='Linear Regression', alpha=0.8, color='#F18F01')
bars2 = ax.bar(x_pos + width/2, results_df['RF_R2_Val'], width,
               label='Random Forest', alpha=0.8, color='#10b981')

ax.set_xlabel('Train/Test Split', fontsize=12)
ax.set_ylabel('R² Score (Validation)', fontsize=12)
ax.set_title('R² Score Comparison Across Different Train/Test Ratios', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{row['Train_Size']}/{row['Test_Size']}" for _, row in results_df.iterrows()])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: r2_comparison.png")

# Plot 2: RMSE Comparison
print("Creating RMSE comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x_pos - width/2, results_df['LR_RMSE_Val'], width,
               label='Linear Regression', alpha=0.8, color='#F18F01')
bars2 = ax.bar(x_pos + width/2, results_df['RF_RMSE_Val'], width,
               label='Random Forest', alpha=0.8, color='#10b981')

ax.set_xlabel('Train/Test Split', fontsize=12)
ax.set_ylabel('RMSE (kW) - Validation', fontsize=12)
ax.set_title('RMSE Comparison Across Different Train/Test Ratios', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{row['Train_Size']}/{row['Test_Size']}" for _, row in results_df.iterrows()])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rmse_comparison.png")

# Plot 3: Performance Trends
print("Creating performance trend charts...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

test_sizes = [int(ts.strip('%')) for ts in results_df['Test_Size']]

# R² Trend
ax1.plot(test_sizes, results_df['LR_R2_Val'], marker='o', linewidth=2.5, markersize=10,
         label='Linear Regression', color='#F18F01')
ax1.plot(test_sizes, results_df['RF_R2_Val'], marker='s', linewidth=2.5, markersize=10,
         label='Random Forest', color='#10b981')
ax1.set_xlabel('Test Set Size (%)', fontsize=12)
ax1.set_ylabel('R² Score (Validation)', fontsize=12)
ax1.set_title('R² Score Trend', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# RMSE Trend
ax2.plot(test_sizes, results_df['LR_RMSE_Val'], marker='o', linewidth=2.5, markersize=10,
         label='Linear Regression', color='#F18F01')
ax2.plot(test_sizes, results_df['RF_RMSE_Val'], marker='s', linewidth=2.5, markersize=10,
         label='Random Forest', color='#10b981')
ax2.set_xlabel('Test Set Size (%)', fontsize=12)
ax2.set_ylabel('RMSE (kW) - Validation', fontsize=12)
ax2.set_title('RMSE Trend', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('performance_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: performance_trends.png")

# Plot 4: Comprehensive Analysis
print("Creating comprehensive analysis chart...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: R² Comparison
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(results_df))
width = 0.35
ax1.bar(x_pos - width/2, results_df['LR_R2_Val'], width, label='Linear Regression', 
        alpha=0.8, color='#F18F01')
ax1.bar(x_pos + width/2, results_df['RF_R2_Val'], width, label='Random Forest',
        alpha=0.8, color='#10b981')
ax1.set_xlabel('Train/Test Split')
ax1.set_ylabel('R² Score (Validation)')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{row['Train_Size']}/{row['Test_Size']}" for _, row in results_df.iterrows()],
                     rotation=0)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x_pos - width/2, results_df['LR_RMSE_Val'], width, label='Linear Regression',
        alpha=0.8, color='#F18F01')
ax2.bar(x_pos + width/2, results_df['RF_RMSE_Val'], width, label='Random Forest',
        alpha=0.8, color='#10b981')
ax2.set_xlabel('Train/Test Split')
ax2.set_ylabel('RMSE (kW) - Validation')
ax2.set_title('RMSE Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{row['Train_Size']}/{row['Test_Size']}" for _, row in results_df.iterrows()],
                     rotation=0)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: R² Trends
ax3 = fig.add_subplot(gs[1, 0])
test_sizes = [int(ts.strip('%')) for ts in results_df['Test_Size']]
ax3.plot(test_sizes, results_df['LR_R2_Val'], marker='o', linewidth=2, markersize=8,
         label='Linear Regression', color='#F18F01')
ax3.plot(test_sizes, results_df['RF_R2_Val'], marker='s', linewidth=2, markersize=8,
         label='Random Forest', color='#10b981')
ax3.set_xlabel('Test Set Size (%)')
ax3.set_ylabel('R² Score (Validation)')
ax3.set_title('Performance Trend by Test Size')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# Plot 4: Overfitting Analysis
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(x_pos - width/2, results_df['LR_Overfit'], width, label='Linear Regression',
        alpha=0.8, color='#F18F01')
ax4.bar(x_pos + width/2, results_df['RF_Overfit'], width, label='Random Forest',
        alpha=0.8, color='#10b981')
ax4.set_xlabel('Train/Test Split')
ax4.set_ylabel('Overfitting Gap (Train R² - Val R²)')
ax4.set_title('Overfitting Analysis')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"{row['Train_Size']}/{row['Test_Size']}" for _, row in results_df.iterrows()],
                     rotation=0)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfit threshold')

fig.suptitle('Train/Test Split Ratio Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: comprehensive_analysis.png")

# Statistical Analysis
print("\n" + "="*70)
print("PERFORMANCE STABILITY ANALYSIS")
print("="*70)

r2_variance_lr = results_df['LR_R2_Val'].std()
r2_variance_rf = results_df['RF_R2_Val'].std()
rmse_variance_lr = results_df['LR_RMSE_Val'].std()
rmse_variance_rf = results_df['RF_RMSE_Val'].std()

print(f"\nR² Score Stability (Standard Deviation):")
print(f"  Linear Regression: {r2_variance_lr:.4f}")
print(f"  Random Forest:     {r2_variance_rf:.4f}")

print(f"\nRMSE Stability (Standard Deviation):")
print(f"  Linear Regression: {rmse_variance_lr:.2f} kW")
print(f"  Random Forest:     {rmse_variance_rf:.2f} kW")

if r2_variance_rf < 0.01:
    print("\n✓ Random Forest shows stable performance across all ratios")
else:
    print("\n⚠ Random Forest performance varies with split ratio")

# 80/20 Standard Analysis
ratio_80_20 = results_df[results_df['Test_Size'] == '20%']

if not ratio_80_20.empty:
    print("\n" + "="*70)
    print("80/20 SPLIT (INDUSTRY STANDARD)")
    print("="*70)
    print(f"\nLinear Regression:")
    print(f"  Validation R²: {ratio_80_20['LR_R2_Val'].values[0]:.4f}")
    print(f"  Validation RMSE: {ratio_80_20['LR_RMSE_Val'].values[0]:.2f} kW")
    print(f"  Validation MAE: {ratio_80_20['LR_MAE_Val'].values[0]:.2f} kW")
    
    print(f"\nRandom Forest:")
    print(f"  Validation R²: {ratio_80_20['RF_R2_Val'].values[0]:.4f}")
    print(f"  Validation RMSE: {ratio_80_20['RF_RMSE_Val'].values[0]:.2f} kW")
    print(f"  Validation MAE: {ratio_80_20['RF_MAE_Val'].values[0]:.2f} kW")

# Generate Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

optimal_idx = results_df['RF_R2_Val'].idxmax()
optimal = results_df.iloc[optimal_idx]

print(f"\nRecommended Train/Test Split: {optimal['Train_Size']}/{optimal['Test_Size']}")
print(f"\nJustification:")
print(f"  • Achieves highest validation R²: {optimal['RF_R2_Val']:.4f}")
print(f"  • Validation RMSE: {optimal['RF_RMSE_Val']:.2f} kW")
print(f"  • Training samples: {optimal['Train_Samples']:,}")
print(f"  • Test samples: {optimal['Test_Samples']:,} (sufficient for validation)")
print(f"  • Overfitting gap: {optimal['RF_Overfit']:.4f}", end="")

if optimal['RF_Overfit'] < 0.1:
    print(" (acceptable)")
else:
    print(" (monitor for overfitting)")

# Check if more data helps
if results_df['RF_R2_Val'].iloc[0] > results_df['RF_R2_Val'].iloc[-1]:
    print(f"\n  • Model benefits from more training data")
else:
    print(f"\n  • Model is stable even with less training data")

# Compare to 80/20 standard
if not ratio_80_20.empty and optimal['Test_Size'] == '20%':
    print(f"\n  ✓ Aligns with industry standard 80/20 split")
elif not ratio_80_20.empty:
    diff = optimal['RF_R2_Val'] - ratio_80_20['RF_R2_Val'].values[0]
    print(f"\n  • Improvement over 80/20 standard: {diff:.4f} R²")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  • train_test_ratio_results.csv")
print("  • r2_comparison.png")
print("  • rmse_comparison.png")
print("  • performance_trends.png")
print("  • comprehensive_analysis.png")
print("\nAll results saved successfully!")
print("="*70)