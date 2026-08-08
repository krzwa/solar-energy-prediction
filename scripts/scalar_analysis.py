import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
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
print("SCALER COMPARISON ANALYSIS")
print("="*70)
print("\nObjective: Determine optimal feature scaling method")
print("Scalers: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, None")
print("Models: Linear Regression and Random Forest")
print()

# Load data
print("Loading data...")
# TODO: Update these paths to match your data location
X = pd.read_csv('data/processed/X_full.csv')
y = pd.read_csv('data/processed/y_full.csv').squeeze()

print(f"Dataset loaded successfully!")
print(f"  Total samples: {len(X):,}")
print(f"  Features: {X.shape[1]}")
print()

# Split data once (80/20 split)
print("Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False, random_state=42
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")
print()

# Define scalers to test
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'No Scaling': None
}

print("Scalers to test:")
for name, scaler in scalers.items():
    if scaler is None:
        print(f"  • {name}: Raw features (baseline)")
    else:
        print(f"  • {name}: {scaler.__class__.__name__}")
print()

# Store results
results = []

print("="*70)
print("TRAINING AND EVALUATING MODELS WITH DIFFERENT SCALERS")
print("="*70)

for scaler_name, scaler in scalers.items():
    print(f"\n{'='*70}")
    print(f"Testing: {scaler_name}")
    print(f"{'='*70}")
    
    # Scale features (or use raw if None)
    if scaler is None:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        print("Using raw features (no scaling)")
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"Applied {scaler_name}")
        
        # Show scaling statistics
        print(f"  Training data range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"  Training data mean: {X_train_scaled.mean():.3f}")
        print(f"  Training data std: {X_train_scaled.std():.3f}")
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_lr_train = lr_model.predict(X_train_scaled)
    y_pred_lr_test = lr_model.predict(X_test_scaled)
    
    # Metrics
    r2_lr_train = r2_score(y_train, y_pred_lr_train)
    r2_lr_test = r2_score(y_test, y_pred_lr_test)
    rmse_lr_train = np.sqrt(mean_squared_error(y_train, y_pred_lr_train))
    rmse_lr_test = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))
    mae_lr_test = mean_absolute_error(y_test, y_pred_lr_test)
    
    print(f"  Training   - R²: {r2_lr_train:.4f}, RMSE: {rmse_lr_train:.2f} kW")
    print(f"  Validation - R²: {r2_lr_test:.4f}, RMSE: {rmse_lr_test:.2f} kW, MAE: {mae_lr_test:.2f} kW")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_rf_train = rf_model.predict(X_train_scaled)
    y_pred_rf_test = rf_model.predict(X_test_scaled)
    
    # Metrics
    r2_rf_train = r2_score(y_train, y_pred_rf_train)
    r2_rf_test = r2_score(y_test, y_pred_rf_test)
    rmse_rf_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
    rmse_rf_test = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    mae_rf_test = mean_absolute_error(y_test, y_pred_rf_test)
    
    print(f"  Training   - R²: {r2_rf_train:.4f}, RMSE: {rmse_rf_train:.2f} kW")
    print(f"  Validation - R²: {r2_rf_test:.4f}, RMSE: {rmse_rf_test:.2f} kW, MAE: {mae_rf_test:.2f} kW")
    
    # Check for overfitting
    overfit_lr = r2_lr_train - r2_lr_test
    overfit_rf = r2_rf_train - r2_rf_test
    
    if overfit_lr > 0.1:
        print(f"  ⚠ Linear Regression: Potential overfitting (gap: {overfit_lr:.4f})")
    if overfit_rf > 0.1:
        print(f"  ⚠ Random Forest: Potential overfitting (gap: {overfit_rf:.4f})")
    
    # Store results
    results.append({
        'Scaler': scaler_name,
        'LR_R2_Train': r2_lr_train,
        'LR_R2_Val': r2_lr_test,
        'LR_RMSE_Train': rmse_lr_train,
        'LR_RMSE_Val': rmse_lr_test,
        'LR_MAE_Val': mae_lr_test,
        'LR_Overfit': overfit_lr,
        'RF_R2_Train': r2_rf_train,
        'RF_R2_Val': r2_rf_test,
        'RF_RMSE_Train': rmse_rf_train,
        'RF_RMSE_Val': rmse_rf_test,
        'RF_MAE_Val': mae_rf_test,
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
results_df.to_csv('scaler_comparison_results.csv', index=False)
print("\n✓ Results saved to 'scaler_comparison_results.csv'")

# Find best scalers
best_lr_idx = results_df['LR_R2_Val'].idxmax()
best_rf_idx = results_df['RF_R2_Val'].idxmax()

print("\n" + "="*70)
print("BEST PERFORMING SCALERS")
print("="*70)
print(f"\nLinear Regression:")
print(f"  Best scaler: {results_df.iloc[best_lr_idx]['Scaler']}")
print(f"  Validation R²: {results_df.iloc[best_lr_idx]['LR_R2_Val']:.4f}")
print(f"  Validation RMSE: {results_df.iloc[best_lr_idx]['LR_RMSE_Val']:.2f} kW")
print(f"  Overfitting gap: {results_df.iloc[best_lr_idx]['LR_Overfit']:.4f}")

print(f"\nRandom Forest:")
print(f"  Best scaler: {results_df.iloc[best_rf_idx]['Scaler']}")
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

ax.set_xlabel('Scaler Type', fontsize=12)
ax.set_ylabel('R² Score (Validation)', fontsize=12)
ax.set_title('R² Score Comparison Across Different Scalers', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('scaler_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: scaler_r2_comparison.png")

# Plot 2: RMSE Comparison
print("Creating RMSE comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x_pos - width/2, results_df['LR_RMSE_Val'], width,
               label='Linear Regression', alpha=0.8, color='#F18F01')
bars2 = ax.bar(x_pos + width/2, results_df['RF_RMSE_Val'], width,
               label='Random Forest', alpha=0.8, color='#10b981')

ax.set_xlabel('Scaler Type', fontsize=12)
ax.set_ylabel('RMSE (kW) - Validation', fontsize=12)
ax.set_title('RMSE Comparison Across Different Scalers', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('scaler_rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: scaler_rmse_comparison.png")

# Plot 3: Comprehensive Comparison
print("Creating comprehensive comparison chart...")
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
ax1.set_xlabel('Scaler')
ax1.set_ylabel('R² Score (Validation)')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x_pos - width/2, results_df['LR_RMSE_Val'], width, label='Linear Regression',
        alpha=0.8, color='#F18F01')
ax2.bar(x_pos + width/2, results_df['RF_RMSE_Val'], width, label='Random Forest',
        alpha=0.8, color='#10b981')
ax2.set_xlabel('Scaler')
ax2.set_ylabel('RMSE (kW) - Validation')
ax2.set_title('RMSE Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: MAE Comparison (Random Forest only)
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(x_pos, results_df['RF_MAE_Val'], width*2, alpha=0.8, color='#10b981')
ax3.set_xlabel('Scaler')
ax3.set_ylabel('MAE (kW) - Validation')
ax3.set_title('MAE Comparison (Random Forest)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(results_df['RF_MAE_Val']):
    ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Overfitting Analysis
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(x_pos - width/2, results_df['LR_Overfit'], width, label='Linear Regression',
        alpha=0.8, color='#F18F01')
ax4.bar(x_pos + width/2, results_df['RF_Overfit'], width, label='Random Forest',
        alpha=0.8, color='#10b981')
ax4.set_xlabel('Scaler')
ax4.set_ylabel('Overfitting Gap (Train R² - Val R²)')
ax4.set_title('Overfitting Analysis')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

fig.suptitle('Scaler Comparison Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('scaler_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: scaler_comprehensive_analysis.png")

# Plot 4: Performance differences from baseline (No Scaling)
print("Creating performance delta chart...")
baseline_idx = results_df[results_df['Scaler'] == 'No Scaling'].index[0]
baseline_lr_r2 = results_df.iloc[baseline_idx]['LR_R2_Val']
baseline_rf_r2 = results_df.iloc[baseline_idx]['RF_R2_Val']

results_df['LR_R2_Delta'] = results_df['LR_R2_Val'] - baseline_lr_r2
results_df['RF_R2_Delta'] = results_df['RF_R2_Val'] - baseline_rf_r2

fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(results_df))
bars1 = ax.bar(x_pos - width/2, results_df['LR_R2_Delta'], width,
               label='Linear Regression', alpha=0.8, color='#F18F01')
bars2 = ax.bar(x_pos + width/2, results_df['RF_R2_Delta'], width,
               label='Random Forest', alpha=0.8, color='#10b981')

ax.set_xlabel('Scaler Type', fontsize=12)
ax.set_ylabel('R² Improvement vs No Scaling', fontsize=12)
ax.set_title('Performance Improvement Over Baseline (No Scaling)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Scaler'], rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        label = f'+{height:.4f}' if height > 0 else f'{height:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('scaler_improvement_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: scaler_improvement_comparison.png")

# Analysis
print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)

# Performance variability
print("\nPerformance Variability (Standard Deviation):")
print(f"  Linear Regression R²: {results_df['LR_R2_Val'].std():.4f}")
print(f"  Random Forest R²:     {results_df['RF_R2_Val'].std():.4f}")

# Check if scaling matters for each model
lr_no_scale = results_df[results_df['Scaler'] == 'No Scaling']['LR_R2_Val'].values[0]
lr_best_scale = results_df['LR_R2_Val'].max()
lr_improvement = lr_best_scale - lr_no_scale

rf_no_scale = results_df[results_df['Scaler'] == 'No Scaling']['RF_R2_Val'].values[0]
rf_best_scale = results_df['RF_R2_Val'].max()
rf_improvement = rf_best_scale - rf_no_scale

print(f"\nScaling Impact:")
print(f"  Linear Regression: {lr_improvement:+.4f} R² improvement")
print(f"  Random Forest:     {rf_improvement:+.4f} R² improvement")

if abs(lr_improvement) > 0.01:
    print(f"\n  ✓ Scaling significantly improves Linear Regression")
else:
    print(f"\n  • Scaling has minimal impact on Linear Regression")

if abs(rf_improvement) > 0.01:
    print(f"  ✓ Scaling significantly improves Random Forest")
else:
    print(f"  • Scaling has minimal impact on Random Forest")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

best_overall_lr = results_df.iloc[best_lr_idx]
best_overall_rf = results_df.iloc[best_rf_idx]

print(f"\nFor Linear Regression:")
print(f"  Recommended scaler: {best_overall_lr['Scaler']}")
print(f"  Validation R²: {best_overall_lr['LR_R2_Val']:.4f}")
print(f"  Validation RMSE: {best_overall_lr['LR_RMSE_Val']:.2f} kW")
print(f"  Improvement over no scaling: {best_overall_lr['LR_R2_Delta']:+.4f} R²")

print(f"\nFor Random Forest:")
print(f"  Recommended scaler: {best_overall_rf['Scaler']}")
print(f"  Validation R²: {best_overall_rf['RF_R2_Val']:.4f}")
print(f"  Validation RMSE: {best_overall_rf['RF_RMSE_Val']:.2f} kW")
print(f"  Improvement over no scaling: {best_overall_rf['RF_R2_Delta']:+.4f} R²")

# Overall recommendation
if best_overall_lr['Scaler'] == best_overall_rf['Scaler']:
    print(f"\n🎯 Overall Recommendation: {best_overall_lr['Scaler']}")
    print(f"   This scaler works best for both models")
else:
    print(f"\n🎯 Overall Recommendation:")
    print(f"   • Use {best_overall_rf['Scaler']} (optimal for Random Forest - our best model)")
    print(f"   • Note: {best_overall_lr['Scaler']} is better for Linear Regression")

# Scaler characteristics
print("\n" + "="*70)
print("SCALER CHARACTERISTICS")
print("="*70)

scaler_notes = {
    'StandardScaler': 'Best for normally distributed features. Scales to mean=0, std=1.',
    'MinMaxScaler': 'Scales features to [0,1] range. Sensitive to outliers.',
    'RobustScaler': 'Uses median and IQR. Robust to outliers. Good for skewed data.',
    'MaxAbsScaler': 'Scales to [-1,1] range. Preserves sparsity.',
    'No Scaling': 'Raw features. Baseline for comparison.'
}

for scaler in results_df['Scaler']:
    print(f"\n{scaler}:")
    print(f"  {scaler_notes[scaler]}")
    row = results_df[results_df['Scaler'] == scaler].iloc[0]
    print(f"  LR Performance: R²={row['LR_R2_Val']:.4f}, RMSE={row['LR_RMSE_Val']:.2f} kW")
    print(f"  RF Performance: R²={row['RF_R2_Val']:.4f}, RMSE={row['RF_RMSE_Val']:.2f} kW")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  • scaler_comparison_results.csv")
print("  • scaler_r2_comparison.png")
print("  • scaler_rmse_comparison.png")
print("  • scaler_comprehensive_analysis.png")
print("  • scaler_improvement_comparison.png")
print("\nAll results saved successfully!")
print("="*70)