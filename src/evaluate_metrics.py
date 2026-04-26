import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, mean_squared_error
import os

def calculate_metrics(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg):
    # Classification Metrics
    precision = precision_score(y_true_clf, y_pred_clf, average='binary')
    recall = recall_score(y_true_clf, y_pred_clf, average='binary')
    sensitivity = recall # Sensitivity is another name for recall
    f1 = f1_score(y_true_clf, y_pred_clf, average='binary')
    
    # Regression Metrics
    r2 = r2_score(y_true_reg, y_pred_reg)
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rms = np.sqrt(mse)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'Sensitivity': sensitivity,
        'F1 Score': f1,
        'R^2': r2,
        'MSE': mse,
        'RMS': rms
    }
    return metrics

def plot_bar_graphs(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    clf_metrics = ['Precision', 'Recall', 'Sensitivity', 'F1 Score']
    clf_vals = [metrics[m] for m in clf_metrics]
    axes[0].bar(clf_metrics, clf_vals, color=['blue', 'orange', 'green', 'red'])
    axes[0].set_title('Classification Metrics')
    axes[0].set_ylim([0, 1])
    
    reg_metrics = ['R^2', 'MSE', 'RMS']
    reg_vals = [metrics[m] for m in reg_metrics]
    axes[1].bar(reg_metrics, reg_vals, color=['purple', 'brown', 'pink'])
    axes[1].set_title('Regression Metrics')
    
    plt.tight_layout()
    plt.savefig('plots_bar_graphs.png')
    plt.close()

def plot_radar(metrics):
    categories = ['Precision', 'Recall', 'Sensitivity', 'F1 Score']
    values = [metrics[c] for c in categories]
    values += [values[0]] # Close the loop
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Radar Plot for Classification Metrics')
    
    plt.savefig('plots_radar.png')
    plt.close()

def plot_taylor():
    # A simple Taylor plot approximation
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Angle is standard deviation (simulated), radius is correlation
    # This is a basic mock of Taylor plot visually
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_thetamax(90)
    
    # Add dummy data
    stdevs = [1.0, 1.2, 0.8, 1.1]
    corrs = [1.0, 0.9, 0.85, 0.7]
    angles = np.arccos(corrs)
    
    colors = ['black', 'red', 'blue', 'green']
    labels = ['Observation', 'Model A', 'Model B', 'Model C']
    
    for ang, std, col, lab in zip(angles, stdevs, colors, labels):
        ax.plot(ang, std, marker='o', color=col, label=lab, markersize=8)
        
    ax.set_xlabel('Standard Deviation')
    plt.legend(loc='upper right')
    plt.title('Mock Taylor Plot')
    
    plt.savefig('plots_taylor.png')
    plt.close()

def plot_heatmap(data):
    plt.figure(figsize=(8, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots_heatmap.png')
    plt.close()

def plot_violin(data):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data)
    plt.title('Violin Plot of Data')
    plt.tight_layout()
    plt.savefig('plots_violin.png')
    plt.close()

if __name__ == '__main__':
    print("Generating Synthetic Data for Metrics Calculation...")
    
    # Synthetic Classification Data (0s and 1s)
    np.random.seed(42)
    y_true_clf = np.random.randint(0, 2, 100)
    y_pred_clf = np.random.randint(0, 2, 100)
    
    # Synthetic Regression Data
    y_true_reg = np.random.normal(50, 10, 100)
    y_pred_reg = y_true_reg + np.random.normal(0, 5, 100)
    
    print("Calculating Metrics...")
    metrics = calculate_metrics(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg)
    
    print("\n--- Calculated Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nGenerating Plots...")
    plot_bar_graphs(metrics)
    plot_radar(metrics)
    plot_taylor()
    
    # Data for Heatmap and Violin
    df = pd.DataFrame({
        'Feature_1': np.random.normal(0, 1, 100),
        'Feature_2': np.random.normal(5, 2, 100),
        'Feature_3': np.random.uniform(0, 10, 100),
        'Target': y_true_reg
    })
    
    plot_heatmap(df)
    plot_violin(df)
    
    print("All plots generated successfully. Check the current directory for PNG files.")
