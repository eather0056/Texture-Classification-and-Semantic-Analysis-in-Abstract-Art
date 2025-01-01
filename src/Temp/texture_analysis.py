import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Report Metrics (From your evaluation results)
eval_metrics = pd.DataFrame({
    'class': ['chaotic', 'circular', 'dots', 'lines', 'rough', 'smooth'],
    'precision': [1.00, 0.12, 0.06, 0.26, 0.51, 0.50],
    'recall': [0.05, 0.11, 0.33, 0.52, 0.71, 0.26],
    'f1_score': [0.09, 0.12, 0.11, 0.34, 0.59, 0.34],
    'support': [44, 9, 3, 21, 48, 23]
})

# Grouped Sentiments (Simulated example or replace with actual grouped data)
grouped_sentiments = pd.DataFrame({
    'emotion': ['chaotic', 'circular', 'dots', 'lines', 'rough', 'smooth'],
    'polarity': [-0.02, 0.01, 0.15, 0.22, -0.01, 0.30],
    'subjectivity': [0.38, 0.36, 0.42, 0.40, 0.35, 0.45]
})

# Merge metrics and sentiments
merged_data = pd.merge(eval_metrics, grouped_sentiments, left_on='class', right_on='emotion', how='inner')

# Calculate Correlations
correlation_results = {}
for metric in ['precision', 'recall', 'f1_score']:
    correlations = {}
    for sentiment in ['polarity', 'subjectivity']:
        corr, p_value = pearsonr(merged_data[metric], merged_data[sentiment])
        correlations[sentiment] = {'correlation': corr, 'p_value': p_value}
    correlation_results[metric] = correlations

# Print Correlation Results
print("Correlation Results:")
for metric, results in correlation_results.items():
    print(f"\nMetric: {metric}")
    for sentiment, stats in results.items():
        print(f"  Sentiment: {sentiment}, Correlation: {stats['correlation']:.2f}, p-value: {stats['p_value']:.4f}")

# Visualize Correlations (Heatmap)
correlation_matrix = merged_data[['precision', 'recall', 'f1_score', 'polarity', 'subjectivity']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Save Results
merged_data.to_csv("data/processed/correlation_analysis.csv", index=False)
print("Merged data saved to data/processed/correlation_analysis.csv")
