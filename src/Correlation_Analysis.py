import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load Data
textures_df = pd.read_csv("data/processed/texture_predictions.csv")
emotions_df = pd.read_csv("data/processed/processed_feelingblue_annotations.csv")

# Normalize filenames
textures_df['filename'] = textures_df['filename'].apply(lambda x: os.path.basename(x))
emotions_df['filename'] = emotions_df['filename'].apply(lambda x: os.path.basename(x))

# Merge textures and emotions data
combined_df = pd.merge(textures_df, emotions_df, on="filename", how="inner")

# Debugging: Check for non-empty merge
print("Combined DataFrame Info:")
print(combined_df.info())
print("Sample Combined Data:")
print(combined_df.head())

# One-hot encode the textures for correlation analysis
texture_one_hot = pd.get_dummies(combined_df['texture'], prefix="texture")

# Add one-hot encoded textures to the combined DataFrame
combined_df = pd.concat([combined_df, texture_one_hot], axis=1)

# Group data by texture and emotion
grouped_data = combined_df.groupby(['texture', 'emotion']).agg({
    'polarity': 'mean',
    'subjectivity': 'mean',
    'filename': 'count'
}).rename(columns={'filename': 'image_count'}).reset_index()

# Debugging: Check grouped data
print("Grouped Data:")
print(grouped_data.head())

# Save grouped data for reference
grouped_data.to_csv("data/processed/grouped_texture_emotion.csv", index=False)
print("Grouped data saved to data/processed/grouped_texture_emotion.csv")

# Merge `image_count` back into `combined_df`
combined_df = pd.merge(
    combined_df,
    grouped_data[['texture', 'emotion', 'image_count']],
    on=['texture', 'emotion'],
    how='left'
)

# Debugging: Check correlation matrix input
print("Correlation Matrix Input Data:")
print(combined_df[['polarity', 'subjectivity', 'image_count'] + list(texture_one_hot.columns)].head())

# Correlation Analysis
correlation_matrix = combined_df[['polarity', 'subjectivity', 'image_count'] + list(texture_one_hot.columns)].corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Save correlation matrix to file
correlation_matrix.to_csv("data/processed/correlation_matrix.csv")
print("Correlation matrix saved to data/processed/correlation_matrix.csv")

# Visualize Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Textures, Emotions, and Features")
plt.tight_layout()
plt.savefig("data/processed/texture_emotion_correlation_heatmap.png")
plt.show()

# Emotion-Texture Correlation Scatter Plot (Example)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=grouped_data, x='polarity', y='subjectivity', hue='texture', style='emotion', size='image_count')
plt.title("Polarity vs Subjectivity by Texture and Emotion")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig("data/processed/polarity_subjectivity_scatter.png")
plt.show()
