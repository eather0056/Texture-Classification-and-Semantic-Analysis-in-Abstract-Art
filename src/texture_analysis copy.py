import pandas as pd
import json
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the JSON file
with open('data/processed/feelingbluelemas.json', 'r') as file:
    data = json.load(file)

# Process JSON data into a DataFrame
records = []
for filename, emotions in data.items():
    for emotion, categories in emotions.get("emotion", {}).items():
        for category, annotations in categories.items():
            for anno_id, details in annotations.items():
                words = details.get("words", {})
                polarity = sum(words.get("polarity", [0])) / len(words.get("polarity", [1])) if words.get("polarity") else 0
                subjectivity = sum(words.get("subjectivity", [0])) / len(words.get("subjectivity", [1])) if words.get("subjectivity") else 0
                rationale = details.get("rationale", "")
                annotator = details.get("annotator", "")
                records.append({
                    "filename": filename,
                    "emotion": emotion,
                    "category": category,
                    "rationale": rationale,
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "annotator": annotator
                })

# Convert records to DataFrame
df = pd.DataFrame(records)

# Save the processed DataFrame
df.to_csv('data/processed/processed_feelingblue_annotations.csv', index=False)
print("Processed data saved to data/processed/processed_feelingblue_annotations.csv")

# Display sample data
print("Sample Data:")
print(df.head())

# Group by emotion and calculate average polarity and subjectivity
grouped_sentiments = df.groupby('emotion')[['polarity', 'subjectivity']].mean()
print("Grouped Sentiments:")
print(grouped_sentiments)

# Perform correlation analysis between polarity, subjectivity, and other numerical features
correlation_matrix = df[['polarity', 'subjectivity']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.index)
plt.title("Correlation Matrix")
plt.savefig('data/processed/correlation_matrix.png')
plt.show()

# Save grouped sentiments to CSV
grouped_sentiments.to_csv('data/processed/grouped_sentiments.csv')
print("Grouped sentiments saved to data/processed/grouped_sentiments.csv")

# Debugging: Ensure 'emotion' is not multi-dimensional before grouping for word counts
word_counts = df['rationale'].str.split().explode().value_counts().reset_index()
word_counts.columns = ['word', 'count']
word_counts = pd.merge(df[['emotion']], word_counts, how='left', left_index=True, right_index=True)
word_counts['emotion'] = df['emotion']  # Ensure correct alignment

# Aggregate word counts by emotion
emotion_word_counts = word_counts.groupby('emotion', as_index=False).sum()
print("Emotion Word Counts:")
print(emotion_word_counts.head())

# Save emotion word counts to CSV
emotion_word_counts.to_csv('data/processed/emotion_word_counts.csv', index=False)
print("Emotion word counts saved to data/processed/emotion_word_counts.csv")
