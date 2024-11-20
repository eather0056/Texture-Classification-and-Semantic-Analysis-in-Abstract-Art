import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Lowercase and tokenize
    doc = nlp(text.lower())
    # Remove stopwords and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def preprocess_annotations(input_csv, output_csv):
    # Load annotations
    df = pd.read_csv(input_csv)
    # Apply cleaning
    df['cleaned_text'] = df['text_column'].apply(clean_text)  # Replace 'text_column' with your column name
    # Save processed annotations
    df.to_csv(output_csv, index=False)
    print("Text annotations preprocessed and saved.")

# Set paths
input_csv = "data/annotations/wikiartabstractcolors.csv"
output_csv = "data/processed/cleaned_annotations.csv"
preprocess_annotations(input_csv, output_csv)
