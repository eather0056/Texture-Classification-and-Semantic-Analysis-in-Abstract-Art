# Texture Classification and Semantic Analysis in Abstract Art

This repository contains code and resources for analyzing abstract art by classifying textures and correlating them with viewer-generated emotional responses. The project bridges computational vision and sentiment analysis to uncover patterns in abstract art.

## Project Overview
Abstract art evokes diverse emotional responses, making it challenging to analyze computationally. This project focuses on:
- **Texture Classification:** Classifying textures (e.g., chaotic, smooth, rough) in abstract art using a fine-tuned CLIP model.
- **Sentiment Analysis:** Extracting polarity and subjectivity from viewer-generated text annotations.
- **Correlation Analysis:** Exploring relationships between textures and emotional responses.

## Repository Structure
```
├── src/
│   ├── Correlation_Analysis.py           # Code for texture-emotion correlation analysis
│   ├── fine_tune_with_vision_model.py    # Code for fine-tuning the CLIP model
│   ├── generate_splits.py                # Script to create training, validation, and test splits
│   ├── model_evaluation.py               # Code to evaluate the trained model
│   ├── save_datafrme.py                  # Helper script to save data
│   ├── texture_analysis.py               # Main script for texture and sentiment analysis
│   ├── texture_categorized_annotations.py  # Code to prepare categorized annotations
│   ├── verify_data.py                    # Script for data verification
│   └── Temp/                             # Temporary scripts and files
├── data/
│   ├── raw/                              # Raw dataset of abstract art images
│   ├── processed/                        # Processed data files
│   └── results/                          # Analysis results and visualizations
├── models/                               # Trained model weights and configurations
├── results/                              # Generated plots and outputs
└── README.md                             # Repository documentation
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Texture-Classification-and-Semantic-Analysis-in-Abstract-Art.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Texture-Classification-and-Semantic-Analysis-in-Abstract-Art
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. **Obtain Dataset:** Ensure you have a dataset of abstract art images with texture and emotional annotations.
2. **Organize Dataset:** Structure your dataset as follows:
   ```
   data/raw/wikiart/
   ├── images/
   │   ├── chaotic/
   │   ├── smooth/
   │   ├── rough/
   │   └── ... (other texture categories)
   ├── annotations/
   │   ├── emotions.json
   │   └── textures.json
   ```
3. Run `generate_splits.py` to create train, validation, and test splits:
   ```bash
   python src/generate_splits.py
   ```

## Running the Experiment
### 1. Fine-Tuning the Model
Fine-tune the CLIP model using the texture data:
```bash
python src/fine_tune_with_vision_model.py
```
Model checkpoints and final weights will be saved in the `models/` directory.

### 2. Evaluating the Model
Evaluate the trained model on the test dataset:
```bash
python src/model_evaluation.py
```
This script generates a classification report and confusion matrix.

### 3. Sentiment Analysis and Correlation
Perform sentiment analysis and correlate textures with emotions:
```bash
python src/texture_analysis.py
python src/Correlation_Analysis.py
```
Results, including correlation matrices and scatter plots, will be saved in the `results/` directory.

## Results
- **Classification Performance:** Model accuracy, precision, recall, and F1-score for each texture class.
- **Sentiment Insights:** Average polarity and subjectivity for each emotion.
- **Correlation Analysis:** Heatmaps and scatter plots showing texture-emotion relationships.

## Future Work
- Refine texture categories to reduce semantic overlaps.
- Explore additional models and datasets for improved texture classification.
- Develop filters to handle ambiguous annotations and reduce noise.

## Acknowledgments
This work was supported and guided by **Professor Lledó Museros Cabedo**, who provided the dataset and valuable feedback.
