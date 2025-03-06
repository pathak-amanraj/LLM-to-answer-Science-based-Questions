# Science Exam Multiple Choice Question Answering Model

## Project Overview
A sophisticated machine learning solution for answering multiple-choice science exam questions using advanced retrieval and prediction techniques.

## Key Features
- Hybrid document retrieval system
- Longformer-based multiple-choice question answering
- Advanced performance metrics and visualization
- Confidence-based prediction mechanism

## Model Architecture
- *Retrieval*: Two-pronged TF-IDF based document search
  - Parsed Wikipedia dataset
  - Cohere-based dataset
- *Model*: Longformer for multiple-choice inference
- *Prediction Strategy*: 
  - Average probabilities from two retrievals
  - Fallback prediction with 0.4 confidence threshold

## Performance Metrics
- AUC-ROC Score
- Mean Average Precision
- Logarithmic Loss
- F1 Score
- Cohen's Kappa

## Visualization
Includes comprehensive performance visualization:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Performance Metrics Bar Plot

## Requirements
- Python 3.10+
- Seaborn
- tokenizers>=0.10.3
- torch>=1.6.0
- torchvision
- scipy
- nltk
- sentencepiece
- huggingface-hub
- torch==2.0.1
- pandas==2.0.1
- numpy==1.24.3
- datasets==2.14.4
- scikit-learn==1.2.2
- transformers==4.31.0
- matplotlib==3.7.1
- tqdm==4.65.0
- sentence-transformers==2.2.2
- faiss-gpu==1.7.2
- blingfire==0.1.8
- peft==0.4.0
- trl==0.5.0
- unicodedata


## Usage
python


## Model Limitations
- Depends on quality of retrieved documents
- Performance varies with question complexity
- Requires significant computational resources



## Acknowledgments
- Kaggle LLM Science Exam Competition
- Longformer Model
- Open-source ML community
- Dr. Prerna Mukherjee Mam
- Mohammadreza Banaei
