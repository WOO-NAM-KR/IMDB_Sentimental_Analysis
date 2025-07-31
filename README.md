# IMDB Sentiment Analysis using LSTM (PyTorch)

This project performs binary sentiment classification on the IMDB movie reviews dataset using an LSTM-based Recurrent Neural Network built with PyTorch.

## Project Structure

```imdb-sentiment-analysis/
├── notebook/
│ └── imdb_sentiment_lstm.ipynb
├── data/
│ └── IMDB_dataset.csv
├── models/
│ └── state_dict.pt
├── assets/
│ └── training_curves.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Description

- **Dataset**: IMDB movie reviews – 25,000 training and 25,000 testing samples from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- **Task**: Binary sentiment classification — predict whether a review is *positive* or *negative*
- **Model**: LSTM (Long Short-Term Memory) Recurrent Neural Network
- **Preprocessing**:
  - Tokenization using `Tokenizer` from Keras
  - Out-of-vocabulary handling using `<OOV>` token
  - Padding all sequences to a fixed length (`max_length = 300`)
- **Training Setup**:
  - Loss function: Binary Cross Entropy Loss (`BCELoss`)
  - Optimizer: Adam
  - Metric: Accuracy
  - Batch size: 50
  - Gradient clipping to prevent exploding gradients

## Training Metrics

The following graph shows how accuracy and loss changed over training epochs:

![Training Curve](https://raw.githubusercontent.com/WOO-NAM-KR/IMDB_Sentimental_Analysis/main/imdb-sentiment-analysis/assets/training_curves.png)



## Sample Predictions

- **Text**: "The storyline was predictable, but the acting saved the film."  
  **Prediction**: negative (score: 0.1459)

- **Text**: "Absolutely terrible. I want my two hours back."  
  **Prediction**: negative (score: 0.2474)

- **Text**: "One of the best movies I’ve seen this year – touching and inspiring."  
  **Prediction**: positive (score: 0.9770)

> **Note**: The first example includes both negative and positive sentiment cues, making it ambiguous for binary classification. LSTM models, while effective, can struggle to disambiguate such mixed-tone inputs, especially when trained from scratch. The result reflects this uncertainty. More advanced models like BERT tend to perform better in these nuanced cases.

## How to Run


1. Clone the repository:
   ```bash
   git clone https://github.com/WOO-NAM-KR/IMDB_Sentimental_Analysis.git 
   cd imdb-sentiment-analysis

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Jupyter notebook:
   ```bash
   jupyter notebook notebook/imdb_sentiment_analysis.ipynb
   ```

4. (Optional) Load the trained model:
   ```python
   model.load_state_dict(torch.load("models/state_dict.pt"))
   model.eval()
   ```

## Notes

- All data preprocessing, training, validation, and inference are handled in the notebook.
- The model was trained from scratch using PyTorch.
- You can extend this project with:
  - Pretrained word embeddings (GloVe, Word2Vec)
  - Bidirectional LSTM layers
  - Transformer-based architectures like BERT for better performance and context understanding.
