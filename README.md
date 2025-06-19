#  Sentiment Analyzer

A modular Python project for sentiment analysis on user reviews using traditional machine learning methods.

##  Features

- Clean architecture with modular components
- Text preprocessing with NLTK
- TF-IDF vectorization
- Model training using Logistic Regression
- Evaluation with classification report and confusion matrix
- REST API with Flask
- Interactive UI with Streamlit
- Unit tests with unittest framework

##  How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run main training script:
   ```
   python src/main.py
   ```

3. Launch Streamlit app:
   ```
   streamlit run app/sentiment_ui.py
   ```

4. Run API server:
   ```
   python src/api.py
   ```

##  Project Structure

```
sentiment-analyzer/
├── src/
├── data/
├── models/
├── outputs/
├── app/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
```

##  License

This repository is released under the MIT License.
