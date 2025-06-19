# Sentiment Analyzer

A modular Python project for sentiment analysis on user reviews using traditional ML and optional integration with deep learning.

##  Features

-  Modular code architecture with reusable components
-  Text preprocessing with NLTK (cleaning, lemmatization, stopword removal)
-  Multiple model support (Logistic Regression, SVM, etc.)
-  Evaluation with classification report and confusion matrix
-  Unit testing using `unittest` framework
-  Interactive API using Flask for real-time inference
-  Model persistence using joblib
-  Visualization using seaborn and matplotlib
-  Jupyter Notebook for end-to-end demo and reporting
-  Optional: Integration with pretrained transformers like BERT for advanced performance
-  Streamlit UI for simple front-end interaction

##  How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python src/main.py
   ```

3. Run Streamlit app:
   ```
   streamlit run app/sentiment_ui.py
   ```

4. Run API server:
   ```
   python src/api.py
   ```

5. Run tests:
   ```
   python -m unittest discover tests
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

This project is licensed under the MIT License.
