from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_model(vectorizer):
    return Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression())
    ])
