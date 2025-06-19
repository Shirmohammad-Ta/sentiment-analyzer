from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_model(model_name="logreg"):
    vectorizer = TfidfVectorizer(max_features=5000)

    if model_name == "logreg":
        clf = LogisticRegression()
    elif model_name == "svm":
        clf = SVC(probability=True)
    elif model_name == "randomforest":
        clf = RandomForestClassifier()
    elif model_name == "xgboost":
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Model {model_name} not supported.")

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
    ])
    return pipeline
