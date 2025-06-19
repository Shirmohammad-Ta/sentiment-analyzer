from src.data_loader import load_data
from src.preprocessing import clean_text
from src.vectorizer import get_vectorizer
from src.model_builder import build_model
from src.evaluator import evaluate_model
from src.predictor import save_model

def main():
    data = load_data("data/raw/reviews.csv")
    data["cleaned"] = data["review"].apply(clean_text)

    X = data["cleaned"]
    y = data["label"]

    vectorizer = get_vectorizer()
    model = build_model(vectorizer)
    model.fit(X, y)

    preds = model.predict(X)
    evaluate_model(y, preds)

    save_model(model, "models/sentiment_model.pkl")

if __name__ == "__main__":
    main()
