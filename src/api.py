from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("models/sentiment_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data["texts"]
    preds = model.predict(texts)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
