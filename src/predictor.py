import joblib

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, texts):
    return model.predict(texts)
