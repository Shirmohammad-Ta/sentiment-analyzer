from transformers import pipeline

def get_bert_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):

    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
    return sentiment_analysis


if __name__ == "__main__":
    model = get_bert_sentiment_model()
    results = model(["I love this!", "I hate that!"])
    print(results)
