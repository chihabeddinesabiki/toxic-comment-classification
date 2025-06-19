import joblib

model_path = "models/model_LP.pkl"
vectorizer_path = "models/vectorizer_LP.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_lp(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "Toxique" if pred == 1 else "Non toxique"