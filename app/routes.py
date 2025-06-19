from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from app.model_lp import predict_lp
from app.model_hp import predict_hp
from app.utils import clean_text, generate_visuals

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    uploaded_results = []
    report = {}

    if request.method == "POST":
        model_type = request.form.get("model", "").lower()
        file = request.files.get("file")

        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            filepath = os.path.join("data/uploads", filename)
            file.save(filepath)

            try:
                # Try reading with headers first
                df = pd.read_csv(filepath)
            except pd.errors.ParserError:
                # Try again as headerless
                df = pd.read_csv(filepath, header=None)

            # Step 1: Try known column names
            possible_columns = ["text", "comment", "message", "content", "body", "text_clean"]
            text_column = next((col for col in df.columns if str(col).lower() in possible_columns), None)

            # Step 2: If no column names match, fallback to the first column
            if text_column:
                texts = df[text_column].astype(str).tolist()
            else:
                # Use first column if no named column found
                texts = df.iloc[:, 0].astype(str).tolist()

            # Step 3: Clean the text
            cleaned = [clean_text(t) for t in texts]

            # Step 4: Predict
            if model_type == "lp":
                predictions = [1 if predict_lp(t) == "Toxique" else 0 for t in cleaned]
            elif model_type == "hp":
                predictions = [1 if predict_hp(t) == "Toxique" else 0 for t in cleaned]
            else:
                return render_template("index.html", error="❌ Unknown model selected.")

            # Step 5: Results
            uploaded_results = list(zip(texts, predictions))
            report = generate_visuals(predictions, texts)

        else:
            return render_template("index.html", error="❌ Invalid file type. Please upload a .csv file.")

    return render_template("index.html", uploaded_results=uploaded_results, report=report)
