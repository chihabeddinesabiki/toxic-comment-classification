# Toxic Comment Classification

Machine learning and deep learning web application for detecting toxic comments from CSV files, built with Flask, Scikit-learn, and a BERT + CNN high-performance pipeline.

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/Deep%20Learning-BERT%20%2B%20CNN-49DCB1?style=for-the-badge" alt="BERT CNN" />
  <img src="https://img.shields.io/badge/Backend-Flask-111827?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/NLP-Toxicity%20Detection-7AA2FF?style=for-the-badge" alt="NLP toxicity detection" />
</p>

## Overview

This project classifies text comments as toxic or non-toxic and generates visual analysis from uploaded CSV files.

It includes two model paths:

- **LP - Light Pipeline:** TF-IDF vectorization with Logistic Regression for fast inference.
- **HP - High-Performance Pipeline:** BERT embeddings with a CNN classifier for deeper NLP modeling.

The application is designed as a practical NLP moderation tool: upload comments, choose a model, run predictions, and review summary charts.

## Key Features

- CSV upload and automatic comment-column detection.
- Text preprocessing for noisy user-generated content.
- Fast ML inference with TF-IDF + Logistic Regression.
- Optional deep learning inference with BERT + CNN model weights.
- Toxicity summary with total comments, toxic count, non-toxic count, and toxicity rate.
- Visual reports using pie, bar, and heatmap charts.
- Flask web interface for a simple end-to-end demo.

## Tech Stack

| Area | Tools |
|---|---|
| Backend | Flask, Werkzeug |
| Machine Learning | Scikit-learn, TF-IDF, Logistic Regression |
| Deep Learning | PyTorch, Transformers, BERT, CNN |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Interface | HTML, CSS, JavaScript |

## Project Structure

```text
toxic-comment-classification/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── model_lp.py
│   ├── model_hp.py
│   └── utils.py
├── data/
│   ├── uploads/
│   └── results/
├── models/
│   ├── model_LP.pkl
│   └── vectorizer_LP.pkl
├── notebooks/
│   ├── 1_preprocessing.ipynb
│   ├── 2_model_LP.ipynb
│   └── 3_model_HP.ipynb
├── static/
├── templates/
├── requirements.txt
└── run.py
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/chihabeddinesabiki/toxic-comment-classification.git
cd toxic-comment-classification
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

## CSV Format

Upload a `.csv` file containing comments. The app automatically looks for one of these columns:

```text
text, comment, message, content, body, text_clean
```

If no matching header is found, the first column is used.

Example:

```csv
comment
I hate this product.
You are amazing!
This is disgusting.
Great job.
```

## Model Notes

The light pipeline model is included in the repository:

```text
models/model_LP.pkl
models/vectorizer_LP.pkl
```

The high-performance BERT + CNN pipeline expects a model weights file at:

```text
models/model_HP.pt
```

Large deep learning weights are intentionally not required for the light demo. If `model_HP.pt` is not present, the app still runs with the LP model and shows a clear message when HP is selected.

## Results

The app generates:

- total number of comments
- toxic and non-toxic counts
- toxicity rate
- pie chart
- bar chart
- heatmap-style summary

## Author

**Chihab Eddine Sabiki**  
Machine Learning & Deep Learning Engineer  
[GitHub](https://github.com/chihabeddinesabiki) · [LinkedIn](https://www.linkedin.com/in/chihabsab) · [Email](mailto:chihabeddinesabiki@gmail.com)

## License

This project is provided for educational and portfolio purposes.
