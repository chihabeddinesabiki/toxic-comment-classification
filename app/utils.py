import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def clean_text(text):
    """Nettoyage de texte brut."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip().lower()

def generate_report(predictions, texts, filename="results"):
    """Enregistre les prédictions dans un fichier CSV et retourne un résumé."""
    df = pd.DataFrame({
        "text": texts,
        "prediction": predictions
    })
    df["label"] = df["prediction"].map({1: "Toxique", 0: "Non toxique"})
    os.makedirs("data/results", exist_ok=True)
    df.to_csv(f"data/results/{filename}.csv", index=False)

    toxic = df["prediction"].sum()
    non_toxic = len(df) - toxic
    total = len(df)
    toxic_rate = round((toxic / total) * 100, 2)

    return {
        "toxic": toxic,
        "non_toxic": non_toxic,
        "total": total,
        "toxic_rate": toxic_rate
    }

def generate_visuals(predictions, texts):
    """Génère des graphiques et retourne le résumé des statistiques."""
    df = pd.DataFrame({
        "text": texts,
        "prediction": predictions
    })
    df["label"] = df["prediction"].map({1: "Toxique", 0: "Non toxique"})

    toxic = df["prediction"].sum()
    non_toxic = len(df) - toxic
    total = len(df)
    toxic_rate = round((toxic / total) * 100, 2)

    # 📊 Pie chart
    plt.figure(figsize=(4, 4), dpi=150)
    plt.pie([toxic, non_toxic],
            labels=["Toxique", "Non toxique"],
            autopct="%1.1f%%",
            colors=["crimson", "forestgreen"])
    plt.gca().set_facecolor('none')
    plt.savefig("static/plot_pie.png", transparent=True, bbox_inches='tight')
    plt.close()

    # 📊 Bar chart
    plt.figure(figsize=(4.5, 3.5), dpi=150)
    sns.countplot(data=df, x="label", palette=["crimson", "forestgreen"])
    plt.gca().set_facecolor('none')
    plt.title("Répartition des commentaires")
    plt.savefig("static/plot_bar.png", transparent=True, bbox_inches='tight')
    plt.close()

    # 📊 Confusion Matrix (simulation)
    y_true = predictions
    y_pred = predictions
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 3.5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu", xticklabels=["Non toxique", "Toxique"],
                yticklabels=["Non toxique", "Toxique"])
    plt.gca().set_facecolor('none')
    plt.title("Matrice de confusion simulée")
    plt.savefig("static/plot_heatmap.png", transparent=True, bbox_inches='tight')
    plt.close()

    return {
        "pie": "static/plot_pie.png",
        "bar": "static/plot_bar.png",
        "heatmap": "static/plot_heatmap.png",
        "stats": {
            "total": total,
            "toxic": toxic,
            "non_toxic": non_toxic,
            "toxic_rate": toxic_rate
        }
    }
