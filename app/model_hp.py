import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Détection GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classe du modèle
class BertCNNClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BertCNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        x = hidden_states.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        return x

# Chargement du tokenizer et du modèle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertCNNClassifier()
model.load_state_dict(torch.load("models/model_HP.pt", map_location=device))
model.to(device)
model.eval()

# Fonction de prédiction
def predict_hp(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return "Toxique" if predicted_class == 1 else "Non toxique"
