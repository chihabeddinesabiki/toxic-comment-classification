from pathlib import Path


MODEL_WEIGHTS_PATH = Path("models/model_HP.pt")
_tokenizer = None
_model = None
_device = None


def _load_hp_model():
    global _tokenizer, _model, _device

    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model, _device

    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "HP model weights were not found. Add models/model_HP.pt or use the LP model."
        )

    import torch
    import torch.nn as nn
    from transformers import BertModel, BertTokenizer

    class BertCNNClassifier(nn.Module):
        def __init__(self, hidden_size=768, num_classes=2):
            super().__init__()
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
            return self.fc(x)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    _model = BertCNNClassifier()
    _model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()

    return _tokenizer, _model, _device


def predict_hp(text):
    import torch

    tokenizer, model, device = _load_hp_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return "Toxique" if predicted_class == 1 else "Non toxique"
