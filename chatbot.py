import torch
import json
import random

# Load
with open("medical_intents.json") as f:
    data = json.load(f)

vocab = torch.load("vocab.pth")
lbl_encoder = torch.load("label_encoder.pth")

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 64)
        self.lstm = torch.nn.LSTM(64, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

model = LSTMModel(len(vocab)+1, len(lbl_encoder.classes_))
model.load_state_dict(torch.load("model.pth"))
model.eval()

def encode(text):
    words = text.lower().split()
    seq = [vocab.get(w, 0) for w in words]
    seq = seq[:10] + [0]*(10-len(seq))
    return torch.tensor([seq])

def predict(text):
    with torch.no_grad():
        output = model(encode(text))
        pred = torch.argmax(output).item()
        return lbl_encoder.inverse_transform([pred])[0]

def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

print("Medical Chatbot Ready!")

while True:
    text = input("You: ")
    if text.lower() == "quit":
        break

    tag = predict(text)
    print("Bot:", get_response(tag))