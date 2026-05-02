import json
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

# Load data
with open("medical_intents.json") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Encode labels
lbl_encoder = LabelEncoder()
labels = lbl_encoder.fit_transform(labels)

# Build vocab
words = []
for text in texts:
    words.extend(text.split())

vocab = {w: i+1 for i, w in enumerate(set(words))}

def encode(text):
    return [vocab.get(w, 0) for w in text.split()]

max_len = 10

X = [encode(t)[:max_len] + [0]*(max_len-len(encode(t))) for t in texts]
y = labels

X = torch.tensor(X)
y = torch.tensor(y)

class ChatDataset(Dataset):
    def __len__(self):
        return len(X)
    def __getitem__(self, idx):
        return X[idx], y[idx]

loader = DataLoader(ChatDataset(), batch_size=8, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = LSTMModel(len(vocab)+1, len(set(labels)))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(200):
    for X_batch, y_batch in loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save
torch.save(model.state_dict(), "model.pth")
torch.save(vocab, "vocab.pth")
torch.save(lbl_encoder, "label_encoder.pth")

print("Training complete!")