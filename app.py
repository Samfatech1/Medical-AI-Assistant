import streamlit as st
import torch
import json
import random

st.sidebar.title("About")
st.sidebar.info("This chatbot provides educational medical information only.")
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

# =========================
# LOAD MODEL (ONLY ONCE)
# =========================
@st.cache_resource
def load_model():
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

    return model, vocab, lbl_encoder

model, vocab, lbl_encoder = load_model()

# =========================
# LOAD DATA
# =========================
with open("medical_intents.json") as f:
    data = json.load(f)

# =========================
# FUNCTIONS
# =========================
def encode(text):
    words = text.lower().split()
    seq = [vocab.get(w, 0) for w in words]
    seq = seq[:10] + [0]*(10-len(seq))
    return torch.tensor([seq])

def predict(text):
    with torch.no_grad():
        output = model(encode(text))

        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

        tag = lbl_encoder.inverse_transform([predicted.item()])[0]

        return tag, confidence.item()

def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# =========================
# CHAT HISTORY (KEY PART 🔥)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# UI
# =========================
st.title("🩺 Medical Chatbot")
st.caption("Ask about diseases, symptoms, and treatments")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box (ChatGPT-style)
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    tag, confidence = predict(user_input)

    if confidence < 0.6:
        response = "🤔 I'm not sure I understand. Try asking about a disease or symptoms."
    else:
        response = get_response(tag)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(f"{response}\n\n(confidence: {confidence:.2f})")