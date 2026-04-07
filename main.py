import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from urllib.parse import urlparse

# ==========================
# CREDIBILITY SCORE
# ==========================

def get_credibility_score(url):
    domain = urlparse(url).netloc.lower()

    if "who.int" in domain:
        return 0.9
    elif ".gov" in domain:
        return 0.8
    elif ".edu" in domain:
        return 0.7
    else:
        return 0.3


# ==========================
# MODEL
# ==========================

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(769, 2)

    def forward(self, input_ids, attention_mask, credibility):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        credibility = credibility.float().unsqueeze(1)
        x = torch.cat((pooled, credibility), dim=1)
        return self.fc(x)


# ==========================
# DATASET CLASS
# ==========================

class HealthDataset(Dataset):
    def __init__(self, texts, labels, cred, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels
        self.cred = cred

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["credibility"] = torch.tensor(self.cred[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


# ==========================
# TRAIN FUNCTION
# ==========================

def train_model():

    print("\nLoading dataset...\n")

    fake_df = pd.read_csv("data/NewsFakeCOVID-19.csv")[["content", "news_url"]].dropna()
    real_df = pd.read_csv("data/NewsRealCOVID-19.csv")[["content", "news_url"]].dropna()

    fake_df["label"] = 1
    real_df["label"] = 0

    real_df = real_df.sample(n=len(fake_df), random_state=42)

    df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)

    df["credibility"] = df["news_url"].apply(get_credibility_score)

    train_texts, test_texts, train_labels, test_labels, train_cred, test_cred = train_test_split(
        df["content"].tolist(),
        df["label"].tolist(),
        df["credibility"].tolist(),
        test_size=0.2,
        random_state=42
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = HealthDataset(train_texts, train_labels, train_cred, tokenizer)
    test_dataset = HealthDataset(test_texts, test_labels, test_cred, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = Model()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining started...\n")

    model.train()

    for epoch in range(3):
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                credibility=batch["credibility"]
            )

            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss}")

    print("\nSaving model...\n")

    torch.save(model.state_dict(), "model.pth")

    print("✅ Model saved successfully as model.pth\n")

    # ==========================
    # EVALUATION
    # ==========================

    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                credibility=batch["credibility"]
            )

            p = torch.argmax(logits, dim=1)

            preds.extend(p.tolist())
            labels.extend(batch["labels"].tolist())

    acc = accuracy_score(labels, preds)

    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(labels, preds))


# ==========================
# MAIN MENU
# ==========================

while True:

    print("\n===== HEALTH MISINFORMATION DETECTOR =====")
    print("1. Train Model")
    print("2. Exit")

    choice = input("\nEnter choice: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        print("\nExiting...\n")
        break

    else:
        print("\nInvalid choice\n")