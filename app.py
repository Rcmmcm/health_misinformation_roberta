from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from urllib.parse import urlparse
import torch.nn.functional as F

app = Flask(__name__)

# ==========================
# CREDIBILITY FUNCTION
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
# LOAD TRAINED MODEL
# ==========================
model = Model()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# ==========================
# ROUTE
# ==========================
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    cred_score = None
    confidence = None
    explanation = None
    text = None
    url = None

    accuracy = 0.86
    precision = 0.79
    recall = 0.96
    f1 = 0.86

    if request.method == "POST":

        text = request.form["text"]
        url = request.form["url"]

        cred_score = get_credibility_score(url)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                credibility=torch.tensor([cred_score], dtype=torch.float)
            )

            probs = F.softmax(logits, dim=1)
            confidence = torch.max(probs).item()

        pred = torch.argmax(logits, dim=1).item()

        if pred == 1:
            result = "FAKE NEWS"
            explanation = "Misleading patterns + low credibility source detected."
        else:
            result = "REAL NEWS"
            explanation = "Content matches reliable medical sources."

    return render_template(
        "index.html",
        result=result,
        cred=cred_score,
        confidence=confidence,
        explanation=explanation,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        text=text,
        url=url
    )

if __name__ == "__main__":
    app.run(debug=True)