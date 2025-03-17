from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from transformers import pipeline                                                             
from transformers import pipeline

# ==============================
# Initialisation de l'application FastAPI
# ==============================
app = FastAPI()

# Templates (si tu veux une page d’accueil)
templates = Jinja2Templates(directory="templates")

# ==============================
# Chargement des modèles et vectorizers
# ==============================
cv = joblib.load('ft_title_N_cv.pkl')
tfidf = joblib.load('ft_title_N_tfidf.pkl')
model_supervise = joblib.load('model_supervise.pkl')
lda_model = joblib.load('best_lda_model_ft_title_N.pkl')

print("✅ Vectorizer, TF-IDF Transformer, modèle supervisé et modèle LDA chargés.")

# Modèle BERT Zero-Shot (optionnel si tu veux)
nlp_bert = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
print("✅ Pipeline BERT chargé.")

# ==============================
# Pydantic Request Model
# ==============================
class QuestionRequest(BaseModel):
    question: str

# ==============================
# Page d'accueil (optionnel)
# ==============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})

# ==============================
# Endpoint de prédiction
# ==============================
@app.post("/predict-tags/")
async def predict_tags(request: QuestionRequest):
    question = request.question

    try:
        # Vectorisation
        X_cv = cv.transform([question])
        X_tfidf = tfidf.transform(X_cv)

        # Prédiction supervisée
        supervised_prediction = model_supervise.predict(X_tfidf)

        # Extraction des topics avec LDA
        lda_topics = lda_model.transform(X_tfidf)

        # Prédiction Zero-shot avec BERT
        candidate_labels = ["python", "data science", "machine learning", "api", "backend", "deep learning", "flask", "fastapi"]
        bert_tags = nlp_bert(question, candidate_labels=candidate_labels)

        return {
            "supervised_tags": supervised_prediction.tolist(),
            "lda_topics": lda_topics.tolist(),
            "bert_tags": bert_tags["labels"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}") 
