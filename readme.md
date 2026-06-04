# 📝 Text Summarizer

A text summarization web app built with a fine-tuned T5 model, FastAPI, and a clean HTML frontend.

---

## How It Works

The T5 model is fine-tuned on the SAMSum dataset (dialogue summarization) and served via a FastAPI backend. You paste text, hit Summarize, and get a clean summary back.

---

## Project Structure

```
├── app.py                  # FastAPI backend
├── text_summarizer.ipynb   # Model training notebook
├── templates/
│   └── index.html          # Frontend UI
├── parameters.yaml         # Training hyperparameters
└── saved_summary_model/    # Fine-tuned model (generate by training)
```

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/TextSum.git
cd TextSum

# Create and activate environment
conda create -n textsum python=3.10 -y
conda activate textsum

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.0 datasets pandas accelerate sentencepiece fastapi uvicorn
```

---

## Training

Open `text_summarizer.ipynb` and run all cells. The model will be saved to `saved_summary_model/`.

---

## Running the App

```bash
python -m uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Tech Stack

- Model: T5 (fine-tuned on SAMSum)
- Backend: FastAPI
- Frontend: HTML, CSS, Vanilla JS
- Training: PyTorch + HuggingFace Transformers