from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
app = FastAPI(title="Text Summarization APP")

#model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)
#templates
# templates = Jinja2Templates(directory="templates")

# input schema for dialouge => string
class DialogueInput(BaseModel):
    dialogue: str
def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n", " ",text) #lines
    text = re.sub(r"\s+", " ",text) #spaces
    text = re.sub(r"<.*?>", " ",text) #html tags
    text = text.strip().lower()
    return text
def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length=512,
        truncation=True,
        return_tensors = "pt"
    ).to(device)
    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4, # generate 4 seq of output and then give us the best one
        early_stopping=True
    )
    # token ids => convert to text ( decoding )
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary
#API endpoint
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}
@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)