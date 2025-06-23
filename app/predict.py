import torch 
from app.model import model, tokenizer

MAX_LEN = 128
label_map = {0: "Fake", 1: "True"}

def predict_fake_news(text):
    inputs = tokenizer.encode_plus(
        text,
        max_length = MAX_LEN,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        _, prediction = torch.max(outputs, dim = 1)
        
    return label_map[prediction.item()]