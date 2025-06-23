import torch
from transformers import BertTokenizer
from app.bert_classifier import FakeNewsDetection

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

# Load model
model = FakeNewsDetection(n_classes = 2)
model.load_state_dict(torch.load("bert_fakenews_model.bin",  map_location=torch.device('cpu')))
model.eval()