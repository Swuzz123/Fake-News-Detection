import torch
import torch.nn as nn
from transformers import BertModel

class FakeNewsDetection(nn.Module):
    def __init__(self, n_classes, model_name = "bert-base-uncased"):
        super(FakeNewsDetection, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)