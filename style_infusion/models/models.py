import torch
import torch.nn as nn
from transformers import BertModel

class StyleClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=2):
        super(StyleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        outputs = self.bert(**input_batch)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        output = self.softmax(output)
        return torch.max(output)

class StyleClassifierScore(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=3):
        super(StyleClassifierScore, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        outputs = self.bert(**input_batch)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        output = self.sigmoid(output.flatten())
        # Since 0.5 corresponds to equal confidence, we use the following to get confidence on a scale of 0 to 1
        output = 2 * torch.abs(0.5 - output)
        return output