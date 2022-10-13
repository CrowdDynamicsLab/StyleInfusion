import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, models

# design choice:
# binary classification? -> output a score for persuasiveness of each
# classify which one is more persuasive?


class StyleClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=3):
        super(StyleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(ids, attention_mask=mask,
                            token_type_ids=token_type_ids)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        # output = self.softmax(output)
        return output

class StyleClassifierScore(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=3):
        super(StyleClassifierScore, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(ids, attention_mask=mask,
                            token_type_ids=token_type_ids)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        output = self.sigmoid(output.flatten())
        # output = self.softmax(output)
        return output

class SiameseStyleClassifier(nn.Module):
    def __init__(self, embedding_model='bert-base-uncased', hidden_dim=None, n_classes=3, device=None):
        super(SiameseStyleClassifier, self).__init__()

        if type(embedding_model) == str:
            embedding_model = models.Transformer(embedding_model)
        else:
            assert type(embedding_model) == models.Transformer
            embedding_model = embedding_model

        hidden_dim = embedding_model.get_word_embedding_dimension()
        pooling_model = models.Pooling(hidden_dim, pooling_mode='mean',
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        if not device:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = SentenceTransformer(
            modules=[embedding_model, pooling_model], device=device)

        self.linear = nn.Linear(3*hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.n_classes = n_classes

    def forward(self, sentences):
        assert len(sentences) == 2
        pairs = [[sentences[0][i], sentences[1][i]]
                 for i in range(len(sentences[0]))]

        outputs = []
        for pair in pairs:
            embeddings = torch.tensor(self.model.encode(
                pair, show_progress_bar=False), device=self.device)
            diff = torch.abs(torch.sub(embeddings[1], embeddings[0]))
            output = torch.cat([embeddings[0], embeddings[1], diff], dim=0)
            outputs.append(output)
        outputs = torch.stack(outputs)

        outputs = self.linear(outputs)
        if self.n_classes == 1:
            outputs = self.sigmoid(outputs.flatten())
        else:
            outputs = self.softmax(outputs)
        return outputs
