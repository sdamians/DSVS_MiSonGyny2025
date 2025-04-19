import torch as t
import torch.nn as nn
from transformers import AutoModel


class TextClassifier(nn.Module):
    """
    We use a custom small language model to manipulate the last dropout parameter
    """
    def __init__(self, modelpath, num_labels=2, get_att=False, get_hs=False, dropout=0.05, device='cpu'):
        super().__init__()
        self.llm = AutoModel.from_pretrained(modelpath
                                            ,num_labels=num_labels
                                            ,output_attentions=get_att
                                            ,output_hidden_states=get_hs).to(device)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.llm.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids
                            ,token_type_ids=None
                            ,attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Multiple instance learning (MIL)
class MILClassifier(nn.Module):
    def __init__(self, modelpath, num_labels=2, get_att=False, get_hs=False, dropout=0.05, pooling_type='max'):
        super().__init__()
        self.pooling_type = pooling_type

        self.llm = AutoModel.from_pretrained(modelpath
                                            ,num_labels=num_labels
                                            ,output_attentions=get_att
                                            ,output_hidden_states=get_hs)
        
        if self.pooling_type == 'attention':
            self.pooling = AttentionPooling(self.llm.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.llm.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, num_verses):
        B, V, S = input_ids.shape # [batch d_verse seq_len
        input_ids = input_ids.view(-1, S) # [batch*d_verse seq_len]
        attention_mask = attention_mask.view(-1, S) # [batch*d_verse seq_len]

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)        
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token = [ batch*d_verse 1 d_model ]
        cls_embeddings = cls_embeddings.view(B, V, -1) # [batch d_verse d_model]
        
        # MIL pooling: max over all instances (versos)
        if self.pooling_type == 'max':
            bag_logits = self.classifier(cls_embeddings) # [batch d_verse num_classes]

            songs = t.unbind(bag_logits) # [d_verse num_classes]
            
            logits = []
            for idx, verses in enumerate(songs):
                logits.append(verses[: num_verses[idx]].max(dim=-1).values) # [num_classes]

            logits = t.tensor(logits)

        elif self.pooling_type == 'attention':
            weighted_embeddings = self.pooling(cls_embeddings) # [ batch d_model ]
            logits = self.classifier(weighted_embeddings) # [batch num_classes]
        
        return logits
    

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, H, num_verses):  # H: (B, V, L)
        # Compute attention weights
        attn_scores = self.attention(H)  # (B, V, 1)
        attn_scores = attn_scores[:, :num_verses, :]
        attn_weights = t.softmax(attn_scores, dim=1)  # (B, V, 1)
        weighted_sum = (H[:, :num_verses, :] * attn_weights).sum(dim=1)  # (B, L)
        
        return weighted_sum