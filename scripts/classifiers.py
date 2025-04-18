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
        self.fc = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        B, T, L = input_ids.shape # [batch seq_length, d_model]
        input_ids = input_ids.view(-1, L)
        attention_mask = attention_mask.view(-1, L)

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_embeddings = cls_embeddings.view(B, T, -1)
        
        if self.pooling_type == 'max':
            # MIL pooling: max over all instances (versos)
            bag_logits = cls_embeddings.max(dim=1).values
        elif self.pooling_type == 'attention':
            bag_logits = self.pooling(cls_embeddings)
        
        logits = self.classifier(bag_logits)
        
        return logits
    

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, H):  # H: (B, T, H)
        # Compute attention weights
        attn_scores = self.attention(H)  # (B, T, 1)
        attn_weights = t.softmax(attn_scores, dim=1)  # (B, T, 1)
        weighted_sum = (H * attn_weights).sum(dim=1)  # (B, H)
        return weighted_sum