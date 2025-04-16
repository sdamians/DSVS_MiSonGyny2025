import torch.nn as nn
from transformers import AutoModel


class TextClassifier(nn.Module):
    def __init__(self, modelpath, num_labels=2, get_att=False, get_hs=False, dropout=0.05, device='cpu'):
        super(TextClassifier, self).__init__()
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

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

