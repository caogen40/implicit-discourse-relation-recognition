import torch
import torch.nn as nn
from Config import config
from transformers import AutoModel


class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.method = config.method
        self.backbone = AutoModel.from_pretrained(config.backbone)
        output_dim = None
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base":
            output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large":
            output_dim = 1024
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, config.total_conn_num if self.method == 'prompt' else 4),
        )
        if config.order:
            self.order_classifier = nn.Sequential(
                nn.Linear(output_dim, 256),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(256, 2),
            )

    def forward(self, arg):
        result = None
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            result = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            result = self.backbone(input_ids=arg[0], attention_mask=arg[1])
        if self.method == 'prompt':
            out = result.last_hidden_state[:, config.max_length, :].squeeze()
            out = self.classifier(out)
            if config.label == 'relationship':
                final_out = torch.zeros((out.size()[0], 4)).to(config.device)
                for key, value in config.class2conn.items():
                    for idx in value:
                        final_out[:, int(key)] += out[:, idx]
                out = final_out
        else:
            out = torch.mean(result.last_hidden_state, dim=1)
            out = self.classifier(out)
        if config.order:
            order_out = torch.mean(result.last_hidden_state, dim=1)
            order_out = self.order_classifier(order_out)
            out = out, order_out
        return out
