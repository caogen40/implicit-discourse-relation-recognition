import torch
import torch.nn as nn
from Config import config
from transformers import AutoModel


class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.backbone = AutoModel.from_pretrained(config.backbone)
        output_dim = None
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base":
            output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large":
            output_dim = 1024
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, config.total_conn_num),
        )

    def forward(self, arg):
        result = None
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            result = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            result = self.backbone(input_ids=arg[0], attention_mask=arg[1])
        out = result.last_hidden_state[:, config.max_length, :].squeeze()
        out = self.classifier(out)
        final_out = torch.zeros((out.size()[0], 4)).to(config.device)
        for key, value in config.class2conn.items():
            for idx in value:
                final_out[:, int(key)] += out[:, idx]
        return final_out
