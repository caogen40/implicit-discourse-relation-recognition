import torch
import torch.nn as nn
from Config import config
from transformers import AutoModel


class DiscourseBert(nn.Module):
    def __init__(self, ensemble=False):
        super(DiscourseBert, self).__init__()
        self.ensemble = ensemble
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.output_dim = None
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base":
            self.output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large":
            self.output_dim = 1024
        self.classifier = nn.Sequential(nn.Linear(self.output_dim, 256), nn.Dropout(), nn.ReLU(), nn.Linear(256, 4)) if config.method == 'common' else nn.Linear(int(self.output_dim / 3), config.total_conn_num)
        if self.ensemble:
            self.ensemble_classifier = nn.Linear(int(self.output_dim / 3), 2)
        if config.order:
            self.backbone2 = AutoModel.from_pretrained(config.backbone)
            self.classifier2 = nn.Sequential(nn.Linear(self.output_dim, 256), nn.Dropout(), nn.ReLU(), nn.Linear(256, 4)) if config.method == 'common' else nn.Linear(int(self.output_dim / 3), config.total_conn_num)
            self.final_classifier = nn.Linear(2 * config.total_conn_num if config.method == 'prompt' and config.label == 'label' else 8, 4)

    def forward(self, arg):
        result = None
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            result = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            result = self.backbone(input_ids=arg[0], attention_mask=arg[1])
        if config.method == 'prompt':
            out = result.last_hidden_state[:, config.max_length:config.max_length + 3, :]
            out = nn.AvgPool2d(kernel_size=(3, 3))(out).squeeze(1)
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
            result2 = self.backbone2(input_ids=torch.cat((arg[0][config.max_length + 3:], arg[0][config.max_length:config.max_length + 3], arg[0][:config.max_length]), dim=0), attention_mask=torch.cat((arg[1][config.max_length + 3:], arg[1][config.max_length:config.max_length + 3], arg[1][:config.max_length]), dim=0))
            if config.method == 'prompt':
                out2 = result2.last_hidden_state[:, config.max_length:config.max_length + 3, :]
                out2 = nn.AvgPool2d(kernel_size=(3, 3))(out2).squeeze(1)
                out2 = self.classifier2(out2)
                if config.label == 'relationship':
                    final_out2 = torch.zeros((out2.size()[0], 4)).to(config.device)
                    for key, value in config.class2conn.items():
                        for idx in value:
                            final_out2[:, int(key)] += out2[:, idx]
                    out2 = final_out2
            else:
                out2 = torch.mean(result2.last_hidden_state, dim=1)
                out2 = self.classifier2(out2)
            out = self.final_classifier(torch.cat((out, out2), dim=1))
        return out
