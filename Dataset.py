import json
import random
import torch
from Config import config
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DiscourseDataset(Dataset):
    def __init__(self, mode="train", max_length=30, use_explicit=False):
        assert mode in ["train", "dev", "test"], "mode must be train, dev or test"
        self.mode, self.method = mode, config.method
        self.data_path = "dataset/implicit_" + mode + ".json"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.data = []
        self.data = self.load_data(self.data, self.data_path)
        if use_explicit:
            self.data = self.load_data(self.data, "dataset/explicit.json")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arg1, arg2, train_label, conn, total_label = self.data[index]
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            arg_input_ids = torch.cat((arg1["input_ids"], torch.tensor([0]*3), arg2["input_ids"]), dim=0) if self.method == 'prompt' else torch.cat((arg1["input_ids"], arg2["input_ids"]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]*3), arg2["attention_mask"][0]), dim=0) if self.method == 'prompt' else torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            arg_token_type_ids = torch.cat((arg1["token_type_ids"][0], torch.tensor([0]*3), arg2["token_type_ids"][0]), dim=0) if self.method == 'prompt' else torch.cat((arg1["token_type_ids"][0], arg2["token_type_ids"][0]), dim=0)
            return arg_input_ids, arg_token_type_ids, arg_attention_mask, train_label, total_label, config.conn2idx[conn]

        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            arg_input_ids = torch.cat((arg1["input_ids"][0], torch.tensor([0]*3), arg2["input_ids"][0]), dim=0) if self.method == 'prompt' else torch.cat((arg1["input_ids"][0], arg2["input_ids"][0]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]*3), arg2["attention_mask"][0]), dim=0) if self.method == 'prompt' else torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            return arg_input_ids, arg_attention_mask, train_label, total_label, config.conn2idx[conn]

    def load_data(self, data_list, path):
        with open(path, "r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                conn = item['conn']
                if conn not in config.conn2idx.keys() or (self.mode == 'train' and path == "dataset/explicit.json" and int(item["label"][0]) == 2):
                    continue
                arg1 = self.tokenizer(item["arg1"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                arg2 = self.tokenizer(item["arg2"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                train_label = int(item["label"][0])
                total_label = torch.tensor([int(label) for label in item["label"]]) if len(item["label"]) > 1 \
                    else torch.tensor([int(item["label"][0])] * 2)
                data_list.append((arg1, arg2, train_label, conn, total_label))
        return data_list


def argAnalysis(mode="test"):
    import matplotlib.pyplot as plt
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    length_ls = []
    with open(f"./dataset/implicit_{mode}.json", "r") as f:
        jsonFile = json.load(f)
        for item in jsonFile:
            arg1 = tokenizer(item["arg1"])
            arg2 = tokenizer(item["arg2"])
            length_ls.append(len(arg1["input_ids"]))
            length_ls.append(len(arg2["input_ids"]))
    plt.hist(length_ls, bins=100)
    plt.savefig("dataset/{mode}_length.png")
    print("max length:", max(length_ls))
    print("min length:", min(length_ls))
    print("mean length:", sum(length_ls) / len(length_ls))

argAnalysis(mode="test")
