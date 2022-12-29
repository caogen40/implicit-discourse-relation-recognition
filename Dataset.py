import json
import random
import torch
from Config import config
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight


class DiscourseDataset(Dataset):
    def __init__(self, mode="train", max_length=30, use_explicit=False, use_order=False):
        assert mode in ["train", "dev", "test"], "mode must be train, dev or test"
        self.mode, self.method = mode, config.method
        self.data_path = "dataset/implicit_" + mode + ".json"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.data = []
        self.data = self.load_data(self.data, self.data_path, use_order)
        if use_explicit:
            self.data = self.load_data(self.data, "dataset/explicit.json", use_order)
        if config.weight_loss:
            self.weight = torch.tensor(compute_class_weight('balanced', classes=[0, 1, 2, 3],
                                                            y=[example[2] for example in self.data]), dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        order, arg1, arg2, train_label, conn, total_label = self.data[index]
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            arg_input_ids = torch.cat((arg1["input_ids"], torch.tensor([0]), arg2["input_ids"]), dim=0) \
                if self.method == 'prompt' else torch.cat((arg1["input_ids"], arg2["input_ids"]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]), arg2["attention_mask"][0]),
                                           dim=0) \
                if self.method == 'prompt' else torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            arg_token_type_ids = torch.cat((arg1["token_type_ids"][0], torch.tensor([0]), arg2["token_type_ids"][0]),
                                           dim=0) \
                if self.method == 'prompt' else torch.cat((arg1["token_type_ids"][0], arg2["token_type_ids"][0]), dim=0)
            return arg_input_ids, arg_token_type_ids, arg_attention_mask, train_label, total_label, config.conn2idx[
                conn], order

        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            arg_input_ids = torch.cat((arg1["input_ids"][0], torch.tensor([0]), arg2["input_ids"][0]), dim=0) \
                if self.method == 'prompt' else torch.cat((arg1["input_ids"][0], arg2["input_ids"][0]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]), arg2["attention_mask"][0]),
                                           dim=0) \
                if self.method == 'prompt' else torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            return arg_input_ids, arg_attention_mask, train_label, total_label, config.conn2idx[conn], order

    def load_data(self, data_list, path, use_order):
        with open(path, "r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                order = random.choice([0, 1]) if use_order else 0
                arg1 = self.tokenizer(item["arg1" if order == 0 else "arg2"], padding="max_length", truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
                arg2 = self.tokenizer(item["arg2" if order == 0 else "arg1"], padding="max_length", truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
                conn = item['conn']
                train_label = int(item["label"][0])
                total_label = torch.tensor([int(label) for label in item["label"]]) if len(item["label"]) > 1 \
                    else torch.tensor([int(item["label"][0])] * 2)
                data_list.append((order, arg1, arg2, train_label, conn, total_label))
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
