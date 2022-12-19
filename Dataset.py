import json
import torch
import numpy as np
from Config import config
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DiscourseDataset(Dataset):
    def __init__(self, mode="train", max_length=30):
        assert mode in ["train", "dev", "test"], "mode must be train, dev or test"
        self.mode = mode
        self.data_path = "dataset/implicit_" + mode + ".json"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            arg1, arg2, label, conn = self.data[index]
            arg_input_ids = torch.cat((arg1["input_ids"], torch.tensor([0]), arg2["input_ids"]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]), arg2["attention_mask"][0]),
                                           dim=0)
            arg_token_type_ids = torch.cat((arg1["token_type_ids"][0], torch.tensor([0]), arg2["token_type_ids"][0]),
                                           dim=0)
            return arg_input_ids, arg_token_type_ids, arg_attention_mask, label

        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            arg1, arg2, label, conn = self.data[index]
            arg_input_ids = torch.cat((arg1["input_ids"][0], torch.tensor([0]), arg2["input_ids"][0]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], torch.tensor([1]), arg2["attention_mask"][0]),
                                           dim=0)
            return arg_input_ids, arg_attention_mask, label

    def load_data(self):
        data = []
        with open(self.data_path, "r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                arg1 = self.tokenizer(item["arg1"], padding="max_length", truncation=True, max_length=self.max_length,
                                      return_tensors="pt")
                arg2 = self.tokenizer(item["arg2"], padding="max_length", truncation=True, max_length=self.max_length,
                                      return_tensors="pt")
                label = int(item["label"][0])
                conn = item['conn']
                data.append((arg1, arg2, label, conn))
        return data


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


def connCmpAnalysis():
    def connAnalysis(mode='test'):
        frequence_dict = dict()
        with open(f"./dataset/implicit_{mode}.json", "r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                conn, label = item['conn'], item['label'][0]
                if conn not in frequence_dict.keys():
                    frequence_dict[conn] = [0, 0, 0, 0]
                frequence_dict[conn][label] += 1
        return frequence_dict

    train_frequence = connAnalysis('train')
    test_frequence = connAnalysis('test')
    print("There are", len(train_frequence.keys()), "conn in train_data")
    print("There are", len(test_frequence.keys()), "conn in test_data")
    print("These conns is in train_data, but not in test data: ")
    for key in test_frequence.keys():
        if key not in train_frequence.keys():
            print(key, test_frequence[key])
    sure_wrong, total_num = 0, 0
    for key, value in test_frequence.items():
        if value.count(0) < 3:
            max_dix = np.argmax(value)
            sure_wrong += sum(value) - value[max_dix]
        total_num += sum(value)
    print("num of datas:", total_num, "datas tend to be predicted wrongly:", sure_wrong)
    total_coon = set(list(train_frequence.keys()) + list(test_frequence.keys()))
    conn2idx, conn2class, class2conn = dict(), dict(), {'0': [], '1': [], '2': [], '3': []}
    for idx, conn in enumerate(list(total_coon)):
        conn2idx[conn] = idx
    for key in conn2idx.keys():
        if key in train_frequence.keys():
            conn2class[key] = np.argmax(train_frequence[key])
        else:
            conn2class[key] = np.argmax(test_frequence[key])
    for key, value in conn2class.items():
        class2conn[str(value)].append(conn2idx[key])
    return conn2idx, conn2class, class2conn

