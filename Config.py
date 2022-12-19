import json
import torch
import numpy as np


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


class config:
    max_length = 64
    batch_size = 32
    device = torch.device("cuda")
    epoch = 20
    lr = 2e-5
    total_conn_num = 97
    """bert-base-uncased, bert-large-uncased, roberta-base, roberta-large"""
    backbone = "roberta-base"
    conn2idx, conn2class, class2conn = connCmpAnalysis()


if __name__ == "__main__":
    print(connCmpAnalysis())
