import torch
import numpy as np
from Config import config
from Model import DiscourseBert
from Dataset import DiscourseDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def epoch_task(epoch, dataloader, model, optimizer, criterion, model2, optimizer2, criterion2, mode='train'):
    if mode == 'train':
        model.train()
        if config.ensemble:
            model2.train()
    elif model == 'test':
        model.eval()
        if config.ensemble:
            model2.eval()
    loss, y_true, y_pred = 0, np.array([]), np.array([])
    if config.ensemble:
        batch0, batch1, ens_label, count, total_count = torch.zeros((config.batch_size, 131), dtype=torch.int64).to(config.device), torch.zeros((config.batch_size, 131), dtype=torch.int64).to(config.device), torch.zeros(config.batch_size, dtype=torch.int64).to(config.device), 0, 0
    for batch in dataloader:
        if mode == 'train':
            optimizer.zero_grad()
        arg, label, conn_idx, total_labels = None, None, None, None
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            label, conn_idx, total_labels = batch[3].to(config.device), batch[5].to(config.device), batch[4]
            arg = (batch[0].to(config.device), batch[1].to(config.device), batch[2].to(config.device))
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            label, conn_idx, total_labels = batch[2].to(config.device), batch[4].to(config.device), batch[3]
            arg = (batch[0].to(config.device), batch[1].to(config.device))
        outputs = model(arg)
        new_y_pred = torch.argmax(outputs[0] if config.order else outputs, dim=-1).cpu().numpy()
        if config.label == 'conn':
            new_y_pred = torch.tensor([config.conn2class[config.idx2conn[i]] for i in new_y_pred])
        new_y_true = np.array([pred if pred in total_labels[idx] else total_labels[idx][0] for idx, pred in enumerate(new_y_pred)])
        loss = criterion(outputs, label if config.label == 'relationship' else conn_idx)
        if config.ensemble:
            if mode == 'train':
                for idx in range(len(new_y_true)):
                    if new_y_true[idx] != new_y_pred[idx]:
                        batch0[count, :], batch1[count, :], ens_label[count] = batch[0][idx], batch[1][idx], label[idx]
                        total_count += 1
                        count += 1
                        if count == config.batch_size:
                            optimizer2.zero_grad()
                            outputs2 = model2((batch0, batch1))
                            loss2 = criterion2(outputs2, ens_label)
                            loss2.backward()
                            optimizer2.step()
                            batch0, batch1, ens_label, count = torch.zeros((config.batch_size, 131), dtype=torch.int64).to(config.device), torch.zeros((config.batch_size, 131), dtype=torch.int64).to(config.device), torch.zeros(config.batch_size, dtype=torch.int64).to(config.device), 0
            outputs_child = model2(arg)
            new_y_pred = torch.argmax(outputs + outputs_child, dim=-1).cpu().numpy()
        if mode == 'train':
            loss.backward()
            optimizer.step()
        loss += loss.item()
        y_true = np.append(y_true, new_y_true)
        y_pred = np.append(y_pred, new_y_pred)
    if config.ensemble and mode == 'train':
        print("There are " + str(total_count) + " datas trained twice.")
    print(mode + "_epoch: {}, loss: {}, acc: {}, f1: {}".format(
        epoch, loss / len(dataloader), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))
    print(result_analyse(y_true, y_pred))


def pipline():
    train_dataset = DiscourseDataset(mode="train", max_length=config.max_length, use_explicit=False)
    test_dataset = DiscourseDataset(mode="test", max_length=config.max_length, use_explicit=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    model = DiscourseBert(ensemble=True if config.ensemble else False).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss().to(config.device)
    model2 = DiscourseBert(ensemble=False).to(config.device) if config.ensemble else None
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=config.lr) if config.ensemble else None
    criterion2 = torch.nn.CrossEntropyLoss().to(config.device) if config.ensemble else None
    for epoch in range(config.epoch):
        epoch_task(epoch, train_dataloader, model, optimizer, criterion, model2, optimizer2, criterion2, mode='train')
        epoch_task(epoch, test_dataloader, model, optimizer, criterion, model2, optimizer2, criterion2, mode='test')
    torch.save(model.state_dict(), "models/" + config.backbone + "_model.pt")


def result_analyse(y_true, y_pred):
    matric = np.zeros((4, 4))
    for idx in range(len(y_true)):
        matric[int(y_true[idx])][int(y_pred[idx])] += 1
    return matric


if __name__ == "__main__":
    pipline()
