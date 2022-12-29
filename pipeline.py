import torch
import numpy as np
from torch import nn
from Config import config
from Model import DiscourseBert
from Dataset import DiscourseDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def epoch_task(model, dataloader, optimizer, criterion, epoch, mode='train'):
    if mode == 'train':
        model.train()
    elif model == 'test':
        model.eval()
    loss, y_true, y_pred = 0, np.array([]), np.array([])
    for batch in dataloader:
        if mode == 'train':
            optimizer.zero_grad()
        arg, label, conn_idx, total_labels, orders = None, None, None, None, None
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            label, conn_idx, orders, total_labels = batch[3].to(config.device), batch[5].to(config.device), batch[6].to(config.device), batch[4]
            arg = (batch[0].to(config.device), batch[1].to(config.device), batch[2].to(config.device))
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            label, conn_idx, orders, total_labels = batch[2].to(config.device), batch[4].to(config.device), batch[5].to(config.device), batch[3]
            arg = (batch[0].to(config.device), batch[1].to(config.device))
        outputs = model(arg)
        if config.order:
            loss = criterion(outputs[1], orders) if epoch < config.freeze_epoch else \
                criterion(outputs[0], label if config.label == 'relationship' else conn_idx)
        else:
            loss = criterion(outputs, label if config.label == 'relationship' else conn_idx)
        if mode == 'train':
            loss.backward()
            optimizer.step()
        loss += loss.item()
        if config.label == 'conn':
            final_out = torch.zeros((outputs.size()[0], 4)).to(config.device)
            for key, value in config.class2conn.items():
                for idx in value:
                    final_out[:, int(key)] += outputs[:, idx]
            outputs = final_out
        new_y_pred = torch.argmax(outputs[0] if config.order else outputs, dim=-1).cpu().numpy()
        y_true = np.append(y_true, np.array([pred if pred in total_labels[idx] else total_labels[idx][0]
                                             for idx, pred in enumerate(new_y_pred)]))
        y_pred = np.append(y_pred, new_y_pred)
    print(mode + "_epoch: {}, loss: {}, acc: {}, f1: {}".format(
        epoch, loss / len(dataloader), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))
    print(result_analyse(y_true, y_pred))

def pipline():
    train_dataset = DiscourseDataset(mode="train", max_length=config.max_length, use_explicit=False, use_order=False)
    order_dataset = DiscourseDataset(mode="train", max_length=config.max_length, use_explicit=False, use_order=True)
    test_dataset = DiscourseDataset(mode="test", max_length=config.max_length, use_explicit=False, use_order=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    order_dataloader = DataLoader(order_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    model = DiscourseBert()
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(weight=train_dataset.weight) if config.weight_loss else torch.nn.CrossEntropyLoss()
    criterion.to(config.device)
    for epoch in range(config.epoch):
        epoch_task(model, order_dataloader if config.order and epoch < config.freeze_epoch else train_dataloader,
                   optimizer, criterion, epoch, mode='train')
        epoch_task(model, test_dataloader, optimizer, criterion, epoch, mode='test')
    torch.save(model.state_dict(), "models/" + config.backbone + "_model.pt")

def result_analyse(y_true, y_pred):
    matric = np.zeros((4, 4))
    for idx in range(len(y_true)):
        matric[int(y_true[idx]-1)][int(y_pred[idx]-1)] += 1
    return matric


if __name__ == "__main__":
    pipline()
