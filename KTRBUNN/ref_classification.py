import argparse
import os
import numpy as np
from Utils.logger import setlogger

from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from Backbone import ResNet1D, MLPNet, CNN1D
from PreparData.CWRU import CWRUloader
import Utils.utils as utils

from tqdm import *
import warnings
import logging



# ===== Build Model =====
class FeatureNet(nn.Module):
    def __init__(self, args):
        super(FeatureNet, self).__init__()
        if args.backbone == "ResNet1D":
            self.feature_net = ResNet1D.resnet18()
        elif args.backbone == "ResNet2D":
            self.model_ft = models.resnet18(pretrained=True)
            self.bottleneck = nn.Sequential(nn.Linear(self.model_ft.fc.out_features, 512), nn.ReLU(), nn.Dropout(0.5))
            self.feature_net = nn.Sequential(self.model_ft, self.bottleneck)
        elif args.backbone == "MLPNet":
            if args.fft:
                self.feature_net = MLPNet.MLPNet(num_in=512)
            else:
                self.feature_net = MLPNet.MLPNet()
        elif args.backbone == "CNN1D":
            self.feature_net = CNN1D.CNN1D()
        else:
            raise Exception("model not implement")

    def forward(self, x):
        logits = self.feature_net(x)

        return logits

class Classifier(nn.Module):
    def __init__(self, args, num_out=10):
        super(Classifier, self).__init__()
        if args.backbone in ("ResNet1D", "ResNet2D"):
            self.classifier = nn.Sequential(nn.Linear(512,num_out, nn.Dropout(0.5)))
        if args.backbone in ("MLPNet", "CNN1D"):
            self.classifier = nn.Sequential(nn.Linear(64,num_out, nn.Dropout(0.5)))

    def forward(self, logits):
        outputs = self.classifier(logits)

        return outputs

# ===== Load Data =====
# def loaddata(args):
#     data, label = CWRUloader(args, args.load, args.label_set)
#     data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
#
#     train_loader, val_loader, test_laoder = utils.DataSplite(args, data, label)
#
#     return train_loader, val_loader, test_laoder
def loaddata(args):
    if args.source_dataname == "CWRU":
        source_data, source_label = CWRUloader(args, args.s_load, args.s_label_set)

    source_data, source_label = np.concatenate(source_data, axis=0), np.concatenate(source_label, axis=0)

    if args.target_dataname == "CWRU":
        target_data, target_label = CWRUloader(args, args.t_load, args.t_label_set)

    target_data, target_label = np.concatenate(target_data, axis=0), np.concatenate(target_label, axis=0)

    source_loader, _, _ = utils.DataSplite(args, source_data, source_label)
    target_trainloader, target_valloader, target_testloader = utils.DataSplite(args, target_data, target_label)

    return source_loader, target_trainloader, target_valloader, target_testloader
# ===== Test the Model =====
def ptester(featurenet, classifier, dataloader):
    featurenet.eval()
    classifier.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    correct_num, total_num = 0, 0
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # compute model cotput and loss
        logtis_batch = featurenet(x_batch)
        output_batch = classifier(logtis_batch)

        pre = torch.max(output_batch.cpu(), 1)[1].numpy()
        y = y_batch.cpu().numpy()
        correct_num += (pre == y).sum()
        total_num += len(y)
    accuracy = (correct_num / total_num) * 100.0
    return accuracy


# ===== Train the Model =====
def trainer(args):
    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert args.batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))
    
    # load the dataset
    # trainloader, valloader, testloader = loaddata(args)
    trainloader, _, valloader,testloader = loaddata(args)

    # load the model
    featurenet = FeatureNet(args)
    classifier = Classifier(args, num_out=len(args.s_label_set))

    parameter_list = [{"params": featurenet.parameters(), "lr": args.lr},
                       {"params": classifier.parameters(), "lr": args.lr}]

    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()
    featurenet.to(device)
    classifier.to(device)
    # train
    best_acc = 0.0
    meters = {"acc_train": [], "acc_val": []}

    for epoch in range(args.max_epoch):
        featurenet.train()
        classifier.train()
        with tqdm(total=len(trainloader), leave=False) as pbar:
            for i, (x_batch, y_batch) in enumerate(trainloader):
                # move to GPU if available
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # compute model cotput and loss
                logtis_batch = featurenet(x_batch)
                output_batch = classifier(logtis_batch)
                loss = loss_fn(output_batch, y_batch.long())

                # clear previous gradients, compute gradients
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # evaluate
                # training accuracy
                train_acc = utils.accuracy(output_batch, y_batch)

                pbar.update()
        
        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_acc = ptester(featurenet, classifier, valloader)
        if val_acc > best_acc:
            best_acc = val_acc
            if args.savemodel:
                utils.save_model(featurenet, args)
        
        logging.info("Epoch: {:>3}/{}, loss: {:.4f}, train_acc: {:>6.2f}%, val_acc: {:>6.2f}%".format(\
                epoch+1, args.max_epoch, loss, train_acc, val_acc))
        meters["acc_train"].append(train_acc)
        meters["acc_val"].append(val_acc)

    logging.info("Best accuracy: {:.4f}%".format(best_acc))
    utils.save_log(meters, "./logs/cls_{}_{}_meters.pkl".format(args.backbone, args.max_epoch))

    logging.info("="*15+"Done!"+"="*15)

