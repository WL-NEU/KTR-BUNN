import argparse
import os
import numpy as np
from Utils.logger import setlogger

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function
import scipy.io as sio
from Backbone import ResNet1D, MLPNet, CNN1D
from PreparData.CWRU import CWRUloader
import Utils.utils as utils

from tqdm import *
import warnings
import logging



# ===== Build Model =====
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output, None

def grad_reverse(x):
    return GradReverse.apply(x)

# define the model
class FeatureNet(nn.Module):
    def __init__(self, args):
        super(FeatureNet, self).__init__()
        if args.backbone == "ResNet1D":
            self.feature_net = ResNet1D.resnet18()
        elif args.backbone == "ResNet2D":
            self.model_ft = models.resnet18(pretrained=True)
            self.bottleneck = nn.Sequential(nn.Linear(self.model_ft.fc.out_features, 512), nn.ReLU(), nn.Dropout(0.2))
            self.feature_net = nn.Sequential(self.model_ft, self.bottleneck)
        elif args.backbone == "MLPNet":
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

    def forward(self, logits, reverse = False):
        if reverse:
            logits = grad_reverse(logits)
        outputs = self.classifier(logits)

        return outputs

# ===== Load Data =====
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

# ===== Define Loss Function =====
def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)

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
    source_trainloader, target_trainloader, target_valloader, target_testloader = loaddata(args)

    num_out = len(args.s_label_set)+1

    # load the model
    featurenet = FeatureNet(args)
    classifier = Classifier(args, num_out=num_out)

    # load the checkpoint
    if args.pretrained:
        if args.backbone != "ResNet2D": # pretrained ResNet2D model is downloaded from torchvision module
            if not args.fft:
                path = "./checkpoints/{}_checkpoint.tar".format(args.backbone)
            else:
                path = "./checkpoints/{}FFT_checkpoint.tar".format(args.backbone)
            featurenet.load_state_dict(torch.load(path))

    parameter_list = [{"params": featurenet.parameters(), "lr": args.lr},
                       {"params": classifier.parameters(), "lr": args.lr}]

    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    ## define loss function
    criterion = nn.CrossEntropyLoss()

    featurenet.to(device)
    classifier.to(device)

    # train
    best_acc = 0.0
    meters = {"acc_source_train":[], "acc_target_train": [], "acc_target_val": []}

    for epoch in range(args.max_epoch):
        featurenet.train()
        classifier.train()
        with tqdm(total=len(target_trainloader), leave=False) as pbar:
            feats_train=[]
            labels_train=[]
            for i, ((x_s_batch, y_s_batch), (x_t_batch, y_t_batch)) in enumerate(zip(source_trainloader,target_trainloader)):

                if len(y_s_batch) != len(y_t_batch):
                    break
                batch_num = x_s_batch.size(0)

                target_funk = torch.FloatTensor(batch_num, 2).fill_(0.5).cuda()

                # clear previous gradients, compute gradients
                optimizer.zero_grad()

                # move to GPU if available
                x_s = x_s_batch.to(device)
                x_t = x_t_batch.to(device)
                y_s = y_s_batch.to(device)
                y_t = y_t_batch.to(device)

                # compute model output and loss
                # source data
                logits_s = featurenet(x_s)
                feats_train.append(logits_s)
                y_s_imed = y_s.cpu().numpy()
                labels_train.append(y_s_imed)
                outputs_s = classifier(logits_s)
                loss_s = criterion(outputs_s, y_s.long())
                loss_s.backward()

                # target data
                logits_t = featurenet(x_t)
                outputs_t = classifier(logits_t, reverse=True)
                outputs_t = F.softmax(outputs_t)
                prob1 = torch.sum(outputs_t[:, :num_out-1], 1).view(-1, 1)
                prob2 = outputs_t[:, num_out-1].contiguous().view(-1, 1)
                prob = torch.cat([prob1, prob2], 1)

                loss_t = bce_loss(prob, target_funk)
                loss_t.backward()

                # performs updates using calculated gradients
                optimizer.step()
                # clear previous gradients
                optimizer.zero_grad()            

                pbar.update()
            feats_train_array=torch.cat(feats_train, axis=0)
            feats_train_array=feats_train_array.cpu().detach().numpy()
            print(feats_train_array.shape)
            labels_train_array =np.concatenate(labels_train, axis=0)
            print(labels_train_array.shape)


            # update lr
            if lr_scheduler is not None:
                lr_scheduler.step()

            # validation
            featurenet.eval()
            classifier.eval()
            feats_val=[]
            labels_val=[]
            correct_num = 0
            val_num = 0
            per_class_num = np.zeros((num_out))
            per_class_correct = np.zeros((num_out)).astype(np.float32)
            # for step, (x_val_batch, y_val_batch) in enumerate(target_valloader):
            for step, (x_val_batch, y_val_batch) in enumerate(target_trainloader):
                # move to GPU if available
                x_val= x_val_batch.to(device)
                y_val = y_val_batch.to(device)
                batch_size_val = y_val.data.size()[0]

                logits_val = featurenet(x_val)
                outputs_val = classifier(logits_val)
                feats_val.append(logits_val)
                pre = torch.max(outputs_val.cpu(), 1)[1].numpy()
                y_val = y_val.cpu().numpy()
                labels_val.append(y_val)
                correct_num += (pre == y_val).sum() # the number of correct preditions per batch
                val_num += batch_size_val       # the number of predictions per batch

                for i in range(num_out):
                    if i==num_out-1:
                        index = np.where(y_val == i)
                        index+=(np.where(y_val == i+1))
                        index+=(np.where(y_val == i+2))
                        # index += (np.where(y_val == i + 3))
                        # index += (np.where(y_val == i + 4))
                        correct_ind = np.where(pre[index[0]] == i)+ np.where(pre[index[1]] == i)+ np.where(pre[index[2]] == i)
                        per_class_correct[i] += (float(len(correct_ind[0])))+float(len(correct_ind[1]))+float(len(correct_ind[2]))
                        per_class_num[i] += (float(len(index[0]))) +float(len(index[1]))+float(len(index[2]))
                    if i!=num_out-1:
                        index = np.where(y_val == i)
                        correct_ind = np.where(pre[index[0]]==i)
                        per_class_correct[i] += float(len(correct_ind[0]))
                        per_class_num[i] += float(len(index[0]))
            feats_val_array=torch.cat(feats_val, axis=0)
            feats_val_array=feats_val_array.cpu().detach().numpy()
            print(feats_val_array.shape)
            labels_val_array =np.concatenate(labels_val, axis=0)
            print(labels_val_array.shape)
            per_class_acc = (per_class_correct / per_class_num) * 100.0
            known_acc = (per_class_correct[:-1].sum() / per_class_num[:-1].sum()) * 100.0
            unknown_acc = (per_class_correct[-1].sum() / per_class_num[-1].sum()) * 100.0
            # print(per_class_correct[:-1].sum(),per_class_correct[-1].sum(),per_class_num[:-1].sum(),per_class_num[-1].sum())
            # print(correct_num)
            # print(val_num)
            # all_acc = (correct_num / val_num) * 100.0
            all_acc = (per_class_correct.sum() / val_num) * 100.0
            H_score=2*known_acc*unknown_acc/(known_acc+unknown_acc)
            if all_acc > best_acc:
                    best_acc = all_acc
                    data_to_save = {'feats_train_array': feats_train_array,'labels_train_array':labels_train_array,'feats_val_array': feats_val_array,'labels_val_array':labels_val_array}
                    sio.savemat('tsne.mat', data_to_save)
            logging.info("Epoch: {:>3}/{}, loss_s: {:.4f}, loss_t: {:.4f}, all_acc: {:>6.2f}, known_acc: {:>6.2f}, unknown_acc: {:>6.2f}, H_score: {:>6.2f}%".format(\
                    epoch+1, args.max_epoch, loss_s, loss_t, all_acc, known_acc,unknown_acc,H_score))
            #for i in range(num_out):
            #    logging.info("Label {}: {:>6.2f}%".format(i, per_class_acc[i]))

    logging.info("Best all accuracy: {:.4f}".format(best_acc))

    logging.info("="*10+"Done!"+"="*10)

