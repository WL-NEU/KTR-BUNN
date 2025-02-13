from data import *
from utilities import *
from networks import *
# import matplotlib.pyplot as plt
import numpy as np

def skip(data, label, is_train):
    return False
batch_size = 16 

import logging
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset,DataLoader
import os
import scipy.io as scio
import PIL.Image as Image
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from Backbone import ResNet1D, MLPNet, CNN1D
from PreparData.CWRU import CWRUloader
import Utils.utils as utils

from tqdm import *
import warnings
import logging
import argparse
# domain_train = sys.argv[1]
# domain_test = sys.argv[2]
# setGPU(sys.argv[3])
# store_name = sys.argv[4]
# loss_MSE_value = float(sys.argv[5])
# three_domain_loss_value = float(sys.argv[6])
setGPU("0")
store_name = "A_C"
loss_MSE_value = float(0.2)
# three_domain_loss_value = float(0.2)


# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Deep Domain Confusion networks')

    # task setting
    parser.add_argument("--log_file", type=str, default="./logs/OSDABP.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="./datasets", help="data directory")
    parser.add_argument("--source_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--target_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--s_load", type=int, default=3, help="source domain working condition")
    parser.add_argument("--t_load", type=int, default=2, help="target domain working condition")
    # parser.add_argument("--s_label_set", type=list, default=[0,1,2,3,4,5], help="source domain label set")
    parser.add_argument("--s_label_set", type=list, default=[0,1, 2, 3, 4, 5,6], help="source domain label set")
    parser.add_argument("--t_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="target domain label set")
    parser.add_argument("--val_rat", type=float, default=0.3, help="training-validation rate")
    parser.add_argument("--test_rat", type=float, default=0.5, help="validation-test rate")
    parser.add_argument("--seed", type=int, default="29")

    # pre-processing
    parser.add_argument("--fft", type=bool, default=1, help="FFT preprocessing")
    parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["None", "0-1", "mean-std"], help="normalization option")
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")

    # backbone
    parser.add_argument("--backbone", type=str, default="ResNet2D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)


    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


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


args = parse_args()

Scouce_train_loader, Target_train_loader, Target_test_loader, target_testloader = loaddata(args)

class MyDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder,label_duiying,transform=None):
        label = label_duiying
        self.dictlabel={}
        for i in range(0,label.shape[0]):
            # self.dictlabel.setdefault(label[i][0][0][0][0],label[i][1][0][0]) yuan lai
            name=str(label[i][0])+'.jpg'
            self.dictlabel.setdefault(name, label[i][1])
        # label=one_hot(10,label[:,1])
        self.folder = folder
        self.transform = transform
        self.label = label
        self.axial = os.listdir(self.folder)


    def __getitem__(self, index):
        axial_index = self.axial[index]
        label = self.dictlabel[axial_index]
        axial_path = os.path.join(self.folder,axial_index)
        axial = Image.open(axial_path)


        if self.transform is not None:
            # axial = self.transform(axial)
            axial,label = self.transform(axial,label)
        return axial,label


    def __len__(self):
        return len(self.axial)

myTransforms = transforms.Compose([transforms.Resize(size=(32,32)),
    transforms.ToTensor()
])

def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


def transform(data, label):
    label = one_hot(10, label)
    resize=transforms.Resize(size=(32,32))
    totensor=transforms.ToTensor()
    data=resize(data)
    data=totensor(data)
    return data, label

def source_transform(data, label):
    label = one_hot(8, label)
    resize=transforms.Resize(size=(32,32))
    totensor=transforms.ToTensor()
    data=resize(data)
    data=totensor(data)
    return data, label

discriminator_p = Discriminator(n = 7).cuda()  # 10 binary classifier
discriminator = LargeAdversarialNetwork(256).cuda()

feature_extractor_fix = ResNetFc(model_name='resnet50',model_path='/home/username/data/pytorchModels/resnet50.pth')
feature_extractor_nofix = ResNetFc(model_name='resnet50',model_path='/home/username/data/pytorchModels/resnet50.pth')

cls_upper = CLS(feature_extractor_fix.output_num(), 8, bottle_neck_dim=256)
cls_down = CLS(feature_extractor_nofix.output_num(), 8, bottle_neck_dim=256)

net_upper = nn.Sequential(feature_extractor_fix, cls_upper).cuda()
net_down = nn.Sequential(feature_extractor_nofix, cls_down).cuda()


three_domain_discriminator = Discriminator2(n = 3).cuda() 



scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)

optimizer_discriminator = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor_fix = OptimWithSheduler(optim.SGD(feature_extractor_fix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_feature_extractor_nofix = OptimWithSheduler(optim.SGD(feature_extractor_nofix.parameters(), lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls_upper = OptimWithSheduler(optim.SGD(cls_upper.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_cls_down = OptimWithSheduler(optim.SGD(cls_down.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_discriminator_p = OptimWithSheduler(optim.SGD(discriminator_p.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
optimizer_three_domain_discriminator = OptimWithSheduler(optim.SGD(three_domain_discriminator.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True),
                            scheduler)
KL = nn.KLDivLoss()
MSE = nn.MSELoss()

# =========================weighted adaptation of the source and target domains      
best_acc = 0.0
k=0
while k <150:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(Scouce_train_loader, Target_train_loader)):
        
        im_source = Variable(im_source).cuda()
        label_source = Variable(label_source).cuda()
        im_target = Variable(im_target).cuda()
         
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        label_source_onethot = F.one_hot(label_source.long(), num_classes=len(args.s_label_set) + 1)  # n为类别数
        d1_upper = BCELossForMultiClassification(label_source_onethot[:,0:7],p0_upper)

        ce_upper = CrossEntropyLoss(label_source_onethot, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        domain_prob_discriminator_1_source_list = []
        domain_prob_discriminator_1_target_list = []
        for _ in range(3):
            domain_prob_discriminator_1_source = discriminator.forward(feature_source)
            domain_prob_discriminator_1_target= discriminator.forward(feature_target)
            domain_prob_discriminator_1_source_list.append(domain_prob_discriminator_1_source)
            domain_prob_discriminator_1_target_list.append(domain_prob_discriminator_1_target)
        domain_prob_discriminator_1_source_tensor = torch.cat((domain_prob_discriminator_1_source_list),dim=1)
        domain_prob_discriminator_1_target_tensor = torch.cat((domain_prob_discriminator_1_target_list),dim=1)


        var_source = torch.var(domain_prob_discriminator_1_source_tensor, dim=1, keepdim=True)  #  (batch, 1)
        var_target = torch.var(domain_prob_discriminator_1_target_tensor, dim=1, keepdim=True)

        # var_source=discriminator.forward_with_variance(feature_source)
        # var_target=discriminator.forward_with_variance(feature_target)
        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)




        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_konwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_konwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        se_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)



        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,7)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source_onethot, predict_prob_source_down)
        u_loss=torch.sum(var_source+var_target)





        loss_MSE_upper =  MSE(p0_upper,p0_down.detach())
        loss_MSE_upper += MSE(p1_upper,p1_down.detach())



        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_upper, optimizer_discriminator_p,optimizer_feature_extractor_fix]):
            loss = loss_MSE_value * loss_MSE_upper + d1_upper + ce_upper +ce_ep*3+u_loss
            loss.backward()


#---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, predict_prob_source_upper = net_upper.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_upper.forward(im_target)


        p0_upper = discriminator_p.forward(fs1)
        p1_upper = discriminator_p.forward(ft1)
        
        p2_upper = torch.sum(p1_upper, dim = -1).detach()
        p3_upper = torch.sum(p0_upper, dim = -1).detach()

        d1_upper = BCELossForMultiClassification(label_source_onethot[:,0:7],p0_upper)

        ce_upper = CrossEntropyLoss(label_source_onethot, predict_prob_source_upper)



        fs1, feature_source, __, predict_prob_source_down = net_down.forward(im_source)
        ft1, feature_target, __, predict_prob_target = net_down.forward(im_target)

        # domain_prob_discriminator_1_source = discriminator.forward(feature_source)
        # domain_prob_discriminator_1_target = discriminator.forward(feature_target)
        domain_prob_discriminator_1_source_list = []
        domain_prob_discriminator_1_target_list = []
        for _ in range(3):
            domain_prob_discriminator_1_source = discriminator.forward(feature_source)
            domain_prob_discriminator_1_target = discriminator.forward(feature_target)
            domain_prob_discriminator_1_source_list.append(domain_prob_discriminator_1_source)
            domain_prob_discriminator_1_target_list.append(domain_prob_discriminator_1_target)
        domain_prob_discriminator_1_source_tensor = torch.cat((domain_prob_discriminator_1_source_list), dim=1)
        domain_prob_discriminator_1_target_tensor = torch.cat((domain_prob_discriminator_1_target_list), dim=1)

        var_source = torch.var(domain_prob_discriminator_1_source_tensor, dim=1, keepdim=True)  # (batch, 1)
        var_target = torch.var(domain_prob_discriminator_1_target_tensor, dim=1, keepdim=True)

        p0_down = discriminator_p.forward(fs1)
        p1_down = discriminator_p.forward(ft1)
        var_source=discriminator.forward_with_variance(feature_source)
        var_target=discriminator.forward_with_variance(feature_target)



        r_unk = torch.sort(p2_upper,dim = 0)[1][:2]  # 001
        feature_otherep = torch.index_select(ft1, 0, r_unk.view(2)) 
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)


        r = torch.sort(p2_upper,dim = 0)[1][-2:]  # 010
        feature_otherep = torch.index_select(ft1, 0, r.view(2)) 
        _, feature_target_konwn, _, _ = cls_down.forward(feature_otherep)

        r = torch.sort(p3_upper,dim = 0)[1][-2:]  # 100
        feature_otherep = torch.index_select(fs1, 0, r.view(2)) 
        _, feature_source_konwn, _, _ = cls_down.forward(feature_otherep)  



  
        feature_target_unkonwn_labels = Variable(torch.from_numpy(np.concatenate((np.zeros((2,2)), np.ones((2,1))), axis = -1).astype('float32'))).cuda() #001
        feature_target_konwn_labels   = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110
        feature_source_konwn_labels = Variable(torch.from_numpy(np.concatenate((np.ones((2,2)),  np.zeros((2,1))), axis = -1).astype('float32'))).cuda()#110

                
        three_domain_discriminator_feature = torch.cat([feature_target_unkonwn,feature_target_konwn, feature_source_konwn], 0)
        three_domain_discriminator_labels  = torch.cat([feature_target_unkonwn_labels,feature_target_konwn_labels, feature_source_konwn_labels], 0)

        p0 = three_domain_discriminator.forward(three_domain_discriminator_feature)
        se_loss = BCELossForMultiClassification(three_domain_discriminator_labels,p0)
        u_loss=torch.sum(var_source+var_target)


        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,7)), np.ones((2,1))), axis = -1).astype('float32'))).cuda(), \
                                    predict_prob_otherep)


        ce_down = CrossEntropyLoss(label_source_onethot, predict_prob_source_down)




        loss_MSE_down =  MSE(p0_down,p0_upper.detach())
        loss_MSE_down += MSE(p1_down,p1_upper.detach())


        
        entropy  = EntropyLoss(predict_prob_target, instance_level_weight= p2_upper.contiguous())
        adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                 predict_prob=domain_prob_discriminator_1_source)
        adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                  predict_prob=1 - domain_prob_discriminator_1_target, 
                                                  instance_level_weight = p2_upper.contiguous())


        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix,optimizer_discriminator,optimizer_three_domain_discriminator]):
            loss = loss_MSE_value * loss_MSE_down  + ce_down +  0.3 * adv_loss + 0.1 * entropy +  0.2*se_loss +ce_ep*3+u_loss

            loss.backward()


        counter_upper = AccuracyCounter()
        counter_upper.addOntBatch(variable_to_numpy(predict_prob_source_upper), variable_to_numpy(label_source_onethot))
        acc_train_upper = Variable(torch.from_numpy(np.asarray([counter_upper.reportAccuracy()], dtype=np.float32))).cuda()

        counter_down = AccuracyCounter()
        counter_down.addOntBatch(variable_to_numpy(predict_prob_source_down), variable_to_numpy(label_source_onethot))
        acc_train_down = Variable(torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).cuda()
        # track_scalars(log, ['ce_down', 'acc_train_down', 'acc_train_upper','adv_loss','entropy',"d1_upper", "loss_MSE_upper","loss_MSE_down","three_domain_loss","ce_ep"], globals())

    print(k)
    with TrainingModeManager([feature_extractor_fix,feature_extractor_nofix, cls_down,cls_upper], train=False) \
                                as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(Target_test_loader):
            correct_num = 0
            val_num = 0
            per_class_num = np.zeros((8))
            per_class_correct = np.zeros((8)).astype(np.float32)


            im = Variable(im, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()


            ft1, feature_target, __, predict_prob = net_down.forward(im)

            p1_upper = discriminator_p.forward(ft1)
            p2_upper = torch.sum(p1_upper, dim = -1).detach()

            label = F.one_hot(label.long(), num_classes=10)

            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob,label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)

            correct_num += (predict_index == label).sum()
            val_num += label.shape[0]
            num_out=8
            for i in range(num_out):
                if i == num_out - 1:
                    index = np.where(label == i)
                    index += (np.where(label == i + 1))
                    index += (np.where(label == i + 2))
                    correct_ind = np.where(predict_index[index[0]] == i) + np.where(predict_index[index[2]] == i) + np.where(
                        predict_index[index[4]] == i)
                    per_class_correct[i] += (
                                float(len(correct_ind[0])) + float(len(correct_ind[2])) + float(len(correct_ind[4])))
                    per_class_num[i] += (float(len(index[0])) + float(len(index[2])) + float(len(index[4])))
                if i != num_out - 1:
                    index = np.where(label == i)
                    correct_ind = np.where(predict_index[index[0]] == i)
                    per_class_correct[i] += float(len(correct_ind[0]))
                    per_class_num[i] += float(len(index[0]))

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
        print("Epoch: {:>3}/{},all_acc: {:>6.2f},  known_acc: {:>6.2f}, unknown_acc: {:>6.2f}".format(\
                k+1,20,  all_acc, known_acc,unknown_acc))
        # for i in range(num_out):
        #    logging.info("Label {}: {:>6.2f}%".format(i, per_class_acc[i]))

        accumulator.updateData(globals())
            # if i % 10 == 0:
            #     print(i)
    k += 1

    for x in accumulator.keys():
        globals()[x] = accumulator[x]





