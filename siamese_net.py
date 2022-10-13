import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.models import efficientnet_b0


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        #Outputs batch X 512 X 1 X 1 
        self.net = efficientnet_b0()


    def forward_once(self, x):
        output = self.net(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        
        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2,input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1,output2,output3

        return output1, output2



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * euclidean_distance + (1 + (-1 * label) ).float() * F.relu(self.margin - (euclidean_distance + self.eps).sqrt()).pow(2))
        loss_contrastive = torch.mean(losses)

        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor,positive) #Each is batch X 512 
        distance_negative = F.cosine_similarity(anchor,negative)  # .pow(.5)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case. 
        return losses.mean() if size_average else losses.sum()
