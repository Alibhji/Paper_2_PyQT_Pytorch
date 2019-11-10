
import numpy as np
import helper
import simulation
import time
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchsummary import summary

from collections import defaultdict
from loss import dice_loss

from my_utils import gui_tools
from PyQt5 import  QtGui





class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]

# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.ToTensor(),
])




def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class AliNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,3,padding=1)
        self.conv2=nn.Conv2d(6,6,3,padding=1)
        self.out=nn.Conv2d(6,6,3,padding=1)
        
        
        
    def forward(self, input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.out(x)        
               
        
        return x



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase,tools=None):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    tools.logging("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, dataloaders,optimizer, scheduler, num_epochs=25, tools=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        tools.logging('Epoch {}/{}'.format(epoch, num_epochs - 1))
        tools.logging('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    tools.logging("LR {}".format(param_group['lr']))
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase,tools)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        tools.logging('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        QtGui.QGuiApplication.processEvents()
    print('Best val loss: {:4f}'.format(best_loss))
    tools.logging('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def training(ui):
    tools=gui_tools.utils(ui)
    
    trans = transforms.Compose([
    transforms.ToTensor(),])
    
    train_set = SimDataset(200, transform = trans)
    val_set = SimDataset(100, transform = trans)
        
    image_datasets = {
    'train': train_set, 'val': val_set
                                    }

    batch_size = 50

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    # print(dataset_sizes)
    tools.logging(str(dataset_sizes))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=AliNet()
    model=model.to(device)

    summary(model,input_size=(3,25,25))
    
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
    QtGui.QGuiApplication.processEvents()
    model = train_model(model,dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=10,tools=tools)