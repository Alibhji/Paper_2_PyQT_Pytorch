import os


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

from PyQt5 import  QtGui
import pickle



# import hiddenlayer as hl


class SimDataset(Dataset):
    def __init__(self, count, transform=None , config= None):

        # self.config.update({'dataset_obj_shape_triangle': self.ui.in_shape_triangle.isChecked()})
        # self.config.update({'dataset_obj_shape_circle': self.ui.in_shape_circle.isChecked()})
        # self.config.update({'dataset_obj_shape_mesh': self.ui.in_shape_mesh.isChecked()})
        # self.config.update({'dataset_obj_shape_square': self.ui.in_shape_squre.isChecked()})
        # self.config.update({'dataset_obj_shape_plus': self.ui.in_shape_plus.isChecked()})
        
        H_ , W_ = config['img_H'] , config['img_W']
        T_, C_, M_, S_, P_= config['dataset_obj_shape_triangle'] , \
                            config['dataset_obj_shape_circle'] ,  \
                            config['dataset_obj_shape_mesh'] , \
                            config['dataset_obj_shape_square'], \
                            config['dataset_obj_shape_plus']

        self.input_images, self.target_masks = simulation.generate_random_data(H_, W_,
                                                                               count=count,
                                                                               triangle=T_,
                                                                               circle=C_,
                                                                               mesh=M_ ,
                                                                               square=S_,
                                                                               plus=P_)
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





def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    # target.size(0) is the batch size
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    # print('In the loss ----->>>>>>>>>>', bce.data.cpu().numpy())
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase,ui=None):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    ui.tools.logging("{}: {}".format(phase, ", ".join(outputs)))
    return outputs

def train_model(model, dataloaders,optimizer, scheduler, num_epochs=25, ui=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ui.modelList[ui.model_name]['trained'] = True

    # print(model)

    # if(ui.tools.check_dir(ui.module_dir_name)):
    # with open(ui.model_txt_file, 'a') as f:
    #     f.writelines('\n'+ '*'*30)

    flag_gen_txt=hasattr(ui,'model_txt_file')
    # print('??????????????---->>>>>>',len(ui.modelList))
    # print('??????????????---->>>>>>', ui.model_txt_file)
    txt_file_content=''

    if(flag_gen_txt):
        txt_file_content='[Model_txt_log_path] = '+ui.model_txt_file
    txt_file_content+='\n' + '*' * 30 + '\n'
    txt_file_content+='[Model_name] = '+ui.modelList[ui.model_name]['name']
    txt_file_content += '\n'+'[Model_code] = '+ui.modelList[ui.model_name]['code']
    txt_file_content+='\n' + '*' * 30 + '\n'

    temp_list_bce_dice_total_t, temp_list_bce_dice_total_v = [], [] # (sample_number, loss_value , epoch_number)


    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ui.tools.logging('Epoch {}/{}'.format(epoch, num_epochs - 1))
        ui.tools.logging('-' * 10)

        txt_file_content+=('\n'+ 'Epoch {}/{}'.format(epoch, num_epochs - 1))
        txt_file_content+=('\n'+ '-' * 10)


        dict_loss={}




        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    ui.tools.logging("LR {}".format(param_group['lr']))
                    txt_file_content+=('\n' + "LR {}".format(param_group['lr']))

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
                        sample_number= epoch_samples+ui.config['dataset_train_size'] * epoch
                        loss_bce_train= (sample_number,epoch,
                                         metrics['bce'] / (epoch_samples+inputs.size(0)),
                                         metrics['dice'] / (epoch_samples + inputs.size(0)),
                                         metrics['loss'] / (epoch_samples + inputs.size(0)),
                                         param_group['lr']
                                         )

                        temp_list_bce_dice_total_t.append(loss_bce_train)
                    else:
                        sample_number = epoch_samples + ui.config['dataset_val_size'] * epoch

                        loss_bce_val =  (sample_number,epoch,
                                         metrics['bce'] / (epoch_samples+inputs.size(0)),
                                         metrics['dice'] / (epoch_samples + inputs.size(0)),
                                         metrics['loss'] / (epoch_samples + inputs.size(0)),
                                         param_group['lr']
                                         )
                        temp_list_bce_dice_total_v.append(loss_bce_val)

                # statistics
                epoch_samples += inputs.size(0)

                # print('metric ---Epoch_sampel--->>>>', metrics, epoch_samples)
                # print('metric ---Epoch_sampel--->>>>', temp_list_bce_t)


            print_out=print_metrics(metrics, epoch_samples, phase,ui)
            txt_file_content+=('\n' + "{}: {}".format(phase, ", ".join(print_out)))




            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                bset_epoch= epoch
                best_model_wts = copy.deepcopy(model.state_dict())



        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        ui.tools.logging('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        txt_file_content+=('\n' + '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        QtGui.QGuiApplication.processEvents()
    print('Best val loss: {:4f} at {} epoch.'.format(best_loss , bset_epoch))
    ui.tools.logging('Best val loss: {:4f} at {} epoch.'.format(best_loss , bset_epoch),'red')
    txt_file_content += '\n' + '*' * 30
    txt_file_content+=('\n' + 'Best val loss: {:4f} at {} epoch.'.format(best_loss , bset_epoch))

    dict_loss.update({'loss_bce_train': temp_list_bce_dice_total_t})
    dict_loss.update({'loss_bce_Val': temp_list_bce_dice_total_v})

    ui.modelList[ui.model_name].update({'loss': dict_loss})
    # print(ui.modelList[/ui.model_name]['loss'])
    txt_file_content+='\n' + '*' * 30
    txt_file_content+= '\n'+"The loss is in this format [sample number, epoch number , binary_cross_entropy_with_logits , defined_loss , total loss, learning rate]"
    txt_file_content+='\n' + '[loss] =' + str(ui.modelList[ui.model_name]['loss'])



    if (flag_gen_txt):
        with open(ui.model_txt_file, 'a') as f:
            f.writelines('\n' + '*' * 30 +'\n'+txt_file_content+'\n' + '*' * 30 +'\n')


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def Dataset_create(ui):
    trans = transforms.Compose([
    transforms.ToTensor(),])


    
    train_set = SimDataset(ui.config['dataset_train_size'], transform = trans, config=ui.config)
    val_set = SimDataset(ui.config['dataset_val_size'], transform = trans, config=ui.config)
    ui.tools.logging("Datasets are created.",'red')
    
        
    image_datasets = {
    'train': train_set, 'val': val_set
                                    }
    # print('Number of classes',image_datasets['train'][0][1].shape)
    ui.image_datasets=image_datasets
    ui.number_of_classess=image_datasets['train'][0][1].shape[0]

def Model_create(ui,architect_file=None):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(architect_file):
        model=AliNet(architect_file=architect_file)

    else:
        model=AliNet()


    # summary(ui.model,input_size=(3,25,25),tools=ui.tools)
    model = model.to(device)
    summary(model, input_size=(3, 25, 25))
    ui.tools.logging("The model is created.",'red')

    # hl.build_graph(ui.model, torch.zeros([1, 3, 25, 25]).to(device))


    root=os.path.join(os.getcwd(),ui.module_dir_name,'models')
    ui.tools.check_dir(root,create_dir=True)
    # print('ui.modelListname------->',ui.modelList[ui.Module_name]['code'])

    print('\n'*3)
    model_name=os.path.join(root,ui.modelList[ui.Module_name]['code']+'.'+ui.modelList[ui.Module_name]['name']+'.model')


    ui.tools.save_object(path=model_name , object=model)
    ui.modelList[ui.Module_name].update({'model_address': model_name})
    # with open(model_name, 'wb') as uiFile:
    #     # Step 3
    #     pickle.dump(model, uiFile)

    torch.cuda.empty_cache()
    del model


def training(ui):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ui.model.to(device)
    num_epochs=ui.config['train_Epoch_number']
    tools=ui.tools
    image_datasets=ui.image_datasets
    batch_size = ui.config['dataset_batch_size']

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    # print(dataset_sizes)
    tools.logging(str(dataset_sizes))
    

    
    optimizer_ft = optim.Adam(ui.model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
    QtGui.QGuiApplication.processEvents()
    ui.ui_state='training'

    train_model(ui.model,dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,ui=ui)
    ui.ui_state = 'idle'



def model_architecture(ui):
    # input_ch = list([3])
    # out_ch   = list([pow(2,i) for i in range(2)])
    # kernels  = list(range(3,8,2))


    s1 = ui.ui.in_models_layers.text()
    s2 = ui.ui.in_models_kernels.text()
    s3 = ui.ui.in_models_channels.text()

    input_ch = eval(s1)
    kernels  = eval(s2)
    out_ch   = eval(s3)




    print('O..'*20,kernels)
    num_classes = ui.number_of_classess

    ui.config.update({'models_inputs' :input_ch})
    ui.config.update({'models_outputs': out_ch})
    ui.config.update({'models_kernels': kernels})
    ui.config.update({'models_num_classes': num_classes})



    ui.config['model_counter']=0

    # ui.module_dir_name='designed_module'

    root=os.path.join(os.getcwd(),ui.module_dir_name)
    ui.tools.check_dir(ui.module_dir_name,create_dir=True)


    for out__ in out_ch:
        for k__ in kernels:
            model_dic = {}
            in__=input_ch[0]
            # k__=17
            p__=int(k__/2)
            conv=[]
            conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format(in__,out__,k__,p__))
            conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format(out__, out__, k__, p__))
            conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format(out__, num_classes, k__, p__))
            # conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format('6','6',str(ch),str(int(ch/2))))
            # conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format('6','6',str(ch),str(int(ch/2))))
            # conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format('6','6',str(ch),str(int(ch/2))))
            # conv.append('nn.Conv2d( {}, {}, {}, padding={})'.format('6','6',str(ch),str(int(ch/2))))




            ui.tools.fill_out_table(ui.modelList)
            QtGui.QGuiApplication.processEvents()

            ui.tools.logging(str(conv))
            print(str(conv))

            Module_name= 'Module_{:02d}L_{:02d}ich_{:003d}och_{:02d}k_{:02d}p'.format(len(conv),in__,out__,k__,p__)
            # print(Module_name)
            ui.model_txt_file=os.path.join(root,Module_name+'.txt')
            ui.Module_name=Module_name

            ui.config['model_counter'] += 1
            model_code='{:04d}_{:03d}L'.format(ui.config['model_counter'],len(conv))
            model_dic.update({'name':Module_name})
            model_dic.update({'text_log': ui.model_txt_file})
            model_dic.update({'struct':conv})

            model_dic.update({'trained': False})
            model_dic.update({'code': model_code})
            ui.modelList.update({Module_name : model_dic})

            Model_create(ui, architect_file=conv)
            # model_dic.update({'model': model})

            with open(ui.model_txt_file , 'w') as f:
                f.writelines('\n'.join(conv[0:]))


    # print(ui.modelList)

class AliNet(nn.Module):
    
    def __init__(self, architect_file=None):
        super().__init__()
        self.convlen=0
        
        if (architect_file):
            # self.conv=(eval(architect_file))
            for idx,layer in enumerate(architect_file):
                exec('self.conv{}='.format(idx)+str(architect_file[idx]))
            self.convlen=len(architect_file)
        else :
            self.conv0=nn.Conv2d(3,6,3,padding=1)
            self.conv1=nn.Conv2d(6,6,3,padding=1)
            self.conv2=nn.Conv2d(6,6,3,padding=1)
            self.convlen=3
            

        
        
        
    def forward(self, x):
        # x=eval('self.conv0(x)')
        # x=self.conv1(x)
        # x=self.conv2(x) 
        for layer in range(self.convlen):
            x=eval('self.conv{}(x)'.format(layer))      
               
        
        return x        