from PIL import Image
import timm
from torch import nn
import torch
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import transforms 
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image as ImFunc
import numpy as np
from tqdm.auto import tqdm
from torch import optim
import pickle
import cv2
from PIL import ImageFile
import sys
from torchmetrics import Accuracy

import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    device=torch.device("cuda") 
else:  
    device = torch.device("cpu")



class Yoga82_Data(Dataset):
    def __init__(self,Imagepath,textpath,transform=None):
        self.textpath=textpath
        self.Imagepath=Imagepath
        self.transform=transform
        f = open(self.textpath);
        self.file=f.readlines();
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self,idx):
        path=self.file[idx].split(",")[0]
        Img=os.path.join(self.Imagepath,path)
        IMGFILE=ImFunc.open(Img).convert("RGB")
        IMGFILE=IMGFILE.resize((224, 224), resample=ImFunc.ANTIALIAS)

        try:
          Image=np.array(IMGFILE)/255.0
        except:
          print(Img)
        
        if len(Image.shape)<3 or Image.shape[2]>3 or Image.shape[2]<3 :
          new = IMGFILE.convert(mode='RGB')
          Image=np.array(new)/255.0
        Image=torch.from_numpy(Image).permute(2,0,1)
        Image=Image.type(torch.float32)
        
        out1=torch.tensor(int(self.file[idx].split(",")[1]))
        out2=torch.tensor(int(self.file[idx].split(",")[2]))
        out3=torch.tensor(int(self.file[idx].split(",")[3].split()[0]))

        if self.transform:
          Image = self.transform(Image)
        
        return Image,out1,out2,out3
        
        
TrainPath="/home/22mces01/Yoga82/DataFiles/Train"
TrainLable="/home/22mces01/Yoga82/DataFiles/yoga_train.txt"
TestPath="/home/22mces01/Yoga82/DataFiles/Test/"
TestLable="/home/22mces01/Yoga82/DataFiles/yoga_test.txt"


Train_Data=Yoga82_Data(TrainPath,TrainLable,transform=transforms.Compose([transforms.Resize([224, 224])]))
Test_Data=Yoga82_Data(TestPath,TestLable,transform=transforms.Compose([transforms.Resize([224, 224])]))



print("Length of Trainset",len(Train_Data))
print("Length of Testset",len(Test_Data))

trainset=DataLoader(Train_Data,batch_size=32,shuffle=True)
testset=DataLoader(Test_Data,batch_size=32,shuffle=True)


Pretrained_Model = timm.create_model(
    'coatnet_rmlp_2_rw_224',
    pretrained=True,
    num_classes=82,
)

for param in Pretrained_Model.parameters():
    param.requires_grad=True 
# avail_pretrained_models = timm.list_models(pretrained=True)

# for i in Pretrained_Model.head.parameters():
#     i.requires_grad=True  



def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)
    
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x
        
        
class CoatNet(nn.Module):
    def __init__(self,Pretrained_Model):
        super(CoatNet,self).__init__()
        self.model=Pretrained_Model
        self.lin1=nn.Linear(in_features=512,out_features=6,bias=True)
        self.lin2=nn.Linear(in_features=512,out_features=20,bias=True)
        self.norm_o1=LayerNorm2d((512,), eps=1e-06)
        self.norm_o2=LayerNorm2d((512,), eps=1e-06)
        
    def forward(self,x):
        
        x=self.model.stem(x)
        x=self.model.stages[0](x)
        x=self.model.stages[1](x)
        
        for i in range(0,7):
            x=self.model.stages[2].blocks[i](x)
        out1=self.norm_o1(x)
        out1=self.model.head.global_pool(out1)
        out1=self.lin1(out1)
        
        
        for i in range(7,14):
            x=self.model.stages[2].blocks[i](x)
        out2=self.norm_o2(x)
        out2=self.model.head.global_pool(out2)
        out2=self.lin2(out2)
        
        
        x=self.model.stages[3](x)
        x=self.model.norm(x)
        out3=self.model.head(x)
        
        
        return out1,out2,out3
    
model=CoatNet(Pretrained_Model)
model=model.to(device)




lr=0.003
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9, nesterov=False)

criterion1=nn.CrossEntropyLoss()



checkpoint = torch.load("/home/22mces01/Hierarchical_CoatNet/Hierarchical_CoatNet_Best_BESTMODEL.pth.tar")

# Load the model's state_dict
# model = mo(*args, **kwargs)
model.load_state_dict(checkpoint['state_dict'])

# Load any other objects you saved, such as the optimizer state
optimizer.load_state_dict(checkpoint['optimizer'])


# def calculate_top_5_accuracy(output, target):
#     with torch.no_grad():
#         batch_size = target.size(0)
#         _, pred = output.topk(k=int(batch_size * 0.05), dim=1, largest=True, sorted=True)
#         correct = pred.eq(target.view(-1, 1).expand_as(pred))
#         top_5_accuracy = correct[:, :int(batch_size * 0.05)].reshape(-1).float().sum(0) / batch_size
#     return top_5_accuracy


def EvalMode():
  top51=[];top52=[];top53=[]
  Elossloop1 = [];Elossloop2=[];Elossloop3=[]
  Eaccuracy1=[];Eaccuracy2=[];Eaccuracy3=[]
  model.eval()
  with torch.no_grad():
    for data in tqdm(testset,desc="Eval Loop",position=1):
        img,y1,y2,y3=data
        img=img.to(device)
        y1=y1.to(device)
        y2=y2.to(device)
        y3=y3.to(device)
        orig1=torch.reshape(y1,(32,1))
        orig2=torch.reshape(y2,(32,1))
        orig3=torch.reshape(y3,(32,1))
        Out1,Out2,Out3=model.forward(img)

       
        top51.append((torch.topk(Out1.data, 5).indices==orig1).sum().item()) # list of top 5 highest value indices
        top52.append((torch.topk(Out2.data, 5).indices==orig2).sum().item())
        top53.append((torch.topk(Out3.data, 5).indices==orig3).sum().item())
                
        Eaccuracy1.append((Out1.data.max(1)[1] == y1.data).sum().item())
        Eaccuracy2.append((Out2.data.max(1)[1] == y2.data).sum().item())
        Eaccuracy3.append((Out3.data.max(1)[1] == y3.data).sum().item())

  return top51,top52,top53,Eaccuracy1,Eaccuracy2,Eaccuracy3


top51,top52,top53,Eaccuracy1,Eaccuracy2,Eaccuracy3=EvalMode()

print(f">>>[Eval] Epoch :  >> TOP5_1:{(sum(top51) / len(testset.dataset))*100} >>> TOP5_2:{(sum(top52) / len(testset.dataset))*100} >>> TOP5_3:{(sum(top53) / len(testset.dataset))*100} >>> EAccuracy 1:{(sum(Eaccuracy1) / len(testset.dataset))*100} >>> EAccuracy 2:{(sum(Eaccuracy2) / len(testset.dataset))*100} >>> EAccuracy 3:{(sum(Eaccuracy3) / len(testset.dataset))*100}")
print("==================================================================================================================================================")

        