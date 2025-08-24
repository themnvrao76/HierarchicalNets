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
    
class LayerNorm2d(nn.LayerNorm): # Layer Normalization Code
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
        out1=self.norm_o1(x)  # H Class 6
        out1=self.model.head.global_pool(out1)
        out1=self.lin1(out1)
        
        
        for i in range(7,14):
            x=self.model.stages[2].blocks[i](x)
        out2=self.norm_o2(x)
        out2=self.model.head.global_pool(out2)
        out2=self.lin2(out2) # H Class 20
        
        
        x=self.model.stages[3](x)
        x=self.model.norm(x)
        out3=self.model.head(x)
        
        
        return out1,out2,out3
    
model=CoatNet(Pretrained_Model)
model=model.to(device)

lr=0.003
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9, nesterov=False)

criterion1=nn.CrossEntropyLoss()
criterion2=nn.CrossEntropyLoss()
criterion3=nn.CrossEntropyLoss()


TrainLoss={};
TestLoss={};
TrainAccuracy={};
TestAccuracy={};
for i in range(1,4):
  TrainLoss[f"Loss {i}"]=[]
  TestLoss[f"Loss {i}"]=[]
  TrainAccuracy[f"Accuracy {i}"]=[]
  TestAccuracy[f"Accuracy {i}"]=[]

def _checkpoint(statedict,file_path):
    print("*Saving Model*")
    torch.save(statedict,file_path)


def EvalMode():
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

        Out1,Out2,Out3=model.forward(img)

        loss1=criterion1(Out1,y1)
        loss2=criterion2(Out2,y2)
        loss3=criterion2(Out3,y3)

        Elossloop1.append(loss1.data.item() / len(testset))
        Elossloop2.append(loss2.data.item() / len(testset))
        Elossloop3.append(loss3.data.item() / len(testset))

        Eaccuracy1.append((Out1.data.max(1)[1] == y1.data).sum().item())
        Eaccuracy2.append((Out2.data.max(1)[1] == y2.data).sum().item())
        Eaccuracy3.append((Out3.data.max(1)[1] == y3.data).sum().item())

  return Elossloop1,Elossloop2,Elossloop3,Eaccuracy1,Eaccuracy2,Eaccuracy3



def _checkpoint(statedict,file_path):
    print("Saving Model ==========>")
    torch.save(statedict,file_path)

def TrainModel(epochs=50):
  best_accuracy = 0.0
  for epo in range(1,epochs):
    model.train()
    lossloop1 = [];lossloop2=[];lossloop3=[]
    accuracy1=[];accuracy2=[];accuracy3=[]
    #saving every epochs of model 
      

    for data in tqdm(trainset,desc="Train loop",position=1,leave=False):
      img,y1,y2,y3=data
      img=img.to(device)
      y1=y1.to(device)
      y2=y2.to(device)
      y3=y3.to(device)
   
      Out1,Out2,Out3=model.forward(img)

      loss1=criterion1(Out1,y1)
      loss2=criterion2(Out2,y2)
      loss3=criterion2(Out3,y3)

      lossloop1.append(loss1.data.item() / len(trainset))
      lossloop2.append(loss2.data.item() / len(trainset))
      lossloop3.append(loss3.data.item() / len(trainset))

      accuracy1.append((Out1.data.max(1)[1] == y1.data).sum().item())
      accuracy2.append((Out2.data.max(1)[1] == y2.data).sum().item())
      accuracy3.append((Out3.data.max(1)[1] == y3.data).sum().item())

      loss = loss1+loss2+loss3
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    ELoss1,ELoss2,ELoss3,EAcc1,EAcc2,EAcc3=EvalMode()
    
    
    
    original_stdout=sys.stdout
    with open("Output.txt",'a+') as f:
        sys.stdout=f
        print("===============================================================================================================")
    
        print(f">>>[Train] Epoch : {epo} Loss 1: { sum(lossloop1):.4f} >>> Loss 2: { sum(lossloop2):.4f} >>> Loss 3: { sum(lossloop3):.4f} >> Accuracy 1: {sum(accuracy1) / len(trainset.dataset):.4f} >>> Accuracy 2 : {sum(accuracy2) / len(trainset.dataset):.4f} >>> Accuracy 3: {sum(accuracy3) / len(trainset.dataset):.4f}")
    
        print(f">>>[Eval] Epoch : {epo} ELoss 1: { sum(ELoss1):.4f} >>> ELoss 2: { sum(ELoss2):.4f} >>> ELoss 3:{ sum(ELoss3):.4f} >> EAccuracy 1:{sum(EAcc1) / len(testset.dataset):.4f} >>> EAccuracy 2:{sum(EAcc2) / len(testset.dataset):.4f} >>> EAccuracy 3:{sum(EAcc3) / len(testset.dataset):.4f}")
        print("==================================================================================================================================================")

        
        sys.stdout=original_stdout
    
    
    print("===============================================================================================================")
    
    print(f">>>[Train] Epoch : {epo} Loss 1: { sum(lossloop1):.4f} >>> Loss 2: { sum(lossloop2):.4f} >>> Loss 3: { sum(lossloop3):.4f} >> Accuracy 1: {sum(accuracy1) / len(trainset.dataset):.4f} >>> Accuracy 2 : {sum(accuracy2) / len(trainset.dataset):.4f} >>> Accuracy 3: {sum(accuracy3) / len(trainset.dataset):.4f}")
    
    print(f">>>[Eval] Epoch : {epo} ELoss 1: { sum(ELoss1):.4f} >>> ELoss 2: { sum(ELoss2):.4f} >>> ELoss 3:{ sum(ELoss3):.4f} >> EAccuracy 1:{sum(EAcc1) / len(testset.dataset):.4f} >>> EAccuracy 2:{sum(EAcc2) / len(testset.dataset):.4f} >>> EAccuracy 3:{sum(EAcc3) / len(testset.dataset):.4f}")
    print("==================================================================================================================================================")
    
    val_accuracy=sum(EAcc3) / len(testset.dataset)
    if val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      checkpoint={"state_dict":model.state_dict(),"optimizer":optimizer.state_dict()}
      filename=f"Hierarchical_CoatNet_Best_BESTMODEL.pth.tar"
      print(f"This is Best Accuracy{best_accuracy} and Best Epoch{epo}")
      print("************************************************************************")
      _checkpoint(checkpoint,filename)

  
    TrainLoss["Loss 1"].append(sum(lossloop1));TrainLoss["Loss 2"].append(sum(lossloop2));TrainLoss["Loss 3"].append(sum(lossloop3)) 
    TrainAccuracy["Accuracy 1"].append(sum(accuracy1) / len(trainset.dataset));TrainAccuracy["Accuracy 2"].append(sum(accuracy2) / len(trainset.dataset));TrainAccuracy["Accuracy 3"].append(sum(accuracy3) / len(trainset.dataset));

    TestLoss["Loss 1"].append(sum(ELoss1));TestLoss["Loss 2"].append(sum(ELoss2));TestLoss["Loss 3"].append(sum(ELoss3)) 
    TestAccuracy["Accuracy 1"].append(sum(EAcc1) / len(testset.dataset));TestAccuracy["Accuracy 2"].append(sum(EAcc2) / len(testset.dataset));TestAccuracy["Accuracy 3"].append(sum(EAcc3) / len(testset.dataset));

    if epo%5==0:
        checkpoint={"state_dict":model.state_dict(),"optimizer":optimizer.state_dict()}
        filename=f"{epo}_epoch_SEDenseNet201.pth.tar"
        _checkpoint(checkpoint,filename)

  return TrainLoss,TrainAccuracy,TestLoss,TestAccuracy




if __name__ == "__main__":
   
  torch.backends.cudnn.benchmark = True
  TrainLoss,TrainAccuracy,TestLoss,TestAccuracy=TrainModel(epochs=51)

  with open('TrainLoss.pickle', 'wb') as handle:
    pickle.dump(TrainLoss, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('TrainAccuracy.pickle', 'wb') as handle:
      pickle.dump(TrainAccuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('TestLoss.pickle', 'wb') as handle:
      pickle.dump(TestLoss, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('TestAccuracy.pickle', 'wb') as handle:
      pickle.dump(TestAccuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)
