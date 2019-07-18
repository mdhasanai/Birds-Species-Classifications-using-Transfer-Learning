import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_loader, set_transform
from models.model import Encoder

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

import pickle

#transform = set_transform()
#dataloader = get_loader("./test_corpus.csv",8,transform=transform)

save_path = "./save"
epoch = 200

if not os.path.exists(save_path):
    os.makedirs(save_path)


def train(epoch=5,freeze=True):
    

    #Defining Model
    model = Encoder()
    print(model)

    if freeze:
        for param in model._resnet_extractor.parameters():
            param.require_grad = False
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    transform = set_transform()
    
    test_loader = get_loader("./test_corpus.csv",8,transform=transform)
    
    

    

    
    
    
    if torch.cuda.is_available():
        model = model.cuda()

    total_train_loss = []
    total_val_loss = []
    
    best_train = 100000000
    best_valid = 100000000
    not_improve  = 0


    
    for e in range(1):

        loss_train = []
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train()
        num_iter = 1

        model.eval()
        num_iter_val = 1
        for i, (images,classes) in enumerate(test_loader):
            
            optimizer.zero_grad()

            feature_image = model(images)
            
            if torch.cuda.is_available():
                feature_image = feature_image.cuda()
                classes=classes.cuda()
            
            
            
            _, preds = torch.max(feature_image.data, 1)
            
            loss = criterion(feature_image, classes)
            
            loss_val += loss.cpu().detach().numpy()
            acc_val += torch.sum(preds == classes)
            
            num_iter_val = i+1
            del feature_image, classes, preds
            torch.cuda.empty_cache()

        avg_val =  loss_val/num_iter_val   
        print(f"\t\tValid Loss: {avg_val}")

        
        tb.add_scalar("Validation_Loss", avg_val, e)
        tb.add_scalar("Validation_Accuracy", 100-avg_val, e)
        
        if avg_val<best_valid:
            total_val_loss.append(avg_val)
            model_save = save_path+"/best_model.th"
            torch.save(model.state_dict(), model_save)
            best_valid = avg_val
            print(f"Model saved to path save/")
            not_improve = 0            

        else:
            not_improve +=1
            print(f"Not Improved {not_improve} times ")
        if not_improve==6:
            break
            
    save_loss = {"train":total_train_loss, "valid":total_val_loss}
    with open(save_path+"/losses.pickle","wb") as files:
        pickle.dump(save_loss,files)
            
    tb.close()
#         print(f"AVG loss: {loss_train/e}")
        
#         print(f"AVG loss: {acc_train/e}")
        
        
#valid_file = "./valid_corpus.csv"  
    
train(200) 
