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



save_path = "./save/"
batch_size = 8
learning_rate = 0.001
momentum = 0.9
epoch = 200
train_csv = "./train_corpus.csv"
valid_csv = "./valid_corpus.csv"
#load_path = "./save/best_model.th"

if not os.path.exists(save_path):
    os.makedirs(save_path)


def train(epoch=5,freeze=True):
    
    tb = SummaryWriter()
    #Defining Model
    model = Encoder()
    print(model)
    

    
    #model.load_state_dict(torch.load(load_path))

    if freeze:
        for param in model._resnet_extractor.parameters():
            param.require_grad = False
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    transform = set_transform()
    train_loader = get_loader(train_csv,batch_size,transform=transform)
    
    valid_loader = get_loader(valid_csv,batch_size,transform=transform)
    
    
    img,cls = next(iter(train_loader))
       
    #print(img.shape)
    grid = torchvision.utils.make_grid(img)
    tb.add_image('images', grid, 0)
   # tb.add_graph(model,img[0])
    

    
    
    
    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    total_train_loss = []
    total_val_loss = []
    
    best_train = 100000000
    best_valid = 100000000
    not_improve  = 0

    #train_avg_list = []
    #valid_avg_list = []
    tb.add_graph(model,img)
    for e in range(1,epoch):

        loss_train = []
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train()
        num_iter = 1
        for i, (images,classes) in enumerate(train_loader):
            
            optimizer.zero_grad()
            

             
            if torch.cuda.is_available():
                images = images.cuda()
                classes=classes.cuda()
            
            feature_image = model(images)            
            _, preds = torch.max(feature_image.data, 1)
            
            loss = criterion(feature_image, classes)
            
            loss.backward()
            
            optimizer.step()
            
            loss_train.append(loss.cpu().detach().numpy())
            acc_train += torch.sum(preds == classes)
            
            del feature_image, classes, preds
            torch.cuda.empty_cache()
            
            #print(f"Loss i: {i}")
            num_iter = i+1
            if i %10 == 0:
                print(f"Epoch ({e}/{epoch}) Iter: {i+1} Loss: {loss}")
        
        avg_loss =  sum(loss_train)/num_iter   
        print(f"\t\tTotal iter: {num_iter} AVG loss: {avg_loss}")
        tb.add_scalar("Train_Loss", avg_loss, e)
        tb.add_scalar("Train_Accuracy", 100-avg_loss, e)


        total_train_loss.append(avg_loss)
       
    
        model.eval()
        num_iter_val = 1
        for i, (images,classes) in enumerate(valid_loader):
            
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


        
 
    
train(epoch) 




