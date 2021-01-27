import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,classification_report, precision_recall_fscore_support
import pickle
import numpy as np
from datetime import datetime

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoAudio, self.videoLabels, self.videoSpeakers, self.trainVid, self.testVid  = pickle.load(open(path, 'rb'), encoding='latin1')
        self.dataset = []
        self.label = []
        dialogs = [x for x in (self.trainVid if train else self.testVid)]

        for dialog in dialogs:
          for utterence,label in zip(self.videoAudio[dialog],self.videoLabels[dialog]):
            self.dataset.append(utterence)
            self.label.append(label)
        
        self.len = len(self.label)

    def __getitem__(self, index):
        return torch.FloatTensor(self.dataset[index]), torch.tensor([self.label[index]])

    def __len__(self):
        return self.len

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(path, batch_size=32):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, 0.1)
    train_loader = DataLoader(trainset, batch_size=batch_size,sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size,sampler=valid_sampler)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader,optimizer, train=True):
  preds = []
  labels = []
  if train:
    model.train()
  else:
    model.eval()

  for data in dataloader:
    if train:
      optimizer.zero_grad()

    features, label = data
    log_prob = model(features)  
    
    label = label.view(-1)  
    loss = loss_function(log_prob, label)

    if train:
      loss.backward()
      optimizer.step()
      
    pred_ = torch.argmax(log_prob,1)
    preds.append(pred_.data.cpu().numpy())
    labels.append(label.data.cpu().numpy())

  if preds!=[]:
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)

  return preds,labels

n_classes=6
gru_input_dim=512
gru_hidden_dim=512
linear_hidden_dim=512
dropout=0.5

batch_size = 32
n_epochs = 500
loss_weights = torch.FloatTensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668])

train_loader,valid_loader, test_loader = get_IEMOCAP_loaders('/content/drive/My Drive/EmotionRNN2/dataformodel_20_bins.pkl',batch_size)
model = Model(n_classes,gru_input_dim,gru_hidden_dim,linear_hidden_dim,dropout)
loss_function = CrossEntropyLoss(loss_weights) 
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

for e in range(n_epochs): 
  train_pred,train_label = train_or_eval_model(model, loss_function, train_loader,optimizer)
  valid_pred,valid_label = train_or_eval_model(model,loss_function,valid_loader,None,train=False)
  test_pred,test_label = train_or_eval_model(model,loss_function,test_loader,None,train=False)
  train_acc = round(accuracy_score(train_label,train_pred)*100,2)
  valid_acc = round(accuracy_score(valid_label,valid_pred)*100,2)
  test_acc = round(accuracy_score(test_label,test_pred)*100,2)
  print("Train Accuracy: ",train_acc," Validation Accuracy ",valid_acc,"   Test Accuracy: ",test_acc)
  if test_acc>46:
    print(confusion_matrix(test_label,test_pred))
    print(classification_report(test_label,test_pred,digits=4))
    if test_acc>47:
      torch.save(model.state_dict(), '/content/drive/My Drive/FYP_Final_Work/'+str(datetime.now())+"-"+str(test_acc)+'wave2vec_20_mean_utterence_pediction_model.pt')