import numpy as np
np.random.seed(1234)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from datetime import datetime
import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from ConversationModel import BiModel

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoAudio, self.videoLabels, self.videoSpeakers, self.trainVid, self.testVid  = pickle.load(open('/content/drive/My Drive/EmotionRNN2/dataformodel.pkl', 'rb'), encoding='latin1')

        
        '''
         joy, trust, fear, surprise, sadness, anticipation, anger, and disgust.= basic 8 emotions

         
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5} -= we have here
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        acouf, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        #log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes
        log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask,umask) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()      
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':
    #batch_size = args.
    batch_size = 2
    n_classes  = 6
    #cuda       = args.cuda
    cuda       = False
    #n_epochs   = args.epochs
    n_epochs   = 100

    
    D_m = 2000
    D_g = 300
    D_p = 300
    D_e = 200
    D_h = 200

    D_a = 200 # concat attention
      
    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                  n_classes=n_classes,
                  listener_state=False,
                  context_attention='general',
                  dropout_rec=0.1,
                  dropout=0.1)             
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])

    loss_function = MaskedNLLLoss(loss_weights) 
                         
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001,
                           weight_decay=0.00001)

    train_loader, valid_loader, test_loader =\
            get_IEMOCAP_loaders('/content/drive/My Drive/Emotion RNN/IEMOCAP_features_raw.pkl',
                                valid=0.1,
                                batch_size=batch_size,
                                num_workers=2)

    best_test, best_label, best_pred, best_mask = None, None, None, None
    best_uwa = None
    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore,_= train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc, _,_,_,val_fscore,_= train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        dict1 = classification_report(test_label,test_pred,sample_weight=test_mask,digits=4,output_dict=True)
        test_uwa = dict1['macro avg']['recall']
        print(test_uwa)
        if test_acc > 61 and test_uwa > 0.58:
          print("***"+str(test_acc)+"***"+str(test_uwa)+"***")
          print(classification_report(test_label,test_pred,sample_weight=test_mask,digits=4))
          print(confusion_matrix(test_label,test_pred,sample_weight=test_mask))
          torch.save(model.state_dict(), '/content/drive/My Drive/Emotion RNN/sadil/'+str(datetime.now())+"-"+str(test_acc)+"-"+str(test_uwa)+'rnn_model_loss_wiegts.pt')

        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
