import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EmotionRNNcell(nn.Module):
  
  def __init__(self,gru_input_dim,gru_hidden_dim,dropout):

    super(EmotionRNNcell, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.gru_cell = nn.GRUCell(gru_input_dim,gru_hidden_dim)
    
  def forward(self, U):
    u_ = torch.zeros(U.size()[0],gru_hidden_dim)
    U = U.permute(1,0,2)
    for u in U:
      u_ = self.gru_cell(u,u_)
      u_ = self.dropout(u_)
    return u_

class Model(nn.Module):
  
  def __init__(self,n_classes,gru_input_dim,gru_hidden_dim,linear_hidden_dim,dropout):
    
    super(Model, self).__init__()
    self.n_classes = n_classes
    self.dropout   = nn.Dropout(dropout)
    self.emotion_rnn_cell = EmotionRNNcell(gru_input_dim,gru_hidden_dim,dropout)

    self.linear = nn.Linear(gru_hidden_dim,linear_hidden_dim)
    self.smax_fc = nn.Linear(linear_hidden_dim,n_classes)

  def forward(self,x):
    x_ = self.emotion_rnn_cell(x)
    x_ = F.relu(self.linear(x_))
    x_ = self.dropout(x_)
    log_prob = F.log_softmax(self.smax_fc(x_)) 
    return log_prob

class CrossEntropyLoss(nn.Module):
  
  def __init__(self, weight=None):
    super(CrossEntropyLoss, self).__init__()
    self.weight = weight
    self.loss = nn.CrossEntropyLoss(weight=weight)

  def forward(self, pred, target):
    loss = self.loss(pred, target)
    return loss