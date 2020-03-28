# -*- coding: utf-8 -*-
"""
ESE545-Data Mining Project: Binary classification with SVM (Pegasos v.s Adagrad) & 
                            Multi-class classification with SVM (One-Versus-One) and CNN
Author: Mian Wang  
Time: 3/27/20
"""

import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt

# import the dataset
train = np.load('train.npy')
train_labels = np.load('train_labels.npy')
test = np.load('test.npy')
test_labels = np.load('test_labels.npy')


*********************Extract data with label 2,5 for binary classifier —— SVM (Pegasos & Adagrad)******************************

def extract_data(target_labels, train, train_labels, test, test_labels):
  '''
  Extract data with target labels from given dataset.
  '''
  # step 1: find indices of train data and test data with target labels
  train_idx, test_idx = [], []
  for i in target_labels:
    idx1, idx2 = np.where(train_labels==i)[0], np.where(test_labels==i)[0]
    train_idx.extend(idx1)
    test_idx.extend(idx2)

  # step 2: shuffle the indices
  np.random.shuffle(train_idx)
  np.random.shuffle(test_idx)

  # step 3: screen out the dataset according to indices
  train_target, train_labels_target = train[train_idx], train_labels[train_idx]
  test_target, test_labels_target = test[test_idx], test_labels[test_idx]
  return train_target, train_labels_target, test_target, test_labels_target


# extract data with label 2,5 to train Pegasos & Adagrad SVM
train_bin, train_labels_bin, test_bin, test_labels_bin = extract_data([2,5], train, train_labels, test, test_labels)
train_labels_bin[np.argwhere(train_labels_bin==2)], test_labels_bin[np.argwhere(test_labels_bin==2)] = -1, -1
train_labels_bin[np.argwhere(train_labels_bin==5)], test_labels_bin[np.argwhere(test_labels_bin==5)] = 1, 1


def pegasos(train_data, train_label, T, n, lbd, test_data, test_label):
  '''
  Train on n randomly picked data, with regularization parameter lbd for T iterations.
  '''
  w = np.zeros(train_data[0].size)
  train_acc, test_acc = [], []
  for t in range(1,T+1):
    idx = np.random.choice(range(len(train_data)), n, replace=False)
    imgs, labels = train_data[idx]/255, train_label[idx]
    imgs = imgs.reshape(imgs.shape[0], imgs[0].size)

    # update all points with wrong labels
    lr = 1/(lbd*t)
    for i in range(len(labels)):
      x, y = imgs[i], labels[i]
      margin = np.dot(y, np.matmul(x,w))
      delta_f = -np.dot(x, y)     
      # update weight when ywx<1
      if margin >= 1: continue
      w = w - lr*(lbd*w + delta_f)
      # project w into the boundary
      w_norm = np.linalg.norm(w)
      w = w/w_norm * min(w_norm, 1/np.sqrt(lbd))

    # calculate the training accuracy
    predict = np.sign(np.matmul(imgs,w))
    predict = predict.reshape(len(predict))
    correct = np.where(labels==predict)[0].size
    train_acc.append(correct/n)

    # calculate the test accuracy
    predict = np.sign(np.matmul(test_data.reshape(test_data.shape[0],test_data[0].size)/255,w))
    predict = predict.reshape(len(predict))
    correct = np.where(test_label==predict)[0].size
    test_acc.append(correct/len(test_data))
    
  return w, train_acc, test_acc

# train Pegasos SVM for classification
T, n, lbd = 150, 800, 1
w, train_acc_pega, test_acc_pega = pegasos(train_bin, train_labels_bin, T, n, lbd, test_bin, test_labels_bin)
# plot Pegasos's training accuracy & test accuracy v.s number of iteration
plt.plot(1-np.array(train_acc_pega), label='training error')
plt.plot(1-np.array(test_acc_pega), label='test error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.title(f'pegasos with T{T}_n{n}_lambda{lbd}')
plt.legend()
plt.show()
plt.savefig(f'pegasos with T{T}_n{n}_lambda{lbd}.png')


def adagrad(train_data, train_label, T, n, lbd, test_data, test_label):
  '''
  Train on n data with regularization parameter lbd for T iterations.
  '''
  w = np.zeros(train_data[0].size)
  train_acc, test_acc = [], []
  G = 0
  for t in range(1, T+1):
    idx = np.random.choice(range(len(train_data)), n, replace=False)
    imgs, labels = train_data[idx]/255, train_label[idx]
    imgs = imgs.reshape(imgs.shape[0], imgs[0].size)
    
    lr = 1/lbd
    for i in range(len(labels)):
      x, y = imgs[i], labels[i]
      output = np.matmul(x, w)
      if np.dot(y, output) >= 1: continue
      # update the weight if ywx<1
      delta_f = -np.dot(x, y)
      G += np.matmul(delta_f, delta_f)
      w = w - lr/np.sqrt(G) * delta_f
      # project w into the boundary
      w_norm = np.linalg.norm(w)
      w = w/w_norm * min(w_norm, G/np.sqrt(lbd))

    # compute the training accuracy
    predict = np.sign(np.matmul(imgs, w))
    predict = predict.reshape(len(predict))
    correct = np.where(predict==labels)[0].size
    train_acc.append(correct/n)

    # compute the test accuracy
    test_x = test_data.reshape(test_data.shape[0], test_data[0].size) / 255
    predict = np.sign(np.matmul(test_x, w))
    predict = predict.reshape(len(predict))
    correct = np.where(predict==test_label)[0].size
    test_acc.append(correct/len(test_label))
  
  return w, train_acc, test_acc

# train Adagrad SVM for classification 
T, n, lbd = 20, 100, 0.1
w, train_acc_ada, test_acc_ada = adagrad(train_bin, train_labels_bin, T, n, lbd, test_bin, test_labels_bin)
# plot Adagrad's training accuracy & test accuracy v.s number of iteration
plt.plot(1-np.array(train_acc_ada), label='training error')
plt.plot(1-np.array(test_acc_ada), label='test error')
plt.xticks(range(len(train_acc_ada)))
plt.xlabel('iteration')
plt.ylabel('error')
plt.title(f'adagrad with T{T}_n{n}_lambda{lbd}')
plt.legend()
plt.show()
plt.savefig(f'adagrad with T{T}_n{n}_lambda{lbd}.png')


*********************Extract data with label 2,5,7 for multi-class classifier —— One-versus-One SVM & CNN************************

# define a class to train 3 SVM at the same time for multi-class classification (one-versus-one SVM)
class ovo():
  def __init__(self, train_data, train_label, T, n, lbd, test_data, test_label):
    self.train_data, self.test_data = train_data, test_data
    self.train_label, self.test_label = train_label, test_label
    self.T, self.n, self.lbd = T, n, lbd  
    self.w25 = np.zeros(self.train_data[0].size)
    self.w27 = np.zeros(self.train_data[0].size)
    self.w57 = np.zeros(self.train_data[0].size)
    self.G25, self.G27, self.G57 = 0, 0, 0

  def ada25(self, train_data, train_label, n, lbd):
    ''' Binary classifier on data with label 2,5 '''
    train_label[np.argwhere(train_label==2)], train_label[np.argwhere(train_label==5)] = -1, 1   
    idx = np.random.choice(range(len(train_data)), n, replace=False)
    imgs, labels = train_data[idx]/255, train_label[idx]
    imgs = imgs.reshape(imgs.shape[0], imgs[0].size)
      
    lr = 1/self.lbd
    for i in range(len(labels)):
      x, y = imgs[i], labels[i]
      output = np.matmul(x, self.w25)
      if np.dot(y, output) >= 1: continue
      # update the weight if ywx<1
      delta_f = -np.dot(x, y)
      self.G25 += np.matmul(delta_f, delta_f)
      self.w25 = self.w25 - lr/np.sqrt(self.G25) * delta_f
      # project w into the boundary
      w_norm = np.linalg.norm(self.w25)
      self.w25 = self.w25/w_norm * min(w_norm, self.G25/np.sqrt(lbd))

    # compute the training accuracy
    predict = np.sign(np.matmul(imgs, self.w25))
    predict = predict.reshape(len(predict))
    correct = np.where(predict==labels)[0].size
    train_acc = correct/n
    return train_acc
  
  def ada27(self, train_data, train_label, n, lbd):
    ''' Binary classifier on data with label 2,7 '''
    train_label[np.argwhere(train_label==2)], train_label[np.argwhere(train_label==7)] = -1, 1
    idx = np.random.choice(range(len(train_data)), n, replace=False)
    imgs, labels = train_data[idx]/255, train_label[idx]
    imgs = imgs.reshape(imgs.shape[0], imgs[0].size)
      
    lr = 1/self.lbd
    for i in range(len(labels)):
      x, y = imgs[i], labels[i]
      output = np.matmul(x, self.w27)
      if np.dot(y, output) >= 1: continue
      # update the weight if ywx<1
      delta_f = -np.dot(x, y)
      self.G27 += np.matmul(delta_f, delta_f)
      self.w27 = self.w27 - lr/np.sqrt(self.G27) * delta_f
      # project w into the boundary
      w_norm = np.linalg.norm(self.w27)
      self.w27 = self.w27/w_norm * min(w_norm, self.G27/np.sqrt(lbd))

    # compute the training accuracy
    predict = np.sign(np.matmul(imgs, self.w27))
    predict = predict.reshape(len(predict))
    correct = np.where(predict==labels)[0].size
    train_acc = correct/n
    return train_acc
    
  def ada57(self, train_data, train_label, n, lbd):
    ''' Binary classifier on data with label 5,7 '''
    train_label[np.argwhere(train_label==5)], train_label[np.argwhere(train_label==7)] = -1, 1
    idx = np.random.choice(range(len(train_data)), n, replace=False)
    imgs, labels = train_data[idx]/255, train_label[idx]
    imgs = imgs.reshape(imgs.shape[0], imgs[0].size)
      
    lr = 1/self.lbd
    for i in range(len(labels)):
      x, y = imgs[i], labels[i]
      output = np.matmul(x, self.w57)
      if np.dot(y, output) >= 1: continue
      # update the weight if ywx<1
      delta_f = -np.dot(x, y)
      self.G57 += np.matmul(delta_f, delta_f)
      self.w57 = self.w57 - lr/np.sqrt(self.G57) * delta_f
      # project w into the boundary
      w_norm = np.linalg.norm(self.w57)
      self.w57 = self.w57/w_norm * min(w_norm, self.G57/np.sqrt(lbd))
    
    # compute the training accuracy
    predict = np.sign(np.matmul(imgs, self.w57))
    predict = predict.reshape(len(predict))
    correct = np.where(predict==labels)[0].size
    train_acc = correct/n
    return train_acc

  def train(self):
    train25, train_labels25, _, _ = extract_data([2,5], self.train_data, self.train_label, self.test_data, self.test_label)
    train27, train_labels27, _, _ = extract_data([2,7], self.train_data, self.train_label, self.test_data, self.test_label)
    train57, train_labels57, _, _ = extract_data([5,7], self.train_data, self.train_label, self.test_data, self.test_label)

    train_acc25, train_acc27, train_acc57, test_acc = [], [], [], []
    for t in range(1, self.T+1):
      train_acc25.append(self.ada25(train25, train_labels25, self.n, self.lbd))
      train_acc27.append(self.ada27(train27, train_labels27, self.n, self.lbd))
      train_acc57.append(self.ada57(train57, train_labels57, self.n, self.lbd))

      # compute the test accuracy
      # 1) get the result from 3 svm models
      test_x = self.test_data.reshape(self.test_data.shape[0], self.test_data[0].size) / 255
      margin25, margin27, margin57 = np.matmul(test_x, self.w25), np.matmul(test_x, self.w27), np.matmul(test_x, self.w57)
      output25, output27, output57 = np.sign(margin25), np.sign(margin27), np.sign(margin57) 
      # 2) transform the output from -1,1 to respective labels(2,5,7)
      output25[np.argwhere(output25==-1)], output25[np.argwhere(output25==1)] = 2, 5
      output27[np.argwhere(output27==-1)], output27[np.argwhere(output27==1)] = 2, 7
      output57[np.argwhere(output57==-1)], output57[np.argwhere(output57==1)] = 5, 7
      # 3) merge the label and get the mode
      output_merge = np.zeros((3, len(output25)), dtype=int)
      output_merge[0,:], output_merge[1,:], output_merge[2,:] = output25, output27, output57
      predict = stats.mode(output_merge)[0][0]
      # 4) compare our predict with the real label
      correct = np.where(predict==self.test_label)[0].size
      test_acc.append(correct/len(self.test_label))
      
    return train_acc25, train_acc27, train_acc57, test_acc


# train one-versus-one SVM
train_tri, train_labels_tri, test_tri, test_labels_tri = extract_data([2,5,7], train, train_labels, test, test_labels)
svm = ovo(train_tri, train_labels_tri, 20, 150, 0.1, test_tri, test_labels_tri)
train_acc25, train_acc27, train_acc57, test_acc_ovo = svm.train()
print(f'The final test accuracy: {test_acc_ovo[-1]}')

# plot one-versus-one SVM's training accuracy & test accuracy v.s number of iteration
plt.plot(1-np.array(train_acc25))
plt.xticks(range(len(train_acc25)))
plt.xlabel('iteration')
plt.ylabel('training error')
plt.title('training error of SVM with label 2,5')
plt.show()
plt.savefig('svm25')
plt.plot(1-np.array(train_acc27))
plt.xticks(range(len(train_acc27)))
plt.xlabel('iteration')
plt.ylabel('training error')
plt.title('training error of SVM with label 2,7')
plt.show()
plt.savefig('svm27')
plt.plot(1-np.array(train_acc57))
plt.xticks(range(len(train_acc57)))
plt.xlabel('iteration')
plt.ylabel('training error')
plt.title('training error of SVM with label 5,7')
plt.show()
plt.savefig('svm57')
plt.plot(1-np.array(test_acc_ovo))
plt.xticks(range(len(test_acc_ovo)))
plt.xlabel('iteration')
plt.ylabel('test error')
plt.title('test error of ovo SVM')
plt.show()
plt.savefig('ovo')


***************************************Establish CNN for multi-class classification*********************************************

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

# prepare the dataset with label 2,5,7
train_tri, train_labels_tri, test_tri, test_labels_tri = extract_data([2,5,7], train, train_labels, test, test_labels)
train_labels_tri[np.argwhere(train_labels_tri==2)], test_labels_tri[np.argwhere(test_labels_tri==2)] = 0, 0
train_labels_tri[np.argwhere(train_labels_tri==5)], test_labels_tri[np.argwhere(test_labels_tri==5)] = 1, 1
train_labels_tri[np.argwhere(train_labels_tri==7)], test_labels_tri[np.argwhere(test_labels_tri==7)] = 2, 2

# pack dataset in mini batch
train_dataset = TensorDataset(torch.tensor(train_tri).float(), torch.tensor(train_labels_tri).long())
batch_size = 20
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)


class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=1, padding=0)
    self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=1, padding=0)  
    self.l1 = nn.Linear(16*26*26, 64)
    self.l2 = nn.Linear(64, 3)

  def forward(self, x):
    x = F.relu(self.c1(x))
    x = self.p1(x)
    x = F.relu(self.c2(x))
    x = self.p2(x)
    x = x.view(-1,16*26*26)
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return x

model = CNN()
summary(model, (1,28,28))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1
train_err, test_err = [], []
for epoch in range(epochs):
  correct = 0
  total = 0
  for i,(x,y) in enumerate(train_dataloader):
    x = x.unsqueeze(1).float()
    optimizer.zero_grad()

    output = model(x)   
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    _,predict = torch.max(output,1)
    correct += (predict==y).sum().item()
    total += y.size(0)
    train_err.append(1-correct/total)

    with torch.no_grad():
      test_x, test_y = torch.tensor(test_tri).float().unsqueeze(1), torch.tensor(test_labels_tri).long()
      test_output = model(test_x)
      _,predict = torch.max(test_output,1)
      corr = (predict==test_y).sum().item()
      test_err.append(1-corr/test_y.size(0))

    if i % 20 == 0:
      print('Epoch: [{}|{}: {}], Training accuracy: {:.4f}, Test accuracy: {:.4f}'.format(epoch+1, epochs, i, 1-train_err[-1], 1-test_err[-1]))
print('Done')

# plot CNN's training error & test error v.s number of iteration
plt.plot(train_err[:20], label='training error')
plt.plot(test_err[:20], label='test error')
plt.xticks(range(20))
plt.xlabel('iteration')
plt.ylabel('error')
plt.title('CNN error')
plt.legend()
plt.show()
plt.savefig('CNN error')
