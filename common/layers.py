#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from common.functions import softmax, cross_entropy_error, sigmoid


# In[2]:


# 배열 곱
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.param
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.t)
        dW = np.dout(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# In[6]:


class Affine:
    '''
    x shape (batch_size, hidden_size)
    W shape (hidden_size, 1)
    b shape (1)
    out shape (batch_size, 1)

    dout shape (batch_size) -> (batch_size, 1)
    dx shape (4, 128)
    dW shape (128, 1)
    db shape (1,)
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dout = dout.reshape(-1, 1)
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        # print('Affine dout shape', dout.shape)
        # print('dx shape', dx.shape)
        # print('dW shape', dW.shape)
        # print('db shape', db.shape)


        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# In[7]:


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


# In[8]:


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis = 1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy() 
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


# In[11]:


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# In[12]:


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.y = self.y.reshape(t.shape[0])

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # print('y shape', self.y.shape)
        # print('t shape', self.t.shape)
        # print('y', self.y)
        # print('t', self.t)
        # print('y-t', (self.y-self.t).shape)

        dx = (self.y - self.t) * dout / batch_size
        return dx

# %%
class WeightedSigmoidWithLoss:
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        self.params, self.grads = [], []
        self.loss = None
        self.pos_weight = pos_weight   # 장애아동의 가중치
        self.neg_weight = neg_weight   # 비장애아동의 가중치
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.y = self.y.reshape(t.shape[0])
        # 샘플별 가중치 조정
        self.sample_weight = np.where(self.t == 1,
                                      self.pos_weight,
                                      self.neg_weight)
        
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        # 가중치 반영
        dx = (self.y - self.t)
        dx *= self.sample_weight
        dx = dx * dout / batch_size

        return dx
# In[16]:


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# In[17]:


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None

