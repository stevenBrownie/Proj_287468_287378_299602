#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep learning - Miniproject 2 - Model

Steven Brown, Guillaume Briand and Paulin de Schoulepnikoff
"""

# Importations Modules + torch + pickle

from .modules.py import Conv2d
from .modules.py import ReLU
from .modules.py import Sequential
from .modules.py import Sigmoid
from .modules.py import NearestUpsampling
from .modules.py import MSELoss
from .modules.py import SGD
import torch
#from torch.nn.functional import fold, unfold
import pickle

class Model():
  def __init__(self) -> None:

    self.model = Sequential(Conv2d(3,32,stride=2),
                            ReLU(),
                            Conv2d(32,64,stride=2),
                            ReLU(),
                            NearestUpsampling(2), #by default upsampling *2 the dimension is squared
                            Conv2d(64,32,kernel_size=2,stride=1,padding=1),     #conv must match the dimension
                            ReLU(),
                            NearestUpsampling(2),
                            Conv2d(32,3,kernel_size=3,stride=1),
                            Sigmoid())

    self.criterion = MSELoss(self.model) #creates andlinks the MSELoss too the model

    # Stochastic Gradient Descent (SGD) optimizer
    self.learning_rate = 0.05
    self.optimizer = SGD(self.model,lr=self.learning_rate)    

  def load_pretrained_model(self) -> None:
    ## This loads the parameters saved in bestmodel.pth into the model
    filename = 'bestmodel.pth'
    infile = open(filename,'rb')
    param_list = pickle.load(infile)
    infile.close()
    self.model.set_params(param_list)
    
  def save_model(self) -> None:
    filename = 'bestmodel.pth'
    outfile = open(filename,'wb')
    pickle.dump(self.model.params(),outfile)
    outfile.close()

  def train(self, train_input, train_target, nb_epochs=10, mb_size = 4, print_evolution = False) -> None:
    '''
    train_input:      tensor of size (N, C, H, W) containing a noisy version of the images.
    train_target:     tensor of size (N, C, H, W) containing another noisy version of the
                      same images, which only differs from the input by their noise.
    nb_epochs:        number of epochs to train the model
    mb_size:          minibatch size
    print_evolution:  bool, if True, the loss and the number of epoch will be printed during
                      the training 
    '''
    # if the data in in range: 0-255, we normalize them
    if train_input.max() > 200:
        train_input = train_input / 255.

    self.loss_history = train_input.new_zeros(nb_epochs) #this creates new tensor with space position as train inout (CPU/GPU)
   
    for e in range(nb_epochs):
      for b in range(0, train_input.size(0), mb_size):
          output = self.model(train_input[b:b + mb_size])
          loss = self.criterion(output, train_target[b:b + mb_size])
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

      self.loss_history[e] = loss.loss
      if print_evolution:
        print("======> epoch: {}/{}, Loss:{}".format(e+1,nb_epochs,loss.loss))

  def predict(self, test_input) -> torch.Tensor:
    '''
    test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained
                or the loaded network
    return:     tensor of the size (N1, C, H, W)
    '''
    if test_input.max() > 200:  # if the data are not normalized
        test_input = test_input / 255.
    predicted_tensor = self.model.forward(test_input) * 255.
    return (predicted_tensor > 0)*predicted_tensor  + 1e-13  # we want positive prediction

