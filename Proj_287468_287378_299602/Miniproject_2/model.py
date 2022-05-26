#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep learning - Miniproject 2 - Model

Steven Brown, Guillaume Briand and Paulin de Schoulepnikoff
"""

# Importations Modules + torch + pickle

import torch
from torch.nn.functional import fold, unfold
import pickle
from pathlib import Path

best_model_path=str(Path(__file__).parent)+'/bestmodel.pth'

class Model():
  def __init__(self) -> None:

    self.model = Sequential(Conv2d(3,32,stride=2),
                            ReLU(),
                            Conv2d(32,64,stride=2),
                            ReLU(),
                            TransposeConv2d(64,32,kernel_size=2,stride=1,padding=1),
                            ReLU(),
                            TransposeConv2d(32,3,kernel_size=3,stride=1),
                            Sigmoid())

    self.criterion = MSELoss(self.model) #creates andlinks the MSELoss too the model

    # Stochastic Gradient Descent (SGD) optimizer
    self.learning_rate = 0.05
    self.optimizer = SGD(self.model,lr=self.learning_rate)    

  def load_pretrained_model(self) -> None:
    ## This loads the parameters saved in bestmodel.pth into the model
    filename = best_model_path
    infile = open(filename,'rb')
    param_list = pickle.load(infile)
    infile.close()
    self.model.set_params(param_list)
    
  def save_model(self) -> None:
    filename = best_model_path
    outfile = open(filename,'wb')
    pickle.dump(self.model.params(),outfile)
    outfile.close()

  def train(self, train_input, train_target, num_epochs=10, mb_size = 4, print_evolution = False) -> None:
    '''
    train_input:      tensor of size (N, C, H, W) containing a noisy version of the images.
    train_target:     tensor of size (N, C, H, W) containing another noisy version of the
                      same images, which only differs from the input by their noise.
    num_epochs:        number of epochs to train the model
    mb_size:          minibatch size
    print_evolution:  bool, if True, the loss and the number of epoch will be printed during
                      the training 
    '''
    # if the data in in range: 0-255, we normalize them
    if train_input.type()=='torch.ByteTensor':
      train_input=train_input.float()/255.
    if train_target.type()=='torch.ByteTensor':
      train_target = train_target.float()/255.

    self.loss_history = train_input.new_zeros(num_epochs) #this creates new tensor with space position as train inout (CPU/GPU)
   
    for e in range(num_epochs):
      for b in range(0, train_input.size(0), mb_size):
          output = self.model(train_input[b:b + mb_size])
          loss = self.criterion(output, train_target[b:b + mb_size])
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

      self.loss_history[e] = loss.loss
      if print_evolution:
        print("======> epoch: {}/{}, Loss:{}".format(e+1,num_epochs,loss.loss))

  def predict(self, test_input) -> torch.Tensor:
    '''
    test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained
                or the loaded network
    return:     tensor of the size (N1, C, H, W)
    '''
   
    if test_input.type()=='torch.ByteTensor':
           test_input = test_input.float()/255.

    predicted_tensor = self.model(test_input) * 255.
    return (predicted_tensor > 0)*predicted_tensor  + 1e-13  # we want positive prediction


#############
#Modules

class Module () :
  def forward (self, input) : 
      # should get for input and returns, a tensor or a tuple of tensors
      raise NotImplementedError
  def backward (self, gradwrtoutput): 
      #should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect tothe module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input
      raise NotImplementedError 
  def param (self) : 
      #should return a list of pairs composed of a parameter tensor and a gradient tensor of the same size. This list should be empty for parameterless modules
      return []
  

# Convolution layer
  
class Conv2d (Module) :
  def __init__(self, in_channels = 3, out_channels = 3, kernel_size = 2, stride = 1, padding = 0) -> None:
    '''
    Initialization of conv layer
    '''
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = padding

    self.init_weights()

  def init_weights(self) -> None:
    '''
    Weights and bias initialization for convolution
    '''
    std = (2./ (self.kernel_size**2*self.in_channels + self.out_channels)) ** 0.5
    self.weight = torch.stack([torch.empty((self.in_channels,self.kernel_size,self.kernel_size)).normal_(0,std) for i in range(self.out_channels)])
    self.bias = torch.empty(self.out_channels).normal_(0,std)
    self.grad_weight=torch.zeros_like(self.weight)#init weight grad
    self.grad_bias=torch.zeros_like(self.bias)#init bias grad

  
  def __call__(self,input):
    return self.forward(input)

  def forward(self, input):
    '''
    Does the forward pass for a convolutional layer
    '''

    if len(input.size())==3:#makes it possible to recieve single tensor as input
      input = torch.unsqueeze(input, dim=0)

    N_ie, N_ich, N_iW, N_iH = input.size()
    # No dilation, size of output image :
    N_oW = (N_iW + 2*self.pad - self.kernel_size)//self.stride + 1
    N_oH = (N_iH + 2*self.pad - self.kernel_size)//self.stride + 1

    output = torch.empty((N_ie,N_ich,N_oW,N_oH))
    
    unfolded_input = unfold(input, self.kernel_size, stride = self.stride, padding = self.pad)
    convolution = self.weight.view(self.out_channels, -1) @ unfolded_input + self.bias.view(1, -1, 1)
    output = convolution.view(N_ie,self.out_channels, N_oW, N_oH)
    
    # conserved parameters and variables for backward pass
    self.N_oW = N_oW
    self.N_oH = N_oH
    self.input = input
    self.unfolded_input = unfolded_input
    
    return output
  
  def backward(self, gradwrtoutput):
      '''
      Does the backward pass of a convolutional layer
      Takes as input the loss wrt the input of the previous layer
      Return the loss wrt input of this layer
      Update the gradients for the weight and bias
      '''
      grad_input = grad_weight = grad_bias = None

      N_s, N_ich, N_iW, N_iH = self.input.shape

      # Gradient of bias : sum over all dimension except the one with out_channels dimension
      grad_bias = gradwrtoutput.sum(dim=[0, 2, 3])  # done for grad_bias
     
      # Gradient of weight : Using convolution between the input and the gradient wrt output
      convolution = (gradwrtoutput.view(N_s,self.out_channels,-1) @ self.unfolded_input.transpose(1,2)).sum(axis = 0)
      grad_weight = convolution.view(self.out_channels,self.in_channels, self.kernel_size, self.kernel_size)
      
      # Gradient of input X : Using full convolution between the filter and the gradient wrt output
      fullconvolution = gradwrtoutput.view(N_s,self.out_channels,-1).transpose(1,2) @ self.weight.view(self.out_channels,-1)
      fct = fullconvolution.transpose(1,2)
      grad_input = fold(fct,(N_iW, N_iH),self.kernel_size,padding = self.pad, stride = self.stride)
        
      self.grad_weight += grad_weight #add grad
      self.grad_bias += grad_bias #add grad
      
      return grad_input #grad_weight, grad_bias /!\ only grad input needs to be returned, grad_input is used to calculate recursively the grad of other modules

  def param (self) : 
      #should return a list of pairs composed of a parameter tensor and a gradient tensor of the same size. This list should be empty for parameterless modules
      return [[self.weight,self.grad_weight],[self.bias,self.grad_bias]]


  def set_params(self,new_params):
    self.weight=new_params[0]
    self.bias=new_params[1]

  def zero_grad(self):
    self.grad_weight = torch.zeros_like(self.grad_weight)
    self.grad_bias = torch.zeros_like(self.grad_bias)
    
# Nearest neighbour Upsampling

class NearestUpsampling (Module) : #This is a 2D Nearest upsampling module

  def __init__(self, upsampling_dim_power=2):
    self.dim_mult = upsampling_dim_power #this gives x of the 2^x dimension augmentation of the tensor
    
  def forward (self, input) : #foward for NearestUpsampling
    """ 
    IN:  a tensor or a vector of tensors 
    OUT: a tensor a vector of tensors
    """
   
    if len(input.size())==3:
      input = torch.unsqueeze(input, dim=0)


    N_ie, N_ich, N_iW, N_iH =input.size()
    
    N_oW = N_iW*self.dim_mult # output row length
    N_oH = N_iH*self.dim_mult # output column length
  
    out_tensor=torch.empty((N_ie,N_ich,N_oW,N_oH)) # create output tensor
     
    #fill tensor
    for i in range(N_oW):
        xi = int(i/self.dim_mult)
        for j in range(N_oH):
            xj = int(j/self.dim_mult)
            out_tensor[:,:,i,j] = input[:,:,xi,xj]

    return out_tensor

  def backward (self, gradwrtoutput): #sums the loss for each upsampled block and return the loss wrt to the input
    """ 
    IN:  a tensor of gradients wrt to output
    OUT: a tensor of gradients wrt to input
    """

    if len(gradwrtoutput.size())==3:
      gradwrtoutput = torch.unsqueeze(gradwrtoutput, dim=0) #corrects the format of ouput if it is a single tensor

    N_ie, N_ch, N_oW, N_oH =gradwrtoutput.size()
    N_iW = int(N_oW/self.dim_mult)
    N_iH = int(N_oH/self.dim_mult)
    gradwrtinput=torch.empty((N_ie, N_ch, N_iW, N_iH))
    
    for iW in range(N_iW):
        for iH in range(N_iH):
            gradwrtinput[:,:,iW,iH]=gradwrtoutput[:,:,self.dim_mult*iW:self.dim_mult*(1+iW),self.dim_mult*iH:self.dim_mult*(1+iH)].sum(dim=(2,3))
            
    return gradwrtinput

    #no paramaters
    
# TranposeConv2d - combination of NNUpsampling + conv2d

class TransposeConv2d (Module): #this module combines two modules: 1 NearestUpsampling and 1 Conv2d

  def __init__(self, in_channels = 3, out_channels = 3, kernel_size = 2, scale_factor = 2, stride = 1, padding = 0):
    self.up=NearestUpsampling(scale_factor)
    self.conv2d=Conv2d(in_channels,out_channels,kernel_size,stride,padding)

  def forward (self, input) :
   return self.conv2d.forward(self.up.forward(input))

  def backward (self, gradwrtoutput):
    return self.up.backward(self.conv2d.backward(gradwrtoutput))

  def param (self) : #return a list of pairs composed of a parameter tensor and a gradient tensor of the same size.
    return self.conv2d.param()

  def set_params(self,new_params):
    self.conv2d.weight=new_params[0]
    self.conv2d.bias=new_params[1]

  def zero_grad(self):
    self.conv2d.grad_weight = torch.zeros_like(self.conv2d.grad_weight)
    self.conv2d.grad_bias = torch.zeros_like(self.conv2d.grad_bias)

# ReLU

class ReLU (Module): #This is a 2D Nearest upsampling module
  '''
  IN:  a tensor of image data or a vector of tensors 
  OUT: a tensor of image or a vector of tensors
  '''
  def __init__(self) -> None:
    self.input = None
    #no grad needed since no parameters. The grad variable is only used by optimiser thus only necessary if has parameters

  def forward (self, input):
    self.input = input #rememeber the input, needed for the backward
    return (input>0)*input #max(input,0)

  def backward (self, gradwrtoutput):
    return (self.input>0)*gradwrtoutput #grad if input>0, 0 otherwise

# Sigmoid

class Sigmoid (Module):
    def __init__(self):
        self.input = None
    def forward (self, input):
        self.input = input
        output = 1/(1 + torch.exp(-self.input))
        return output
    def backward (self, gradwrtoutput):
        sig = 1/(1 + torch.exp(-self.input))
        dL = gradwrtoutput * sig * (1-sig)
        return dL
    
# Sequential

class Sequential(Module):
 
  def __init__(self,*argv):
   
    self.modules_list=[]

    for arg in argv:
      self.modules_list.append(arg)

  def forward(self , input):
    output = input
    for mod in self.modules_list:
      output = mod.forward(output)
    return output

  def backward(self , gradwrtoutput) -> None:
    output = gradwrtoutput
    for mod in reversed(self.modules_list): # reversed order as backward takes gradwrtouput as argument
      output=mod.backward(output)  #update ouput
      
  def __call__(self,input):
    return self.forward(input)

  def params(self): 
    params = []
    for mod in self.modules_list:
      if mod.param() !=[]: #if empty list we don't add it the the list of params
        params.append(mod.param())
    return params

  def set_params(self,param_list) -> None:
    i = 0
    for mod in self.modules_list:
      if mod.param() !=[]: #if empty list, module don't have weights and bias
        mod.weight = param_list[i][0][0]
        mod.bias = param_list[i][1][0]
        i += 1
        
# MSELoss - Mean squared error loss

class MSELoss (Module):

  def __init__(self,sequential): #connect loss to sequential
    
    self.sequential=sequential #gives acces of the loss to the sequential
    self.model_output=None #model_output
    self.target=None
      
  def __call__(self,model_output,target):
    self.model_output=model_output
    self.target=target
    try:
      self.loss=torch.mean((model_output-target)**2) #mean over every pixel
    except:
      print('model ouput size = '+str(model_output.size())) #error if model ouput size!= target size
      print('target size = '+ str(target.size()))
    return self

  def dMSELoss(self,model_output,target):
    return 2*(model_output-target)/torch.prod(torch.tensor(model_output.size()))  #mean over every pixel

  def backward (self) -> None:
   
    dloss=self.dMSELoss(self.model_output,self.target)
    self.sequential.backward(dloss) #dloss is the gradwrtoutput of sequential
    
# SGD - Stochastic gradient descent 

class SGD ():
  def __init__(self,model,lr = 0.05) -> None :
    self.model = model
    self.lr = lr

  def zero_grad(self) -> None:
     for module in self.model.modules_list:
       if module.param()!=[]:
          module.zero_grad()

  def step(self) -> None: #need to updates the params and the grad before doing a step
    for module in self.model.modules_list:
      if module.param()!=[]:
        new_params=[]
        for param in module.param():
          new_params.append(param[0]-self.lr*param[1]) #[0] is param and [1] is grad

        module.set_params(new_params)

