# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:26:21 2018

@author: Amichayf
"""
import numpy as np


def generate_data():
    X = np.matrix([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    Y = np.logical_xor(np.logical_xor(X[:,0],X[:,1]) , X[:,2])*1  # 0-even.  1 -odd
    
    return X,Y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def derivative_of_sigmoid(x):
    return np.multiply(sigmoid(x) , (1 - sigmoid(x)))




    
    

class model:
    
    def __init__(self):

#        self.input,self.labels = generate_data()
        
        self.w1 = np.random.randn(4,3)
        self.w2 =  np.random.randn(4,1)
        self.Z=None
        self.a=None
        self.y=None
        self.y_hat=None
        self.loss=0
        self.loss_derivative=0
        self.loss_according_to_W2=0
        self.loss_according_to_W1=0
        
    def feed_forward(self,X):
        
        self.Z =  np.matmul( X , (self.w1))
        self.a = np.concatenate( (np.ones((8,1)) , sigmoid(self.Z)) , axis=1)
        
        self.y = np.matmul( self.a , self.w2 )
        self.y_hat = sigmoid(self.y)
        
        
        
    def batch_gradient_descent_step(self,X):
        
        
        #W2
        self.loss_according_to_W2 = np.multiply(self.loss_derivative , np.matmul( (self.a).transpose() ,  derivative_of_sigmoid(self.y)  ))
        
        #W1
#        s=derivative_of_sigmoid(self.Z)
#        help_matrix = np.matrix([ [0,0,0],[s[0],0,0],[0,s[1],0],[0,0,s[2]]  ])
#        temp1 = np.matmul( (self.w2).transpose() , help_matrix) # result dim : 1X3
#        temp2=np.matmul(temp1.transpose() , X.transpose())
#        self.loss_according_to_W1 = np.multiply(self.loss_derivative , np.multiply(( derivative_of_sigmoid(self.y) ) ,  temp2.transpose()))
#        
        temp1 = np.matmul(self.w2.transpose() , np.concatenate( ((np.eye(3)) ,np.zeros((3,1)).transpose() ))) #dim : 1X3
        temp2 = np.matmul(derivative_of_sigmoid(self.y) , temp1) #dim: Nx3
        temp3 = np.multiply( temp2 , derivative_of_sigmoid(self.Z) ) #dim: Nx3 (element wise mult)
        temp4 = np.matmul( temp3.transpose(), X ).transpose()
        temp5= np.multiply (self.loss_derivative , temp4)
        #UPDATE
        self.w1 = self.w1 -2 * temp5
        self.w2 = self.w2 -2 * self.loss_according_to_W2
        
    def MSE_loss(self,t):
        self.loss =  (1/8) * np.sum(np.power((self.y_hat - t),2)) 
        self.loss_derivative =  (1/4) *np.sum(self.y_hat - t)
    
        
        
def main():
    
    
    inputs,label = generate_data()
#    data = np.concatenate( (np.concatenate((np.ones((8,1)) , inputs), axis=1) , label),axis=1)
    inputs = np.concatenate((np.ones((8,1)) , inputs), axis=1)
    
    results=np.zeros((100,2000))
    
    for row_index in range (100):
        Model =model()

    
        for epoch in range(2000):

#            l=np.arange(8)
#            np.random.shuffle(l) 
#            
            
#            
#            for example_index in l:
#                    
#                    Model.feed_forward(inputs[example_index].transpose())
#                    Model.MSE_loss(label[example_index])
            Model.feed_forward(inputs)
            Model.MSE_loss(label)
            Model.batch_gradient_descent_step(inputs)
            
            print("Iter: {} , epoch: {} , loss: {}".format(row_index,epoch,Model.loss))        
#            Model.batch_gradient_descent_step(inputs[example_index].transpose() )
            results[row_index,epoch] = Model.loss
            Model.loss=0
            Model.loss_derivative=0
            
            
            
                
            
        
        

    
    
    
    
    
    
    
    
    
    
    
    
