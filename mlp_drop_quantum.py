# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:36:36 2019

@author: valter.e.junior
"""

import numpy as np
from random import sample
import pandas as pd
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())


class Mlp():
    
    def __init__(self, size_layers, act_funct='sigmoid', reg_lambda=0, bias_flag=True,dropout=0.2,type_dropout='classic'):
        
        self.size_layers = size_layers
        self.size_layers_mod = size_layers
        self.n_layers    = len(size_layers)
        self.act_f       = act_funct
        self.lambda_r    = reg_lambda
        self.bias_flag   = bias_flag
 
        # Ramdomly initialize theta (MLP weights)
        self.initialize_theta_weights()
        self.perc_dropout=dropout
        self.index_dropout=[]
        #self.theta_weights_mod=[]
        self.type_dropout=type_dropout
        
    def train(self,X, Y, iterations=400, reset=False):
    
        if reset:
            self.initialize_theta_weights()
        for iteration in range(iterations):
            gradients,index_dropout,theta_weights_mod = self.backpropagation(X, Y)
            gradients_vector = self.unroll_weights(gradients)
            theta_vector = self.unroll_weights(theta_weights_mod)
            theta_vector = theta_vector - gradients_vector
            self.size_layers_mod=self.size_layers.copy()
            for j in range(len(theta_weights_mod)):
                if j>0:
                    self.size_layers_mod[j]=theta_weights_mod[j-1].shape[0]
            
                    
            theta_weights_mod = self.roll_weights(theta_vector)
            
            for i in index_dropout:
                if i==0:
                    index=np.where(np.isin(np.array(range(self.theta_weights[i].shape[0])),np.array(index_dropout[i]))!=True)[0]
                    self.theta_weights[i][index,:]=theta_weights_mod[i]
                elif i>0 and i<(len(index_dropout)-1):
                    index=np.where(np.isin(np.array(range(self.theta_weights[i].shape[0])),np.array(index_dropout[i]))!=True)[0]
                    index1=np.where(np.isin(np.array(range(self.theta_weights[i].shape[1])),np.array(index_dropout[i-1])+1)!=True)[0]
                    for j,value1 in enumerate(index):
                        for k,value2 in enumerate(index1):
                            self.theta_weights[i][value1,value2]=theta_weights_mod[i][j,k]
                            
                elif i==(len(index_dropout)-1):
                    index1=np.where(np.isin(np.array(range(self.theta_weights[i].shape[1])),np.array(index_dropout[i-1])+1)!=True)[0]
                    self.theta_weights[i][:,index1]=theta_weights_mod[i]
            print(self.theta_weights)
            
    def predict(self,X):
   
        A , Z = self.feedforward_pred(X)
        Y_hat = A[-1]
        return Y_hat
    
    def product_notable(self,A_eq,b_eq,var):
        contrainst=[]
        features=[]
        for index,k in enumerate(A_eq):
            var1=pd.Series(var)
            map_p=list(var1[np.where(np.array(k)!=0)[0]])
            g=list(pd.Series(k)[np.where(np.array(k)!=0)[0]])
            g.append(b_eq[index]*-1)
            map_p.append('c')
            g1=[]
            tipo=[]
            for i in range(len(g)):
                for j in range(len(g)):
                    g1.append(g[i]*g[j])
                    if i==j and map_p[i]!='c':
                        tipo.append(map_p[i])
                    elif i!=j and map_p[i]=='c' and map_p[j]!='c':
                        tipo.append(map_p[j])
                    elif i!=j and map_p[i]!='c' and map_p[j]=='c':
                        tipo.append(map_p[i])
                    else:
                        value=[map_p[i],map_p[j]]
                        value.sort()
                        value='||'.join(value)
                        tipo.append(value)
                
            
            
            g1=pd.Series(g1)  
            dt=pd.DataFrame({'var':tipo,'value':g1}).groupby('var').sum()
            value=list(dt['value'])
            maping=list(dt.index)
            
      
            contrainst.extend(value)
            features.extend(maping)
            #print(index)
            
    
        dt=pd.DataFrame({'var':features,'value':contrainst}).groupby('var').sum()
        value=list(dt['value'])
        maping=list(dt.index)
    
         
        
        if value.count(0)>0:
           value=list(pd.Series(value)[np.where(np.array(value)!=0)[0]])
           maping=list(pd.Series(maping)[np.where(np.array(value)!=0)[0]])
        return maping,value
        
    def model_dwave(self,maping,value,const=False):
        Q={}
        if const==False:
            index=np.where(np.array(maping)=='c||c')[0][0]
            maping.pop(index)
            value.pop(index)
        for i,values in enumerate(maping):
            if values.count('||')==1:
                Q[(values[0:values.find('||')],values[values.find('||')+2:len(values)])]=value[i]
            else:
                Q[(values,values)]=value[i]
        return Q
    
    def result_dwave(self,response,var):
        amostras=[]
        energy=[]
        occurences=[]
        
        for datum in response.data(['sample', 'energy', 'num_occurrences']):
            amostras.append(datum.sample)
            energy.append(datum.energy)
            occurences.append(datum.num_occurrences)
        
        data={}
        data['samples']=amostras
        data['energy']=energy
        data['ocurrences']=occurences
        data=pd.DataFrame(data)
        data=data[data['energy']==min(data['energy'])]
        
        drop=[int(np.where(value==np.array(var))[0]) for i,value in enumerate(data['samples'][0]) if data['samples'][0][value]==0]
        
                 
        return drop
    
    def dropout_dwave(self,value,dropout=0.2):
      
        A_eq=[list(j) for j in value.transpose()]
        var=['neuron_'+str(j) for j in range(value.shape[0])]
        b_eq=[sum(j)*(1-dropout) for j in A_eq ]
        g=self.product_notable(A_eq=A_eq,b_eq=b_eq,var=var)
        Q=self.model_dwave(g[0],g[1],const=False)
        response = sampler.sample_qubo(Q, num_reads=1000)
        result_neuron=self.result_dwave(response,var)
        return result_neuron
    
    def dropout_random(self,x,split_add=0.8):
        
        select_neurons=sample(list(range(x)),round(split_add*x))
        select_neurons.sort()
        return select_neurons
    
    
    def initialize_theta_weights(self):
    	
    	self.theta_weights = []
    	size_next_layers = self.size_layers.copy()
    	size_next_layers.pop(0)
    	for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
    		if self.act_f == 'sigmoid':
    			
    			epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
    			
    			if self.bias_flag:  
    				theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer + 1) * 2.0 ) - 1)
    			else:
    				theta_tmp = epsilon * ( (np.random.rand(size_next_layer, size_layer) * 2.0 ) - 1)            
    		elif self.act_f == 'relu':
    			
    			epsilon = np.sqrt(2.0 / (size_layer * size_next_layer) )
    			
    			if self.bias_flag:
    				theta_tmp = epsilon * (np.random.randn(size_next_layer, size_layer + 1 ))
    			else:
    				theta_tmp = epsilon * (np.random.randn(size_next_layer, size_layer))                    
    		self.theta_weights.append(theta_tmp)
    	return self.theta_weights
    
    def backpropagation(self, X, Y):
            
        if self.act_f == 'sigmoid':
            g_dz = lambda x: self.sigmoid_derivative(x)
        elif self.act_f == 'relu':
            g_dz = lambda x: self.relu_derivative(x)
    
        n_examples = X.shape[0]
        # Feedforward
        A, Z, index_dropout,theta_weights_mod = self.feedforward(X)
    
        # Backpropagation
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - Y
        # For the second last layer to the second one
        for ix_layer in np.arange(self.n_layers - 1 - 1 , 0 , -1):
            theta_tmp = theta_weights_mod[ix_layer]
            if self.bias_flag:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose() ) ).transpose() * g_dz(Z[ix_layer])
    
        # Compute gradients
        gradients = [None] * (self.n_layers - 1)
        for ix_layer in range(self.n_layers - 1):
            grads_tmp = np.matmul(deltas[ix_layer + 1].transpose() , A[ix_layer])
            grads_tmp = grads_tmp / n_examples
            if self.bias_flag:
                # Regularize weights, except for bias weigths
                grads_tmp[:, 1:] = grads_tmp[:, 1:] + (self.lambda_r / n_examples) * theta_weights_mod[ix_layer][:,1:]
            else:
                # Regularize ALL weights
                grads_tmp = grads_tmp + (self.lambda_r / n_examples) * theta_weights_mod[ix_layer]       
            gradients[ix_layer] = grads_tmp;
        return gradients, index_dropout,theta_weights_mod
    
    def feedforward(self, X):
            
        if self.act_f == 'sigmoid':
            g = lambda x: self.sigmoid(x)
        elif self.act_f == 'relu':
            g = lambda x: self.relu(x)

        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X
        theta_weights_mod=self.theta_weights.copy()
        save_neurons_dropout={}
        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            
            if self.bias_flag:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)
            
            if self.type_dropout=='classic':
                neuron_dropout=self.dropout_random(theta_weights_mod[ix_layer].shape[0],self.perc_dropout)
            else:
                neuron_dropout=self.dropout_dwave(theta_weights_mod[ix_layer],dropout=self.perc_dropout)
                
            if ix_layer<(self.n_layers - 2):
                save_neurons_dropout[ix_layer]=neuron_dropout
            else:
                save_neurons_dropout[ix_layer]=save_neurons_dropout[ix_layer-1]
            
            if ix_layer==0:
                theta_weights_mod[ix_layer]=np.delete(theta_weights_mod[ix_layer],neuron_dropout,0)
            elif ix_layer>0 and ix_layer<(self.n_layers - 2):
                theta_weights_mod[ix_layer]=np.delete(theta_weights_mod[ix_layer],np.array(save_neurons_dropout[ix_layer-1])+1,1)
                theta_weights_mod[ix_layer]=np.delete(theta_weights_mod[ix_layer],save_neurons_dropout[ix_layer],0)
            elif ix_layer==(self.n_layers - 2):
                theta_weights_mod[ix_layer]=np.delete(theta_weights_mod[ix_layer],np.array(save_neurons_dropout[ix_layer])+1,1)
                   
            A[ix_layer] = input_layer
            # Multiplying input_layer by theta_weights for this layer
            Z[ix_layer + 1] = np.matmul(input_layer,  theta_weights_mod[ix_layer].transpose() )
            # Activation Function
            output_layer = g(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        return A, Z, save_neurons_dropout,theta_weights_mod
        
        
    def feedforward_pred(self,X):
            
        if self.act_f == 'sigmoid':
            g = lambda x: self.sigmoid(x)
        elif self.act_f == 'relu':
            g = lambda x: self.relu(x)

        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X
        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            
            if self.bias_flag:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate((np.ones([n_examples ,1]) ,input_layer), axis=1)
            
            A[ix_layer] = input_layer
            # Multiplying input_layer by theta_weights for this layer
            Z[ix_layer + 1] = np.matmul(input_layer,  self.theta_weights[ix_layer].transpose() )
            # Activation Function
            output_layer = g(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        return A, Z
    
    
    def unroll_weights(self, rolled_data):
    	
    	unrolled_array = np.array([])
    	for one_layer in rolled_data:
    		unrolled_array = np.concatenate((unrolled_array, one_layer.flatten(1)) )
    	return unrolled_array
    
    def roll_weights(self, unrolled_data):
    	
    	size_next_layers = self.size_layers_mod.copy()
    	size_next_layers.pop(0)
    	rolled_list = []
    	if self.bias_flag:
    		extra_item = 1
    	else:
    		extra_item = 0
    	for size_layer, size_next_layer in zip(self.size_layers_mod, size_next_layers):
    		n_weights = (size_next_layer * (size_layer + extra_item))
    		data_tmp = unrolled_data[0 : n_weights]
    		data_tmp = data_tmp.reshape(size_next_layer, (size_layer + extra_item), order = 'F')
    		rolled_list.append(data_tmp)
    		unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
    	return rolled_list    
        
    def sigmoid(self, z):
    	
    	result = 1.0 / (1.0 + np.exp(-z))
    	return result
    
    def relu(self, z):
    	
    	if np.isscalar(z):
    		result = np.max((z, 0))
    	else:
    		zero_aux = np.zeros(z.shape)
    		meta_z = np.stack((z , zero_aux), axis = -1)
    		result = np.max(meta_z, axis = -1)
    	return result
    
    def sigmoid_derivative(self, z):
    	
    	result = self.sigmoid(z) * (1 - self.sigmoid(z))
    	return result
    
    def relu_derivative(self, z):
    	
    	result = 1 * (z > 0)
    	return result
        
        