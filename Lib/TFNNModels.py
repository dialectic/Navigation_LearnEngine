import numpy as np
import tensorflow as tf
import keras
import os as os
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers import schedules 
from tensorflow.keras import regularizers



class TFNNModels:
    def __init__(self):
        print("Created NN Model Instance")

    

    def DNN_FFSeqSGD(self,Learn_Rate_Schedule = True):
        model = keras.Sequential()
        #model.add(layers.Flatten())
        model.add(layers.Dense(6,activation='relu')   ) # extra layer 1  ,input_shape=(train_input.shape[1],)
        model.add(layers.Dense(24,activation='relu')   )# extra layer 2
        model.add(layers.Dense(48,activation='relu')   )# extra layer 3
        model.add(layers.Dense(24,activation='relu')   )# extra layer 2
        model.add(layers.Dense(36,activation='elu')   ) # extra layer 3
        model.add(layers.Dense(1)) # extra layer 4
        
        if(Learn_Rate_Schedule):
                        
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2,
                decay_steps=10000,
                decay_rate=0.9)
        else:

            lr_schedule = .01
        

        model.compile(
            optimizer= tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=0.2,
            nesterov=False,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            name="SGD"),
            loss='mse', metrics=['mean_squared_error'],)
        return model


    def DNN_FFSeqSGD_Regulized(self,Learn_Rate_Schedule = True):
        model = keras.Sequential()
        #model.add(layers.Flatten())
        model.add(layers.Dense(6,activation='relu') ) # extra layer 1  ,input_shape=(train_input.shape[1],)
        model.add(layers.Dense(24,activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),  bias_regularizer=regularizers.L2(1e-4), activity_regularizer=regularizers.L2(1e-5)) )# extra layer 2
        model.add(layers.Dense(48,activation='relu',activity_regularizer=regularizers.L2(1e-5)) )# extra layer 3
        model.add(layers.Dense(24,activation='relu',activity_regularizer=regularizers.L2(1e-5)) )# extra layer 2
        model.add(layers.Dense(36,activation='elu')   ) # extra layer 3
        model.add(layers.Dense(1)) # extra layer 4
        
        if(Learn_Rate_Schedule):
                        
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2,
                decay_steps=10000,
                decay_rate=0.9)
        else:

            lr_schedule = .01
        

        model.compile(
            optimizer= tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=0.2,
            nesterov=False,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            name="SGD"),
            loss='mse', metrics=['mean_squared_error'],)
        return model
    


    def PredictMidPoint(self,model1x,model1y,model2x,model2y,input,maxlen):
        m1x = model1x.predict(input[0:maxlen])
        m1y = model1y.predict(input[0:maxlen])
        m2x = model2x.predict(input[0:maxlen])
        m2y = model2y.predict(input[0:maxlen])
        
        
        return m1x,m1y,m2x,m2y

    def PredictMidArray(self,model1x,model1y,model2x,model2y,input,maxlen):
        m1x,m1y,m2x,m2y = self.PredictMidPoint(model1x,model1y,model2x,model2y,input,maxlen)
        
        dim = 4 # m1x,m1y,m2x,m2y 
        MidPoints = np.zeros([maxlen,dim])
        
        for i in range(0,maxlen):
            MidPoints[i,0] = m1x[i]
            MidPoints[i,1] = m1y[i]
            MidPoints[i,2] = m2x[i]
            MidPoints[i,3] = m2y[i]
        
        return MidPoints   
    