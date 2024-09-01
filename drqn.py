import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class QNetwork(tf.keras.Model):
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, step_size=1, name='QNetwork'):
        super(QNetwork, self).__init__(name=name)
        
        self.lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.dense1 = layers.Dense(hidden_size, activation='relu')
        self.dense2 = layers.Dense(action_size)

        self.opt = tf.keras.optimizers.Adam(learning_rate)

    def call(self, inputs):
        lstm_out, _, _ = self.lstm(inputs)
        lstm_out = lstm_out[:, -1, :]
        x = self.dense1(lstm_out)
        output = self.dense2(x)
        return output

    def compute_loss(self, Q, targetQs):
        return tf.reduce_mean(tf.square(targetQs - Q))





from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size,step_size):
        idx = np.random.choice(np.arange(len(self.buffer)-step_size), 
                               size=batch_size, replace=False)
        
        res = []                       
                             
        for i in idx:
            temp_buffer = []  
            for j in range(step_size):
                temp_buffer.append(self.buffer[i+j])
            res.append(temp_buffer)
        return res    
        

