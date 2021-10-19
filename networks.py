import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Cov1DModel(tf.keras.Model):
    def __init__(self):
        he_initializer = tf.keras.initializers.he_normal()
        norm_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.01)
        super(Cov1DModel, self).__init__()
        self.conv1d1 = layers.Conv1D(128, kernel_size=(9,), kernel_initializer=he_initializer, padding='same', activation='relu', input_shape=(40,200,12))
        self.norm1 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)
        self.conv1d2 = layers.Conv1D(128, kernel_size=(9,), kernel_initializer=he_initializer, padding='same', activation='relu')
        self.norm2 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)
        self.conv1d3 = layers.Conv1D(1, kernel_size=(1,), kernel_initializer=he_initializer, activation='sigmoid')
        self.pooling = layers.AveragePooling1D(pool_size=10)

    def call(self, x):
        x = self.conv1d1(x)
        x = self.norm1(x)
        x = self.conv1d2(x)
        x = self.norm2(x)
        x = self.conv1d3(x)    
        x = self.pooling(x)
        return x

def single_loss(result, m1, m2):
    batch_size = result.shape[0]//2
    sum1 = 0.0
    sum2 = 0.0 
    loss = 0.0 
    l_sum = [] 
    for i in range(batch_size):
        y_pos_max = tf.math.reduce_max(result[i])
        y_neg_max = tf.math.reduce_max(result[i+batch_size])
        l = tf.math.reduce_max([0, tf.math.add(1.0, tf.math.subtract(y_neg_max, y_pos_max))]) #波形1セット分の損失
        l_sum.append(l)

    loss = tf.math.reduce_sum(l_sum)

    pos_result = np.squeeze(result[0:batch_size], axis=2).flatten()
    neg_result = np.squeeze(result[batch_size:], axis=2).flatten()
    number = []
    for k in range(len(pos_result)-1):
        number.append(tf.square(tf.math.subtract(pos_result[k], pos_result[k+1])))
    sum1 = tf.math.reduce_sum(number)
    sum2 = tf.math.reduce_sum(pos_result)
    smooth = tf.math.scalar_mul(m1, sum1)
    sparse = tf.math.scalar_mul(m2, sum2)
    lasso = tf.math.add(smooth, sparse)
    loss = tf.math.add(loss, lasso)

    return loss