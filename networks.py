import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
#from tensorflow.python.keras.layers import Layer

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

class AttentionCov1DModel(tf.keras.Model):
    def __init__(self):
        he_initializer = tf.keras.initializers.he_normal()
        norm_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.01)
        super(AttentionCov1DModel, self).__init__()
        self.conv1d1 = layers.Conv1D(128, kernel_size=(9,), kernel_initializer=he_initializer, padding='same', activation='relu', input_shape=(40,200,12))
        self.norm1 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)
        self.conv1d2 = layers.Conv1D(128, kernel_size=(9,), kernel_initializer=he_initializer, padding='same', activation='relu')
        self.norm2 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)
        self.conv1d3 = layers.Conv1D(1, kernel_size=(1,), kernel_initializer=he_initializer, activation='sigmoid')
        self.pooling = layers.AveragePooling1D(pool_size=10)
        self.attention1 = Attention()

    def call(self, x):
        x = self.conv1d1(x)
        x = self.norm1(x)
        x = self.conv1d2(x)
        x = self.norm2(x)
        x = self.attention1(x) 
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

def create_padding_mask(input):
    input = tf.cast(tf.math.equal(input,0),tf.float32)
    input = input[:,:,0]
    input = input[:,tf.newaxis,tf.newaxis,:]
    mask = tf.tile(input,multiples=(1,1,input.shape[-1],1))
    return mask

def self_attention(q,k,v,mask,d_model): #d_model vector_size
    qk = tf.matmul(q,k,transpose_b = True) #(batch*seq_len*seq_len)
    scaled_qk = tf.divide(qk,tf.math.sqrt(tf.cast(d_model,tf.float32)))

    if mask is not None:
        scaled_qk += (mask*-1e9)
    attention_weights = tf.nn.softmax(scaled_qk,axis=-1)

    final_output = tf.matmul(attention_weights,v) #(batch*seq_len*vocab_Size)

    return attention_weights,final_output

class MultiHeadAttention(tf.keras.layers.Layer): #vector size = d_model--third dimension
    def __init__(self,num_heads,d_model):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads ==0 

        self.depth = self.d_model//self.num_heads
        #dense layer
        self.wq = tf.keras.layers.Dense(units = d_model,trainable = True)
        self.wk = tf.keras.layers.Dense(units = d_model,trainable = True)
        self.wv = tf.keras.layers.Dense(units = d_model,trainable = True)
        self.dense1 = tf.keras.layers.Dense(units = d_model,trainable = True)

#input--input data generate weight and then divide into multi-heads part for calculate different attention 
    def split_into_heads(self,input,batch_size):

        input = tf.reshape(input,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(input,perm = [0,2,1,3]) # batch size, heads, len, vector size
    
    def call(self,input,mask):
        batch_size = input.shape[0]

        #linear layer to generate the weight matrix
        q = self.wq(input)
        k = self.wk(input)
        v = self.wq(input)

        #splitting into heads
        q = self.split_into_heads(q,batch_size)
        k = self.split_into_heads(k,batch_size)
        v = self.split_into_heads(v,batch_size)

        #mask = self.split_into_heads(mask,batch_size)

        attention_weights,multi_head_output = self_attention(q,k,v,mask,self.depth) #for every head

        pre_concat_multi_head_output =  tf.transpose(multi_head_output,perm = [0,2,1,3])
        
        post_concat_multi_head_output =  tf.reshape(pre_concat_multi_head_output,(batch_size,-1,self.d_model))

        final_output = self.dense1(post_concat_multi_head_output)

        return attention_weights,final_output

class MultiHeadAttention2(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model):
        super(MultiHeadAttention2,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads ==0 

        self.depth = self.d_model//self.num_heads

        self.wq = tf.keras.layers.Dense(units = d_model,trainable = False)
        self.wk = tf.keras.layers.Dense(units = d_model,trainable = False)
        self.wv = tf.keras.layers.Dense(units = d_model,trainable = False)
        self.dense1 = tf.keras.layers.Dense(units = d_model,trainable = False)

    #x--input series divide into heads part, add the dimension of the turn of the head
    def split_into_heads(self,x,batch_size):

        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(x,perm = [0,2,1,3])

    def call(self,enc_output,dec_input,mask):
        batch_size = enc_output.shape[0]
        
        #linear layer 
        self.q = self.wq(dec_input)
        self.k = self.wk(enc_output)
        self.v = self.wq(enc_output)

        #splitting into heads
        q = self.split_into_heads(q,batch_size)
        k = self.split_into_heads(k,batch_size)
        v = self.split_into_heads(v,batch_size)
        #mask = self.split_into_heads(mask,batch_size)
        attention_weights,multi_head_output = self_attention(q,k,v,mask,self.depth) #for every head

        pre_concat_multi_head_output =  tf.transpose(multi_head_output,perm = [0,2,1,3])
        post_concat_multi_head_output =  tf.reshape(pre_concat_multi_head_output,(batch_size,-1,self.d_model))
        
        final_output = self.dense1(post_concat_multi_head_output)

        return attention_weights,final_output

#embedding --similarity, without the positional distance information. Thus add the position on embeddings
class get_positional_encoding(tf.keras.layers.Layer):
    def __init__(self,position,d_model):
        super(get_positional_encoding,self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position,d_model)
    
    def get_angle(self,position,i,d_model):
        angle = 1 / tf.pow(10000, (2*(i//2))/ tf.cast(d_model,tf.float32))
        return position * angle

    def positional_encoding(self, position, d_model): #d_model = 512 if 128
        angle_final = self.get_angle(tf.expand_dims(tf.range(tf.cast(position,tf.float32)),axis = 1),tf.expand_dims(tf.range(tf.cast(d_model,tf.float32)),axis = 0),d_model)
        np_angle_final = angle_final.numpy()
        np_angle_final[:,0::2] = np.sin(np_angle_final[:,0::2])
        np_angle_final[:,1::2] = np.cos(np_angle_final[:,1::2])
        angle_final = tf.convert_to_tensor(np_angle_final)
        pos_encoding = angle_final[tf.newaxis,...]
        return tf.cast(pos_encoding,dtype=tf.float32)
    
    def call(self,input):
        return input + self.pos_encoding[:, :tf.shape(input)[1], : ]

#hidden_layer_shape--units(how many to enlarge) generally, d_model*4 
class FeedForward_Layer(tf.keras.layers.Layer): 
    def __init__(self,d_model,hidden_layer_shape):
        super(FeedForward_Layer,self).__init__()
        self.d_model = d_model
        self.hidden_layer_shape = hidden_layer_shape
        self.dense1 = tf.keras.layers.Dense(hidden_layer_shape, activation='relu') # (batch_size, seq_len, dff) relu-->gelu
        self.dense2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model) normalization done on the final d_model axis

    def call(self,input):
        output = self.dense1(input)
        final_output = self.dense2(output)
        return final_output

class Encoding_Layer(tf.keras.layers.Layer): #LN
    def __init__(self,num_heads,d_model,hidden_layer_shape,rate=0.1):
        super(Encoding_Layer,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(num_heads,d_model)
        self.ffd_net = FeedForward_Layer(d_model,hidden_layer_shape)
        #        self.norm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        #        self.norm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        norm_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.01)
        self.norm1 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)
        self.norm2 = layers.BatchNormalization(momentum=0.90, epsilon=10**-5, gamma_initializer=norm_initializer)

    def call(self,input,mask,training = True):
        #part 1
        _,multihead_output = self.multi_head_attention(input,mask)
        dropout_multihead_output = self.dropout1(multihead_output,training = training)
        skip_dropout_output = tf.add(input,dropout_multihead_output)
        norm1_skip_output  = self.norm1(skip_dropout_output)        
        #part 2
        point_norm1_output = self.ffd_net(norm1_skip_output)
        dropout_point_output = self.dropout2(point_norm1_output,training = training)
        skip_dropout_output = tf.add(dropout_point_output,norm1_skip_output)
        norm2_skip_output = self.norm2(skip_dropout_output)
        return norm2_skip_output

class Decoding_Layer(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model,hidden_layer_shape,rate = 0.1):
        super(Decoding_Layer,self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(num_heads,d_model)
        self.multi_head_attention2 = MultiHeadAttention2(num_heads,d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6) #mean=1 variance=0
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.point_ffd_net = FeedForward_Layer(d_model,hidden_layer_shape)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self,enc_output,dec_input,look_ahead_mask,padding_mask,training = True):
        #decoder attention with look ahead mask(include padding mask)
        # enc_output.shape --> (batch_size, input_seq_len, d_model)
        _,multihead1_output = self.multi_head_attention1(dec_input,look_ahead_mask)
        dropout1_multihead1_output = self.dropout1(multihead1_output)
        skip1_dropout1_output = tf.add(dropout1_multihead1_output,dec_input)
        norm1_skip1_output = self.norm1(skip1_dropout1_output)

        #decoder&encoder attention with padding mask
        _,multihead2_output  = self.multi_head_attention2(enc_output,dec_input,padding_mask)
        dropout2_multihead2_output = self.dropout2(multihead2_output)
        skip2_dropout2_output = tf.add(dropout2_multihead2_output,norm1_skip1_output)
        norm2_skip2_output = self.norm2(skip2_dropout2_output)

        #sum and normalize
        point_norm2_output = self.point_ffd_net(norm2_skip2_output)
        dropout3_point_output = self.dropout3(point_norm2_output)
        skip3_dropout3_output = tf.add(norm2_skip2_output,dropout3_point_output)
        norm3_skip3_output = self.norm3(skip3_dropout3_output)
        return norm3_skip3_output

#without embeddings
class Encoder(tf.keras.Model):
    def __init__(self,d_model,num_heads,hidden_layer_shape,max_pos_encoding,num_enc_layers,rate):
        super(Encoder,self).__init__()
        self.max_pos_encoding = max_pos_encoding #max_pos_encoding--longest position 
        self.num_enc_layers = num_enc_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_shape = hidden_layer_shape
        self.position_encoding = get_positional_encoding(max_pos_encoding,d_model)#using part of the position
        self.encoder_layers = [Encoding_Layer(num_heads,d_model,hidden_layer_shape,rate) for _ in range(num_enc_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,inputs,training):
        #enc_padding_mask = create_padding_mask(inputs)
        enc_padding_mask = None
        pos_scaled_embedding = tf.math.add(inputs,self.position_encoding(inputs))
        output_input_encoder = self.dropout(pos_scaled_embedding,training = training)

        for i in range(self.num_enc_layers):
            output_input_encoder = self.encoder_layers[i](output_input_encoder,enc_padding_mask,training)

        return output_input_encoder

#without decoder
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, d_model, hidden_layer_shape, max_pos_encoding, rate = 0.1):
        super(Transformer,self).__init__()
        self.encoder = Encoder(d_model,num_heads,hidden_layer_shape,max_pos_encoding,num_layers,rate) #pe_input--max_pos_encoding
#        Decoder(d_model,num_heads,hidden_layer_shape,max_pos_encoding,num_layers,rate) #pe_output--max_pos_encoding dec_input_Size--detect input size
        self.pooling = layers.AveragePooling1D(pool_size=10)
        self.final_layer = tf.keras.layers.Dense(1,activation='sigmoid')

    def call(self,input,training = True):
        
        #mid = int(input.shape[0])
#        enc_input=input[0]
#        dec_input=input[1]
        enc_output = self.encoder(input,training)
#        final_dec_input = tf.concat(dec_input,enc_output,0)
#        final_dec_input = [dec_input,enc_output]
        enc_output = self.pooling(enc_output)
        final_output = self.final_layer(enc_output)

        return final_output

class Attention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()
        
    def build(self, input_shape):
        
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention, self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
    ### https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    ### I'm not doing this right... 
    ### https://www.tensorflow.org/guide/keras/save_and_serialize
    def get_config(self):

        config = super().get_config().copy()
        config.update({
#             'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        })
        return config
