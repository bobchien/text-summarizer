import numpy as np
import functools
import tensorflow as tf
import tensorflow_addons as tfa

import official
from official.nlp import modeling      # to sample (ops/beam_search.py, ops/ampling_module) or built layers
from official.nlp import optimization  # to create AdamW optimizer
from official.nlp.metrics import bleu as bleu_metric

###################################################################################
#################################### Optimizer ####################################
###################################################################################

### Optimization: can be replaced via tf-offical-models

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, warmup_steps=4000):
        super().__init__()

        self.embed_dim = tf.cast(embed_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

###################################################################################
#################################### Embedding ####################################
###################################################################################

### Positional Encoding

def get_angles(pos, i, embed_dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
    return pos * angle_rates

def positional_encoding(position, embed_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embed_dim)[np.newaxis, :],
                            embed_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

### Token + Position Embedding

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, vocab_size, embed_dim, embed_pos=False):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_pos = embed_pos        
        
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        if self.embed_pos:
            self.position_embeddings = tf.keras.layers.Embedding(input_dim=max_length, output_dim=embed_dim)
        else:
            self.position_embeddings = positional_encoding(max_length, embed_dim)
        
    def call(self, inputs):
        time_length = tf.shape(inputs)[1]
        
        # Token
        embedded_tokens = self.token_embeddings(inputs)  # (batch_size, target_seq_len, embed_dim)
        embedded_tokens *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        
        # Position
        if self.embed_pos:
            embedded_positions = self.position_embeddings(
                tf.range(start=0, limit=time_length, delta=1))
            embedded_positions *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        else:
            embedded_positions = self.position_embeddings[:, :time_length]
        
        return embedded_tokens + embedded_positions
    
    # (1) any layer that produces a tensor with a different time dimension than its input, 
    # such as a Concatenate layer that concatenates on the time dimension, 
    # will need to modify the current mask so that downstream layers will be able to properly 
    # take masked timesteps into account.
    # (2) A CustomEmbedding layer that is capable of generating a mask from input values
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
###################################################################################
################################ Padding & Masking ################################
###################################################################################

### https://www.tensorflow.org/guide/keras/masking_and_padding

# __init__
# (1) By default, a custom layer will destroy the current mask 
# (since the framework has no way to tell whether propagating the mask is safe to do).
# (2) If you have a custom layer that does not modify the time dimension,
# and if you want it to be able to propagate the current input mask, 
# you should set self.supports_masking = True in the layer constructor.

# __call__
# (1) Some layers are mask consumers: they accept a mask argument in call and 
# use it to determine whether to skip certain time steps.
# To write such a layer, you can simply add a mask=None argument in your call signature. 
# The mask associated with the inputs will be passed to your layer whenever it is available.
    
### modified for tf.keras.layers.MultiHeadAttention (0: masked, 1: not masked)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

###################################################################################
################################### Transformer ###################################
###################################################################################
    
### Point wise feed forward network

def point_wise_feed_forward_network(embed_dim, dense_dim, activation=tf.nn.relu):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dense_dim, activation=activation),  # (batch_size, seq_len, dense_dim)
        tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, embed_dim)
    ])

### TransformerEncoder layer 

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dense_dim, activation=tf.nn.relu, dropout=0.1, cross_attention=False):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = point_wise_feed_forward_network(embed_dim, dense_dim, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        attn_output = self.mha(inputs, inputs, inputs, mask)  # (batch_size, input_seq_len, embed_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, embed_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embed_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embed_dim)

        return out2
    
### TransformerDecoder Layer

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dense_dim, activation=tf.nn.relu, dropout=0.1, cross_attention=True):
        super().__init__()

        self.cross_attention = cross_attention
        
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = point_wise_feed_forward_network(embed_dim, dense_dim, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

        if self.cross_attention:
            self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, encoder_outputs, combined_mask=None, inp_padding_mask=None, training=None):
        # encoder_outputs.shape == (batch_size, input_seq_len, embed_dim)
    
        ### Self attention
        attn1, attn_weights_block1 = self.mha1(query=inputs, value=inputs, key=inputs, 
                                               attention_mask=combined_mask, 
                                               return_attention_scores=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs) # (batch_size, target_seq_len, embed_dim)
        
        ### Cross attention
        if self.cross_attention:
            attn2, attn_weights_block2 = self.mha2(query=out1, value=encoder_outputs, key=encoder_outputs,
                                                   attention_mask=inp_padding_mask,
                                                   return_attention_scores=True)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)   # (batch_size, target_seq_len, embed_dim)
        else:
            attn_weights_block2 = None
            out2 = out1

        ### FFN
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embed_dim)

        return out3, attn_weights_block1, attn_weights_block2
    
### TransformerEncoder

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, dense_dim, 
                 input_vocab_size=None, maximum_position_encoding=None, activation=tf.nn.relu, dropout=0.1, 
                 embed_pos=False, embed_outside=None, embedding=True):
        super().__init__()

        self.embedding = embedding        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        if self.embedding:
            if embed_outside:
                self.embedding_layer = embed_outside
            else:
                self.embedding_layer = TokenAndPositionEmbedding(maximum_position_encoding, 
                                                                 input_vocab_size, 
                                                                 embed_dim, 
                                                                 embed_pos)
        self.enc_layers = [TransformerEncoderLayer(embed_dim, num_heads, dense_dim, activation, dropout)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        ### Embedding
        if self.embedding:
            # Use the built embedding layer
            inputs = self.embedding_layer(inputs) # (batch_size, input_seq_len, embed_dim)
            
        ### Masking
            inp_padding_mask = tf.cast(inputs._keras_mask[:, tf.newaxis], tf.int32) # (batch_size, 1, inp_len)
        else:
            inp_padding_mask = tf.cast(mask[:, tf.newaxis], tf.int32) # (batch_size, 1, inp_len)
            
        inputs = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            inputs = self.enc_layers[i](inputs, inp_padding_mask, training)

        return inputs, inp_padding_mask # (batch_size, input_seq_len, embed_dim)
    
### TransformerDecoder

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, dense_dim, 
                 target_vocab_size=None, maximum_position_encoding=None, activation=tf.nn.relu, dropout=0.1, 
                 embed_pos=False, embed_outside=None, embedding=True, cross_attention=True):
        super().__init__()

        self.embedding = embedding        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        if self.embedding:
            if embed_outside:
                self.embedding_layer = embed_outside
            else:
                self.embedding_layer = TokenAndPositionEmbedding(maximum_position_encoding, 
                                                                 target_vocab_size, 
                                                                 embed_dim, 
                                                                 embed_pos)
        self.dec_layers = [TransformerDecoderLayer(embed_dim, num_heads, dense_dim, activation, dropout, cross_attention)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, enc_outputs, inp_padding_mask, mask=None, training=None):
        attention_weights = {}

        ### Embedding
        if self.embedding:
            # Use the built embedding layer
            inputs = self.embedding_layer(inputs) # (batch_size, input_seq_len, embed_dim)
            
        ### Masking
            tar_padding_mask = tf.cast(inputs._keras_mask[:, tf.newaxis], tf.int32) # (batch_size, 1, inp_len)
        else:
            tar_padding_mask = tf.cast(mask[:, tf.newaxis], tf.int32) # (batch_size, 1, inp_len)
        combined_mask = self.compute_combined_mask(inputs, tar_padding_mask)        
            
        inputs = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            inputs, block1, block2 = self.dec_layers[i](inputs, enc_outputs, combined_mask, inp_padding_mask, training)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return inputs, attention_weights # (batch_size, target_seq_len, embed_dim)
    
    # Compute combined_mask and zero means the masked location 
    def compute_combined_mask(self, inputs, tar_padding_mask):        
        # Compute look_ahead_mask : (1, tar_len, tar_len)        
        seq_len = tf.shape(inputs)[1]
        look_ahead_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        look_ahead_mask = tf.cast(look_ahead_mask, tf.int32)[tf.newaxis, ...]
        
        # Compute padding_mask : (batch_size, 1, tar_len)
        tar_padding_mask = tf.cast(tar_padding_mask[:, tf.newaxis], tf.int32)
        
        # Compute combined_mask : (batch_size, tar_len, tar_len)
        combined_mask = tf.minimum(tar_padding_mask, look_ahead_mask)
            
        return combined_mask    
    
### Transformer

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, dense_dim, 
                 input_vocab_size, target_vocab_size, pe_input, pe_target, 
                 activation=tf.nn.relu, dropout=0.1, embed_pos=False, embed_same=False):
        super().__init__()

        ### Model Detail

        run_id  = f"{num_layers}layers_{num_heads}heads_{embed_dim}embed_{dense_dim}hidden"
        run_id += f"{'_embedpos' if embed_pos else ''}{'_embsame' if embed_same else ''}"
        run_id += f"_{dropout}dropout_{activation if type(activation)==str else activation.__name__}"
        run_id += f"_{input_vocab_size}inpvocab_{target_vocab_size}tarvocab"
        self.run_id = run_id
        
        ### Initialization
        
        self.embed_same = embed_same
        if self.embed_same:
            self.embedding_layer = TokenAndPositionEmbedding(max_length=pe_input, 
                                                             vocab_size=input_vocab_size, 
                                                             embed_dim=embed_dim, 
                                                             embed_pos=embed_pos)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, dense_dim,
                                          input_vocab_size, pe_input, activation, dropout, 
                                          embed_pos, embedding=not embed_same)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, dense_dim,
                                          target_vocab_size, pe_target, activation, dropout, 
                                          embed_pos, embedding=not embed_same)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        ### Metric Trackers
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]
    
    def compile(self, optimizer, loss_function=None, accuracy_function=None):
        super().compile()

        if not loss_function:
            
            def loss_function(real, pred):
                mask = tf.math.not_equal(real, 0)
                loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction='none')(real, pred)

                mask = tf.cast(mask, dtype=loss_.dtype)
                loss_ *= mask

                return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
       
        if not accuracy_function:

            def accuracy_function(real, pred):
                accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

                mask = tf.math.not_equal(real, 0)
                accuracies = tf.math.logical_and(mask, accuracies)

                mask = tf.cast(mask, dtype=tf.float32)
                accuracies = tf.cast(accuracies, dtype=tf.float32)

                return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
         
        self.optimizer = optimizer
        self.loss_fn = loss_function
        self.accuracy_fn = accuracy_function

    def return_model_name(self):
        model_name = 'Transformer'
        return model_name  

    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        
        if self.embed_same:
            inp = self.embedding_layer(inp)
            tar = self.embedding_layer(tar)
            
        # enc_output.shape == (batch_size, inp_seq_len, embed_dim)
        enc_outputs, inp_padding_mask = self.encoder(inp, training=training)

        # dec_output.shape == (batch_size, tar_seq_len, embed_dim)
        dec_outputs, attention_weights = self.decoder(tar, enc_outputs, inp_padding_mask, training=training)

        outputs = self.final_layer(dec_outputs)

        return outputs, attention_weights
    
    def symbols_to_logits_fn(self, ids, index, cache):
        """Define the logits function based on the model structure for sampler"""
        
        if self.embed_same:            
            target_ids = self.embedding_layer(ids)
        else:
            target_ids = ids

        decoder_outputs, attention_weights = self.decoder(
            target_ids, cache['encoder_outputs'], cache['inp_padding_mask'], training=False)

        logits = self.final_layer(decoder_outputs)[:, -1]

        return logits, cache
        
    def build_graph(self):
        """
        The most convenient method to print model.summary() similar to 
        the sequential or functional API like.
        """
        inp_ids = tf.keras.layers.Input(shape=(None,), name='input_ids', dtype='int32')
        tar_ids = tf.keras.layers.Input(shape=(None,), name='target_ids', dtype='int32')
        
        inputs = [inp_ids, tar_ids]
        
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))    
    
    def train_step(self, dataset):
        inp, tar = dataset

        # For training process, using t-1~T-1 to as decoder inputs to predict t~T outputs.
        # It is also known as teacher forcing.
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            tar_pred, _ = self.call([inp, tar_inp], training=True)
            loss = self.loss_fn(tar_real, tar_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(self.accuracy_fn(tar_real, tar_pred))

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }
        
    def test_step(self, dataset):
        inp, tar = dataset

        # For training process, using t-1~T-1 to as decoder inputs to predict t~T outputs.
        # It is also known as teacher forcing.
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        tar_pred, _ = self.call([inp, tar_inp], training=False)
        loss = self.loss_fn(tar_real, tar_pred)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(self.accuracy_fn(tar_real, tar_pred))

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

###################################################################################
###################################### Record #####################################
###################################################################################

""" 
This test_step is inspired by the export format of model, 
but is now replaced by beam search and other sampling methods

@tf.function(input_signature=input_signature)
def test_step(dataset):
    inp, tar = dataset
    
    tar_real = tar[:, 1:]

    # Define the max_length of sentences
    max_length = tar.shape[1]
    
    # Setup the first and the last
    start = tf.math.multiply(BOS_IDS['tar'], tf.ones_like(tar[:, 0]))
    end   = tf.math.multiply(EOS_IDS['tar'], tf.ones_like(tar[:, 0]))
        
    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    logits_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)
        
    # tf.while_loop
    # https://www.tensorflow.org/probability/examples/Estimating_COVID_19_in_11_European_countries
    i = tf.constant(0)
    cond = lambda i, *_: tf.less(i, max_length)
    
    def body(i, logits_array, output_array):
        output = tf.transpose(output_array.stack())
        predictions, _ = model([inp, output], training=False)

        # select the last token from the seq_len dimension
        predictions = predictions[:, -1:]  # (batch_size, 1, vocab_size)
 
        # Beam Search or Greedy Search
        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)[:, 0]
 
        # concatentate the predicted_id to the output which is given to the TransformerDecoder
        # as its input.
        logits_array = logits_array.write(i, predictions[:, -1])
        output_array = output_array.write(i+1, predicted_id)
        
        return [tf.add(i, 1), logits_array, output_array]
    
    _, logits_array, output_array = tf.while_loop(cond, body, [i, logits_array, output_array])
    
    tar_pred = tf.transpose(logits_array.stack(), perm=[1, 0, 2])[:, 1:]
    tar_pred_ids = tf.transpose(output_array.stack())

    loss = loss_function(tar_real, tar_pred)

    # Track progress
    valid_loss(loss)
    valid_accuracy(accuracy_function(tar_real, tar_pred))
"""
