### Hugging Face: https://blog.tensorflow.org/2019/11/hugging-face-state-of-art-natural.html

import datasets
import transformers
from transformers import (AutoConfig, AutoTokenizer, BertTokenizerFast, 
                          TFAutoModel, 
                          TFAutoModelForSequenceClassification, 
                          TFAutoModelForTokenClassification, 
                          TFAutoModelForQuestionAnswering)
from transformers import (TFAutoModelForSeq2SeqLM)
from transformers import (TFTrainer, TFTrainingArguments,
                          AdamWeightDecay, WarmUp)

### self-defined

from .transformer import *

###################################################################################
#################################### Auxiliary ####################################
###################################################################################

BERT_NAMES = {
    'en':{
        'bert':['bert-base-uncased'],
        'distilbert':['distilbert-base-uncased'],
        'roberta':['roberta-base'],
    },
    'zh':{
        'bert':['bert-base-chinese'],
        'albert':['ckiplab/albert-tiny-chinese'],
        'roberta':['hfl/chinese-roberta-wwm-ext']
    }
}

HF_TORCH_ONLY = ['ckiplab']

def generate_bert_configuration(bert_names, cache_dirs):
    # configuration
    bert_configs = {}
    if bert_names['inp']:
        bert_configs['inp'] = AutoConfig.from_pretrained(bert_names['inp'], cache_dir=cache_dirs['inp'])
    else:
        bert_configs['inp'] = None
    if bert_names['tar']:
        bert_configs['tar'] = AutoConfig.from_pretrained(bert_names['tar'], cache_dir=cache_dirs['tar'])
    else:
        bert_configs['tar'] = None    

    # setup bert parameters
    bert_params = {
        tag:{'pretrained_model_name_or_path':bert_names[tag], 
             'config':bert_configs[tag], 
             'cache_dir':cache_dirs[tag],
             'from_pt':any([name in bert_names['inp'] for name in HF_TORCH_ONLY])} 
        for tag in ['inp', 'tar']
    }
    return bert_params

###################################################################################
#################################### Components ###################################
###################################################################################

### Embedding Projection

# TODO: BERT adaptor for each attention weight
def embedding_projector(embeddings, num_projection_layers, projection_dim, activation, dropout):
    projected_embeddings = tf.keras.layers.Dense(units=projection_dim)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.keras.layers.Activation(activation=activation)(projected_embeddings)
        x = tf.keras.layers.Dense(projection_dim)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = tf.keras.layers.LayerNormalization()(x)
    return projected_embeddings


class EmbeddingProjector(tf.keras.layers.Layer):
    def __init__(self, num_projection_layers, projection_dim, activation=tf.nn.relu, dropout=0.1):
        super().__init__()
        
        self.num_projection_layers = num_projection_layers
        
        self.init_layer = tf.keras.layers.Dense(units=projection_dim)
        self.activation_layer = tf.keras.layers.Activation(activation=activation)
        
        self.dense_layers = {}
        self.drouput_layers = {}
        self.layernorm_layers = {}
        
        # https://stackoverflow.com/questions/57517992/can-i-use-dictionary-in-keras-customized-model
        for l in range(self.num_projection_layers):
            self.dense_layers[str(l)] = tf.keras.layers.Dense(projection_dim)
            self.drouput_layers[str(l)] = tf.keras.layers.Dropout(dropout)
            self.layernorm_layers[str(l)] = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, embeddings, training=None):
        embeddings_projected = self.init_layer(embeddings)
        for l in range(self.num_projection_layers):
            x = self.activation_layer(embeddings_projected)
            x = self.dense_layers[str(l)](x)
            x = self.drouput_layers[str(l)](x)
            embeddings_projected = self.layernorm_layers[str(l)](x+embeddings_projected)
        return embeddings_projected
    
###################################################################################
###################################### Models #####################################
###################################################################################

### TransformerDecoder

class GptTransformerDecoder(tf.keras.Model):
    """ Use transformer decoder as a language model
    *** TODO:
        1. define GPT pretrained model as feature extractor
    """
    def __init__(self, tar_pretrained_model, num_tune, num_projection_layers, 
                 num_layers, embed_dim, num_heads, dense_dim, 
                 target_vocab_size, pe_target, activation=tf.nn.relu, dropout=0.1, embed_pos=False):
        super().__init__()

        ### Model Detail

        run_id  = f"{num_layers}layers_{num_projection_layers}projlayers_tune{num_tune}layers"
        run_id += f"_{num_heads}heads_{embed_dim}embed_{dense_dim}hidden{'_embedpos' if embed_pos else ''}"
        run_id += f"_{dropout}dropout_{activation if type(activation)==str else activation.__name__}"
        run_id += f"_{target_vocab_size}tarvocab"
        self.run_id = run_id
        
        ### Initialization
                
        self.embedding = False if tar_pretrained_model else True
                
        ### Load pretrained GPT model

        self.tar_pretrained_model = tar_pretrained_model
        if self.embedding:
            self.embedding_projector = None
        else:
            self.embedding_projector = EmbeddingProjector(num_projection_layers, embed_dim, 
                                                          activation=activation, dropout=dropout)
            # Whether the fine-tune process including the pretrained model
            for layer in self.tar_pretrained_model.layers[-num_tune:]:
                layer.trainable = bool(num_tune)

        ### Build the downstream model
        
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, dense_dim, 
                                          target_vocab_size=target_vocab_size, maximum_position_encoding=pe_target,
                                          activation=activation, dropout=dropout, 
                                          embed_pos=embed_pos, embedding=self.embedding, cross_attention=False)        

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
        model_name = 'GptTransformerDecoder'
        return model_name    
    
    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        tar = inputs
                    
        # GPT Embedding 
        #if not self.embedding:
        #    tar = self.tar_pretrained_model(tar, attention_mask=tar_mask)[0]
        #    tar = self.embedding_projector(tar, training=training)

        # Decoder
        dec_outputs, attention_weights = self.decoder(tar, None, None, training=training)

        # Output
        outputs = self.final_layer(dec_outputs)

        return outputs, attention_weights
    
    def symbols_to_logits_fn(self, ids, index, cache):
        """Define the logits function based on the model structure for sampler"""
        target_ids = ids

        decoder_outputs, attention_weights = self.decoder(target_ids, None, None, training=False)

        logits = self.final_layer(decoder_outputs)[:, -1]

        return logits, cache 
    
    def build_graph(self):
        """
        The most convenient method to print model.summary() similar to 
        the sequential or functional API like.
        """
        inp_ids = tf.keras.layers.Input(shape=(None, ), name='input_ids', dtype='int32')
        
        inputs = inp_ids
        
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
    
    def train_step(self, dataset):
        tar = dataset

        # For training process, using t-1~T-1 to as decoder inputs to predict t~T outputs.
        # It is also known as teacher forcing.
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            tar_pred, _ = self.call(tar_inp, training=True)
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
        tar = dataset

        # For training process, using t-1~T-1 to as decoder inputs to predict t~T outputs.
        # It is also known as teacher forcing.
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        tar_pred, _ = self.call(tar_inp, training=False)
        loss = self.loss_fn(tar_real, tar_pred)
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(self.accuracy_fn(tar_real, tar_pred))

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

### TransformerEncoder

class BertTransformerEncoder(tf.keras.Model):
    """ Use BERT model as the feature extractor for downstream model.
    *** TODO:
        1. finetune different layers
        2. adapt different attention outputs
      V 3. Compatible to token classification problem 
        4. Compatible to multiple sentences classification problem ( build-graph as well )
      X 5. Built-in metrics
    """
    def __init__(self, inp_pretrained_model, num_tune, num_projection_layers, use_lstm, nn_units, 
                 num_layers, embed_dim, num_heads, dense_dim, num_classes, output_type='cls', activation=tf.nn.relu, dropout=0.1):
        super().__init__()
        
        ### Model Detail

        run_id  = f"{num_layers}layers_{num_projection_layers}projlayers_tune{num_tune}layers"
        run_id += f"_{nn_units}nn_{'lstm' if use_lstm else 'agg'}"
        run_id += f"_{num_heads}heads_{embed_dim}embed_{dense_dim}hidden_{output_type}_{num_classes}classes"
        run_id += f"_{dropout}dropout_{activation if type(activation)==str else activation.__name__}"
        self.run_id = run_id
        
        ### Initialization
        
        self.use_lstm = use_lstm
        self.nn_units = nn_units
        self.output_type = output_type
        
        ### Load pretrained BERT model

        self.inp_pretrained_model = inp_pretrained_model
        self.embedding_projector  = EmbeddingProjector(num_projection_layers, embed_dim, 
                                                       activation=activation, dropout=dropout)
        
        # Whether the fine-tune process including the pretrained model
        for layer in self.inp_pretrained_model.layers[-num_tune:]:
            layer.trainable = bool(num_tune)
            
        ### Build the downstream model
        
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, dense_dim, 
                                          activation=activation, dropout=dropout, embedding=False)
        
        if self.output_type == 'cls':
            if self.use_lstm:
                self.nn_units //= 2
                self.aggregate_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.nn_units, dropout=dropout))
            else:
                self.aggregate_layer = tf.keras.layers.GlobalMaxPool1D()
        elif self.output_type == 'seq':
            if self.use_lstm:
                self.nn_units //= 2
                self.aggregate_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.nn_units, dropout=dropout, return_sequences=True))
            else:
                self.aggregate_layer = None

        self.dense_layer = tf.keras.layers.Dense(self.nn_units, activation=activation)            
        self.dropout_layer = tf.keras.layers.Dropout(dropout)                
        
        if num_classes <= 2:
            num_classes = 1
        self.final_layer = tf.keras.layers.Dense(num_classes, activation=None)
        
    def return_model_name(self):
        model_name = 'BertTransformerEncoder'
        return model_name        
    
    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, inp_mask = inputs
                    
        # BERT Embedding 
        inp_embedded = self.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
        inp_embedded = self.embedding_projector(inp_embedded, training=training)

        # Encoder
        enc_outputs, _ = self.encoder(inp_embedded, mask=inp_mask, training=training)    

        # Output
        if self.aggregate_layer:
            if self.use_lstm:
                x = self.aggregate_layer(enc_outputs, training=training)
            else:
                x = self.aggregate_layer(enc_outputs)
        else:
            x = enc_outputs
    
        x = self.dense_layer(x)
        x = self.dropout_layer(x, training=training)
        outputs = self.final_layer(x)

        return outputs
    
    def build_graph(self):
        """
        The most convenient method to print model.summary() similar to 
        the sequential or functional API like.
        """
        inp_ids = tf.keras.layers.Input(shape=(None, ), name='input_ids', dtype='int32')
        inp_masks = tf.keras.layers.Input(shape=(None, ), name='attention_mask', dtype='int32') 
        
        inputs = [inp_ids, inp_masks]
        
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))

### Transformer

class BertEncoderTransformer(tf.keras.Model):
    """ Use BERT model as the feature extractor of input sequences for downstream model.
    *** TODO:
        1. finetune different layers
        2. adapt different attention outputs
    """
    def __init__(self, inp_pretrained_model, num_tune, num_projection_layers, 
                 num_enc_layers, num_dec_layers, embed_dim, num_heads, dense_dim, 
                 target_vocab_size, pe_target, activation=tf.nn.relu, dropout=0.1, embed_pos=False):
        super().__init__()
        
        ### Model Detail

        run_id  = f"{num_enc_layers}-{num_dec_layers}layers_{num_projection_layers}projlayers_tune{num_tune}layers"
        run_id += f"_{num_heads}heads_{embed_dim}embed_{dense_dim}hidden{'_embedpos' if embed_pos else ''}"
        run_id += f"_{dropout}dropout_{activation if type(activation)==str else activation.__name__}"
        run_id += f"_{target_vocab_size}tarvocab"
        self.run_id = run_id
        
        ### Load pretrained BERT model

        self.inp_pretrained_model = inp_pretrained_model
        self.embedding_projector  = EmbeddingProjector(num_projection_layers, embed_dim, 
                                                       activation=activation, dropout=dropout)
        
        # Whether the fine-tune process including the pretrained model
        for layer in self.inp_pretrained_model.layers[-num_tune:]:
            layer.trainable = bool(num_tune)
            
        ### Build the downstream model
        
        self.encoder = TransformerEncoder(num_enc_layers, embed_dim, num_heads, dense_dim, 
                                          activation=activation, dropout=dropout, embed_pos=False, embedding=False)
        self.decoder = TransformerDecoder(num_dec_layers, embed_dim, num_heads, dense_dim, 
                                          target_vocab_size=target_vocab_size, maximum_position_encoding=pe_target,
                                          activation=activation, dropout=dropout, embed_pos=embed_pos, embedding=True)        
        
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
        model_name = 'BertEncoderTransformer'
        return model_name        
    
    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        [inp, inp_mask], tar = inputs
                    
        # BERT Embedding 
        inp_embedded = self.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
        inp_embedded = self.embedding_projector(inp_embedded, training=training)

        # Encoder
        enc_outputs, inp_padding_mask = self.encoder(inp_embedded, mask=inp_mask, training=training)    

        # Decoder
        dec_outputs, attention_weights = self.decoder(tar, enc_outputs, inp_padding_mask, training=training)

        outputs = self.final_layer(dec_outputs)

        return outputs, attention_weights
    
    def symbols_to_logits_fn(self, ids, index, cache):
        """Define the logits function based on the model structure for sampler"""
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
        inp_ids = tf.keras.layers.Input(shape=(None, ), name='input_ids', dtype='int32')
        inp_masks = tf.keras.layers.Input(shape=(None, ), name='attention_mask', dtype='int32') 
        tar_ids = tf.keras.layers.Input(shape=(None,), name='target_ids', dtype='int32')
        
        inputs = [[inp_ids, inp_masks], tar_ids]
        
        return tf.keras.Model(inputs=inputs, 
                              outputs=self.call(inputs))

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
    