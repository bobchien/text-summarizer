import os
from transformers import AutoConfig

from .decoding import *
from models.tokenizer import HFSelectTokenizer

###################################################################################
################################### Hugging Face ##################################
###################################################################################

#################################### Classifier ###################################

class HF2TFSingleClassifierExporter(tf.Module):
    """ 
    Save the fine-tuned Hugging Face model with detail information, 
    and should be modified based on the model's inputs.    
    """
    def __init__(self, model, bert_names, config_detail, num_classes, inp_lang='', tar_lang=''):
        self.model = model
        self.num_classes = num_classes
        self.config_detail = tf.Variable(config_detail)
        
        self.inp_bert = tf.Variable(bert_names['inp'] or '')        
        self.tar_bert = tf.Variable(bert_names['tar'] or '')        
        self.inp_lang = tf.Variable(inp_lang)
        self.tar_lang = tf.Variable(tar_lang)     

    # Notice: the input should be encoded as token_ids and masks, 
    # and the signature need to be adjusted based on the number of masks
    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                   tf.TensorSpec(shape=[None, None], dtype=tf.int32)], 
                                  tf.TensorSpec(shape=[], dtype=tf.bool)])
    def __call__(self, inputs, return_prob=True):        
        outputs = self.model(inputs, training=False)
        
        # Multiclass Problem
        if self.num_classes <= 2:
            outputs = tf.nn.sigmoid(outputs)
        else:
            outputs = tf.nn.softmax(outputs)
            if not return_prob:
                outputs = tf.math.argmax(outputs, axis=-1) + 1

        return outputs
    
class HF2TFClassifierPipeline(tf.Module):
    def __init__(self, predictor_dir, pretrain_dir, preprocessors=None, tokenizer_params={}):
        self.predictor = tf.saved_model.load(predictor_dir)
        self.inp_lang = self.predictor.inp_lang.numpy().decode()
        self.inp_bert = self.predictor.inp_bert.numpy().decode()
        self.inp_cjk = True if self.inp_lang in ['zh', 'jp', 'kr'] else False
        self.preprocessors = preprocessors
        
        self.bert_dir = os.path.join(pretrain_dir, self.inp_bert)
        self.bert_config = AutoConfig.from_pretrained(self.inp_bert, cache_dir=self.bert_dir)
        self.bert_tokenizer = HFSelectTokenizer(self.inp_bert).from_pretrained(self.inp_bert, 
                                                                               cache_dir=self.bert_dir, 
                                                                               do_lower_case=not self.inp_cjk)
        self.bert_tokenizer_params = {
            'add_special_tokens':True, 
            'padding':True, 'truncation':True, 
            'return_attention_mask':True, 'return_token_type_ids':False
        }
        # e.g. return_token_type_ids = True for multi-sentence problems
        for k, v in tokenizer_params.items():
            self.bert_tokenizer_params[k] = v
        
    def __call__(self, sentence1, sentence2=None, max_length=256, return_prob=True):
        # Set the maxmium length of input sentence
        self.bert_tokenizer_params['max_length'] = max_length
        
        # Preprocessing : bert tokenizer cannot accept byte format so we should set py_function=True
        if self.preprocessors:
            sentence1 = self.preprocessors['inp'](sentence1, py_function=True)
            if sentence2 is not None:
                sentence2 = self.preprocessors['inp'](sentence2, py_function=True)

        # Tokenization
        if self.bert_tokenizer_params['return_token_type_ids']:
            tokens = self.bert_tokenizer(sentence1, sentence2, return_tensors='tf', **self.bert_tokenizer_params)
            # Two sentences
            outputs = self.predictor([tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids']], 
                                     return_prob=tf.constant(return_prob))
        else:
            tokens = self.bert_tokenizer(sentence1, return_tensors='tf', **self.bert_tokenizer_params)            
            # One Sentence
            outputs = self.predictor([tokens['input_ids'], tokens['attention_mask']], 
                                     return_prob=tf.constant(return_prob))

        return outputs

##################################### Seq2Seq #####################################

class HF2TFSeq2SeqExporter(tf.Module):
    """ 
    Save the fine-tuned Hugging Face model with detail information, 
    and should be modified based on the model's inputs.    
    """
    def __init__(self, model, tokenizers, bos_ids, beam_params, sampler_params,
                 bert_names, config_detail, inp_lang='', tar_lang=''):
        self.model = model
        self.tokenizers = tokenizers
        self.config_detail = tf.Variable(config_detail)
        
        self.inp_bos = tf.cast(bos_ids['inp'], tf.int32)
        self.tar_bos = tf.cast(bos_ids['tar'], tf.int32)
        self.beam_params = beam_params
        self.beam_params['symbols_to_logits_fn'] = self.model.symbols_to_logits_fn
        self.sampler_params = sampler_params
        self.sampler_params['symbols_to_logits_fn'] = self.model.symbols_to_logits_fn
        
        self.inp_bert = tf.Variable(bert_names['inp'] or '')        
        self.tar_bert = tf.Variable(bert_names['tar'] or '')        
        self.inp_lang = tf.Variable(inp_lang)
        self.tar_lang = tf.Variable(tar_lang)     

    # Notice: the input should be encoded as token_ids and masks, 
    # and the signature need to be adjusted based on the number of masks
    @tf.function(input_signature=[[tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
                                   tf.TensorSpec(shape=(None, None), dtype=tf.int32)],
                                  tf.TensorSpec(shape=[], dtype=tf.int32)])
    def __call__(self, inputs, max_length=64):
        inp, inp_mask = inputs        
        
        # Initialization
        initial_ids = tf.math.multiply(self.tar_bos, tf.ones_like(inp[:, 0]))
        
        # Create decoding cache based on the model structure
        cache = {}
        inp_embedded = self.model.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
        inp_embedded = self.model.embedding_projector(inp_embedded, training=False)
        cache['encoder_outputs'], cache['inp_padding_mask'] = self.model.encoder(inp_embedded, mask=inp_mask, training=False)
        cache['inp_padding_mask'] = tf.cast(cache['inp_padding_mask'], dtype=tf.float32)

        # Update the max_decode_length in the parameters
        self.beam_params['max_decode_length'] = max_length
        self.sampler_params['max_decode_length'] = max_length
        
        # Decoder
        ids = ids_decoder(initial_ids, cache, 'BeamSearch', self.beam_params, self.sampler_params)
        output = tf.cast(ids, dtype=tf.int64)

        # Detokenization
        text = self.tokenizers.tar.detokenize(output)[0]
        tokens = self.tokenizers.tar.lookup(output)[0]
        
        # recalculate attention_weights after sampling
        output = tf.ensure_shape(ids, [None, None])        
        _, attention_weights = self.model([inputs, output], training=False)      
        
        return text, tokens, attention_weights


class HF2TFSeq2SeqPipeline(tf.Module):
    def __init__(self, predictor_dir, pretrain_dir, preprocessors=None, tokenizer_params={}):
        self.predictor = tf.saved_model.load(predictor_dir)
        self.inp_lang = self.predictor.inp_lang.numpy().decode()        
        self.tar_lang = self.predictor.tar_lang.numpy().decode()        
        self.inp_bert = self.predictor.inp_bert.numpy().decode()        
        self.tar_bert = self.predictor.tar_bert.numpy().decode()
        self.inp_cjk = True if self.inp_lang in ['zh', 'jp', 'kr'] else False
        self.preprocessors = preprocessors
        
        # Only support input data now
        self.bert_dir = os.path.join(pretrain_dir, self.inp_bert)
        self.bert_config = AutoConfig.from_pretrained(self.inp_bert, cache_dir=self.bert_dir)
        self.bert_tokenizer = HFSelectTokenizer(self.inp_bert).from_pretrained(self.inp_bert, 
                                                                               cache_dir=self.bert_dir, 
                                                                               do_lower_case=not self.inp_cjk)
        self.bert_tokenizer_params = {
            'add_special_tokens':True, 
            'padding':True, 'truncation':True, 
            'return_attention_mask':True, 'return_token_type_ids':False
        }
        for k, v in tokenizer_params.items():
            self.bert_tokenizer_params[k] = v
        
    def __call__(self, sentence, max_lengths={'inp':256, 'tar':256}, return_attention=False):
        # Set the maxmium length of input sentence
        self.bert_tokenizer_params['max_length'] = max_lengths['inp']
        
        # Preprocessing : bert tokenizer cannot accept byte format so we should set py_function=True
        if self.preprocessors:
            sentence = self.preprocessors['inp'](sentence, py_function=True)

        # Tokenization
        tokens = self.bert_tokenizer(sentence, return_tensors='tf', **self.bert_tokenizer_params)            
            
        # Model Inference
        text, tokens, attention_weights = self.predictor([tokens['input_ids'], tokens['attention_mask']], 
                                                         max_length=tf.constant(max_lengths['tar']))

        # Postprocessing
        text = self.preprocessors['tar'](text.numpy().decode(), py_function=True)[0]
        
        if return_attention:
            return text, [token.decode() for token in tokens.numpy()], attention_weights
        else:
            return text
        
    
    
###################################################################################
######################### TF : Natural Language Processing ########################
###################################################################################

#################################### Classifier ###################################

class ClassifierPredictor(tf.Module):
    def __init__(self, model, tokenizer, preprocessors, inp_lang, tokenizer_params={}, model_detail=''):
        self.inp_lang = inp_lang
        self.preprocessor = preprocessors.inp        

        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        
        self.model = model
        self.model_detail = model_detail        
        
    @tf.function
    def __call__(self, sentence):
        # Preprocessing : 
        # bert tokenizer cannot accept byte format so the preoprocessed sentences should be decoded
        sentence = self.preprocessor(sentence, py_function=True)

        # Tokenization
        tokens = tokenizer(sentence, return_tensors='tf', **self.tokenizer_params)
        if ('return_token_type_ids' in self.tokenizer_params) & self.tokenizer_params['return_token_type_ids']:
            # Two sentences
            inputs = [tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids']]
        else:
            # One Sentence
            inputs = [tokens['input_ids'], tokens['attention_mask']]

        outputs = self.model(inputs, training=False)
        outputs = tf.nn.sigmoid(outputs)

        return outputs
    
##################################### Seq2Seq #####################################
        
class Seq2SeqPredictor(tf.Module):
    def __init__(self, model, tokenizers, preprocessors, beam_params, sampler_params):
        self.model = model
        self.tokenizers = tokenizers
        self.preprocessors = preprocessors
        self.beam_params = beam_params
        self.beam_params['symbols_to_logits_fn'] = self.model.symbols_to_logits_fn
        self.sampler_params = sampler_params
        self.sampler_params['symbols_to_logits_fn'] = self.model.symbols_to_logits_fn

    def __call__(self, sentence, max_length=64, search_method='BeamSearch'):        
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        
        # Preprocessing
        sentence = self.preprocessors['inp'](sentence)

        # Tokenization
        encoder_input = self.tokenizers.inp.tokenize(sentence).to_tensor()
        
        # Setup the start and end token (here we only have tokenizers and no BOS or EOS)
        start_end = self.tokenizers.tar.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        
        # Initialization
        initial_ids = tf.cast(start, dtype=tf.int32)

        # Create decoding cache based on the model structure
        cache = {}
        cache['encoder_outputs'], cache['inp_padding_mask'] = self.model.encoder(encoder_input)
        cache['inp_padding_mask'] = tf.cast(cache['inp_padding_mask'], dtype=tf.float32)

        # Update the max_decode_length in the parameters
        self.beam_params['max_decode_length'] = max_length
        self.sampler_params['max_decode_length'] = max_length
        
        # Decoder
        ids = ids_decoder(initial_ids, cache, search_method, self.beam_params, self.sampler_params)
        output = tf.cast(ids, dtype=tf.int64)

        # Detokenization & Postprocessing
        text = self.tokenizers.tar.detokenize(output)[0]
        text = self.preprocessors['tar'](text)

        tokens = self.tokenizers.tar.lookup(output)[0]

        # recalculate attention_weights after sampling
        output = tf.ensure_shape(output, [None, None])        
        _, attention_weights = self.model([encoder_input, output], training=False)

        return text, tokens, attention_weights
        
        
class Seq2SeqPipeline(tf.Module):
    def __init__(self, predictor):
        self.predictor = predictor

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string), 
                                  tf.TensorSpec(shape=[], dtype=tf.int32)])
    def __call__(self, sentence, max_length=64):
        (text, 
         tokens,
         attention_weights) = self.predictor(sentence, max_length=max_length, search_method='BeamSearch')
    
        return text
        
###################################################################################
############################### TF : Video Processing #############################
###################################################################################

##################################### Seq2Seq #####################################

class Video2VideoPredictor(tf.Module):
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, time_length=6):
        assert isinstance(inputs, tf.Tensor)
        # Can digest data without batch dimension axis
        if len(inputs.shape) == 4:
            inputs = inputs[tf.newaxis]
        
        # Create a empty tensor array to store the data
        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # Append the last epoch of input data as the start token
        output_array = output_array.write(0, inputs[:, -1])
    
        for i in tf.range(time_length):
            # Rearrange the time axis
            outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2, 3, 4])
            
            # Predict the results based on the latest information
            predictions, _ = self.model([inputs, outputs], training=False)

            # Select the last epoch from the seq_len dimension
            predictions = predictions[:, -1]  # (batch_size, 1, ...)
            
            # Concatentate the predictions to the outputs which is given to the TransformerDecoder
            # as its input.
            output_array = output_array.write(i, predictions)

        outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2, 3, 4])
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.model([inputs, outputs], training=False)

        return outputs, attention_weights
            
class Video2VideoPipeline(tf.Module):
    def __init__(self, predictor, image_shape):
        self.predictor = predictor

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None]+list(image_shape), dtype=tf.float32)])
    def __call__(self, inputs, time_length=6):
        result, _ = self.predictor(inputs, time_length)
    
        return result

###################################################################################
###################################### Record #####################################
###################################################################################

""" 
This Seq2SeqPredictor is inspired by the tutorial of Tensorflow,
but is now replaced by beam search and other sampling methods
using tf-offical-models

class Seq2SeqPredictor(tf.Module):
    def __init__(self, model, tokenizers, preprocessors):
        self.preprocessors = preprocessors
        self.tokenizers = tokenizers
        self.model = model

    def __call__(self, sentence, max_length=20):        
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        
        # Preprocessing
        sentence = self.preprocessors['inp'](sentence)

        # Tokenization
        encoder_input = self.tokenizers.inp.tokenize(sentence).to_tensor()
        
        # Setup the first and the last
        start_end = self.tokenizers.tar.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end   = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
    
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.model([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the TransformerDecoder
            # as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack()) # output.shape (1, tokens)
        
        # Detokenization
        text = self.tokenizers.tar.detokenize(output)[0]
        tokens = self.tokenizers.tar.lookup(output)[0]
        
        # Postprocessing
        text = self.preprocessors['tar'](text)
        tokens = self.preprocessors['tar'](tokens)
        
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.model([encoder_input, output[:,:-1]], training=False)

        return text, tokens, attention_weights
"""