#################################### Notes ####################################
# to use beam search (ops/beam_search.py), top-k/p sampling (ops/sampling_module) or built layers
# https://www.tensorflow.org/text/guide/decoding_api#beam_search_decoding
# https://github.com/tensorflow/models/blob/master/official/nlp/transformer/transformer.py#L236
###############################################################################

import tensorflow as tf

import official
from official.nlp import modeling      
from official.nlp.modeling.ops import sampling_module



def create_decoding_cache(batch_size, max_length, num_layers, num_heads, embed_dim, dense_dim):
    """
    In auto-regressive architectures like Transformer based Encoder-Decoder models, 
    Cache is used for fast sequential decoding. It is a nested dictionary storing pre-computed hidden-states 
    (key and values in the self-attention blocks and in the cross-attention blocks) for every layer.
    
    Actually cache = {} can also run these sampling methods
    """
    cache = {
    'layer_%d' % layer: {
        'k': tf.zeros([batch_size, max_length, num_heads, int(embed_dim/num_heads)], dtype=tf.float32),
        'v': tf.zeros([batch_size, max_length, num_heads, int(embed_dim/num_heads)], dtype=tf.float32)
        } for layer in range(num_layers)
    }
    
    return cache

def length_norm(length, dtype=tf.float32):
    """
    Return length normalization factor.
    This is used for normalizing the final scores of generated sequences and is optional
    """
    return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)

def create_decoding_api(symbols_to_logits_fn, tar_vocab_size, max_length, eos_id, TopK=3, TopP=0.9, temp=1.0):
    decoding_api = {}
    
    ### Greedy sampling
    decoding_api['Greedy'] = sampling_module.SamplingModule(
        symbols_to_logits_fn=symbols_to_logits_fn,
        length_normalization_fn=None,
        vocab_size=tar_vocab_size,
        max_decode_length=max_length,
        eos_id=eos_id, 
        padded_decode=False,
        dtype=tf.float32
    )
    
    ### TopK sampling
    decoding_api['TopK'] = sampling_module.SamplingModule(
        top_k=tf.constant(TopK),
        sample_temperature=tf.constant(temp),
        symbols_to_logits_fn=symbols_to_logits_fn,
        length_normalization_fn=length_norm,
        vocab_size=tar_vocab_size,
        max_decode_length=max_length,
        eos_id=eos_id, 
        padded_decode=False,
        enable_greedy=False,
        dtype=tf.float32)
    
    ### TopP sampling
    decoding_api['TopP'] = sampling_module.SamplingModule(
        top_p=tf.constant(TopP),
        sample_temperature=tf.constant(temp),
        symbols_to_logits_fn=symbols_to_logits_fn,
        length_normalization_fn=length_norm,
        vocab_size=tar_vocab_size,
        max_decode_length=max_length,
        eos_id=eos_id, 
        padded_decode=False,
        enable_greedy=False,
        dtype=tf.float32)    
    
    return decoding_api

def ids_decoder(initial_ids, cache, method, beam_params, sampler_params={}, keep_logits=False):
    
    ### Decode
    
    if method in ['Greedy', 'TopK', 'TopP']:
        decoding_sampler = create_decoding_api(**sampler_params)
        ids, _ = decoding_sampler[method].generate(initial_ids=initial_ids, initial_cache=cache)
    else:
        beam_params['initial_ids'] = initial_ids
        beam_params['initial_cache'] = cache
        beam_params['padded_decode'] = False
        beam_params['dtype'] = tf.float32        
        
        ids, _ = modeling.ops.beam_search.sequence_beam_search(**beam_params)
    
    ### Detokenizer

    # Outputs of Beam Search have shape (batch_size, beam_size, 1+seq_len)
    if tf.size(tf.shape(ids)) == 3:
        ids = ids[:, 0]

    return ids
