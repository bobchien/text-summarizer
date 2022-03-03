import tensorflow as tf
import tensorflow_datasets as tfds

import os
import tqdm
import functools
from sklearn.model_selection import train_test_split

### Autotune

AUTOTUNE = tf.data.AUTOTUNE

### Dataset Optimization

# Experimental tf.data.experimental.OptimizationOptions that are disabled by default 
# can in certain contexts -- such as when used together with tf.distribute -- 
# cause a performance degradation. You should only enable them after you validate 
# that they benefit the performance of your workload in a distribute setting.
options = tf.data.Options()
options.experimental_optimization.apply_default_optimizations = True

###################################################################################
############################### tensorflow tfrecord ###############################
###################################################################################

# The following functions can be used to convert a value to a type compatible with tf.train.Example.

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

###################################################################################
############################### tensorflow datasets ###############################
###################################################################################

def setup_tfds_builder(builder, pcts, as_supervised=True):
    
    ### Set the splitting ratio

    (pct_train, pct_valid), pct_test = pcts
    pct_drop = 100-pct_train-pct_valid
    
    # Training
    
    if pct_train == 100:
        split_train = [f'train', f'train', f'train']
    else:
        split_train = [f'train[:{pct_train}%]', 
                       f'train[{pct_train}%:{pct_train+pct_valid}%]', 
                       f'train[{-pct_drop}%:]']
    
    # Testing
    
    if ('test' in builder.info.splits):
        split_name = 'test'
    elif ('validation' in builder.info.splits):
        split_name = 'validation'
    else:
        split_name = None
    
    if split_name is not None:
        split_test = [f'{split_name}[:{pct_test}%]', f'{split_name}[{-pct_test}%:]']
    else:
        split_test = None
    
    ### Create the datasets
    
    train_dataset, valid_dataset, _ = builder.as_dataset(split=split_train, as_supervised=as_supervised)
        
    if (split_test is not None) & (pct_test!=0):
        test_dataset, _ = builder.as_dataset(split=split_test, as_supervised=as_supervised)
    else:
        test_dataset = None
    
    return train_dataset, valid_dataset, test_dataset

###################################################################################
############################### tensorflow pipeline ###############################
###################################################################################

def make_batches(dataset, batch_size=None, buffer_size=None, cache=True, prefetch=True,
                 fn_interleave=None, fn_before_cache=None, fn_before_batch=None, fn_before_prefetch=None,
                 input_context=None):
    """ Make the dataset object into batches of Dataset or distributedDataset
    """
    if input_context:
        batch_size = input_context.get_per_replica_batch_size(batch_size)
        # Be sure to shard before you use any randomizing operator (such as shuffle).
        dataset = dataset.shard(num_shards=input_context.num_input_pipelines, 
                                index=input_context.input_pipeline_id)   
        
    # Parsing
    if fn_interleave:   
        dataset = dataset.interleave(fn_interleave,
                                     num_parallel_calls=AUTOTUNE, deterministic=False,
                                     cycle_length=AUTOTUNE, block_length=1)       
        
    # Cache
    if fn_before_cache:
        dataset = dataset.map(fn_before_cache, num_parallel_calls=AUTOTUNE, deterministic=None)
    if cache:
        dataset = dataset.cache()
    
    # Shuffle
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)
    
    # Batch
    if fn_before_batch:
        dataset = dataset.map(fn_before_batch, num_parallel_calls=AUTOTUNE, deterministic=None)
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch
    if fn_before_prefetch:
        dataset = dataset.map(fn_before_prefetch, num_parallel_calls=AUTOTUNE, deterministic=None)
    if prefetch:
        dataset = dataset.prefetch(AUTOTUNE)    
    
    return dataset

def make_file_batches(filename, dir_file, batch_size=64, shuffle_size=None, cache=True, prefetch=True,
                      fn_interleave=None, fn_before_cache=None, fn_before_batch=None, fn_before_prefetch=None,
                      input_context=None):
    """ Load files and turn them into tf.data.Dataset object or tf.distribute.DistributedDataset
    """
    dataset = tf.data.Dataset.list_files(os.path.join(dir_file, filename), shuffle=False) 
    dataset = make_batches(dataset, 
                           batch_size=batch_size, 
                           buffer_size=shuffle_size, 
                           cache=cache, 
                           prefetch=prefetch,
                           fn_interleave=fn_interleave, 
                           fn_before_cache=fn_before_cache, 
                           fn_before_batch=fn_before_batch, 
                           fn_before_prefetch=fn_before_prefetch,
                           input_context=input_context)    
    return dataset

def make_custom_token_pair_batches(dataset, tokenizers, max_lengths=None, 
                                   batch_size=64, buffer_size=None, cache=True):

    def tokenize_pairs(inp, tar):
        # Convert from ragged to dense, padding with zeros.
        inp = tokenizers.inp.tokenize(inp)
        tar = tokenizers.tar.tokenize(tar)
        
        # Truncate sentence
        if max_lengths['inp']:
            inp = inp[:, :max_lengths['inp']]
        if max_lengths['tar']:
            tar = tar[:, :max_lengths['tar']]

        # Pad sentence
        return tf.cast(inp, dtype=tf.int32).to_tensor(), tf.cast(tar, dtype=tf.int32).to_tensor()
    
    return make_batches(dataset, batch_size, buffer_size, cache,
                        fn_before_cache=None, fn_before_batch=None, fn_before_prefetch=tokenize_pairs)
