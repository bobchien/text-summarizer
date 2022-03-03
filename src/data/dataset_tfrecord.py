from .dataset_pipeline import *

### Save TFRecord

def serialize_example(text, label):
    """Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    # The inputs here are Hugging Face tokenizer objects
    feature = {
        'inp_input_ids':int64_feature(text.ids),
        'inp_attention_mask':int64_feature(text.attention_mask),
        'tar_input_ids':int64_feature(label),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def saveTFRecord(name, dir_tfrecord, dataset, shard=1):    
    texts, labels = dataset
    
    amount = len(texts['input_ids'])

    # To make every shard have the same amount of data
    sample = amount//shard 
    amount = sample * shard
    
    count = 0
    for n in tqdm.tqdm(range(shard)):
        # Write the `tf.train.Example` observations to the file.
        filename = os.path.join(dir_tfrecord, f'{name}{n}-{amount}.tfrecord')

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(count, count+sample):
                example = serialize_example(texts[i], labels[i])
                writer.write(example)
        
        count += sample
        
### Load TFRecord

def parse_example(example_proto, labeled=True):
    # Create a dictionary describing the features.
    # Reference: https://www.google.com/search?q=tf.io.FixedLenSequenceFeature(&oq=tf.io.FixedLenSequenceFeature(&aqs=chrome..69i57&sourceid=chrome&ie=UTF-8
    feature_description = {
        'inp_input_ids':tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'inp_attention_mask':tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'tar_input_ids':tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32, so cast all int64 to int32.
    for name in list(example.keys()):
        tmp = example[name]
        if tmp.dtype == tf.int64:
            tmp = tf.cast(tmp, tf.int32)
        example[name] = tmp
        
    texts = (example['inp_input_ids'], example['inp_attention_mask'])
    if labeled:
        labels = example['tar_input_ids']
        return texts, labels
    return texts

def loadTFRecord(filename, dir_tfrecord, batch_size=64, shuffle_size=None, cache=True, labeled=True, input_context=None):
    fn_interleave = lambda file:tf.data.TFRecordDataset(file) \
                                  .map(functools.partial(parse_example, labeled=labeled),
                                                         num_parallel_calls=AUTOTUNE, 
                                                         deterministic=False)
    
    dataset = tf.data.Dataset.list_files(os.path.join(dir_tfrecord, f'{filename}*.tfrecord'), shuffle=False)
    dataset = make_batches(dataset, 
                           batch_size=batch_size, 
                           buffer_size=shuffle_size, 
                           cache=cache, 
                           fn_interleave=fn_interleave, 
                           fn_before_cache=None, 
                           fn_before_batch=None, 
                           fn_before_prefetch=None,
                           input_context=input_context)  
                            
    return dataset
