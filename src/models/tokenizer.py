import re
import tqdm
import pathlib
import collections

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset

from transformers import AutoTokenizer, BertTokenizerFast


### Autotune

AUTOTUNE = tf.data.AUTOTUNE

### Global variables

RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]

START = tf.argmax(tf.constant(RESERVED_TOKENS) == "[START]")
END = tf.argmax(tf.constant(RESERVED_TOKENS) == "[END]")

### Hugging Face

def HFSelectTokenizer(bert_name):
    """ 
    Select suitable tokenizers for given bert_name because
    some pretrained models do not support AutoTokenizer
    """
    if 'ckiplab' in bert_name:
        return BertTokenizerFast
    else:
        return AutoTokenizer

###################################################################################
########################## Vocabularies Saving & Loading ##########################
###################################################################################

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)

def load_vocab_file(filepath):
    vocab = []
    with open(filepath, 'r') as f:
        vocab = f.read().splitlines()
    return vocab

###################################################################################
################################### Tokenization ##################################
###################################################################################

def build_bert_tokenizer(vocab_path, dataset, cjk=False,
                         bert_vocab_params={}, vocab_size=None, batch_size=1024, revocab=False):
    if (not pathlib.Path(vocab_path).is_file()) | revocab:
        # For the CJK languages, we build the vocabularies by ourselves
        # by adding spaces among words beforehand (during dataset preprocessing)
        if cjk:
            vocab = []
            vocab_dict = collections.defaultdict(lambda: 0)
            for tokens in tqdm.tqdm(dataset.batch(batch_size).prefetch(AUTOTUNE)):
                for token in tf.strings.split(tf.strings.join(tokens, separator=' ')).numpy():
                    vocab_dict[token.decode()] += 1
        
            vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
            vocab = [token for token, count in vocab]
            if vocab_size:
                vocab = vocab[:vocab_size]
            vocab = RESERVED_TOKENS + vocab
        else:
            if vocab_size:
                bert_vocab_params['vocab_size'] = vocab_size
            vocab = bert_vocab_from_dataset.bert_vocab_from_dataset(
                dataset.batch(batch_size).prefetch(AUTOTUNE),
                **bert_vocab_params
            )
        write_vocab_file(vocab_path, vocab)
    else:
        vocab = load_vocab_file(vocab_path)
    
    tokenizer = tf_text.BertTokenizer(vocab_path, 
                                      lower_case=True, 
                                      unknown_token="[UNK]")
    
    print(f'\nThere are {len(vocab)} words in the dictionary\n')
    
    return tokenizer, vocab

def demo_tokenizer(tokenizer, dataset, sample=5):
    for text in dataset.take(sample):
        # Because we don't need the extra num_tokens dimensions for our current use case
        # we can merge the last two dimensions to obtain a RaggedTensor with shape [batch, num_wordpieces]
        tokens = tokenizer.tokenize(text).merge_dims(-2,-1)
        detoken = tokenizer.detokenize(tokens).merge_dims(-2,-1)
        print('Original  :\n', text.numpy().decode())
        print('Tokenized :\n', tokens.to_list()[0])
        print('Recovered :\n', tf.strings.join(detoken, separator=' ').numpy().decode())
        print('-'*60)
        
### Postprocess

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)

def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result

class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True, unknown_token="[UNK]")
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:   
        # Include a tokenize signature for a batch of strings. 
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

###################################################################################
################################## Load Tokenizer #################################
###################################################################################

def load_tokenizers(custom_path, max_lengths, 
                    inp_lang, inp_bert, inp_cache, inp_mask=True, inp_type=False, 
                    tar_lang=None, tar_bert=None, tar_cache=None, tar_mask=False, tar_type=False):
    """
    """
    tar_lang = tar_lang if tar_lang else inp_lang

    ### Tokenizers

    tokenizers = tf.Module()

    if inp_bert:
        # Pretrained Bert Tokenizer
        tokenizers.inp = HFSelectTokenizer(inp_bert).from_pretrained(inp_bert, 
                                                                     cache_dir=inp_cache, 
                                                                     do_lower_case=True)
    else:
        # Customized Bert Tokenizer
        tokenizers.inp = getattr(tf.saved_model.load(custom_path), inp_lang)

    if tar_bert:
        # Pretrained Bert Tokenizer
        tokenizers.tar = HFSelectTokenizer(tar_bert).from_pretrained(tar_bert, 
                                                                     cache_dir=tar_cache, 
                                                                     do_lower_case=True)
    else:
        # Customized Bert Tokenizer
        tokenizers.tar = getattr(tf.saved_model.load(custom_path), tar_lang)

    ### Parameters

    tokenizer_params = {}
    tokenizer_params['inp'] = {
        'add_special_tokens':True, 'padding':True, 'truncation':True, 'max_length':max_lengths['inp'], 
        'return_attention_mask':inp_mask, 'return_token_type_ids':inp_type
    } if inp_bert else None
    tokenizer_params['tar'] = {
        'add_special_tokens':True, 'padding':True, 'truncation':True, 'max_length':max_lengths['tar'], 
        'return_attention_mask':tar_mask, 'return_token_type_ids':tar_type
    } if tar_bert else None

    ### Reserved Tokens
    
    BOS_IDS = {'inp':tokenizers.inp.convert_tokens_to_ids('[BOS]') if inp_bert else START.numpy(), 
               'tar':tokenizers.tar.convert_tokens_to_ids('[BOS]') if tar_bert else START.numpy()}
    EOS_IDS = {'inp':tokenizers.inp.convert_tokens_to_ids('[EOS]') if inp_bert else END.numpy(), 
               'tar':tokenizers.tar.convert_tokens_to_ids('[EOS]') if tar_bert else END.numpy()}

    ### Vocab Sizes

    inp_vocab_size = tokenizers.inp.vocab_size if inp_bert else tokenizers.inp.get_vocab_size().numpy()
    tar_vocab_size = tokenizers.tar.vocab_size if tar_bert else tokenizers.tar.get_vocab_size().numpy()

    print(f'{inp_lang} Vocabulary Size :', inp_vocab_size)
    print(f'{tar_lang} Vocabulary Size :', tar_vocab_size)

    return tokenizers, tokenizer_params, [BOS_IDS, EOS_IDS], [inp_vocab_size, tar_vocab_size]
