import os
import glob
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from utils.configuration import *
from utils.preprocessor import preprocessors, cc_converter
from data.dataset_pipeline import setup_tfds_builder
from models.tokenizer import RESERVED_TOKENS, build_bert_tokenizer, CustomTokenizer


config = configparser.ConfigParser()
config.read('../config/model.cfg')
lang = config['data']['lang']

vocab_path = os.path.join(DIR_VOCAB, f'{lang}_vocab.txt')


if __name__ == '__main__':
    
    ### build lexicon in tf.Dataset object & build the vocabularies
    
    # basic lexicon
    
    if lang == 'zh':
        # Classical Chinese
        builder = tfds.builder("wikipedia/20190301.zh-classical", data_dir=DIR_DATA)
        builder.download_and_prepare(download_dir=DIR_DATA)
        lexicon_zhc, _, _ = setup_tfds_builder(builder, pcts=((100, 0), 0), as_supervised=False)
        lexicon_zhc = lexicon_zhc.map(lambda text:text['text']). \
                                  map(preprocessors[lang])
    
        # Traditional Chinese
        config = tfds.translate.wmt.WmtConfig(
          version='1.0.0',
          language_pair=("zh", "en"),
          subsets={
            tfds.Split.TRAIN: ["wikititles_v1"],
            tfds.Split.VALIDATION: ["newstest2018"]
          }
        )
        builder = tfds.builder("wmt_translate", data_dir=DIR_DATA, config=config)
        builder.download_and_prepare(download_dir=DIR_DATA)

        lexicon_zh, _, _ = setup_tfds_builder(builder, pcts=((100, 0), 0), as_supervised=True)
        lexicon_zh = lexicon_zh.map(lambda zh,en:tf.py_function(func=cc_converter, inp=[zh], Tout=tf.string)). \
                                map(preprocessors[lang])
        
        #.concatenate(dataset_zhc)

    # training lexicon
    
    lexicon = pd.concat([pd.read_csv(file) for file in glob.glob(os.path.join(DIR_INTERMIN, lang, '*'))]).dropna()
    lexicon = np.concatenate([lexicon.target.values, lexicon.source.values])
    lexicon = tf.data.Dataset.from_tensor_slices(lexicon.tolist()). \
                              map(preprocessors[lang])
    lexicon = lexicon.concatenate(lexicon_zh).concatenate(lexicon_zhc)

    # build the vocabularies
    tokenizer, vocab = build_bert_tokenizer(vocab_path, lexicon, cjk=True, revocab=True)

    ### create the tf.Module object for tokenizers & output the result

    # build
    tokenizers = tf.Module()
    tokenizers.zh = CustomTokenizer(RESERVED_TOKENS, vocab_path)
    tokenizer_path = os.path.join(DIR_TOKEN, 'bert_tokenizers')

    # save
    if os.path.isdir(tokenizer_path):
        shutil.rmtree(tokenizer_path)
    tf.saved_model.save(tokenizers, tokenizer_path)

    # Reload
    
    tokenizers = tf.saved_model.load(tokenizer_path)

    for lang in dir(tokenizers):
        func = getattr(tokenizers, lang)
        if hasattr(func, 'get_vocab_size'):
            print(f'{lang} Vocabulary Size :', func.get_vocab_size().numpy())
