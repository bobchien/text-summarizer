import shutil

from utils import configuration
from utils.configuration import *
from data.dataset_tfrecord import saveTFRecord, loadTFRecord
from models.tokenizer import *
from models.transformer_bert import BERT_NAMES

changeToColabPath(configuration.colab)
createDirectory()

config = configparser.ConfigParser()
config.read('../config/model.cfg')

### read configurations

lang = config['data']['lang']
bert_name = config['model']['bert_name']

bert_names = {'inp':BERT_NAMES[lang][bert_name][0],
              'tar':None}
cache_dirs = {'inp':os.path.join(configuration.DIR_MODELTORCH, bert_names['inp']),
              'tar':None}
max_lengths = {'inp':config['model'].getint('inp_max'), 
               'tar':config['model'].getint('tar_max')}

print(f"\nUsing Pretrained Bert Model: {bert_names}")
print(f"Cache Directory of Model: {cache_dirs}")
print(f"\nMax Length of Text: {max_lengths}")

### load tokenizer

loader = load_tokenizers(custom_path=os.path.join(configuration.DIR_TOKEN, 'bert_tokenizers'),
                         max_lengths=max_lengths, 
                         inp_lang=lang, inp_bert=bert_names['inp'], inp_cache=cache_dirs['inp'], 
                         inp_mask=True, inp_type=False)
tokenizers = loader[0]
tokenizer_params = loader[1]
BOS_IDS, EOS_IDS = loader[2]
inp_vocab_size, tar_vocab_size = loader[3]


if __name__ == '__main__':
                    
    import pandas as pd

    # load intermin dataset
    train_texts, train_labels = pd.read_csv(os.path.join(configuration.DIR_INTERMIN, lang, 'train.zip'))[['source', 'target']].dropna().values.T.tolist()
    valid_texts, valid_labels = pd.read_csv(os.path.join(configuration.DIR_INTERMIN, lang, 'valid.zip'))[['source', 'target']].dropna().values.T.tolist()
    test_texts, test_labels = pd.read_csv(os.path.join(configuration.DIR_INTERMIN, lang, 'test.zip'))[['source', 'target']].dropna().values.T.tolist()    

    # generate tfrecord files
    print('\n Start to reprocess intermediate data to tfrecord files...\n')

    #*** Bert Tokenization: can only accept str, List[str] or List[List[str]]
    encodings = []
    for texts, labels in zip([train_texts, valid_texts, test_texts], 
                             [train_labels, valid_labels, test_labels]):
        if bert_names['inp']:
            inp_encoding = tokenizers.inp(texts, **tokenizer_params['inp'])
        else:
            inp_encoding = tokenizers.inp.tokenize(texts).to_tensor().numpy()[:, :max_lengths['inp']].tolist()

        if bert_names['tar']:
            tar_encoding = tokenizers.tar(labels, **tokenizer_params['tar'])
        else:
            tar_encoding = tokenizers.tar.tokenize(labels).to_tensor().numpy()[:, :max_lengths['tar']].tolist()        
        
        encodings.append([inp_encoding, tar_encoding])
    
    ### Save the tokens to tfrecord files
    
    file_path = os.path.join(configuration.DIR_TFRECORD, lang)
    
    # clean and recreate the folder
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path, exist_ok=True)

    saveTFRecord("train", file_path, encodings[0], shard=config['data'].getint('train_shard'))
    saveTFRecord("valid", file_path, encodings[1], shard=config['data'].getint('valid_shard'))
    saveTFRecord("test", file_path, encodings[2], shard=config['data'].getint('test_shard'))
