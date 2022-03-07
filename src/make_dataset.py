import argparse
import swifter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import configuration
from utils.configuration import *
from utils.preprocessor import *

changeToColabPath(configuration.colab)
createDirectory()

def preprocess_raw_data(from_file, lang, min_len=64, max_len=256, size=None, split=False, plot=False, random_state=24601):
    name = from_file[from_file.rfind('/')+1:]
    print(f"\nStart process raw data - {name}...\n")
    
    # load and parse
    if 'news2016zh' in from_file:
        data_head = {k:v for k, v in enumerate(['news_id', 'keywords', 'desc', 'title', 'source', 'time', 'content'])}
        data = pd.read_csv(from_file, header=None).rename(columns=data_head)[['title', 'content']]
        data = data.assign(**{c:data[c].str.replace(f'"{c}": "', '').str[:-1] for c in data.columns})
    elif 'lcsts_data' in from_file:
        data = pd.read_json(configuration.DIR_DATA+'/zh/lcsts_data.json')
        # remove hashtag & quotation marks in label column
        data = data.assign(title=data.title.str.replace('#|“|”', ''))
    
    # specify the length range for datasets
    print(' Before clean, data size is:, ', len(data))
    data = data.assign(length=data.content.apply(len))
    data = data[(min_len<=data.length)&(data.length<=max_len)]
    print(' After clean, data size is:, ', len(data))

    # sample part of data
    if size:
        size = min(size, len(data))
        data = data.sample(n=size, replace=False, random_state=random_state)

    # show size & plot distribution
    print("  Data Size: ", data.shape[0])
    plt.hist(data.length, bins=100)
    if plot:
        print("  Plot length distribution of content")
        plt.show()
    else:
        plt.savefig(f'../reports/figures/data-{name}.jpg')
    data = data.drop('length', axis=1)

    # preprocess texts & labels
    print("  Preprocess & Transfer from Simplified Chinese to Tranditional Chinese...")
    data = data.swifter.applymap(lambda text:preprocessors[lang](cc.convert(text)).numpy().decode().replace(' ', ''))
    data = data.dropna().reset_index(drop=True)
    
    # split dataset & output
    data = data.rename(columns={'title':'target', 'content':'source'})
    if split:
        data_train, data_valid = train_test_split(data, test_size=split, random_state=random_state)
        return data_train, data_valid
    else:
        return data

    
config = configparser.ConfigParser()
config.read('../config/model.cfg')

### read configurations

seed = config['basic'].getint('seed')

lang = config['data']['lang']
test_file = config['data']['test_file']
train_file = config['data']['train_file']
  
min_len = config['data'].getint('min_len')    
max_len = config['data'].getint('max_len')


if __name__ == '__main__':

    # setup size
    train_size = config['data'].getint('train_size')
    test_size = config['data'].getint('test_size')  

    # setup path
    train_from_path = os.path.join(configuration.DIR_DATA, lang, train_file)
    test_from_path = os.path.join(configuration.DIR_DATA, lang, test_file)
    train_to_path = os.path.join(configuration.DIR_INTERMIN, lang, 'train.zip')
    valid_to_path = os.path.join(configuration.DIR_INTERMIN, lang, 'valid.zip')
    test_to_path = os.path.join(configuration.DIR_INTERMIN, lang, 'test.zip')

    # preprocess 
    if test_file:
        test = preprocess_raw_data(test_from_path, lang, min_len, max_len, size=test_size, random_state=seed)
        test_size = len(test)
    else:
        test = None
        test_size *= 2

    train, valid = preprocess_raw_data(train_from_path, lang, min_len, max_len, 
                                       size=train_size, split=test_size, random_state=seed)
    if not bool(test):
        valid, test = train_test_split(valid, test_size=0.5, random_state=seed)
    
    # save to intermin directory
    train.to_csv(train_to_path, index=None)
    valid.to_csv(valid_to_path, index=None)
    test.to_csv(test_to_path, index=None)
