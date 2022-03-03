import os
import configparser

config_file = '../config/conf.ini'
config = configparser.ConfigParser()
config.read(config_file)

### Configuration

DIR_MODELTENSOR  = config['path']['DIR_MODEL_TF']
DIR_MODELTORCH  = config['path']['DIR_MODEL_PT']
DIR_TOKENIZER = config['path']['DIR_TOKENIZER']

DIR_DATA_TOP = config['path']['DIR_DATA_TOP']
DIR_MODEL_TOP = config['path']['DIR_MODEL_TOP']

# Define Directories

DIR_VOCAB      = os.path.join(DIR_TOKENIZER, 'data', 'vocab')
DIR_TOKEN      = os.path.join(DIR_TOKENIZER, 'models', 'trained')

DIR_DATA       = os.path.join(DIR_DATA_TOP, 'raw')
DIR_INTERMIN   = os.path.join(DIR_DATA_TOP, 'intermin')
DIR_TFRECORD   = os.path.join(DIR_DATA_TOP, 'processed')

DIR_MODEL      = os.path.join(DIR_MODEL_TOP, 'savedmodels')
DIR_CHECKPOINT = os.path.join(DIR_MODEL_TOP, 'checkpoints')
DIR_LOG        = os.path.join(DIR_MODEL_TOP, 'logs')

# Setup Directories

DIRs = {key:value for key, value in globals().items()}
for key, value in DIRs.items():
    if 'DIR_' in key:
        if os.path.isdir(value):
            print(f"Directory {value} exists.")
        else:
            print(f"Creating {value}...")
            os.makedirs(value)
            print(f" Succeeded!!!")
            
# Global Variables

SEED = config.getint('global', 'SEED')
