import os
import configparser

colab = False

config_file = '../config/conf.ini'
config = configparser.ConfigParser()
config.read(config_file)

### Configuration

PROJECT_NAME = config['global']['PROJECT_NAME']

DIR_MODELTENSOR  = config['path']['DIR_MODEL_TF']
DIR_MODELTORCH  = config['path']['DIR_MODEL_PT']
DIR_TOKENIZER = config['path']['DIR_TOKENIZER']

DIR_DATA_TOP = config['path']['DIR_DATA_TOP']
DIR_MODEL_TOP = config['path']['DIR_MODEL_TOP']

# initialize null values

DIR_VOCAB, DIR_TOKEN = None, None
DIR_DATA, DIR_INTERMIN, DIR_TFRECORD = None, None, None
DIR_MODEL, DIR_CHECKPOINT, DIR_LOG = None, None, None
    
### Setup Directories

def changeToColabPath(colab, project_name=PROJECT_NAME, gcs_path="gs://bobscchien-project-data"):
    """
     - colab: whether to run on Colab or not
     - gcs_path: root path of GCP
    """
    global DIR_MODELTENSOR, DIR_MODELTORCH, DIR_TOKENIZER
    global DIR_DATA_TOP, DIR_MODEL_TOP
    
    global DIR_VOCAB, DIR_TOKEN
    global DIR_DATA, DIR_INTERMIN, DIR_TFRECORD
    global DIR_MODEL, DIR_CHECKPOINT, DIR_LOG
    
    if colab:
        DIR_MODELTENSOR = os.path.join(gcs_path, 'Model_Tensorflow')
        DIR_MODELTORCH  = os.path.join(gcs_path, 'Model_Pytorch')
        
        DIR_TOKENIZER = os.path.join(gcs_path, project_name, 'tokenizers')
        DIR_DATA_TOP  = os.path.join(gcs_path, project_name, 'data')
        DIR_MODEL_TOP = os.path.join(gcs_path, project_name, 'models')

    # Define Directories

    DIR_VOCAB      = os.path.join(DIR_TOKENIZER, 'data', 'vocab')
    DIR_TOKEN      = os.path.join(DIR_TOKENIZER, 'models', 'trained')

    DIR_DATA       = os.path.join(DIR_DATA_TOP, 'raw')
    DIR_INTERMIN   = os.path.join(DIR_DATA_TOP, 'intermin')
    DIR_TFRECORD   = os.path.join(DIR_DATA_TOP, 'processed')

    DIR_MODEL      = os.path.join(DIR_MODEL_TOP, 'savedmodels')
    DIR_CHECKPOINT = os.path.join(DIR_MODEL_TOP, 'checkpoints')
    DIR_LOG        = os.path.join(DIR_MODEL_TOP, 'logs')
    
def createDirectory():
    DIRs = {key:value for key, value in globals().items()}
    for key, value in DIRs.items():
        if 'DIR_' in key:
            if os.path.isdir(value):
                print(f"Directory {value} exists.")
            else:
                print(f"Creating {value}...")
                os.makedirs(value, exist_ok=True)
                print(f" Succeeded!!!")       

# Global Variables

SEED = config.getint('global', 'SEED')
