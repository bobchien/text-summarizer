from utils.initialization import *
from utils.preprocessor import preprocessors
from utils.servitization import HF2TFSeq2SeqPipeline
from utils.callback import print_seq2seq

### Reload the pipeline to verify the result

predictor_dir = ''
pretrain_dir = DIR_MODELTORCH
text_preprocessors = {'inp': preprocessors[lang], 'tar': preprocessors[lang]}

pipeline = HF2TFSeq2SeqPipeline(predictor_dir, pretrain_dir, text_preprocessors)
