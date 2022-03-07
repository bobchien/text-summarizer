from utils.initialization import *
from utils.decoding import ids_decoder
from utils.callback import Seq2SeqMonitor
from utils.servitization import HF2TFSeq2SeqExporter
from data.dataset_tfrecord import loadTFRecord
from models.transformer_bert import *

### read configurations from other modules

from make_dataset import *
from make_tfrecord import *

BUFFER_SIZE = config['training'].getint('buffer_size')
BATCH_SIZE = config['training'].getint('batch_size')

EPOCHS = config['training'].getint('epochs')
EARLYSTOP = config['training'].getint('earlystop')
teacher = config['training'].getint('teacher')
init_lr = config['training'].getfloat('init_lr')

num_enc_layers = config['model'].getint('num_enc_layers')
num_dec_layers = config['model'].getint('num_dec_layers')
num_projection_layers = config['model'].getint('num_projection_layers')
embed_pos = config['model'].getboolean('embed_pos') 
embed_dim = config['model'].getint('embed_dim')
dense_dim = config['model'].getint('dense_dim')
num_heads = config['model'].getint('num_heads')
dropout = config['model'].getfloat('dropout')
activation = config['model']['activation']

num_tune = config['model'].getint('num_tune')
#nn_units = config['model'].getint('nn_units')
#use_lstm = config['model'].getboolean('use_lstm') 

lang_prefix = lang.upper()+'_'
bert_params = generate_bert_configuration(bert_names, cache_dirs)
beam_params = {'beam_size':config['inference'].getint('beam_size'), 'alpha':config['inference'].getfloat('beam_alpha'), 
               'vocab_size':tar_vocab_size, 'max_decode_length':max_lengths['tar'], 'eos_id':EOS_IDS['tar']}
sampler_params = {'TopK':config['inference'].getint('topk'), 
                  'TopP':config['inference'].getfloat('topp'), 
                  'temp':config['inference'].getfloat('temperature'), 
                  'tar_vocab_size':tar_vocab_size, 'max_length':max_lengths['tar'], 'eos_id':EOS_IDS['tar']}

score_name = config['training']['metric_name']
text_metric = datasets.load_metric(score_name)
print('\n', text_metric.features)

### define function for inference 

def detokenize_fn(ids, way, dtype=tf.int64):
    fn = getattr(tokenizers, way)
    
    ids = tf.cast(ids, dtype=dtype)
    
    if bool(bert_names[way]):
        return [[text] for text in fn.batch_decode(ids, skip_special_tokens=True)]
    else:
        return [[text.decode()] for text in fn.detokenize(ids).numpy()]

def predict_step(model, dataset, tensor=False):
    global text_metric
    
    inps, tar = dataset
    inp, inp_mask = inps

    # Initialization 
    initial_ids = tf.math.multiply(BOS_IDS['tar'], tf.ones_like(tar[:, 0]))

    # Create decoding cache based on the model structure
    cache = {}
    inp_embedded = model.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
    inp_embedded = model.embedding_projector(inp_embedded, training=False)
    cache['encoder_outputs'], cache['inp_padding_mask'] = model.encoder(inp_embedded, mask=inp_mask, training=False)
    cache['inp_padding_mask'] = tf.cast(cache['inp_padding_mask'], dtype=tf.float32)
    
    beam_params['symbols_to_logits_fn'] = model.symbols_to_logits_fn
    sampler_params['symbols_to_logits_fn'] = model.symbols_to_logits_fn    
    pred = ids_decoder(initial_ids, cache, 'BeamSearch', beam_params, sampler_params)

    if not tensor:
        inp_list = detokenize_fn(inp, 'inp')
        tar_list = detokenize_fn(tar, 'tar')
        pred_list = detokenize_fn(pred, 'tar')

        # Accumulate the results for later computation
        text_metric.add_batch(predictions=pred_list, references=tar_list)

        return inp_list, tar_list, pred_list
    else:
        return pred

########################### read configuration ###########################

GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

############################## setup dataset #############################

# read number of samples
num_samples = [file for file in tf.io.gfile.listdir(os.path.join(configuration.DIR_TFRECORD, lang)) if 'train' in file][0]
num_samples = int(num_samples[num_samples.rfind('-')+1:num_samples.rfind('.')])
print('\nNumber of Samples:', num_samples)

# load tfrecord
train_batches = loadTFRecord('train', os.path.join(configuration.DIR_TFRECORD, lang), GLOBAL_BATCH_SIZE, BUFFER_SIZE)
valid_batches = loadTFRecord('valid', os.path.join(configuration.DIR_TFRECORD, lang), GLOBAL_BATCH_SIZE)
test_batches = loadTFRecord('test',   os.path.join(configuration.DIR_TFRECORD, lang), GLOBAL_BATCH_SIZE, cache=False)

############################### setup model ###############################

with strategy.scope():

    ############################### optimizer

    steps_per_epoch = math.ceil(num_samples/GLOBAL_BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = official.nlp.optimization.create_optimizer(init_lr=init_lr,
                                                            num_train_steps=num_train_steps,
                                                            num_warmup_steps=num_warmup_steps,
                                                            optimizer_type='adamw')
    print(f'\nwarmup_steps: {num_warmup_steps}')

    ############################### model

    inp_pretrained_model = TFAutoModel.from_pretrained(**bert_params['inp'])  
    model = BertEncoderTransformer(inp_pretrained_model=inp_pretrained_model,
                                    num_tune=num_tune, 
                                    num_projection_layers=num_projection_layers, 
                                    num_enc_layers=num_enc_layers, 
                                    num_dec_layers=num_dec_layers, 
                                    embed_dim=embed_dim, 
                                    num_heads=num_heads, 
                                    dense_dim=dense_dim, 
                                    target_vocab_size=tar_vocab_size, 
                                    pe_target=max(max_lengths.values()), 
                                    activation=activation, 
                                    dropout=dropout, 
                                    embed_pos=embed_pos)
    model.compile(optimizer)

    display(tf.keras.utils.plot_model(model.build_graph(), to_file=f'../reports/{lang}-model.png',
                                      show_shapes=False, expand_nested=True))
    print('\n', model.build_graph().summary())

    model_name = lang_prefix+f"{model.return_model_name()}_{bert_names['inp']}".replace('/', '-')
    print('\nModel Name:', model_name)

    ############################### callback

    # Record the training configurations as the name of checkpoints, logs and models to compare the different results
    run_id  = model.run_id
    run_id += f"_{init_lr}initlr_{num_warmup_steps}warmup_{optimizer._name}"
    run_id += f"_{GLOBAL_BATCH_SIZE}batchsize_{BUFFER_SIZE}shuffle_{num_samples}samples_{teacher}%Teacher"
    print('\nrun_id:', run_id, '\n')

    # Setup the saving path
    checkpoint_path = os.path.join(configuration.DIR_CHECKPOINT, model_name+'_'+run_id, 'ckpt')
    log_dir = os.path.join(configuration.DIR_LOG, model_name+'_'+run_id)

    # Setup callbacks
    es_metric = 'val_loss'
        
    outputmonitor = Seq2SeqMonitor(test_batches, tokenizers, predict_step, num_examples=3)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=es_metric, save_weights_only=True, save_best_only=True, verbose=1)
    callbacks = [outputmonitor, tensorboard, checkpoint]

    if EARLYSTOP >= 0:
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor=es_metric, patience=EARLYSTOP, restore_best_weights=True, verbose=1)
        callbacks += [earlystopping]

    # If there are corresponding files, then ckpt will load and restore the status of objects
    try:
        model.load_weights(checkpoint_path)
        print(f'Restore the checkpoint at {checkpoint_path}\n')

        last_epoch = optimizer.iterations.numpy()//steps_per_epoch
        print(f'Latest checkpoint of {last_epoch} epochs restored!!')
    except:
        last_epoch = 0
        print("There is no existed checkpoint ... Start from the beginning.\n")

############################### start training ###############################

if __name__ == '__main__':

    history = model.fit(train_batches.repeat(EPOCHS), validation_data=valid_batches,
                        initial_epoch=last_epoch, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    ### measure the performance

    for dataset in tqdm.tqdm(test_batches):
        inp_list, tar_list, pred_list = predict_step(model, dataset)

    if 'bleu' in score_name:
        text_scores = text_metric.compute(lowercase=True)
        text_score = text_scores['score']
    elif 'rouge' in score_name:
        text_scores = text_metric.compute()
        text_score = text_scores['rouge1'].mid.fmeasure

    score = f"{score_name}-{text_score:.6f}"
    print('\nModel Score:', score)

    ### create and save the predictor

    config_detail = f"{model_name}_{score}_{run_id}"
    predictor = HF2TFSeq2SeqExporter(model, tokenizers, BOS_IDS, beam_params, sampler_params,
                                     bert_names, config_detail, lang, lang)
    predictor_dir = os.path.join(configuration.DIR_MODEL, f"{model_name}_{score}")
    tf.saved_model.save(predictor, export_dir=predictor_dir)
