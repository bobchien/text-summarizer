import numpy as np
import tensorflow as tf



def print_seq2seq(sentence, prediction, ground_truth=None):
    print(f'{"Input:":15s}: {sentence}')
    try:
        print(f'{"Prediction":15s}: {prediction.numpy().decode("utf-8")}')
    except:
        print(f'{"Prediction":15s}: {prediction}')
    if ground_truth:
        print(f'{"Ground truth":15s}: {ground_truth}')


class Seq2SeqMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_texts, tokenizers, predict_step, num_examples=3):
        self.test_texts = test_texts
        self.tokenizers = tokenizers
        self.predict_step = predict_step
        self.num_examples = num_examples

    def on_epoch_end(self, epoch, logs=None):
        for dataset in self.test_texts.unbatch().batch(self.num_examples).take(1):
            inp, tar = dataset
            tar_pred = self.predict_step(self.model, dataset, tensor=True)
            tar_pred = tf.cast(tar_pred, tf.int64)

        pred_tokens = self.tokenizers.tar.detokenize(tar_pred)
        pred_texts = [text.decode() for text in pred_tokens.numpy()]
        real_tokens = self.tokenizers.tar.detokenize(tf.cast(tar, tf.int64))
        real_texts = [text.decode() for text in real_tokens.numpy()]
        try:
            # Tensorflow customized tokenizers
            inp_tokens = self.tokenizers.inp.detokenize(tf.cast(inp, tf.int64))
            inp_texts = [text.decode() for text in inp_tokens.numpy()]
        except:
            # Hugging Face tokenizers
            inp_texts = self.tokenizers.inp.batch_decode(tf.cast(inp[0], tf.int64), skip_special_tokens=True)
            
        for inp, real, pred in zip(inp_texts, real_texts, pred_texts):
            print("\n    Input:", inp)
            print("    Target:", real)
            print("    Predict:", pred)

            
class TextGeneratorMonitor(tf.keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        start_tokens: List of integers, the token indices for the starting prompt.
        tokenizer:
        sample_length: Integer, the number of tokens to be generated after prompt.
        max_length:
        top_k: Integer, sample from the `top_k` token predictions.
    """
    def __init__(self, start_tokens, tokenizer, sample_length, max_length, top_k=10):
        self.start_tokens = start_tokens
        self.tokenizer = tokenizer
        self.sample_length = sample_length
        self.max_length = max_length
        self.top_k = top_k
        
    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")

        preds = tf.nn.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        
        return np.random.choice(indices, p=preds)
    
    def on_epoch_end(self, epoch, logs=None):

        eos = tf.cast(self.tokenizer.tokenize([''])[0][-1], tf.int32).numpy() 

        num_tokens_generated = 0
        tokens_generated = []
        tokens_temporary = self.start_tokens.numpy().tolist()
        while num_tokens_generated <= self.sample_length:
            
            pad_len = self.max_length - len(tokens_temporary)
            sample_index = len(tokens_temporary) - 1
            if pad_len < 0:
                inputs = tokens_temporary[:maxlen]
                sample_index = self.max_length - 1
            elif pad_len > 0:
                inputs = tokens_temporary + [0] * pad_len
            else:
                inputs = tokens_temporary
            
            logits, _ = self.model(np.array(inputs)[np.newaxis])
            sample_token = self.sample_from(logits[0][sample_index])
            tokens_generated.append(sample_token)
            tokens_temporary.append(sample_token)

            num_tokens_generated = len(tokens_generated)
            
            if sample_token == eos:
                break

        tokens = self.tokenizer.detokenize(np.array(tokens_temporary)[np.newaxis])
        texts = ''.join([token.decode().strip() for token in tokens.numpy()]).replace(' ', '')
        print(f"\n Generated Text: {texts}\n")
