from typing import List, Union

import re
import string
import opencc
import tensorflow as tf

### Chinese convertor

cc = opencc.OpenCC('s2twp')

def cc_converter(text: tf.Tensor):
    return cc.convert(text.numpy().decode())

### text_preprocessor

# Given the punctuation list
ZHON_KEEP  = '\u4e00-\u9FFF' # All Chinese words
ZHON_KEEP += '\uFF10-\uFF19' # All fullwidth numbers
ZHON_KEEP += '\uFF41-\uFF5A' # All fullwidth lower case English letters
ZHON_KEEP += '\u3105-\u3129\u02CA\u02CB\u02C7\u02C9' # Mandarin Phonetic Symbols: ㄅ-ㄦ˙ˊˇˋ
ZHON_PUNC = "！＂＃＄％＆＇（）＊＋，－｡。／：；＜＝＞？＠［＼］＾＿｀｛｜｝～–….．､、《》〈〉｢｣「」『』【】〔〕‘'‛“”„‟"

# Define the functions for fullwidth / halfwidth transformation

def strQ2B(ustring: str) -> str:
    """transform fullwidth text to halfwidth text"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # fullwidth space
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def strQ2B_tf(ustring: tf.Tensor) -> tf.Tensor:
    """transform fullwidth text to halfwidth text"""
    ustring = tf.strings.unicode_decode(tf.strings.regex_replace(ustring, '　', ' '), 'UTF-8')
    rstring = tf.map_fn(
        fn=lambda c:tf.where((c >= 65281) & (c <= 65374), c - 65248, c),
        elems=ustring
    )
    return tf.strings.unicode_encode(rstring, 'UTF-8')

def strB2Q(ustring: str) -> str:
    """transform halfwidth text to fullwidth text"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 32:  # halfwidth space
                pass #inside_code = 12288
            elif (inside_code >= 33 and inside_code <= 126): 
                inside_code += 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def strB2Q_tf(ustring: tf.Tensor) -> tf.Tensor:
    """transform halfwidth text to fullwidth text"""
    ustring = tf.strings.unicode_decode(ustring, 'UTF-8')
    rstring = tf.map_fn(
        fn=lambda c:tf.where((c >= 33) & (c <= 126), c + 65248, c),
        elems=ustring
    )
    return tf.strings.unicode_encode(rstring, 'UTF-8')

# Define the function for preprocess

def zh_preprocess(text: Union[List[str], tf.Tensor], py_function: bool = False) -> Union[List[str], tf.Tensor]:
    """ Chinese preprocess function for both list of string & tf.Tensor, based on py_function argument"""
    if py_function:
        # Since the tokenizers from hugging face don't support tensor input,
        # we neet to use the native python replacement mechanism to do it.
        def preprocess(sentence):
            sentence = sentence.lower()
            # transform quote marks
            sentence = re.sub("“", "「", sentence)
            sentence = re.sub("”", "」", sentence)
            # halfwidth -> fullwidth
            sentence = strB2Q(sentence)
            # Clean HTML markers
            sentence = re.sub("<[^>]+>", "", sentence)
            # Clean non-target texts
            sentence = re.sub('[^%s%s%s0-9]' % (re.escape(string.punctuation), ZHON_PUNC, ZHON_KEEP), '', sentence)
            # Pretrained bert tokenizers will handle the subwords so we don't add additional spaces among words
            # Replacing beginning, endding and multiple continuous spaces with a single space
            sentence = re.sub("(?<=.)(?!$)", " ", sentence)
            sentence = re.sub(r"\s\s+", " ", sentence)
            sentence = sentence.strip()
            return sentence
        
        if type(text) == str:
            text = [text]
        text = [preprocess(sentence) for sentence in text]
    else:
        text = tf.strings.lower(text)
        # transform quote marks
        text = tf.strings.regex_replace(text, "“", "「")
        text = tf.strings.regex_replace(text, "”", "」")
        # halfwidth -> fullwidth
        text = strB2Q_tf(text)
        # Clean HTML marker
        text = tf.strings.regex_replace(text, "<[^>]+>", "")
        # Clean non-target texts
        text = tf.strings.regex_replace(text, '[^%s%s%s0-9]' % (re.escape(string.punctuation), ZHON_PUNC, ZHON_KEEP), '')
        # Adding a space amoung words to allow better tokenization 
        # (Therefore we don't need to replace words into spaces in the previous steps)
        text = tf.strings.regex_replace(text, "[^\s]", r" \0 ")
        # Replacing beginning, endding and multiple continuous spaces with a single space
        text = tf.strings.regex_replace(text, r"\s\s+", " ")
        text = tf.strings.strip(text)
    return text

def en_preprocess(text: Union[List[str], tf.Tensor], py_function: bool = False) -> Union[List[str], tf.Tensor]:
    """ English preprocess function for both list of string & tf.Tensor, based on py_function argument"""
    if py_function:
        # Since the tokenizers from hugging face don't support tensor input,
        # we neet to use the native python replacement mechanism to do it.
        def preprocess(sentence):
            sentence = sentence.lower()
            # transform quote marks
            sentence = re.sub("“", '"', sentence)
            sentence = re.sub("”", '"', sentence)
            # fullwidth -> halfwidth
            sentence = strQ2B(sentence)
            # Clean HTML markers
            sentence = re.sub("<[^>]+>", " ", sentence)
            # Clean non-target texts
            sentence = re.sub('[^a-z %s%s0-9]' % (re.escape(string.punctuation), ZHON_PUNC), ' ', sentence)
            # Add space between numbers to make tokenizer to only record 0-9            
            sentence = re.sub("(?<=[0-9])(?!$)", " ", sentence)
            # Replacing beginning, endding and multiple continuous spaces with a single space
            sentence = re.sub(r"\s\s+", " ", sentence)
            sentence = sentence.strip()
            return sentence
        
        if type(text) == str:
            text = [text]
        text = [preprocess(sentence) for sentence in text]
    else:
        text = tf.strings.lower(text)
        # transform quote marks
        text = tf.strings.regex_replace(text, "“", '"')
        text = tf.strings.regex_replace(text, "”", '"')
        # fullwidth -> halfwidth 
        text = strQ2B_tf(text)
        # Clean HTML marker
        text = tf.strings.regex_replace(text, "<[^>]+>", " ")
        # Clean non-target texts
        text = tf.strings.regex_replace(text, '[^a-z %s%s0-9]' % (re.escape(string.punctuation), ZHON_PUNC), ' ')
        # Add space between numbers to make tokenizer to only record 0-9
        text = tf.strings.regex_replace(text, "[0-9]", r" \0 ")
        # Replacing beginning, endding and multiple continuous spaces with a single space
        text = tf.strings.regex_replace(text, r"\s\s+", " ")
        text = tf.strings.strip(text)
    return text

preprocessors = {'zh':zh_preprocess, 'zhc':zh_preprocess, 'en':en_preprocess}
