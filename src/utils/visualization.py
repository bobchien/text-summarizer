import tensorflow as tf

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

zhfont = FontProperties(fname='/usr/share/fonts/SimHei/SimHei.ttf')
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

def plot_attention_head(ax, in_tokens, target_tokens, attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.    
    target_tokens = target_tokens[1:]

    #ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(target_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in target_tokens.numpy()]
    ax.set_yticklabels(labels)
    
    return ax

def plot_attention_weights(sentence, target_tokens, attention_weights, tokenizers, layer=1, head=None, show=True):
    # Count heads & Select layer
    layers = sorted([l for l in attention_weights.keys() if 'block2' in l])
    attention_heads = attention_weights[layers[layer-1]][0]
    
    # Clean tokens
    try:
        index = target_tokens.index('[END]')
    except:
        index = -1
    target_tokens = target_tokens[:index]
    attention_heads = attention_heads[:, 1:index]
    
    # Tokenization
    try:
        in_tokens = tf.convert_to_tensor([sentence])
        in_tokens = tokenizers.inp.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.inp.lookup(in_tokens)[0]
    except:
        in_tokens = tokenizers.inp.tokenize(sentence, add_special_tokens=True)
        in_tokens = tf.convert_to_tensor(in_tokens)
        target_tokens = tf.convert_to_tensor(target_tokens)
    
    # Plot
    
    num_heads = len(attention_heads)
    
    if head:
        attention_head = attention_heads[head-1]
        
        plt.rcParams['figure.figsize'] = (8, 8)
        fig, ax = plt.subplots()
        ax = plot_attention_head(ax, in_tokens, target_tokens, attention_head)
    else:
        rows = math.ceil(num_heads/4)
        
        plt.rcParams['figure.figsize'] = (16, int(8/2)*rows)
        fig, axes = plt.subplots(rows, 4)

        for h, attention_head in enumerate(attention_heads):
            axes[h//4, h%4] = plot_attention_head(axes[h//4, h%4], in_tokens, target_tokens, attention_head)
            axes[h//4, h%4].set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig