import os
import datetime
import configparser

import streamlit as st

from news_crawler import news_crawler
from utils.preprocessor import preprocessors, strQ2B
from utils.servitization import HF2TFSeq2SeqPipeline

config = configparser.ConfigParser()

try:
    # local running
    app_local = True
    
    config.read('../config/streamlit.cfg')
    lang = config['data']['lang']
except:
    # github running 
    app_local = False
    
    config.read('/app/text-summarizer/config/streamlit-deploy.cfg')
    lang = config['data']['lang']
    gdrive_assests_vocab_id   = str(st.secrets["gdrive"]["assests_vocab_id"])
    gdrive_variables_data_id  = str(st.secrets["gdrive"]["variables_data_id"])
    gdrive_variables_index_id = str(st.secrets["gdrive"]["variables_index_id"])
    gdrive_saved_model_id     = str(st.secrets["gdrive"]["saved_model_id"])
    
predictor_dir = config['path']['predictor_dir']

max_lengths = {'inp':config['data'].getint('inp_len'), 'tar':config['data'].getint('tar_len')}
text_preprocessors = {'inp':preprocessors[lang], 'tar':preprocessors[lang]}

### setup page

# https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
st.set_page_config(
    page_title="bobscchien/text-summarizer",
    page_icon="rolled_up_newspaper",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help":None, "Report a Bug":None}
)

### setup functions

@st.cache()
def crawl_news(keyword, time_mark, max_len=max_lengths['inp']):
    """
     - keyword: input words for the search engine
     - time_mark: cache the crawled results for the same time_mark
    """
    articles = news_crawler(keyword, max_len)
    return articles
    
@st.cache(allow_output_mutation=True)
def build_model_pipeline(config, text_preprocessors):
    # Build pipeline
    if not app_local:
        from clouds.connect_gdrive import gdown_file_from_google_drive

        # create tmp directories
        os.makedirs(predictor_dir, exist_ok=True)
        os.makedirs(os.path.join(predictor_dir, 'assets'), exist_ok=True)
        os.makedirs(os.path.join(predictor_dir, 'variables'), exist_ok=True)
        
        # setup checkpoint to avoid reloading
        file_vocab = os.path.join(predictor_dir, f"assets/{lang}_vocab.txt")
        file_data  = os.path.join(predictor_dir, "variables/variables.data-00000-of-00001")
        file_index = os.path.join(predictor_dir, "variables/variables.index")
        file_model = os.path.join(predictor_dir, "saved_model.pb")
        
        for file, gdrive_id in zip(
            [
                file_vocab, file_data, file_index, file_model
            ],
            [
                gdrive_assests_vocab_id, gdrive_variables_data_id,
                gdrive_variables_index_id, gdrive_saved_model_id    
            ]
        ):
            if os.path.isfile(file): continue
            gdown_file_from_google_drive(gdrive_id, file)
            #download_file_from_google_drive(gdrive_id, file)

    pipeline = HF2TFSeq2SeqPipeline(predictor_dir, config['path']['pretrain_dir'], text_preprocessors)
    bert_name = pipeline.inp_bert
    return pipeline, bert_name

def model_inference(pipeline, input_text):
    # Predict
    target_text, target_tokens, attention_weights = pipeline(input_text, 
                                                             max_lengths=max_lengths, 
                                                             return_attention=True)

    # Postprocess
    target_text = target_text.replace(' ', '').replace('[]', '')

    return target_text, target_tokens, attention_weights

### setup information

st.title("Stock News Summarizer")
col_left, col_right = st.columns([1, 1])

text = (
    'This project structures an end-to-end Transformer model using [**Hugging Face**](https://huggingface.co/) & [**Tensorflow**](https://www.tensorflow.org/?hl=zh-tw), '
    'which is composed of the pretrained bert tokenizer & encoder and the [customized tokenizer](https://github.com/bobscchien/text-tokenizer) & decoder, '
    'to get a text summarizer. '
    '[_**Source Code**_](https://github.com/bobscchien/text-summarizer)'
)
col_left.markdown(text)

st.write('')
st.subheader('**Crawl News & Retrieve Summary**')
keyword = st.text_input('Input your keyword for the search engine:', value='台灣')

# create buttons & load models
col1, col2 = st.columns([1, 2])
start_to_summarize = col2.button('Summarize Each News')
start_to_crawl = col1.button('Crawl Finance News') | start_to_summarize

pipeline, bert_name = build_model_pipeline(config, text_preprocessors)

# create expanders
col1, col2, col3 = st.columns([1, 1, 1])
expander3 = col3.expander("Click to Collapse" if start_to_summarize else 'Not Yet Summarized', expanded=True)
expander2 = col2.expander("Click to Collapse" if start_to_summarize else 'Not Yet Summarized', expanded=True)
expander1 = col1.expander("Click to Collapse" if start_to_crawl else 'Current Time', expanded=True)

current_time = datetime.datetime.now()
expander1.caption(str(current_time.strftime("%Y-%m-%d %H:%M:%S")))

### output news results

# crawler

if start_to_crawl:
    # update the news every n minutes
    n = 5
    time_mark = (current_time.hour*60+current_time.minute) // n
    articles_raw = crawl_news(keyword, time_mark)

    count = 0
    articles = []
    for texts in articles_raw:
        # before applying Longformer, only demonstrate those shorter than 2 * max_inp_len texts
        if len(''.join(texts)) > max_lengths['inp']*2: continue
        expander1.write(f'[ NEWS {str(count+1).zfill(2)} ]')
        expander1.write('\n'.join(texts).replace(keyword, f'**{keyword}**'))
        articles.append(texts)
        count += 1
        
# summarizer for each news

if start_to_summarize:
    progress_bar2 = expander2.progress(0)

    each_summaries = []
    for num, texts in enumerate(articles):
        summary = '，'.join([model_inference(pipeline, text)[0] for text in texts])
        summary = strQ2B(summary)
        each_summaries.append(summary)
        progress_bar2.progress((num+1)/len(articles))

    for num, texts in enumerate(each_summaries):
        expander2.write(f'[ NEWS {str(num+1).zfill(2)} ]')
        expander2.write(texts)

    # summarizer for all news

    all_summary = model_inference(pipeline, '，'.join(each_summaries))[0]
    all_summary = strQ2B(all_summary)
    expander3.write(all_summary)

### output input results

# sidebar

st.sidebar.header('Text Summarizer')

text_sample = {}
text_sample[1] = (
    "朋友買了一件衣料，綠色的底子帶白色方格，當她拿給我們看時，一位對圍棋十分感與趣的同學說：「啊，好像棋盤似的。」"
    "「我看倒有點像稿紙。」我說。「真像一塊塊綠豆糕。」一位外號叫「大食客」的同學緊接著說。"
    "我們不禁哄堂大笑，同樣的一件衣料，每個人卻有不同的感覺。"
    "那位朋友連忙把衣料用紙包好，她覺得衣料就是衣料，不是棋盤，也不是稿紙，更不是綠豆糕。"
)
text_sample[2] = (
    "把一隻貓關在一個封閉的鐵容器裏面，並且裝置以下儀器（注意必須確保這儀器不被容器中的貓直接干擾）："
    "在一台蓋格計數器內置入極少量放射性物質，在一小時內，這個放射性物質至少有一個原子衰變的機率為50%，它沒有任何原子衰變的機率也同樣為50%；"
    "假若衰變事件發生了，則蓋格計數管會放電，通過繼電器啟動一個榔頭，榔頭會打破裝有氰化氫的燒瓶。經過一小時以後，假若沒有發生衰變事件，則貓仍舊存活；"
    "否則發生衰變，這套機構被觸發，氰化氫揮發，導致貓隨即死亡。"
    "用以描述整個事件的波函數竟然表達出了活貓與死貓各半糾合在一起的狀態。"
)

num = st.sidebar.selectbox('Select one sample texts if needed', 
                           [2]+list(text_sample.keys()))
additional_text = st.sidebar.text_area(
    "Input your text here:",
    text_sample[num] if num else "",
    height=400,
    max_chars=max_lengths['inp']
)

# sidebar summarizer

single_summarize = st.sidebar.button('Summarize')

if single_summarize | start_to_summarize:
    additional_result = model_inference(pipeline, additional_text)[0]
    additional_result = strQ2B(additional_result)
    st.sidebar.text(additional_result)
