import os
import datetime
import configparser
import streamlit as st

from news_crawler import news_crawler

from utils.preprocessor import preprocessors
from utils.servitization import HF2TFSeq2SeqPipeline

config = configparser.ConfigParser()
config.read('../config/streamlit.cfg')

lang = config['data']['lang']
max_lengths = {'inp':config['data'].getint('inp_len'), 'tar':config['data'].getint('tar_len')}
text_preprocessors = {'inp':preprocessors[lang], 'tar':preprocessors[lang]}

### setup page

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
     - time_mark: cache the crawled results for the same time_mark
    """
    articles = news_crawler(keyword, max_len)
    return articles
    
@st.cache(allow_output_mutation=True)
def build_model_pipeline(config, text_preprocessors):
    # Build pipeline
    pipeline = HF2TFSeq2SeqPipeline(config['path']['predictor_dir'], config['path']['pretrain_dir'], text_preprocessors)
    bert_name = pipeline.inp_bert
    return pipeline, bert_name

def model_inference(pipeline, input_text):
    # Predict
    target_text, target_tokens, attention_weights = pipeline(input_text, max_lengths=max_lengths, return_attention=True)

    # Postprocess
    target_text = target_text.replace(' ', '').replace('[]', '')

    return target_text, target_tokens, attention_weights

### setup information

st.title("Stock News Summarizer")
st.subheader('Crawl the News & Summarize the Points')

st.write('')
keyword = st.text_input('Please input your keyword for search engine:', value='遠傳')

# divide columns
col1, col2, col3 = st.columns([3, 3, 3])

# create buttons & load models
start_to_summarize_all = col3.button('Summarize All News')
start_to_summarize = col2.button('Summarize Each News') | start_to_summarize_all
start_to_crawl = col1.button('Crawl Finance News') | start_to_summarize | start_to_summarize_all

pipeline, bert_name = build_model_pipeline(config, text_preprocessors)

# create expanders
expander3 = col3.expander("Click to Collapse" if start_to_summarize_all else 'Not Yet Summarized', expanded=True)
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
    articles = crawl_news(keyword, time_mark)

    for num, texts in enumerate(articles):
        expander1.write(f'[ NEWS {str(num+1).zfill(2)} ]')
        expander1.write('\n'.join(texts).replace(keyword, f'**{keyword}**'))

# summarizer for each news

if start_to_summarize:
    progress_bar2 = expander2.progress(0)

    each_summaries = []
    for num, texts in enumerate(articles):
        summary = '，'.join([model_inference(pipeline, text)[0] for text in texts])
        each_summaries.append(summary)
        progress_bar2.progress((num+1)/len(articles))

    for num, texts in enumerate(each_summaries):
        expander2.write(f'[ NEWS {str(num+1).zfill(2)} ]')
        expander2.write(texts)

    # summarizer for all news

    if start_to_summarize_all:
        all_summary = model_inference(pipeline, '，'.join(each_summaries))[0]
        expander3.write(all_summary)

### output input results

# sidebar

st.sidebar.header('Text Summarizer')

additional_text = st.sidebar.text_area(
    "Input your text here:",
    ("朋友買了一件衣料，綠色的底子帶白色方格，當她拿給我們看時，一位對圍棋十分感與趣的同學說：「啊，好像棋盤似的。」"
     "「我看倒有點像稿紙。」我說。「真像一塊塊綠豆糕。」一位外號叫「大食客」的同學緊接著說。"
     "我們不禁哄堂大笑，同樣的一件衣料，每個人卻有不同的感覺。那位朋友連忙把衣料用紙包好，她覺得衣料就是衣料，不是棋盤，也不是稿紙，更不是綠豆糕。"),
    height=400,
    max_chars=max_lengths['inp']
)

# sidebar summarizer

single_summarize = st.sidebar.button('Summarize')

if single_summarize:
    additional_result = model_inference(pipeline, additional_text)[0]
    st.sidebar.text(additional_result)
