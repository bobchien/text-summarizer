# **Stock News Summarizer**

## **Description**
The target of this project is to build a model with the ability to gain main information & ideas in short texts and articles, mainly the daily crawled stock news, and thus save a great amount of time to read and comprehend. Therefore, this project structures an end-to-end Transformer model using [**Hugging Face**](https://huggingface.co/) & [**Tensorflow**](https://www.tensorflow.org/?hl=zh-tw), which is composed of the pretrained bert tokenizer & encoder and the [customized tokenizer](https://github.com/bobscchien/text-tokenizer) (https://github.com/bobscchien/text-tokenizer) & decoder, to get a text summarizer. <br>
The second part of this project is to crawl stock news on the internet (currently only support the search engine of https://tw.finance.yahoo.com) with specified keywords. The modules of this crawler are **requests**, **BeautifulSoup4** and **urllib** <br> 
Finally, for demonstration purpose, [**streamlit**](https://streamlit.io/) is used to deploy this service on the [website](https://share.streamlit.io/bobscchien/text-summarizer/src/streamlit_app.py) (https://share.streamlit.io/bobscchien/text-summarizer/src/streamlit_app.py).

<br>

## **Dataset**

* [**LCSTS**](http://icrc.hitsz.edu.cn/Article/show/139.html): A Large-Scale Chinese Short Text Summarization Dataset <br> 
  >  _Automatic text summarization is widely regarded as the highly difficult problem, partially because of the lack of large text summarization data set. Due to the great challenge of constructing the large scale summaries for full text, we introduce a Large-scale Chinese Short Text Summarization dataset constructed from the Chinese microblogging website SinaWeibo. This corpus consists of over 2 million real Chinese short texts with short summaries given by the writer of each text. We also manually tagged the relevance of 10,666 short summaries with their corresponding short texts. Based on the corpus, we introduce recurrent neural network for the summary generation and achieve promising results, which not only shows the usefulness of the proposed corpus for short text summarization research, but also provides a baseline for further research on this topic._

* [**news2016zh**](https://github.com/brightmart/nlp_chinese_corpus) <br>
  This is an additional dataset not used in this project, but the preprocessing process is also included in `src/make_dataset.py` and thus can be used by simply modifying the configuration in `config/model.cfg`.

## **Usage**
**Configuration** <br>
`config/conf.ini` is the configuration file for main program to store `data` & `models` in a seperate data directory, while `config/conf-repo.ini` uses the paths in this repository.<br>

**Main Program** <br>
The usage part and the main program of this project can be divided into four parts:
1. Modify the configuration file in `config` based on your setting & environemnt.
2. Run `cd src && python make_dataset.py` to automatically process & save the raw dataset (default: [**LCSTS**](http://icrc.hitsz.edu.cn/Article/show/139.html)) to intermediate status. This script includes transferring Simplified Chinese to Traditional Chinese, converting halfwidth letters to fullwidth, and other preprocessing processes. 
3. Run `cd src && python make_tokenizer.py` to customize your own tokenizer based on the datasets, which also includes two datasets from [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) to add more words in the lexicon.

<br>

## **Notice**
*  
* 
*  
  
<br>

## **Further**
* 

<br>

## **Reference**

<br>

## **Acknowledgement**

<br>
