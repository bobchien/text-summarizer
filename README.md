# **Stock News Summarizer**

## **Description**
The target of this project is to build a model with the ability to gain main information & ideas in short texts and articles, mainly the daily crawled stock news, and thus save a great amount of time to read and comprehend. Therefore, this project structures an end-to-end Transformer model using [**Hugging Face**](https://huggingface.co/) & [**Tensorflow**](https://www.tensorflow.org/?hl=zh-tw), which is composed of the pretrained bert tokenizer & encoder and the [customized tokenizer](https://github.com/bobscchien/text-tokenizer) & decoder, to get a text summarizer. <br>
The second part of this project is to crawl stock news on the internet (currently only support the search engine of https://tw.finance.yahoo.com) with specified keywords. The modules of this crawler are **requests**, **BeautifulSoup4** and **urllib**. <br> 
Finally, for demonstration purpose, [**streamlit**](https://streamlit.io/) is used to deploy this service on the [demo website](https://share.streamlit.io/bobscchien/text-summarizer/src/streamlit_app.py).

<br>

## **Dataset**

* [**LCSTS**](http://icrc.hitsz.edu.cn/Article/show/139.html): A Large-Scale Chinese Short Text Summarization Dataset <br> 
  >  _Automatic text summarization is widely regarded as the highly difficult problem, partially because of the lack of large text summarization data set. Due to the great challenge of constructing the large scale summaries for full text, we introduce a Large-scale Chinese Short Text Summarization dataset constructed from the Chinese microblogging website SinaWeibo. This corpus consists of over 2 million real Chinese short texts with short summaries given by the writer of each text. We also manually tagged the relevance of 10,666 short summaries with their corresponding short texts. Based on the corpus, we introduce recurrent neural network for the summary generation and achieve promising results, which not only shows the usefulness of the proposed corpus for short text summarization research, but also provides a baseline for further research on this topic._

* [**news2016zh**](https://github.com/brightmart/nlp_chinese_corpus) <br>
  This is an additional dataset not used in this project, but the preprocessing process is also included in `src/make_dataset.py` and thus can be used by simply modifying the configuration in `config/model.cfg`.

<br>

## **Usage**
**Configuration** <br>
1. `config/conf.ini` is the configuration file for main program to store `data` & `models` in a seperate data directory, while `config/conf-repo.ini` uses the paths in this repository. 
2. `config/model.cfg` stores the hyperparamters of model structure & training strategy, and ones can tune their models by changing the values in this configuration file. 
3. As for `config/streamlit.cfg` & `config/streamlit-deploy.cfg`, the former one is for local deployment and the latter one is for sharing. Notice that the latter one should correspond to the paths of streamlit deployment in the GitHub repository <br>

**Main Program - Model Training** <br>
The usage part and the main program of this project can be divided into 5 parts:
1. Modify the configuration file in `config` based on your setting & environemnt.
2. Run `cd src && python make_dataset.py` to automatically process & save the raw dataset (default: [**LCSTS**](http://icrc.hitsz.edu.cn/Article/show/139.html)) to intermediate status. This script includes transferring Simplified Chinese to Traditional Chinese, converting halfwidth letters to fullwidth for Chinese (in reverse for English), and other preprocessing processes. 
3. Run `cd src && python make_tokenizer.py` to customize your own tokenizer based on the datasets, which also includes two datasets from [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) to add more words in the lexicon.
4. Run `cd src && python make_tfrecord.py` to tokenize and transform the CSV dataset (intermediate files) into TFRecord dataset (processed files). For source data, this script uses pretrained tokenizers from [**Hugging Face**](https://huggingface.co/), and results in 2 ~ 3 inputs (tokens, masks, ids) depending on the problem situation. For target data, only 1 output will be generated since here we use the basic form of the Transformer Decoder which refers to [this tutorial](https://www.tensorflow.org/text/tutorials/transformer).
5. Finally, run `cd src && python model_training.py` to train the model with the processed dataset of TFRecord files. The configuration of models and traingings can be modified in `config/model.cfg`, and the outputs of tensorboard logs, model checkpoints, and saved models will be stored based on `config/conf.ini` or `config/conf-repo.ini`.

**Main Program - Streamlit Demonstration** <br>
1. `src/streamlit_app.py` on GitHub was deployed as a web application via [**Streamlit Share**](https://share.streamlit.io/), and the demonstration of this project can be found [here](https://share.streamlit.io/bobscchien/text-summarizer/src/streamlit_app.py).
2. `requirements.txt` is necessary for deployment, which **Streamlit Share** will first build your environment based on the dependency. More details about application deployment can be found on the [official website](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app).
3. To deploy a large trained model in your application, please refer to [this discussion](https://discuss.streamlit.io/t/how-to-download-large-model-files-to-the-sharing-app/7160/5). However, this method does not work for my project, and thus I use `pip install gdown` and use the api `gdown.download`, which is in `src/clouds/connect_gdrive.py`.

<br>

## **Further**
* For now, the structure of this project is an naive end-to-end / seq2seq transformer model. To further improve the summarization quality, the methods below will be tested and applied in the future:
  * [2022, Text summarization based on multi-head self-attention mechanism and pointer network](https://link.springer.com/article/10.1007/s40747-021-00527-2)
  * [2021, Neural Abstractive Text Summarization with Sequence-to-Sequence Models](https://dl.acm.org/doi/abs/10.1145/3419106)
  * [2020, Controlling the Amount of Verbatim Copying in Abstractive Summarization](https://arxiv.org/pdf/1911.10390.pdf)
  * [2020, Self-Attention Guided Copy Mechanism for Abstractive Summarization](https://aclanthology.org/2020.acl-main.125.pdf)
  * [2018, Structure-Infused Copy Mechanisms for Abstractive Summarization](https://aclanthology.org/C18-1146.pdf)
  * [2017, Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* This project now only supports input texts with maximum length equal to 256 words, and thus might not be able to comprehensively capture the meaning of whole texts when inputs are too long. Listed researches can deal with this issue:
  * [2020, Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
  * [2020, CogLTX: Applying BERT to Long Texts](https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf)  
* Try to cope with non-Chinese words in the source and target texts by using sub-token mechanism.

<br>
