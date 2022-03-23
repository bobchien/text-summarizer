import re
import sys
import tqdm
from pprint import pprint
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import quote

def news_crawler(keyword, max_len):

    # define the headers of browsers
    headers = {
        "User-Agent":'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0'
    }

    # search your keywords with target url
    url = 'https://tw.finance.yahoo.com/news_search.html?ei=Big5&q=' + quote(keyword.encode('big5'))
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content, "html.parser")

    # get the link of each news and turn into soup object
    tag_name = 'span a'
    url_news = [link['href'] for link in soup.select(tag_name) if link['href'] != '#']
    soup_news = [BeautifulSoup(requests.get(link, headers=headers).content, "html.parser") 
                 for link in tqdm.tqdm(url_news)]

    # retrieve and clean the content of news
    text_news = [[re.sub(r'(\([^()]*\))|(【[^【】]*】)|(（[^（）]*）)|(\s)', '', text.text) 
                  for text in news.find_all('p') if 'class' not in str(text)] 
                 for news in soup_news
                 if '【公告】' not in str(news.select('header h1'))]

    # define a recursive function to split and concatenate each news based on the maximum length of our model

    def combine_text(texts, article):
        text_concat = ''
        # text: each paragraph of this news
        for text in texts:
            try:
                # for those paragraphs already greater than max_len, recursively use this function  
                if len(text) > max_len:
                    article += combine_text(text.split('，'), article)
                # try to fill text_concat to length max_len 
                elif len(text_concat+text) <= max_len:
                    text_concat += text
                # append and reset text_concat if it reaches max_len
                else:
                    article.append(text_concat)
                    text_concat = ''
            except:
                print('Cannot process this text:')
                print(text)
                pass
        return article

    articles = []
    for texts in text_news:
        article = combine_text(texts, [])
        if not len(article): continue
        articles.append(article)
        
    return articles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', dest='keyword', type=str, required=True, help='the keyword for search engine')
    parser.add_argument('-l', '--max_len', dest='max_len', type=int, default='256', help='the length the model can accept for a single time')    
    args = parser.parse_args()

    articles = news_crawler(args.keyword, args.max_len)
    pprint(articles)
