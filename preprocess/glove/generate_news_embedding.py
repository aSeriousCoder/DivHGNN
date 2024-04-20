import pickle
import tqdm
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def tokenize_sentence(sentence, stop_words):
    words = word_tokenize(sentence)
    words_filtered = []
    for w in words:
        if w not in stop_words:
            words_filtered.append(w)
    return words


def unique(somelist):
    list_filtered = []
    for item in somelist:
        if item not in list_filtered:
            list_filtered.append(item)
    return list_filtered


def tokenize_news():
    '''
    1. mk word list
    2. generate word embedding
    3. mk news-word link list
    4. generate news embedding
    '''
    print('tokenize_news')
    stop_words = set(stopwords.words('english'))
    f = open('/home/gpzhang/Projects/News-Rec-v5/preprocess/glove/glove_dict.pkl', 'rb')
    glove = pickle.load(f)
    glove_words = glove.keys()

    dir = '/home/gpzhang/Projects/News-Rec-v5/data/MIND/MINDsmall'

    news_file_name = 'MINDsmall_train/news.tsv'
    news_file_path = '{}/{}'.format(dir, news_file_name)
    train_news_df = pd.read_csv(news_file_path, sep='\t',
                          names=["News-ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title-Entities",
                                 "Abstract-Entities"], index_col='News-ID')
    
    news_file_name = 'MINDsmall_dev/news.tsv'
    news_file_path = '{}/{}'.format(dir, news_file_name)
    dev_news_df = pd.read_csv(news_file_path, sep='\t',
                          names=["News-ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title-Entities",
                                 "Abstract-Entities"], index_col='News-ID')

    print('1. make word list and generate word embedding')
    word_count = {}  # word list
    df = pd.concat([train_news_df, dev_news_df]).drop_duplicates()
    for news_id in list(df.index):
        title = df['Title'][news_id]
        abstract = df['Abstract'][news_id]
        title_tokens = []
        abstract_tokens = []
        if isinstance(title, str):
            title_tokens = tokenize_sentence(title, stop_words) 
        if isinstance(abstract, str):
            abstract_tokens = tokenize_sentence(abstract, stop_words)
        all_token = unique(title_tokens + abstract_tokens)
        for token in all_token:
            if token not in glove_words:  # glove words里没有的不取
                continue
            if token not in word_count:
                word_count[token] = 0
            word_count[token] += 1

    word_count_filtered = {k: word_count[k] for k in word_count if word_count[k] >= 10 and word_count[k] < 100}
    word2id = {w: i for i, w in enumerate(word_count_filtered.keys())}
    word_embedding = np.array([glove[w] for w in word_count_filtered.keys()])

    print('2. mk news-word link list and empty news list')
    df = pd.concat([train_news_df, dev_news_df]).drop_duplicates()
    nw_link = []  # news-id ~ wid
    empty_news = []
    for news_id in list(df.index):
        title = df['Title'][news_id]
        abstract = df['Abstract'][news_id]
        title_tokens = []
        abstract_tokens = []
        if isinstance(title, str):
            title_tokens = tokenize_sentence(title, stop_words) 
        if isinstance(abstract, str):
            abstract_tokens = tokenize_sentence(abstract, stop_words)
        all_token = [w for w in unique(title_tokens + abstract_tokens) if w in word_count_filtered]
        if len(all_token) == 0:
            empty_news.append(news_id)
        for token in all_token:
            nw_link.append([news_id, word2id[token]])

    print('3. generate news embedding')
    # train
    df = train_news_df
    train_news_title_embedding = np.zeros([len(df.index), 300])
    train_news_abstract_embedding = np.zeros([len(df.index), 300])
    for row_id, news_id in enumerate(list(df.index)):
        title = df['Title'][news_id]
        abstract = df['Abstract'][news_id]
        if isinstance(title, str):
            title_tokens = tokenize_sentence(title, stop_words)
            title_word_embedding = [glove[w] for w in title_tokens if w in glove_words]
            train_news_title_embedding[row_id] = sum(title_word_embedding)
        if isinstance(abstract, str):
            abstract_tokens = tokenize_sentence(abstract, stop_words)
            abstract_word_embedding = [glove[w] for w in abstract_tokens if w in glove_words]
            train_news_abstract_embedding[row_id] = sum(abstract_word_embedding)
    # dev
    df = dev_news_df
    dev_news_title_embedding = np.zeros([len(df.index), 300])
    dev_news_abstract_embedding = np.zeros([len(df.index), 300])
    for row_id, news_id in enumerate(list(df.index)):
        title = df['Title'][news_id]
        abstract = df['Abstract'][news_id]
        if isinstance(title, str):
            title_tokens = tokenize_sentence(title, stop_words)
            title_word_embedding = [glove[w] for w in title_tokens if w in glove_words]
            dev_news_title_embedding[row_id] = sum(title_word_embedding)
        if isinstance(abstract, str):
            abstract_tokens = tokenize_sentence(abstract, stop_words)
            abstract_word_embedding = [glove[w] for w in abstract_tokens if w in glove_words]
            dev_news_abstract_embedding[row_id] = sum(abstract_word_embedding)

    print('4. Save results')
    with open('{}/word_count_filtered.json'.format(dir), 'w') as json_file:
        json_file.write(json.dumps(word_count_filtered))
    with open('{}/word2id.json'.format(dir), 'w') as json_file:
        json_file.write(json.dumps(word2id))
    np.save('{}/word_embedding.npy'.format(dir), word_embedding)

    with open('{}/nw_link.txt'.format(dir), 'w') as file:
        _nw_link = ['{}\t{}'.format(p[0], p[1]) for p in nw_link]
        file.write('\n'.join(_nw_link))
    with open('{}/empty_news.txt'.format(dir), 'w') as file:
        file.write('\n'.join(empty_news))
    
    np.save('{}/MINDsmall_train/title_embedding.npy'.format(dir), train_news_title_embedding)
    np.save('{}/MINDsmall_train/abstract_embedding.npy'.format(dir), train_news_abstract_embedding)
    np.save('{}/MINDsmall_dev/title_embedding.npy'.format(dir), dev_news_title_embedding)
    np.save('{}/MINDsmall_dev/abstract_embedding.npy'.format(dir), dev_news_abstract_embedding)


if __name__ == '__main__':
    tokenize_news()
