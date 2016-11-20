'''
Created on Aug 30, 2016
giving entity descriptions, it caculates the tf-idf and save the most 
important features for each entity. 

@author: yadollah
'''
import sys, os
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from _collections import defaultdict
import numpy
import codecs


def save_sparse_csr(filename,array):
    numpy.savez(filename,data=array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_ent2desc(ent2description_path, top=None):
    ent_list = []
    doc_list = []
    with open(ent2description_path) as fp:
        for i, l in enumerate(fp):
            parts = l.strip().split('\t')
            if len(parts) != 2:
                print parts
                continue
            ent_list.append(parts[0])
            doc_list.append(parts[1])
            if top and i > top:
                break
    return ent_list, doc_list


def save_features(fpath, features):
    with codecs.open(fpath, 'w', encoding='utf-8') as fp:
        for fea in features:
            fp.write(fea + '\n')

if __name__ == '__main__':
    ent2description_path = sys.argv[1]
    n_top_feature = int(sys.argv[2])
    ent_list, doc_list = load_ent2desc(ent2description_path, top=None)
    
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    vectorizer = TfidfVectorizer(lowercase=False, stop_words=stop_words, tokenizer=tokenizer.tokenize, min_df=1)
    
#     tfidf_mat = numpy.asarray(vectorizer.fit_transform(doc_list).todense())
    tfidf_mat = vectorizer.fit_transform(doc_list)
    print tfidf_mat.shape
    save_features(ent2description_path+'.tfidfFeaturesVoc.txt', vectorizer.get_feature_names())
    sys.exit()
    save_sparse_csr(ent2description_path+'.tfidfMatrix', tfidf_mat)
    ent_to_top_features = defaultdict(list)
    features = vectorizer.get_feature_names()
    for i in range(tfidf_mat.shape[0]):
        top_features_ind = numpy.asarray(tfidf_mat.getrow(i).todense())[0].argsort()[-n_top_feature:][::-1]
#         print features[top_features_ind[0]]
        ent_to_top_features[ent_list[i]] = [features[ind] for ind in top_features_ind]
#         phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
#         print phrase_scores
    with codecs.open(ent2description_path + '.tfidf,top' + str(n_top_feature), 'w', encoding='utf-8') as fp:
        for ent in ent_list:
            fp.write(ent + '\t' + ' '.join(ent_to_top_features[ent]) + '\n')
            
        


        