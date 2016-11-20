'''
Created on Aug 29, 2016

@author: yadollah
'''
import sys
import os
from _collections import defaultdict
import random
import math
from src.data_prep.fbclueweb.entname.extractentnames import load_ent2name2freq
from src.data_prep.fbclueweb.entname import write_ds_names
import codecs
from unidecode import unidecode


def load_ent2types(ent2typesfile):
    ent2types = {}
    type2freq = defaultdict(lambda: 0)
    with open(ent2typesfile) as fp:
        for oneline in fp:
            parts = oneline.strip().split('\t')
            ent2types[parts[0]] = parts[1:]
            for t in parts[1:]:
                type2freq[t] += 1
    return ent2types, type2freq


def load_ent(trainfile):
    with open(trainfile) as fp:
        ent_list = [oneline.strip() for oneline in fp]
    return ent_list


def write_types(sorted_t2freq, outpath, top=50):
    top50types2freq = {}
    with open(outpath, 'w') as fp:
        for i in range(top):
            t, freq = sorted_t2freq[i + 1]
            top50types2freq[t] = freq
            fp.write('\t'.join([t, str(freq), str(freq)]) + '\n')
    print top50types2freq
    return top50types2freq


def write_ent_ds_myformat(ent2types, ent_list, selected_t2freq, outpath):
    mye2t = defaultdict(list)
    for mye in ent_list:
        if mye not in ent2types: 
            print "not in ent2types: ", mye
            continue
        mytypes = [t for t in ent2types[mye] if t in selected_t2freq]
        if len(mytypes) == 0:
            print mye, ent2types[mye]
            continue
        mye2t[mye] = mytypes
    with open(outpath, 'w') as fp:
        for mye in mye2t:
            myl = '\t'.join([mye, mye2t[mye][0], ' '.join(mye2t[mye][1:]), '100'])
            fp.write(myl + '\n')


def load_ent2nameFreebase(fpath):
    print "loading ent2name from freebase dump"
    ent2name = {}
    from nltk.tokenize import RegexpTokenizer
    with open(fpath) as fp:
        for l in fp:
            parts = l.strip().split()
            if len(parts) < 2:
                continue
            name = parts[1]
            name = unidecode(name)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(name)
            ent2name[parts[0]] = ' '.join(tokens)
    return ent2name
    

def main(dspath):
    trainfile = os.path.join(dspath, 'train.txt')
    testfile = os.path.join(dspath, 'test.txt')
    ent2typesfile = os.path.join(dspath, 'entity2type.txt')
    
    if True:
        ent2types, type2freq = load_ent2types(ent2typesfile)
        sorted_t2freq = sorted(type2freq.items(), key=lambda x: x[1], reverse=True)
        
        train_ent = load_ent(trainfile)
        test_ent = load_ent(testfile)
        
        #select 20% of train for dev, randomly
        random.shuffle(train_ent)
        devlen = int(math.ceil(len(train_ent) * 0.1))
        dev_ent = train_ent[0: devlen]
        train_ent = train_ent[devlen:]
        
        #pick and write 50 frequent types (skipt /common/topic)
        selected_t2freq = write_types(sorted_t2freq, dspath + '/50types.txt', top=50)
        
        write_ent_ds_myformat(ent2types, train_ent, selected_t2freq, outpath=os.path.join(dspath, 'Etrain'))
        write_ent_ds_myformat(ent2types, dev_ent, selected_t2freq, outpath=os.path.join(dspath, 'Edev'))
        write_ent_ds_myformat(ent2types, test_ent, selected_t2freq, outpath=os.path.join(dspath, 'Etest'))
    
    mye2name2freq = load_ent2name2freq(sys.argv[2])
    fb_ent2name = None
    if False:
        fb_ent2name = load_ent2nameFreebase(sys.argv[3])
    write_ds_names(os.path.join(dspath, 'Etrain'), mye2name2freq, dspath + '/Etrain.names', fb_ent2name=fb_ent2name) 
    write_ds_names(os.path.join(dspath, 'Edev'), mye2name2freq, dspath + '/Edev.names', fb_ent2name=fb_ent2name) 
    write_ds_names(os.path.join(dspath, 'Etest'), mye2name2freq, dspath + '/Etest.names', fb_ent2name=fb_ent2name)

if __name__ == '__main__':
    dspath = sys.argv[1]
    main(dspath)
        
        