'''
Created on Jan 13, 2016

@author: yadollah
'''
import sys
from src.common.myutils import getfilelines, getentparts
import os
from subprocess import Popen
from _collections import defaultdict
from src.common.myutils import startTag, endTag, unkTag, padTag


def filter_sentences(sampled_lines):
    lines = []
    for l in sampled_lines:
        sent = l.split('\t')[4]
        news = []
        for w in sent.split():
            if '/m/' in w:
                _, tokens, _ = getentparts(w)
                news.append(' '.join(tokens).strip())
            else:
                news.append(w)
        lines.append(' '.join(news).strip())
    return lines


def calc_ngram_freq(lines, maxngram):
    ngram2freq = defaultdict(lambda : 0)
    for l in lines:
        for i in range(len(l) - maxngram + 1):
            ng = l[i:i+maxngram]
            ng = ng.replace(' ', '_')
            ngram2freq[ng] += 1
    return ngram2freq


def build_vocab(voc2freq, min_freq):
    voc2idx = {}
    voc2idx[startTag] = 0     
    voc2idx[endTag] = 1   
    voc2idx[unkTag] = 2     
    voc2idx[padTag] = 3     
    idx = 4
    print '#voc before freq threshhold:', len(voc2freq)
    for v in voc2freq:
        if voc2freq[v] < min_freq:
            continue
        voc2idx[v] = idx
        idx += 1
    print '#voc after freq threshhold:', len(voc2idx)
    return voc2idx


def process_corpus(lines, voc2idx, ngram2freq, maxngram, fname):
    unk = unkTag
    unkfreq = 0
    newline_list = []
    for l in lines:
        ngramline = [startTag]
        for i in range(len(l) - maxngram + 1):
            ng = l[i:i+maxngram].replace(' ', '_')
            if ng not in voc2idx:
                ng = unk
                unkfreq += 1
            ngramline.append(ng)
        ngramline.append(endTag)
        newline_list.append(' '.join(ngramline))
    with open(fname, 'w') as f:
        for myline in newline_list:
            f.write(myline.strip() + '\n')
    f.close()


def write_vocab(voc2idx, voc2freq, fname):
    with open(fname, 'w') as fp:
        for v, _ in voc2idx.iteritems():
            fp.write(v + '\t' + str(voc2freq[v]) + '\n')

def build_ngram_corpus(args):
    indir = args[2]
    maxngram = int(args[3])
    mydir = indir + args[3] 
    sampled_lines = getfilelines(indir + '/all.sampled', upto=-1)
    print 'sampled lines are loaded'
    lines = filter_sentences(sampled_lines)
    print 'filtering finished'
    if not os.path.exists(mydir): os.makedirs(mydir)
    ngram2freq = calc_ngram_freq(lines, maxngram) 
    voc2idx = build_vocab(ngram2freq, min_freq=5)
    process_corpus(lines, voc2idx, ngram2freq, maxngram, mydir + '/corpus,processed.txt')
    write_vocab(voc2idx, ngram2freq, mydir + '/vocab-count.txt')

def train_word2vec(args):
    tmpdir = '/nfs/data3/yadollah/tmp/'
    indir = args[2]
    mydir = indir + args[3] 
    tmptrn = tmpdir + 'trn'
    #shutil.copyfile(trnf, tmptrn)
    p = Popen(['cp', mydir + '/corpus,processed.txt', tmptrn]); p.wait()
    embfile = tmpdir + 'embeddings'
    print indir
    print 'skipGram'
    w2vclist = ['./w2v', '-min-count 1', '-train ' + tmptrn, '-output ' + embfile, \
            '-cbow 0', '-size 50', '-window 7', '-negative 0', '-hs 1', '-sample 1e-3', '-threads 40', '-binary 0', '-iter 3']
    print 'running word2vec: ' , ' '.join(w2vclist)
    p = Popen(' '.join(w2vclist), shell=True)
    p.wait()
    if not os.path.exists(mydir): os.makedirs(mydir)
    p = Popen(['cp', embfile, mydir + '/embeddings.txt']); p.wait()

if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'build':
        build_ngram_corpus(args)
    elif args[1] == 'word2vec':
        train_word2vec(args)
    
    
    