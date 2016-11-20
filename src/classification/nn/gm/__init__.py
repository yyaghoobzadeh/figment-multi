from _collections import defaultdict
import numpy
from src.common.myutils import get_ngram_seq,\
    convertTargetsToBinVec, get_ent_names, buildtypevecmatrix, buildcosinematrix

def build_mention_vec_seq(name, wordvectors, maxwords, vectorsize):
    mention_vec = numpy.zeros(shape=(maxwords, vectorsize), dtype='float32')
    i = 0
    for w in name.split():
        if w not in wordvectors:
            continue
        if i > maxwords:
            break
        mention_vec[i] = wordvectors[w]
    
    return numpy.reshape(mention_vec, maxwords * vectorsize)



