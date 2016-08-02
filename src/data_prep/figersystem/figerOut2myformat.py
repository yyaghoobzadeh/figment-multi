'''
Created on May 13, 2015

@author: yadollah
'''
import sys
import logging
from _collections import defaultdict
from lxml.html.builder import BIG
import numpy
from src.common.myutils import sigmoid

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('figerOut2mysystem')
typefilename = '/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/cis1/context-datasets/types'

upto = -1
numtype = 102
def softmax(w, t=1.0):
    """Calculate the softmax of a list of numbers w.
    @param w: list of numbers
    @return a list of the same length as w of non-negative numbers
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = numpy.exp(numpy.array(w) / t)
    dist = e / numpy.sum(e)
    return dist

def mynormalize(scores, softmaxnorm=False, logistic=True):
    normscores = [0.0 for i in range(len(scores))]
    mymax = max(scores)
    mymin = min(scores)
    if mymax == 0:
        return normscores
    if logistic:
        normscores = [sigmoid(scores[i]) for i in range(len(scores))]  # @UndefinedVariable
    elif softmaxnorm:
        normscores = softmax(scores)
    else:
        normscores = [(scores[i] - mymin) / (mymax - mymin) for i in range(len(scores))]
    return normscores


def fill_myt2i(typefilename,numtype):
    logger.info('filling type to index from my 102 figer types')
    t2i = {}
    typefile = open(typefilename)
    i = -1
    for myline in typefile:
        i += 1
        myparts = myline.split('\t')
        assert len(myparts) == 3
        t2i[myparts[0]] = i
    assert i==numtype-1
    return (t2i)

def load_indexfile(indexfile):
    logger.info('loading entity index file - which can be used to process the Figer output and find the entity2scores')
    l2e = {}
    f = open(indexfile)
    ent2lines = defaultdict(list)
    c = 0
    for line in f:
        parts = line.split('\t')
        sentid = int(parts[0])
        entmid = parts[2]
        tokennum = parts[1]
        l2e[sentid] = (entmid, tokennum)
        if c == upto: break
        c += 1
        ent2lines[entmid].append(sentid)
    f.close()
    logger.info('number of distinct entities are: %d', len(ent2lines.keys()))
    return l2e
        

def parse_one_type(type_in_parant): #(/person/terrorist -1.0)
    parts = type_in_parant.split()
#     logger.info(parts)
    assert len(parts) == 2
    assert len(parts[0]) > 2
    mytype = parts[0][1:].replace('/','-')
    score = float(parts[1][:-1])
    return (mytype, score)
    
def getpercentile(mylist, mypercentile=-1):
    if mypercentile == -1:
        return sum(mylist) / len(mylist)
    myindex = int(len(mylist) * mypercentile)
    assert myindex >= 0
    assert myindex < len(mylist)
    return mylist[myindex]

def load_big_from_figer_out(figerfile, numtype, myt2i):
    logger.info('loading figer results in a big matrix from %s', figerfile)
    f = open(figerfile)
    c = 0;
    big = defaultdict(lambda: defaultdict(list))
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        if parts[1].startswith('B-')==False:
            continue
        
        emid = parts[2].strip()
#         
        scores = [0.0 for i in range(numtype)]
        for i in range(3, len(parts)):
            (mytype, score) = parse_one_type(parts[i])
            if mytype not in myt2i:
                continue
            scores[myt2i[mytype]] = score
        assert len(scores) == numtype
        scores = mynormalize(scores)      
        for j in range(numtype):
            big[emid][j].append(scores[j])
        c +=1
        if c == upto:
            break
#         logger.info('entity number: %d', c)
    logger.info('big has %d entities', len(big))
    return big

def big2smallfiger(big):
    ### now big[ent][type]-> [s1, s2, ...] is filled
    logger.info('big to small matrix')
    small = defaultdict(list)
    for mye in big:
        for j in big[mye]:
            small[mye].append(getpercentile(big[mye][j], 0.99))
    return small


def writesmall(small, smallfile):
    f = open(smallfile, 'w')
    for mye in small:
        f.write(mye)
        for sc in small[mye]:
            f.write(' ' + str(sc))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    
    figerfile = sys.argv[1]
#     indexfile = sys.argv[2]
    myt2i = fill_myt2i(typefilename, numtype)
#     line2entity = load_indexfile(indexfile)
    
    big = load_big_from_figer_out(figerfile, numtype, myt2i)
    small = big2smallfiger(big)
    writesmall(small, figerfile + '.matrix')
    