# for debugging: upto = 10000 will only look at the first 10000 lines
# This is a script to summarize entity scores aggregated from all of its context scores
import string, collections, sys
from threading import Thread

from src.common.myutils import *
import logging
from _collections import defaultdict
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('big2small.py')

def load_big_matrix(big_file):
    type2entprobs = defaultdict(lambda: defaultdict(list))
    with open(big_file) as fp:
        for line in fp:
            parts = line.split()
            for i, p in enumerate(parts[1:]):
                type2entprobs[i][parts[0]].append(float(p))
    logger.info('loading the big matrix %s finished', big_file)
    return type2entprobs

def big2small(t2ent_probs):   
    logger.info('aggregating scores for entities')
    e2t_scores = defaultdict(list)
    for t_idx in range(len(t2ent_probs)):
        for ent in t2ent_probs[t_idx]:
            prob_list = t2ent_probs[t_idx][ent]
            summaryscore = getpercentile(prob_list, -1) #-1 is avg
            e2t_scores[ent].append(summaryscore)
    return e2t_scores
    
def write_small_matrix(ent2scores, outfile):
    with open(outfile, 'w') as fp:
        for mye in ent2scores:
            outstr = mye + '\t'
            outstr += ' '.join([str(p) for p in ent2scores[mye]])
            fp.write(outstr + '\n')
    logger.info('small matrix saved in: %s', outfile)

if __name__ == '__main__':
    upto = -1

    config = loadConfig(sys.argv[1])
    logger.info(config)
    big_tst_file = config['matrixtest']
    big_dev_file = config['matrixdev']
    small_tst_file = big_tst_file + '.agg'
    small_dev_file = big_dev_file + '.agg'
    
    type2entprobs = load_big_matrix(big_tst_file)
    ent2scores = big2small(type2entprobs)
    write_small_matrix(ent2scores, small_tst_file)

    type2entprobs = load_big_matrix(big_dev_file)
    ent2scores = big2small(type2entprobs)
    write_small_matrix(ent2scores, small_dev_file)
    
