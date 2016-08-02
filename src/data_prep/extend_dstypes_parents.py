'''
Created on Oct 21, 2015
Extending types of entities in the already generated datasets to their missing parents. 
For example, e1 -building-hospital --> e1 -building-hospital -building 
@author: yadollah
'''
import sys
import os
from src.common.myutils import load_dataset, write_ds, getfilelines,\
    parseents
import logging
from _collections import defaultdict
from src.data_prep.ds_to_figer_types import load_parents, addhighleveltypes
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('lines-gen')


def extend_write_eds(fig2parents, e2types, outfile, e2freq):
    e2types_parents = {}
    for mye in e2types:
        tt = e2types[mye]
        extended_tt = addhighleveltypes(tt, fig2parents)
        e2types_parents[mye] = extended_tt
    write_ds(e2types_parents, e2freq, outfile)
    return e2types_parents
        
def load_lines(lines_file, e2types, upto=-1):
    logger.info('loading lines from %s ...', lines_file)
    e2lines = defaultdict(list)
    e2freq = defaultdict(lambda: 0)
    t2lines = defaultdict(list)
    c = 0
    lines = getfilelines(lines_file)
    for line in lines:
        parts = line.split('\t')
        if len(parts) != 5:
            print len(parts)
            print line
        assert  len(parts) == 5
        mye = parseents(parts[1])[0]
        text = parts[4].strip()
        e2lines[mye].append(text)
        e2freq[mye] += 1
        t2lines[e2types[mye][0]].append((text,mye)) # Add text to the notable type of mye 
        if c == upto:
            break
        c += 1
    logger.info('... lines loaded')
    
    return (e2lines,t2lines, e2freq)
def writelines(e2lines, out_sampled_file, e2types):
    logger.info('%d number of entities in the sampled file', len(e2lines))
    c = 0
    f = open(out_sampled_file, 'w')
    for mye in e2lines:
        etypes = e2types[mye]
        entstr = mye + '\t' + etypes[0] + '\t'
        if len(etypes) > 1:
            for j in range(1, len(etypes)):
                entstr += etypes[j] + ','
        for line in e2lines[mye]:
            c += 1
            f.write(str(c) + '\t' + entstr + '\t' + line)
            f.write('\n')
    f.close()


def extend_dstypes_to_parents(fig2parents, entdsdir, sampledlines_dir, outdir, edsname, linesname):
    e2types, t2ents = load_dataset(entdsdir + edsname, logger)
    e2lines, t2lines_train, e2freq = load_lines(sampledlines_dir + linesname, e2types, -1)
    e2types_extended = extend_write_eds(fig2parents, e2types, outdir + edsname, e2freq)
    writelines(e2lines,outdir + linesname, e2types_extended)

def main(args):
    fig2parents = load_parents('/nfs/data3/yadollah/nlptools_resources/figer/config/yy_type2parents')
    entdsdir = args[1]
    sampledlines_dir = args[2]
    outdir = args[3]
    
    extend_dstypes_to_parents(fig2parents, entdsdir, sampledlines_dir, outdir, 'Etrain', 'train.sampled')
    extend_dstypes_to_parents(fig2parents, entdsdir, sampledlines_dir, outdir, 'Etest', 'test.sampled')
    extend_dstypes_to_parents(fig2parents, entdsdir, sampledlines_dir, outdir, 'Edev', 'dev.sampled')
    extend_dstypes_to_parents(fig2parents, entdsdir, sampledlines_dir, outdir, 'Edev', 'dev.sampled.small')
    
    
if __name__ == '__main__':
    main(sys.argv)