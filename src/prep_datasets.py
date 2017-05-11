'''
Created on Oct 27, 2015

@author: yadollah
'''
import sys, os, string
import theano,numpy, codecs, h5py, yaml, logging
from myutils import convertTargetsToBinVec, get_ent_names
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('makefueldataset')
import myutils as cmn
from fuel.datasets import H5PYDataset
import numpy as np


def gen_new_ds(e2names, e2types, max_name=1, outfile='train.txt'):
    f = open(outfile, 'w')
    for mye in e2names:
        names = get_ent_names(e2names[mye], max_name)
        targets = e2types[mye]
        for nm in names:
            line = '\t'.join([mye, nm.strip(), targets[0]])
            line += '\t' + ' '.join(targets[1:])
            f.write(line + '\n')
    f.close()

def generate_name_dataset(config):
    trainfile = config['Etrain']
    devfile = config['Edev']
    testfile = config['Etest']
    targetTypesFile=config['typefile']
    max_names = [int(n) for n in config['name_num'].split()]
    dsdir = config['dsdir']
    (t2idx, _) = cmn.loadtypes(targetTypesFile)
    _ = len(t2idx)
    (etrain2types, etrain2names, _) = cmn.load_entname_ds(trainfile, t2idx)
    logger.info("number of train examples: %d",len(etrain2names))
    (etest2types, etest2names, _) = cmn.load_entname_ds(testfile, t2idx)
    logger.info("number of test examples: %d",len(etest2names))
    (edev2types, edev2names, _) = cmn.load_entname_ds(devfile, t2idx)
    logger.info("number of dev examples: %d", len(edev2names))

    logger.info('number of names for each entity in trn,dev,test: %s', max_names)
    logger.info('generating new datasets based on entity names')
#     dsdir = dsdir + 'maxname'  + ','.join([str(n) for n in max_names])
#     if not os.path.exists(dsdir): os.makedirs(dsdir)
    gen_new_ds(etrain2names, etrain2types, max_names[0], outfile = dsdir + '/train.txt')
    gen_new_ds(edev2names, edev2types, max_names[1], outfile = dsdir + '/dev.txt')
    gen_new_ds(etest2names, etest2types, max_names[2], outfile = dsdir + '/test.txt')

if __name__ == '__main__':
    print 'loading config file', sys.argv[1]
    config = cmn.loadConfig(sys.argv[1])
    trainfile = config['Etrain']
    devfile = config['Edev']
    testfile = config['Etest']
    targetTypesFile=config['typefile']
    max_names = [int(n) for n in config['name_num'].split()]
    dsdir = config['dsdir']

    upto = -1
    (t2idx, idx2t) = cmn.loadtypes(targetTypesFile)
    numtargets = len(t2idx)
    (etrain2types, etrain2names, _) = cmn.load_entname_ds(trainfile, t2idx)
    logger.info("number of train examples: %d",len(etrain2names))
    (etest2types, etest2names, _) = cmn.load_entname_ds(testfile, t2idx)
    logger.info("number of test examples: %d",len(etest2names))
    (edev2types, edev2names, _) = cmn.load_entname_ds(devfile, t2idx)
    logger.info("number of dev examples: %d", len(edev2names))

    logger.info('number of names for each entity in trn,dev,test: %s', max_names)
    logger.info('generating new datasets based on entity names')
    dsdir = dsdir + 'maxname'  + ','.join([str(n) for n in max_names])
    if not os.path.exists(dsdir): os.makedirs(dsdir)
    gen_new_ds(etrain2names, etrain2types, max_names[0], outfile = dsdir + '/train.txt')
    gen_new_ds(edev2names, edev2types, max_names[1], outfile = dsdir + '/dev.txt')
    gen_new_ds(etest2names, etest2types, max_names[2], outfile = dsdir + '/test.txt')
