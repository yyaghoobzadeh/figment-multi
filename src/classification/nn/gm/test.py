'''
Created on Oct 28, 2015

@author: yadollah
'''
import os, sys, logging

from numpy import argmax
from theano import tensor
import theano

from blocks.bricks import WEIGHT, MLP, Tanh, NDimensionalSoftmax, Linear, Softmax, Logistic
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.serialization import load
import src.common.myutils as cmn
from src.classification.nn.gm.make_dataset_new import load_ent_ds
from src.classification.nn.gm.model import *
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('blocks/gm/test.py')

def applypredict(predict, data_stream, dsMentions, num_samples, batch_size, outfile):
    num_batches = num_samples / batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    goods = 0.;
    epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
    f = open(outfile, 'w') 
    for i in range(num_batches):
        src2vals  = epoch_iter.next()
        inp = [src2vals[src.name] for src in theinputs]
        probs = predict(*inp)
        y_curr = src2vals['targets']
        for j in range(len(probs)):
            index = i * batch_size + j;
            if index >= num_samples: break
            maxtype_ix = argmax(probs[j])
#             max2 = argmax(probs[j][0:maxtype_ix]); max22 = argmax(probs[j][maxtype_ix+1:len(probs)])
#            print idx2type[maxtype_ix], '####', dsMentions[index].name, '###', dsMentions[index].alltypes
            if y_curr[j][maxtype_ix] == 1:
                goods += 1
            onestr = dsMentions[index].entityId + '\t'
            onestr += ' '.join([str(p) for p in probs[j]])
            f.write(onestr.strip() + '\n')
    f.close()
    logger.info('P@1 is = %f ', goods / num_samples)


if __name__ == '__main__':
    logger.info('loading config file: %s', sys.argv[1])
    config = cmn.loadConfig(sys.argv[1])
    dsdir = config['dsdir']
    devfile = dsdir + '/dev.txt'
    testfile = dsdir + '/test.txt'
    networkfile = config['net']
    num_of_hidden_units = int(config['hidden_units'])
    targetTypesFile=config['typefile']
    batch_size = int(config['batchsize'])
    devMentions = load_ent_ds(devfile)
    tstMentions = load_ent_ds(testfile)
    logger.info('#dev: %d #test: %d', len(devMentions), len(tstMentions))
    
    main_loop = load(networkfile + '.best.pkl')
    logger.info('Model loaded. Building prediction function...')
    model = main_loop.model
    logger.info(model.inputs)
    sources = [inp.name for inp in model.inputs]
    theinputs = [inp for inp in model.inputs if inp.name != 'targets']
    
    linear_output = [v for v in model.variables if v.name == 'linear_output'][0]
    y_hat = Logistic().apply(linear_output)
    predict = theano.function(theinputs, y_hat)
    
    fea2obj = build_input_objs(sources, config)
    test_stream, num_samples_test = get_comb_stream(fea2obj, 'test', batch_size, shuffle=False)
    dev_stream, num_samples_dev = get_comb_stream(fea2obj, 'dev', batch_size, shuffle=False)
    
    t2idx = fea2obj['targets'].t2idx
    idx2type = {idx:t for t,idx in t2idx.iteritems()}
    
    logger.info('Starting to apply on test inputs')
    applypredict(predict, test_stream, tstMentions, num_samples_test, batch_size, config['matrixtest'])
    logger.info('apply on test data finished')
    logger.info('Starting to apply on dev inputs')
    applypredict(predict, dev_stream, devMentions, num_samples_dev, batch_size, config['matrixdev'])
    logger.info('apply on dev data finished')








