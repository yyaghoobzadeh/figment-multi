'''
Created on Oct 28, 2015

@author: yadollah
'''
import os, sys, logging
from theano import tensor
import theano
from src.common.myutils import debug_print, logger
from numpy import argmax
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('blocks/gm/test.py')
from blocks.bricks import WEIGHT, MLP, Tanh, NDimensionalSoftmax, Linear, Softmax, Logistic
from blocks.initialization import IsotropicGaussian, Constant

from blocks.bricks.cost import CategoricalCrossEntropy
import src.common.myutils as cmn
from theano import tensor
from blocks.model import Model
from blocks.serialization import load
from src.classification.nn import get_metadata, get_stream, track_best, MainLoop,\
    get_seq_stream, get_entity_metadata

def applypredict(get_mlp_out, data_stream, dsents, num_samples, batch_size, outfile, ix_to_target, numinputs, wlev=False):
    num_batches = num_samples / batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    f = open(outfile, 'w') 
    epoch_iter = data_stream.get_epoch_iterator()
    f.write(str(num_samples) + ' ' + str(num_of_hidden_units) + '\n')
    for i in range(num_batches):
        x_let_curr, y_curr, x_emb_curr, x_mnt_emb = epoch_iter.next() 
        if numinputs == 3:
            mlp_act = get_mlp_out(x_let_curr, x_emb_curr, x_mnt_emb).astype('float32')
        elif numinputs == 2 and wlev:
            mlp_act = get_mlp_out(x_mnt_emb, x_let_curr).astype('float32')
        elif numinputs == 2:
            mlp_act = get_mlp_out(x_let_curr, x_emb_curr).astype('float32')
        else:
            if 'ch_only' in nn_type: mlp_act = get_mlp_out(x_let_curr).astype('float32')
            elif 'w_lev' in nn_type: mlp_act = get_mlp_out(x_mnt_emb).astype('float32')
#             mlp_act = get_mlp_out(x_let_curr).astype('float32')
        for j in range(len(mlp_act)):
            index = i * batch_size + j;
            if index >= num_samples: break
            onestr = dsents[index] + ' '
            onestr += ' '.join([str(p) for p in mlp_act[j]])
            f.write(onestr.strip() + '\n')
    f.close()

print 'loading config file', sys.argv[1]
config = cmn.loadConfig(sys.argv[1])
hdf5_file = config['fuelfile']
networkfile = config['net']
num_of_hidden_units = int(config['hidden_units'])
targetTypesFile=config['typefile']
batch_size =  int(config['batchsize'])
nn_type=config['model_type']

target_to_ix, ix_to_target, num_targets, vocabsize = get_metadata(hdf5_file, feature_name1='letters')

test_stream, num_samples_test = get_seq_stream(hdf5_file, 'test', batch_size)
dev_stream, num_samples_dev = get_seq_stream(hdf5_file, 'dev', batch_size)

main_loop = load(networkfile + '.best.pkl')
print 'Model loaded. Building prediction function...'
model = main_loop.model
print model.inputs
print model.variables

mlp_output = [v for v in model.variables if v.name == 'mlp_apply_output'][0]
wlev=False
if 'all' in nn_type:
    x_seqwords, x_let, y, x_emb = model.inputs
    xin = [x_let, x_emb, x_seqwords]
elif 'ch_only' in nn_type or 'ch_bilstm' in nn_type:
    x_let, y = model.inputs
    xin = [x_let]
elif 'ch_men_cnn' in nn_type:
    x_seqw, x_let, y = model.inputs
    xin = [x_seqw, x_let]
    wlev = True
elif 'ch_men_cnn' in nn_type:
    x_seqw, x_let, y = model.inputs
    xin = [x_seqw, x_let]
    wlev = True
elif 'w_lev' in nn_type: 
    x_mnt, y = model.inputs
    xin = [x_mnt]
else:
    x_let, y, x_emb = model.inputs
    xin = [x_let, x_emb]

get_mlp_out = theano.function(xin, mlp_output)

edev, etest = get_entity_metadata(hdf5_file, feature_name='letters')
print len(edev), len(etest)

logger.info('Starting to apply on test inputs')
applypredict(get_mlp_out, test_stream, etest, num_samples_test, batch_size, sys.argv[1] + '.mlpouts', ix_to_target, len(xin), wlev)








