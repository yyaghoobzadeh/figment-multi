'''
Created on Nov 23, 2015

@author: yadollah
Extracting features from entity names. 
Word shape, Ngram (1 and 2grams), Brown id, Length, head

'''
from src.common.myutils import load_entname_ds, loadtypes,\
    loadConfig, get_ent_names, minimal_of_list, str_to_bool
import sys, yaml
import re
from nltk import word_tokenize, pos_tag  
import numpy
import sklearn
import theano, theano.tensor as T
import layers as layers
from algorithms import compute_ada_grad_updates
import time
import math
import mmap
import cPickle
from operator import itemgetter
from random import shuffle
import os
from _collections import defaultdict
import scipy.sparse as sp
from theano import sparse
from numpy import argmax
from src.classification.nn.gm.nsl import load_brown_clusters_mapping
import argparse

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
        
def make_input_matrix(csr_mat, ent2types, fea2id, whichset='train', maxname=1, npdir='',numtype=102, t2idx=None):
    whichset = npdir + whichset 
    fea_size = len(fea2id)
#     in_trn = numpy.z(shape=(len(ent2fea), fea_size), dtype='int8')
    targets = numpy.zeros(shape=(len(ent2fea) * maxname, numtype), dtype='int8')
    rowsin = []; colsin = []; datain = [];
    ents = []
    i = 0
    print '** ', len(ent2fea) * maxname, whichset
    for mye in ent2fea:
        for onename in ent2fea[mye]:
            ents.append(mye)
            for fea in ent2fea[mye][onename]:
                rowsin.append(i)
                colsin.append(fea2id[fea])
                datain.append(1)
            for myt_idx in ent2types[mye]:
                if t2idx != None: myt_idx = t2idx[myt_idx]
                targets[i][myt_idx] = 1
            i += 1
            
    a = sp.csr_matrix((datain, (rowsin, colsin)), shape=(len(ent2fea) * maxname, fea_size))
    save_sparse_csr(whichset+'.features', a)
    numpy.save(whichset+'.targets', targets)
    numpy.save(whichset+'.ents', ents)
    return a, targets, ents
        
    
def load_input_matrix(whichset):
    whichset = npdir + whichset
    obj1 = load_sparse_csr(whichset + '.features.npz')
    obj2 = numpy.load(whichset + '.targets.npy')
    obj3 = numpy.load(whichset + '.ents.npy')
    return obj1, obj2, obj3
    
def training_model():
        ###############
    # TRAIN MODEL #
    print '... training'
    improvement_threshold = 0.9995  # a relative improvement of this much is
    validation_frequency = n_train_batches#min(n_train_batches, patience)
    
    best_params = []
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    val_losses = []
    max_val_loss = 5#int(math.ceil(n_epochs / 10.))
    epoch = 0
    done_looping = False
    
    possible_indecis = [i for i in xrange(n_train_batches)]
    while (epoch < n_epochs) and (not done_looping):
            shuffle(possible_indecis)
            epoch = epoch + 1
            print 'epoch = ', epoch
            for minibatch_index in xrange(n_train_batches):
                random_index = possible_indecis[minibatch_index] 
                iter = (epoch - 1) * n_train_batches + minibatch_index + 1
                
                if iter % 10000  == 0:
                    print 'training @ iter = ', iter
    
                cost_ij = train_model(random_index)
    
                if (iter + 1) % validation_frequency == 0:
    
                    validation_losses = []
                    for i in xrange(n_valid_batches):
                        validation_losses.append(validate_model(i))
     
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, iteration %i, validation cost %f , train cost %f ' % \
                          (epoch, iter, this_validation_loss, cost_ij))
    
                    # if we got the best validation score until now
                    if this_validation_loss < minimal_of_list(val_losses):
                        del val_losses[:]
                        val_losses.append(this_validation_loss)
                        best_iter = iter
                        if MLP:
                            best_params = [[layer1.params[0].get_value(borrow=False), layer1.params[1].get_value(borrow=False)]
                                       , [out_layer.params[0].get_value(borrow=False), out_layer.params[1].get_value(borrow=False)]]
                        else:
                            best_params = [[out_layer.params[0].get_value(borrow=False), out_layer.params[1].get_value(borrow=False)]]
                            
                        best_validation_loss = this_validation_loss
                        print('**best results updated! waiting for %i more validations!', max_val_loss)
                        print('Saving net.')
                        test_out(test_set_x, target_tst_matrix, batch_size, tstents, config['matrixtest'])
                        test_out(valid_set_x, target_dev_matrix, batch_size, devents, config['matrixdev'])
#                         save_file = open(networkfile, 'wb')
#                         cPickle.dump(best_params[0][0], save_file, -1)
#                         cPickle.dump(best_params[0][1], save_file, -1)
#                         if MLP:
#                             cPickle.dump(best_params[1][0], save_file, -1)
#                             cPickle.dump(best_params[1][1], save_file, -1)
#                         save_file.close()

                        # save best validation score and iteration number
                    elif len(val_losses) < max_val_loss:
                        print 'addinig new validation to the val_losses, len(val_lossses) is ', len(val_losses)
                        val_losses.append(this_validation_loss)
                        if len(val_losses) == max_val_loss:
                            done_looping = True
                            
                            break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f %% obtained at iteration %d, %%', best_validation_loss * 100., best_iter + 1)
    print >> sys.stderr, ('The code for file ' + 
                              os.path.split(__file__)[1] + 
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print('Saving net.')
    save_file = open(networkfile, 'wb')
    cPickle.dump(best_params[0][0], save_file, -1)
    cPickle.dump(best_params[0][1], save_file, -1)
    if MLP:
        cPickle.dump(best_params[1][0], save_file, -1)
        cPickle.dump(best_params[1][1], save_file, -1)
    save_file.close()
    return 
    

def test_out(ds_set_x,target_matrix, batch_size, ents, outfile):
    batch_size = len(ents)
    test_predicted_probs = theano.function([index], predicted_probs,
    givens={
        x: ds_set_x[index * batch_size: (index + 1) * batch_size]
        })
    
    n_test_batches = ds_set_x.get_value(borrow=True).shape[0]
    n_test_batches /= batch_size
    if ds_set_x.get_value(borrow=True).shape[0] % batch_size != 0:
        n_test_batches += 1
    f = open(outfile, 'w')
    goods = 0.
    for i in range(n_test_batches):
        print i
        probslist = test_predicted_probs(i)    
        for j in range(len(probslist)):
            ix = i * batch_size + j;
            maxtype_ix = argmax(probslist[j])
            if target_matrix[ix][maxtype_ix] == 1:
                goods += 1.
            onestr = ents[ix] + '\t'
            onestr += ' '.join([str(p) for p in probslist[j]])
            f.write(onestr.strip() + '\n')
    f.close()    
    print 'output score results saved in:', outfile
    print 'Accuracy of notable type prediction is = ', 100. * goods / len(ents)

def get_argument_parser():
    """ Construct a parser for the command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", help="Path to configuration file")
    
    parser.add_argument(
        "--test", "-t", type=bool, help="Applying the model on the test data, or not")
    
    parser.add_argument(
        "--train", "-tr", type=bool, help="Training the model on the test data, or not")

    parser.add_argument(
        "--loaddata", "-lo", type=bool, help="To load the feature matrices or not?")
    return parser

if __name__ == '__main__':
    UPTO = -1
    parser = get_argument_parser()
    args = parser.parse_args()

    config = loadConfig(args.config)
    brownMappingFile=config['brownclusters']
    trainfile=config['Etrain']
    devfile=config['Edev']
    testfile=config['Etest']
    batch_size=int(config['batchsize'])
    targetTypesFile=config['typefile']
    learning_rate = float(config['lrate'])
    networkfile = config['net']
    num_of_hidden_units = int(config['hidden_units'])
    n_epochs = int(config['nepochs'])
    maxngram = int(config['maxngram'])
    MLP=str_to_bool(config['mlp'])
    featuresToUse= [fea for fea in config['features'].split(' ')]
    npdir = config['npdir']
    if not os.path.exists(npdir): os.makedirs(npdir)
    
    (t2idx, idx2t) = loadtypes(targetTypesFile)
    numtype = len(t2idx)
    (etrain2types, etrain2names,_) = load_entname_ds(trainfile, t2idx, use_ix=True)
    print "number of train examples:" + str(len(etrain2names))
    (etest2types, etest2names,_) = load_entname_ds(testfile, t2idx, use_ix=True)
    print "number of test examples:" + str(len(etest2names))
    (edev2types, edev2names,_) = load_entname_ds(devfile, t2idx, use_ix=True)
    print "number of dev examples:" + str(len(edev2names))
    
    if args.loaddata:
        (in_trn_matrix, target_trn_matrix, trnents) = load_input_matrix('train')
        (in_tst_matrix, target_tst_matrix, tstents) = load_input_matrix('test')
        (in_dev_matrix, target_dev_matrix, devents) = load_input_matrix('dev')
    else:
        word2cluster = load_brown_clusters_mapping(brownMappingFile)
        etrain2rawFea = build_features(etrain2names, etrain2types, word2cluster, featuresToUse, maxnamenum=3, upto=UPTO)
        edev2rawFea = build_features(edev2names, edev2types, word2cluster, featuresToUse, maxnamenum=1, upto=UPTO)
        etest2rawFea = build_features(etest2names, etest2types, word2cluster, featuresToUse, maxnamenum=1, upto=UPTO)
        print '.. building raw features finished!'
        allfeatures = [v for mye in etrain2rawFea for name in etrain2rawFea[mye] for v in etrain2rawFea[mye][name]]
        allfeatures.extend([v for mye in edev2rawFea for name in edev2rawFea[mye] for v in edev2rawFea[mye][name]])
        allfeatures.extend([v for mye in etest2rawFea for name in etest2rawFea[mye] for v in etest2rawFea[mye][name]])
        fea2id = build_feature2id(allfeatures, npdir+'feature2id')
        print 'feature vocabsize = ', len(fea2id)
        (in_trn_matrix, target_trn_matrix, trnents) = make_input_matrix(etrain2rawFea, etrain2types, fea2id, whichset='train',maxname=3,npdir=npdir)
        (in_tst_matrix, target_tst_matrix, tstents) = make_input_matrix(etest2rawFea, etest2types, fea2id, whichset='test',maxname=1,npdir=npdir)
        (in_dev_matrix, target_dev_matrix, devents) = make_input_matrix(edev2rawFea, edev2types, fea2id, whichset='dev',maxname=1,npdir=npdir)    
    
    
    fea2id = ''; allfeatures = ''; etrain2rawFea = ''
    
    dt = numpy.dtype(numpy.int32)
    
    train_set_x = theano.shared(in_trn_matrix)  # @UndefinedVariable
    valid_set_x = theano.shared(in_dev_matrix)
    test_set_x = theano.shared(in_tst_matrix)
    train_set_y = theano.shared(target_trn_matrix)
    valid_set_y = theano.shared(target_dev_matrix)
    test_set_y = theano.shared(target_tst_matrix)
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    
    index = T.lscalar()  # index to a [mini]batch
#     x = T.matrix('x')  # the data is presented as rasterized images
    x = sparse.csr_matrix(name='x', dtype='int64')  # the data is presented as rasterized images
#     y = sparse.csr_matrix(name='y', dtype='int8')  # the labels are presented as 1D vector of  multi
    y = T.bmatrix('y')  # the labels are presented as 1D vector of  multi

    print '... building the computional Graph'
    rng = numpy.random.RandomState(23455)
    
    if MLP:
        layer1 = layers.HiddenLayer(rng, input=x, n_in=in_trn_matrix.shape[1], n_out=num_of_hidden_units,sparse=True)
        out_layer = layers.OutputLayer(input=layer1.output, n_in=num_of_hidden_units, n_out=numtype)
        params = layer1.params + out_layer.params
        #weights = T.concatenate[layer1.W, out_layer.W]
    else:
        out_layer = layers.OutputLayer(input=x, n_in=in_trn_matrix.shape[1], n_out=numtype, sparse=True)
        params = out_layer.params
        #weights = [out_layer.W]
        
    scorelayer = layers.SigmoidLoss(input=out_layer.score_y_given_x, n_in=numtype, n_out=numtype)
    cost = scorelayer.cross_entropy_loss(y)
    predicted_probs = scorelayer.getOutScores()
    #l_weight = float(config['l2_weight'])
    #for w in T.sum(weights ** 2):
    #cost += l_weight * T.sum(weights ** 2)
    updates = compute_ada_grad_updates(cost, params, learning_rate)
    
    train_model = theano.function([index], cost, updates=updates,
                  givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]
                    })
    
    validate_model = theano.function([index], cost, 
                givens={
                    x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                    y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    if args.train:
        print 'TRAINING'
        training_model()
    else:
        print 'ONY TEST'
        netfile = open(networkfile)
        if MLP:
            layer1.params[0].set_value(cPickle.load(netfile), borrow=False)
            layer1.params[1].set_value(cPickle.load(netfile), borrow=False)
        out_layer.params[0].set_value(cPickle.load(netfile), borrow=False)
        out_layer.params[1].set_value(cPickle.load(netfile), borrow=False)

    test_out(test_set_x, target_tst_matrix, batch_size, tstents, config['matrixtest'])
    test_out(valid_set_x, target_dev_matrix, batch_size, devents, config['matrixdev'])
    

        
        
        
