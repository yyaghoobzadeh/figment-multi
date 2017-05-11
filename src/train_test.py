'''
Created on Oct 27, 2015

@author: yadollah
'''
import os, sys, logging

import numpy
from theano import tensor
import theano
from theano.compile import function

from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp, AdaGrad, Scale, AdaDelta, Momentum, Adam
from blocks.extensions import FinishAfter, ProgressBar, Timing, Printing
from blocks.extensions import saveload
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import INPUT, WEIGHT
from blocks.theano_expressions import l2_norm
from myutils import debug_print
from nn import track_best, MainLoop
from model import *
import myutils as cmn
import theano.tensor as T
import argparse
from blocks.bricks import MLP, Tanh, NDimensionalSoftmax, Linear, Softmax, Logistic
from blocks.serialization import load
from make_dataset import load_ent_ds
from numpy import argmax
from subprocess import Popen
from blocks.utils import shared_floatx
import math

dirname, _ = os.path.split(os.path.abspath(__file__))

print theano.__version__
print theano

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('train.py')
seed = 42
if "gpu" in theano.config.device:
    srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
else:
    srng = T.shared_randomstreams.RandomStreams(seed=seed)


class EntityTypingGlobal(object):
    def __init__(self, config):
        self._config = config

    def training(self, fea2obj, batch_size, learning_rate=0.005, steprule='adagrad', wait_epochs=5, kl_weight_init=None, klw_ep=50, klw_inc_rate=0, num_epochs=None):
        networkfile = self._config['net']

        n_epochs = num_epochs or int(self._config['nepochs'])
        reg_weight=float(self._config['loss_weight'])
        reg_type=self._config['loss_reg']
        numtrain = int(self._config['num_train']) if 'num_train' in self._config else None
        train_stream, num_samples_train = get_comb_stream(fea2obj, 'train', batch_size, shuffle=True, num_examples=numtrain)
        dev_stream, num_samples_dev = get_comb_stream(fea2obj, 'dev', batch_size=None, shuffle=False)
        logger.info('sources: %s -- number of train/dev samples: %d/%d', train_stream.sources, num_samples_train, num_samples_dev)

        t2idx = fea2obj['targets'].t2idx
        klw_init = kl_weight_init or float(self._config['kld_weight']) if 'kld_weight' in self._config else 1
        logger.info('kl_weight_init: %d', klw_init)
        kl_weight = shared_floatx(klw_init, 'kl_weight')
        entropy_weight = shared_floatx(1., 'entropy_weight')

        cost, p_at_1, _, KLD, logpy_xz, pat1_recog, misclassify_rate= build_model_new(fea2obj, len(t2idx), self._config, kl_weight, entropy_weight)

        cg = ComputationGraph(cost)

        weights = VariableFilter(roles=[WEIGHT])(cg.parameters)
        logger.info('Model weights are: %s', weights)
        if 'L2' in reg_type:
            cost += reg_weight * l2_norm(weights)
            logger.info('applying %s with weight: %f ', reg_type, reg_weight)

        dropout = -0.1
        if dropout > 0:
            cg = apply_dropout(cg, weights, dropout)
            cost = cg.outputs[0]

        cost.name = 'cost'
        logger.info('Our Algorithm is : %s, and learning_rate: %f', steprule, learning_rate)
        if 'adagrad' in steprule:
            cnf_step_rule = AdaGrad(learning_rate)
        elif 'adadelta' in steprule:
            cnf_step_rule = AdaDelta(decay_rate=0.95)
        elif 'decay' in steprule:
            cnf_step_rule = RMSProp(learning_rate=learning_rate, decay_rate=0.90)
            cnf_step_rule = CompositeRule([cnf_step_rule, StepClipping(1)])
        elif 'momentum' in steprule:
            cnf_step_rule = Momentum(learning_rate=learning_rate, momentum=0.9)
        elif 'adam' in steprule:
            cnf_step_rule = Adam(learning_rate=learning_rate)
        else:
            logger.info('The steprule param is wrong! which is: %s', steprule)

        algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=cnf_step_rule, on_unused_sources='warn')
        #algorithm.add_updates(updates)
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, gradient_norm, step_norm, p_at_1, KLD, logpy_xz, kl_weight, pat1_recog]
        train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                               before_first_epoch=True, prefix='tra')

        dev_monitor = DataStreamMonitoring(variables=[cost, p_at_1, KLD, logpy_xz, pat1_recog, misclassify_rate], after_epoch=True,
                                           before_first_epoch=True, data_stream=dev_stream, prefix="dev")

        extensions = [dev_monitor, train_monitor, Timing(),
                      TrackTheBest('dev_cost'),
                      FinishIfNoImprovementAfter('dev_cost_best_so_far', epochs=wait_epochs),
                      Printing(after_batch=False), #, ProgressBar()
                      FinishAfter(after_n_epochs=n_epochs),
                      saveload.Load(networkfile+'.toload.pkl'),
                      ] + track_best('dev_cost', networkfile+ '.best.pkl')

        #extensions.append(SharedVariableModifier(kl_weight,
        #                                          lambda n, klw: numpy.cast[theano.config.floatX] (klw_inc_rate + klw), after_epoch=False, every_n_epochs=klw_ep, after_batch=False))
#         extensions.append(SharedVariableModifier(entropy_weight,
#                                                   lambda n, crw: numpy.cast[theano.config.floatX](crw - klw_inc_rate), after_epoch=False, every_n_epochs=klw_ep, after_batch=False))

        logger.info('number of parameters in the model: %d', tensor.sum([p.size for p in cg.parameters]).eval())
        logger.info('Lookup table sizes: %s', [p.size.eval() for p in cg.parameters if 'lt' in p.name])

        main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                             model=Model(cost), extensions=extensions)
        main_loop.run()


    def testing(self, fea2obj):
        config = self._config
        dsdir = config['dsdir']
        devfile = dsdir + '/dev.txt'
        testfile = dsdir + '/test.txt'
        networkfile = config['net']
        batch_size = 10000#int(config['batchsize'])
        devMentions = load_ent_ds(devfile)
        tstMentions = load_ent_ds(testfile)
        logger.info('#dev: %d #test: %d', len(devMentions), len(tstMentions))

        main_loop = load(networkfile + '.best.pkl')
        logger.info('Model loaded. Building prediction function...')
        old_model = main_loop.model
        logger.info(old_model.inputs)
        sources = [inp.name for inp in old_model.inputs]
#         fea2obj = build_input_objs(sources, config)
        t2idx = fea2obj['targets'].t2idx
        deterministic = str_to_bool(config['use_mean_pred']) if 'use_mean_pred' in config else True
        kl_weight = shared_floatx(0.001, 'kl_weight')
        entropy_weight= shared_floatx(0.001, 'entropy_weight')


        cost, _, y_hat, _, _,_,_ = build_model_new(fea2obj, len(t2idx), self._config, kl_weight, entropy_weight, deterministic=deterministic, test=True)
        model = Model(cost)
        model.set_parameter_values(old_model.get_parameter_values())

        theinputs = []
        for fe in fea2obj.keys():
            if 'targets' in fe:
                continue
            for inp in model.inputs:
                if inp.name == fe:
                    theinputs.append(inp)

#         theinputs = [inp for inp in model.inputs if inp.name != 'targets']
        print "theinputs: ", theinputs
        predict = theano.function(theinputs, y_hat)

        test_stream, num_samples_test = get_comb_stream(fea2obj, 'test', batch_size, shuffle=False)
        dev_stream, num_samples_dev = get_comb_stream(fea2obj, 'dev', batch_size, shuffle=False)
        logger.info('sources: %s -- number of test/dev samples: %d/%d', test_stream.sources, num_samples_test, num_samples_dev)
        idx2type = {idx:t for t,idx in t2idx.iteritems()}

        logger.info('Starting to apply on dev inputs...')
        self.applypredict(theinputs, predict, dev_stream, devMentions, num_samples_dev, batch_size, os.path.join(config['exp_dir'], config['matrixdev']), idx2type)
        logger.info('...apply on dev data finished')

        logger.info('Starting to apply on test inputs...')
        self.applypredict(theinputs, predict, test_stream, tstMentions, num_samples_test, batch_size, os.path.join(config['exp_dir'], config['matrixtest']), idx2type)
        logger.info('...apply on test data finished')

    def applypredict(self, theinputs, predict, data_stream, dsMentions, num_samples, batch_size, outfile, idx2type):
        num_batches = num_samples / batch_size
        if num_samples % batch_size != 0:
            num_batches += 1
        goods = 0.;
        epoch_iter = data_stream.get_epoch_iterator(as_dict=True)
        f = open(outfile, 'w')
        print num_batches
        for i in range(num_batches):
            src2vals  = epoch_iter.next()
            inp = [src2vals[src.name] for src in theinputs]
            probs = predict(*inp)

            y_curr = src2vals['targets']
            for j in range(len(probs)):
                index = i * batch_size + j;
                if index >= num_samples: break
                maxtype_ix = argmax(probs[j])
#                 max2 = argmax(probs[j][0:maxtype_ix]); max22 = argmax(probs[j][maxtype_ix+1:len(probs)])
#                 print idx2type[maxtype_ix], '####', dsMentions[index].name, '###', dsMentions[index].alltypes, '###', idx2type[argmax(src2vals['tc'][j])], maxtype_ix
                if y_curr[j][maxtype_ix] == 1:
                    goods += 1
                onestr = dsMentions[index].entityId + '\t'
                onestr += ' '.join([str(p) for p in probs[j]])
                f.write(onestr.strip() + '\n')
        f.close()
        logger.info('P@1 is = %f ', goods / num_samples)

    def evaluate(self, configfile):
        cmd = 'python ' + dirname + '/matrix2measures_ents.py ' + configfile + ' > ' + configfile + '.meas.ents'
        p = Popen(cmd, shell=True);
        p.wait()
        logger.info('... entity measures finished!')
        cmd = 'python ' + dirname + '/matrix2measures_types.py ' + configfile + ' > ' + configfile + '.meas.types'
        p = Popen(cmd, shell=True);
        p.wait()

def main(args):
    logger.info('loading config file: %s', args.config)
    exp_dir, _ = os.path.split(os.path.abspath(args.config))
    config = cmn.loadConfig(args.config)
    config['exp_dir'] = exp_dir
    config['net'] = os.path.join(exp_dir, config['net'])
    batch_size =  int(config['batchsize'])
    features = config['features'].split(' ') #i.e. letters words entvec
    if batch_size == 0: batch_size = None
    inp_srcs = []
    for fea in features:
        if 'ngrams' in fea:
            inp_srcs.extend(['ngrams' + ng for ng in config['ngrams_n'].split()])
        else:
            inp_srcs.append(fea)
    our_sources = inp_srcs + ['targets']

    fea2obj = build_input_objs(our_sources, config)

    typer = EntityTypingGlobal(config)
    if args.train:
        import shutil
        #typer.training(fea2obj, batch_size, learning_rate=float(config['lrate']), steprule=config['steprule'], wait_epochs=10, kl_weight_init=1, klw_ep=100, klw_inc_rate=0, num_epochs=50)
        typer.training(fea2obj, batch_size, learning_rate=float(config['lrate']), steprule=config['steprule'], wait_epochs=3, num_epochs=30)
        shutil.copyfile(config['net']+'.best.pkl', config['net']+'.toload.pkl')
        shutil.copyfile(config['net']+'.best.pkl', config['net']+'.best1.pkl')
        # logger.info('One more epoch training...')
        # typer.training(fea2obj, batch_size, learning_rate=float(config['lrate'])/2, steprule=config['steprule'], wait_epochs=2, klw_ep=10, kl_weight_init=0.008, num_epochs=20)
        # shutil.copyfile(config['net']+'.best.pkl', config['net']+'.toload.pkl')
        # shutil.copyfile(config['net']+'.best.pkl', config['net']+'.best2.pkl')
        #logger.info('One more epoch training...')
        #typer.training(fea2obj, batch_size, learning_rate=float(config['lrate'])/2, steprule=config['steprule'], wait_epochs=2, klw_ep=10, kl_weight_init=0.02, num_epochs=10)
        shutil.copyfile(config['net']+'.best.pkl', config['net']+'.toload.pkl')
        logger.info('One more epoch training...')
        typer.training(fea2obj, batch_size=100, learning_rate=0.005, steprule='adagrad', wait_epochs=2, klw_ep=10, kl_weight_init=None, num_epochs=10)


    if args.test:
        typer.testing(fea2obj)

    if args.eval:
        typer.evaluate(args.config)


    #scriptfile = '/mounts/Users/student/yadollah/new_ws/phdworks/src/classification/nn/blocks/joint/test.py'
    #os.system('python ' + scriptfile + ' ' + sys.argv[1])

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
        "--eval", "-ev", type=bool, help="Evaluating the model with f-measures")
    return parser

if __name__ == '__main__':
#     theano.config.dnn.enabled = False
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
