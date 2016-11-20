import sys
import h5py
import yaml
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping, Cast, AgnosticSourcewiseTransformer, FilterSources, Merge
from blocks.extensions import saveload, predicates
from blocks.extensions.training import TrackTheBest
from blocks import main_loop
from blocks.initialization import Uniform
from blocks.roles import add_role, WEIGHT, BIAS
from fuel.utils import do_not_pickle_attributes
import theano.tensor as T
import theano
from blocks.bricks.lookup import LookupTable
import numpy, logging
from src.common.myutils import build_ngram_vocab, get_ngram_seq,\
    str_to_bool, debug_print
import os
# from src.classification.nn.blocks.joint.model import initialize,\
#     create_cnn_general, create_lstm, create_ff, create_mean
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')
rng = numpy.random.RandomState(23455)
from theano.tensor import shared_randomstreams

seq_features = ['letters', 'words', 'ngrams2', 'ngrams3', 'ngrams4', 'ngrams5', 'subwords', 'desc_features']

def scan_func(M):
    def loop (row):
        one_indices = T.nonzero(row)[0]
        zero_indices = T.eq(row, 0).nonzero()[0]
        random = shared_randomstreams.RandomStreams(5)
        ind1=random.random_integers(size=(1,), low=0, high=one_indices.shape[0]-1, ndim=None)
        ind2=random.random_integers(size=(50,), low=0, high=zero_indices.shape[0]-1, ndim=None)
        return one_indices[ind1], zero_indices[ind2]
    
    pos_inds, updates = theano.scan(fn=loop,
                                sequences=M,
                                outputs_info=None) 
    return pos_inds[0], pos_inds[1], updates

def ranking_loss(y_hat, y):
    pos_inds, neg_inds, updates = scan_func(y)
    index = T.arange(y.shape[0])
    pos_scores = y_hat[index, pos_inds.T].T
    neg_scores = y_hat[index, neg_inds.T].T
    pos_scores = T.tile(pos_scores, neg_scores.shape[1])
    cost = T.sum(T.maximum(0., 1. - pos_scores + neg_scores), axis=1)
    return T.mean(cost), updates  


def cross_entropy_loss(y_hat, y):
    return T.mean(T.nnet.binary_crossentropy(y_hat, y))
    
#Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)
        
    def load(self):
        self.extensions = []

class LettersTransposer(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, **kwargs):
        super(LettersTransposer, self).__init__(data_stream=data_stream, produces_examples=data_stream.produces_examples, **kwargs)
  
    def transform_any_source(self, source, _):
        return source.T
    
class CutInput(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, max_len, **kwargs):
        super(CutInput, self).__init__(data_stream=data_stream, produces_examples=data_stream.produces_examples, **kwargs)
        self.max_len = max_len
    def transform_any_source(self, source, _):
        return source[:,0:self.max_len]
    
def sample_transformations(thestream):
    cast_stream = Cast(data_stream=thestream,  dtype='float32', which_sources=('features',))
    return cast_stream
                       
def track_best(channel, save_path):
    tracker = TrackTheBest(channel, choose_best=min, after_epoch=True)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"], predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]

# def load_ds(hdf5_file, ngram=2):
#     new_hdf5file = hdf5_file[:hdf5_file.index('.h5py')] + 'ngram'+str(ngram)+'.h5py'
#     if os.path.exists(new_hdf5file):
#         return new_hdf5file  
#     dataset = H5PYDataset(hdf5_file, which_sets=('train', 'dev', 'test'), load_in_memory=True)
#     if ngram == 1: return dataset
#     letter_matrix = dataset.data_sources[dataset.sources.index('letters')]
#     targets = dataset.data_sources[dataset.sources.index('targets')]
#     ngram2idx, idx2ngram = build_ngram_vocab(letter_matrix, ngram)
#     letts_len = len(letter_matrix[0]) 
#     num_ngrams = letts_len - ngram + 1 + 2 # +2 because of start and end tag
#     ngram_matrix = numpy.zeros(shape=(len(letter_matrix), num_ngrams), dtype='uint8')
#     for i, inst in enumerate(letter_matrix):
#         ngram_matrix[i] = get_ngram_seq(inst, ngram2idx)
#     dataset.data_sources += (ngram_matrix, )
#     f = h5py.File(new_hdf5file, mode='w')
#     f.attrs['split']
#     newsrcname = 'ngram' + str(ngram)
#     if newsrcname in dataset.provides_sources: 
#         logger.info('WARNING: the new source %s already exists in dataset', newsrcname)
#     dataset.provides_sources += (u'ngram' + str(ngram),)
#     
#     return dataset, newsrcname

def get_targets_metadata(dsdir):
    with h5py.File(dsdir + '_targets.h5py') as f:
        t_to_ix = yaml.load(f['targets'].attrs['type_to_ix'])
        ix_to_t = yaml.load(f['targets'].attrs['ix_to_type'])
    return t_to_ix, ix_to_t

        
def transpose_stream(data):
    return (data[0].T, data[1])

def get_comb_stream(fea2obj, which_set, batch_size=None, shuffle=True, num_examples=None):
    streams = []
    for fea in fea2obj:
        obj = fea2obj[fea]
        dataset = H5PYDataset(obj.fuelfile, which_sets=(which_set,),load_in_memory=True)
        if batch_size == None: batch_size = dataset.num_examples
        if num_examples == None: num_examples = dataset.num_examples
        if shuffle: 
            iterschema = ShuffledScheme(examples=num_examples, batch_size=batch_size)
        else: 
            iterschema = SequentialScheme(examples=num_examples, batch_size=batch_size)
        stream = DataStream(dataset=dataset, iteration_scheme=iterschema)
        if fea in seq_features:
            stream = CutInput(stream, obj.max_len)
            if obj.rec == True:
                logger.info('transforming data for recursive input')
                stream = LettersTransposer(stream, which_sources=fea)# Required because Recurrent bricks receive as input [sequence, batch,# features]
        streams.append(stream)
    stream = Merge(streams, tuple(fea2obj.keys()))
    return stream, num_examples

# from blocks.utils import check_theano_variable, shared_floatx_nans

# class MyLookupTable():
#     def __init__(self, length, dim):
#         self.length = length
#         self.dim = dim
#     def allocate(self):
#         self.W = shared_floatx_nans((self.length, self.dim))
#         self.W.name = 'W'
#     def W(self):
#         return self.W
#     def apply(self, indices):
#         check_theano_variable(indices, None, ("int", "uint"))
#         output_shape = [indices.shape[i]
#                         for i in range(indices.ndim)] + [self.dim]
#         return self.W[indices.flatten()].reshape(output_shape)
#     
# class SequentialInput():
#     def __init__(self, name, config):
#         self.config = config
#         self.name = name
#         self.max_len = int(config[self.name + '_max_len'])
#         self.dim_emb = int(config[self.name + '_emb_dim'])
#         self.fuelfile = config['dsdir'] + '_' + name + '.h5py'
#         self.nn_model = config[self.name + '_nn']
#         self.emb_file = self.config['dsdir'] + '_' + self.name + '_embeddings.h5py'
#         self.tune_tune = str_to_bool(self.config[self.name + '_tune_embedding'])
#         self.load_emb = str_to_bool(self.config[self.name + '_load_embedding'])
#         if self.load_emb == False: 
#             assert self.tune_tune == True
#         self.rec = False
#         if 'lstm' in self.nn_model: 
#             self.rec = True    
#             
#     def get_vocab_meta(self):
#         with h5py.File(self.fuelfile) as f:
#             voc2idx = yaml.load(f[self.name].attrs['voc2idx'])
#             idx2voc = yaml.load(f[self.name].attrs['idx2voc'])
#         self.voc2idx, self.idx2voc = voc2idx, idx2voc
#         return voc2idx, idx2voc
#     
#     def get_vocab_size(self):
#         with h5py.File(self.fuelfile) as f:
#             vocabsize = int(f[self.name].attrs['vocabsize'])
#         return vocabsize
#     
#     def build_model(self, x, config):
#         logger.info('building %s model for: %s ', self.nn_model, self.name)
#         vocabsize = self.get_vocab_size()
#         logger.info('%s vocab size is: %d', self.name, vocabsize)
#         self.embeddings, self.dim_emb = self.get_embeddings() 
#         if self.tune_tune:
#             logger.info('%s lookuptable with size (%d, %d) will be tuned.', self.name, vocabsize, self.dim_emb)
#             lookup = LookupTable(length=vocabsize, dim=self.dim_emb)
#             lookup.allocate()
# #             add_role(lookup.W, WEIGHT)
#             lookup.W.name = 'lt.W'
#         else:
#             logger.info('%s lookuptable with size (%d, %d) will NOT be tuned.', self.name, vocabsize, self.dim_emb)
#             lookup = MyLookupTable(length=vocabsize, dim=self.dim_emb)
#             lookup.allocate()
#         lookup.name = self.name + 'lookuptable'
#         lookup.W.set_value(self.embeddings)
#         xemb = lookup.apply(x)
#         xemb = debug_print(xemb, 'xemb', False)
#         if 'cnn' in self.nn_model:
#             logger.info('CNN')
#             feature_vec, feature_vec_len = create_cnn_general(xemb, self.dim_emb, self.max_len, config, self.name)
#         elif self.nn_model == 'lstm':
#             feature_vec, feature_vec_len = create_lstm(xemb, self.dim_emb, False, config, self.name)
#         elif self.nn_model == 'bilstm':
#             feature_vec, feature_vec_len = create_lstm(xemb, self.dim_emb, True, config, self.name)
#         elif self.nn_model == 'ff':
#             feature_vec, feature_vec_len = create_ff(xemb, self.dim_emb, self.max_len, config)
#         elif self.nn_model == 'mean':
#             feature_vec, feature_vec_len = create_mean(xemb, self.dim_emb, self.max_len, config)
#         return feature_vec, feature_vec_len
#     
#     def get_embeddings(self):
#         if self.load_emb == True:        
#             with h5py.File(self.emb_file) as f:
#                 embeddings = f.get('vectors').value.astype('float32')
#         else:
#             embeddings = rng.uniform(-0.08, 0.08, size=(self.get_vocab_size(), self.dim_emb)).astype('float32') #* (4) + (-2)
#         self.dim_emb = embeddings.shape[1]
#         return embeddings, self.dim_emb
#         
# 
# class InputFixed():
#     def __init__(self, name, config):  
#         self.config = config
#         self.name = name
#         self.fuelfile = config['dsdir'] + '_' + self.name + '.h5py'
#         self.dim = self.get_metadata()
#         
#     def get_metadata(self):
#         with h5py.File(self.fuelfile) as f:
#             dim_emb = f[self.name].attrs['vectorsize']
#             return dim_emb
#         
#     def build_model(self, x, config):
#         return x, self.dim
# 
# class Target():
#     def __init__(self, name, config):  
#         self.config = config
#         self.name = name
#         self.fuelfile = config['dsdir'] + '_' + self.name + '.h5py'
#         self.t2idx = self.get_metadata()
#         
#     def get_metadata(self):
#         with h5py.File(self.fuelfile) as f:
#             t2idx = yaml.load(f[self.name].attrs['type_to_ix'])
#             return t2idx
        

from blocks.bricks import Linear, Tanh
from blocks.bricks import MLP, Rectifier, Tanh, Linear, Softmax, Logistic
from blocks import initialization
from blocks.bricks.base import Parameters, application
from blocks.bricks.cost import BinaryCrossEntropy, Cost

class MultiMisclassificationRate(Cost):
    """ Cost function that calculates the misclassification rate for a
        multi-label classification output.
    """
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat):
        """ Apply the cost function.

        :param y:       Expected output, must be a binary k-hot vector
        :param y_hat:   Observed output, must be a binary k-hot vector
        """
        mistakes = T.neq(y, y_hat)
        return mistakes.mean(dtype=theano.config.floatX)

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()
        
def softmax_layer(h, y, hidden_size, num_targets, cost_fn='cross'):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size, output_dim=num_targets)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    y_pred = T.argmax(linear_output, axis=1)
    label_of_predicted = debug_print(y[T.arange(y.shape[0]), y_pred], 'label_of_predicted', False)
    pat1 = T.mean(label_of_predicted)
    updates = None
    if 'ranking' in cost_fn:
        cost, updates = ranking_loss(linear_output, y)
        print 'using ranking loss function!'
    else:
        y_hat = Logistic().apply(linear_output)
        y_hat.name = 'y_hat'
        cost = cross_entropy_loss(y_hat, y)
    cost.name = 'cost'
    pat1.name = 'precision@1'
    misclassify_rate = MultiMisclassificationRate().apply(y, T.ge(linear_output, 0.5))
    misclassify_rate.name = 'error_rate'
    return cost, pat1, updates, misclassify_rate

