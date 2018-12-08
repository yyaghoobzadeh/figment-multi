import logging
import os

import numpy
import numpy, logging
from sphinx.domains.std import Target
import theano

from blocks import initialization
from blocks import main_loop
from blocks.bricks import Linear, Tanh, Initializable, Feedforward, Sequence
from blocks.bricks import MLP, Rectifier, Tanh, Linear, Softmax, Logistic
from blocks.bricks.conv import Convolutional, MaxPooling, ConvolutionalSequence, Flattener
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate, BinaryCrossEntropy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent, Bidirectional
from blocks.extensions import saveload, predicates
from blocks.extensions.training import TrackTheBest
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.initialization import Uniform
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import check_theano_variable, shared_floatx_nans
from fuel.datasets import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping, Cast, AgnosticSourcewiseTransformer, FilterSources, Merge
from fuel.utils import do_not_pickle_attributes
import h5py,yaml
from myutils import build_ngram_vocab, get_ngram_seq, \
    str_to_bool, debug_print
from myutils import debug_print, logger, str_to_bool
import theano.tensor as T
from collections import OrderedDict
logger = logging.getLogger('model.py')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')
rng = numpy.random.RandomState(23455)
from blocks.bricks.cost import Cost
from blocks.bricks.base import Parameters, application

seq_features = ['letters', 'words', 'ngrams2', 'ngrams3', 'ngrams4', 'ngrams5', 'subwords', 'desc_features']
seed = 42
if "gpu" in theano.config.device:
    srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
    srng = T.shared_randomstreams.RandomStreams(seed=seed)
else:
    srng = T.shared_randomstreams.RandomStreams(seed=seed)

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
    tracker = TrackTheBest(channel, choose_best=min)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"], predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]



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
            iterschema = ShuffledScheme(examples=num_examples, batch_size=batch_size, rng=numpy.random.RandomState(seed))
        else: 
            iterschema = SequentialScheme(examples=num_examples, batch_size=batch_size)
        stream = DataStream(dataset=dataset, iteration_scheme=iterschema)
        if fea in seq_features:
            stream = CutInput(stream, obj.max_len)
            if obj.rec == True:
                logger.info('transforming data for recursive input')
                stream = LettersTransposer(stream, which_sources=fea)# Required because Recurrent last_hid receive as input [sequence, batch,# features]
        streams.append(stream)
    stream = Merge(streams, tuple(fea2obj.keys()))
    return stream, num_examples

# def ranking_loss(score_given_x, y, rng):
#         inxs = T.arange(y.shape[0])
#         neg_class = rng.uniform(low=0, high=101, size=y.shape[0])
#         margins = T.maximum(0., 1. - score_given_x[inxs, y] * score_given_x[inxs, neg_class])
#         return 
    

class MyLookupTable():
    def __init__(self, length, dim):
        self.length = length
        self.dim = dim
    def allocate(self):
        self.W = shared_floatx_nans((self.length, self.dim))
        self.W.name = 'W'
    def W(self):
        return self.W
    def apply(self, indices):
        check_theano_variable(indices, None, ("int", "uint"))
        output_shape = [indices.shape[i]
                        for i in range(indices.ndim)] + [self.dim]
        return self.W[indices.flatten()].reshape(output_shape)
    
class SequentialInput():
    def __init__(self, name, config):
        self.config = config
        self.name = name
        self.max_len = int(config[self.name + '_max_len'])
        self.dim_emb = int(config[self.name + '_emb_dim'])
        self.fuelfile = config['dsdir'] + '_' + name + '.h5py'
        self.nn_model = config[self.name + '_nn']
        self.emb_file = self.config['dsdir'] + '_' + self.name + '_embeddings.h5py'
        self.tune_tune = str_to_bool(self.config[self.name + '_tune_embedding'])
        self.load_emb = str_to_bool(self.config[self.name + '_load_embedding'])
        if self.load_emb == False: 
            assert self.tune_tune == True
        self.rec = False
        if 'lstm' in self.nn_model or 'rnn' in self.nn_model or 'gru' in self.nn_model: 
            self.rec = True    
            
    def get_vocab_meta(self):
        with h5py.File(self.fuelfile) as f:
            voc2idx = yaml.load(f[self.name].attrs['voc2idx'])
            idx2voc = yaml.load(f[self.name].attrs['idx2voc'])
        self.voc2idx, self.idx2voc = voc2idx, idx2voc
        return voc2idx, idx2voc
    
    def get_vocab_size(self):
        with h5py.File(self.fuelfile) as f:
            vocabsize = int(f[self.name].attrs['vocabsize'])
        return vocabsize
    
    def build_model(self, x, config):
        logger.info('building %s model for: %s ', self.nn_model, self.name)
        vocabsize = self.get_vocab_size()
        logger.info('%s vocab size is: %d', self.name, vocabsize)
        self.embeddings, self.dim_emb = self.get_embeddings() 
        if self.tune_tune:
            logger.info('%s lookuptable with size (%d, %d) will be tuned.', self.name, vocabsize, self.dim_emb)
            lookup = LookupTable(length=vocabsize, dim=self.dim_emb)
            lookup.allocate()
#             add_role(lookup.W, WEIGHT)
            lookup.W.name = 'lt.W'
        else:
            logger.info('%s lookuptable with size (%d, %d) will NOT be tuned.', self.name, vocabsize, self.dim_emb)
            lookup = MyLookupTable(length=vocabsize, dim=self.dim_emb)
            lookup.allocate()
        lookup.name = self.name + 'lookuptable'
        lookup.W.set_value(self.embeddings)
        xemb = lookup.apply(x)
        xemb = debug_print(xemb, 'xemb', False)
        if 'cnn' in self.nn_model:
            logger.info('CNN')
            feature_vec, feature_vec_len = create_cnn_general(xemb, self.dim_emb, self.max_len, config, self.name)
        elif self.nn_model == 'lstm':
            feature_vec, feature_vec_len = create_lstm(xemb, self.dim_emb, False, config, self.name)
        elif self.nn_model == 'bilstm':
            feature_vec, feature_vec_len = create_lstm(xemb, self.dim_emb, True, config, self.name)
        elif self.nn_model == 'rnn':
            feature_vec, feature_vec_len = create_rnn(xemb, self.dim_emb, config, self.name)
        elif self.nn_model == 'ff':
            feature_vec, feature_vec_len = create_ff(xemb, self.dim_emb, self.max_len, config)
        elif self.nn_model == 'mean':
            feature_vec, feature_vec_len = create_mean(xemb, self.dim_emb, self.max_len, config)
        return feature_vec, feature_vec_len
    
    def get_embeddings(self):
        if self.load_emb == True:        
            with h5py.File(self.emb_file) as f:
                embeddings = f.get('vectors').value.astype('float32')
        else:
            embeddings = rng.uniform(-0.08, 0.08, size=(self.get_vocab_size(), self.dim_emb)).astype('float32') #* (4) + (-2)
        self.dim_emb = embeddings.shape[1]
        return embeddings, self.dim_emb
        

class InputFixed():
    def __init__(self, name, config):  
        self.config = config
        self.name = name
        self.fuelfile = config['dsdir'] + '_' + self.name + '.h5py'
        self.dim = self.get_metadata()
        
    def get_metadata(self):
        with h5py.File(self.fuelfile) as f:
            dim_emb = f[self.name].attrs['vectorsize']
            return dim_emb
        
    def build_model(self, x, config):
        return x, self.dim

class Target():
    def __init__(self, name, config):  
        self.config = config
        self.name = name
        self.fuelfile = config['dsdir'] + '_' + self.name + '.h5py'
        self.t2idx = self.get_metadata()
        
    def get_metadata(self):
        print self.fuelfile
        with h5py.File(self.fuelfile) as f:
            t2idx = yaml.load(f[self.name].attrs['type_to_ix'])
            return t2idx
        
def initialize(to_init, rndstd=0.01):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

def initialize_lasthid(last_hid, matrixfile=None, max_dim=None):
    rng = numpy.random.RandomState(42)
    w = 0.08
    myarray = rng.uniform(-w, +w, size=(last_hid.input_dim, last_hid.output_dim))
    print myarray.shape
    if matrixfile:
        typematrix = (numpy.load(matrixfile))
        if max_dim == None: max_dim = len(typematrix)
        print typematrix.shape
        myarray[0:max_dim, :] = typematrix[0:max_dim, :]
        print myarray
    last_hid.weights_init = initialization.Constant(myarray)
    last_hid.biases_init = initialization.Constant(0)
    last_hid.initialize()
    
def initialize2(brick, num_feature_maps):
    fan_in = numpy.prod(brick.filter_size)
    fan_out = numpy.prod(brick.filter_size) * brick.num_filters / num_feature_maps
    W_bound = numpy.sqrt(6. / (fan_in + fan_out)) 
    brick.weights_init = initialization.Uniform(width=W_bound)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()

def rnn_layer(in_dim, h, h_dim, n, pref=""):
    linear = Linear(input_dim=in_dim, output_dim=h_dim, name='linear' + str(n) + pref)
    rnn = SimpleRecurrent(dim=h_dim, activation=Tanh(), name='rnn' + str(n) + pref)
    initialize([linear, rnn])
    return rnn.apply(linear.apply(h))


def gru_layer(dim, h, n):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates)


def lstm_layer(in_dim, h, h_dim, n, pref=""):
    linear = Linear(input_dim=in_dim, output_dim=h_dim * 4, name='linear' + str(n) + pref)
    lstm = LSTM(dim=h_dim, name='lstm' + str(n) + pref)
    initialize([linear, lstm])
    return lstm.apply(linear.apply(h))[0]

def bilstm_layer(in_dim, inp, h_dim, n, pref=""):
    linear = Linear(input_dim=in_dim, output_dim=h_dim * 4, name='linear' + str(n) + pref)
    lstm = LSTM(dim=h_dim, name='lstm' + str(n) + pref)
    bilstm = Bidirectional(prototype=lstm)
    bilstm.name = 'bilstm' + str(n) + pref
    initialize([linear, bilstm])
    return bilstm.apply(linear.apply(inp))[0]

def softmax_layer_old(h, y, hidden_size, num_targets, cost_fn='softmax'):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size, output_dim=num_targets)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    y_pred = T.argmax(linear_output, axis=1)
    label_of_predicted = debug_print(y[T.arange(y.shape[0]), y_pred], 'label_of_predicted', False)
    pat1 = T.mean(label_of_predicted)
    updates = {}
    if 'softmax' in cost_fn:
        y_hat = Logistic().apply(linear_output)
        y_hat.name = 'y_hat'
        cost = cross_entropy_loss(y_hat, y)
    else:
        cost, updates = ranking_loss(linear_output, y)
    cost.name = 'cost'
    pat1.name = 'precision@1'
    return cost, pat1, updates

def create_yy_cnn(numConvLayer, conv_input, embedding_size, input_len, config, pref):
    '''
     CNN with several layers of convolution, each with specific filter size. 
     Maxpooling at the end. 
    '''
    filter_width_list = [int(fw) for fw in config[pref + '_filterwidth'].split()]
    base_num_filters = int(config[pref + '_num_filters'])
    assert len(filter_width_list) == numConvLayer
    convs = []; fmlist = []
    last_fm = input_len
    for i in range(numConvLayer):
        fw = filter_width_list[i]
        num_feature_map = last_fm - fw + 1 #39
        conv = Convolutional(
            image_size=(last_fm, embedding_size),
            filter_size=(fw, embedding_size),
            num_filters=min(int(config[pref + '_maxfilter']), base_num_filters * fw),
            num_channels=1
        )
        fmlist.append(num_feature_map)
        last_fm = num_feature_map
        embedding_size = conv.num_filters
        convs.append(conv)

    initialize(convs)
    for i, conv in enumerate(convs):
        conv.name = pref+'_conv' + str(i)
        conv_input = conv.apply(conv_input)
        conv_input = conv_input.flatten().reshape((conv_input.shape[0], 1, fmlist[i], conv.num_filters))
        lastconv = conv 
        lastconv_out = conv_input
    pool_layer = MaxPooling(
        pooling_size=(last_fm,1)
    )
    pool_layer.name = pref+'_pool_' + str(fw)
    act = Rectifier(); act.name = 'act_' + str(fw)
    outpool = act.apply(pool_layer.apply(lastconv_out).flatten(2))
    return outpool, lastconv.num_filters


def create_kim_cnn(layer0_input, embedding_size, input_len, config, pref):
    '''
        One layer convolution with different filter-sizes and maxpooling
    '''
    filter_width_list = [int(fw) for fw in config[pref + '_filterwidth'].split()]
    print filter_width_list
    num_filters = int(config[pref+'_num_filters'])
    #num_filters /= len(filter_width_list)
    totfilters = 0
    for i, fw in enumerate(filter_width_list):
        num_feature_map = input_len - fw + 1 #39
        conv = Convolutional(
            image_size=(input_len, embedding_size),
            filter_size=(fw, embedding_size),
            num_filters=min(int(config[pref + '_maxfilter']), num_filters * fw),
            num_channels=1
        )
        totfilters += conv.num_filters
        initialize2(conv, num_feature_map)
        conv.name = pref + 'conv_' + str(fw)
        convout = conv.apply(layer0_input)
        pool_layer = MaxPooling(
            pooling_size=(num_feature_map,1)
        )
        pool_layer.name = pref + 'pool_' + str(fw)
        act = Rectifier()
        act.name = pref + 'act_' + str(fw)
        outpool = act.apply(pool_layer.apply(convout)).flatten(2)
        if i == 0:
            outpools = outpool
        else:
            outpools = T.concatenate([outpools, outpool], axis=1)
    name_rep_len = totfilters
    return outpools, name_rep_len

def create_OLD_kim_cnn(layer0_input, embedding_size, input_len, config, pref):
    '''
        One layer convolution with the same filtersize
    '''
    filter_width_list = [int(fw) for fw in config[pref + '_filterwidth'].split()]
    print filter_width_list
    num_filters = int(config[pref+'_num_filters'])
    totfilters = 0
    for i, fw in enumerate(filter_width_list):
        num_feature_map = input_len - fw + 1 #39
        conv = Convolutional(
                    filter_size=(fw, embedding_size), 
                    num_filters=num_filters,
                    num_channels=1,
                    image_size=(input_len, embedding_size),
                    name="conv" + str(fw))
        pooling = MaxPooling((num_feature_map,1), name="pool"+str(fw))
        initialize([conv])
                
        totfilters += num_filters
        outpool = Flattener(name="flat"+str(fw)).apply(Rectifier(name=pref+'act_'+str(fw)).apply(pooling.apply(conv.apply(layer0_input))))
        if i == 0:
            outpools = outpool
        else:
            outpools = T.concatenate([outpools, outpool], axis=1)
    name_rep_len = totfilters
    return outpools, name_rep_len

def create_cnn_general(xemb, embedding_size, input_len, config, pref):
    numConvLayers = int(config[pref + '_convlayers'])
    xemb = debug_print(xemb, 'afterLookup', False)
    layer0_input = xemb.flatten().reshape((xemb.shape[0], 1, input_len, embedding_size))
    if numConvLayers == 1:
        # return create_kim_cnn(layer0_input, embedding_size, input_len, config, pref)
        return create_OLD_kim_cnn(layer0_input, embedding_size, input_len, config, pref)
    else:        
        return create_yy_cnn(numConvLayers, layer0_input, embedding_size, input_len, config, pref)
    
def create_lstm(xemb, embedding_size, bidirectional, config, pref):
    hiddensize = int(config[pref + '_h_lstm'])
    inpsize = embedding_size
    if bidirectional:
        for i in range(1):
            xemb = bilstm_layer(inpsize, xemb, hiddensize, i, pref)
            xemb.name = 'bilstm' + str(i) + pref
            inpsize = hiddensize * 2
        lstm_outsize = hiddensize * 2
        h = xemb
    else:
        for i in range(1):
            xemb = lstm_layer(embedding_size, xemb, hiddensize, 1, pref)
            embedding_size = hiddensize
            xemb.name = 'lstm' + str(i) + pref
        h = xemb
        lstm_outsize = hiddensize
    h = debug_print(h[h.shape[0] - 1], 'outlstm', False)
    return h, lstm_outsize

def create_rnn(xemb, embedding_size, config, pref):
    hiddensize = int(config[pref + '_h_rnn'])
    for i in range(1):
        xemb = rnn_layer(embedding_size, xemb, hiddensize, 1, pref)
        embedding_size = hiddensize
        xemb.name = 'rnn' + str(i) + pref
    h = xemb
    lstm_outsize = hiddensize
    h = debug_print(h[h.shape[0] - 1], 'rnn', False)
    return h, lstm_outsize


def create_ff(xemb, embedding_size, in_len, config):
    ff_rep = xemb.flatten(2)
    return ff_rep, embedding_size * in_len 

def create_mean(xemb, embedding_size, in_len, config):
    xmean = T.mean(xemb, axis=1) 
    return xmean, embedding_size

def relu(x):
    return T.switch(x<0, 0, x)

def recognition_network(x, n_input, hu_encoder, n_latent):
    logger.info('In recognition_network: n_input: %d, hu_encoder: %d', n_input, hu_encoder)
    x = debug_print(x, 'inp_enc', False)
    mlp1 = MLP(activations=[Rectifier()], dims=[n_input, hu_encoder], name='recog_in_to_hidEncoder')
    initialize([mlp1])
    h_encoder = mlp1.apply(x)
    lin1 = Linear(name='recog_hiddEncoder_to_latent_mu', input_dim=hu_encoder, output_dim=n_latent)
    lin2 = Linear(name='recog_hiddEncoder_to_latent_sigma', input_dim=hu_encoder, output_dim=n_latent)
    initialize([lin1])
    initialize([lin2], rndstd=0.001)
    mu = lin1.apply(h_encoder)
    log_sigma = lin2.apply(h_encoder)
    return mu, log_sigma

def prior_network(x, n_input, hu_encoder, n_latent):
    logger.info('In prior_network: n_input: %d, hu_encoder: %d', n_input, hu_encoder)
    mlp1 = MLP(activations=[Rectifier()], dims=[n_input, hu_encoder], name='prior_in_to_hidEncoder')
    initialize([mlp1])
    h_encoder = mlp1.apply(x)
    h_encoder = debug_print(h_encoder, 'h_encoder', False)
    lin1 = Linear(name='prior_hiddEncoder_to_latent_mu', input_dim=hu_encoder, output_dim=n_latent)
    lin2 = Linear(name='prior_hiddEncoder_to_latent_sigma', input_dim=hu_encoder, output_dim=n_latent)
    initialize([lin1])
    initialize([lin2], rndstd=0.001)
    mu = lin1.apply(h_encoder)
    log_sigma = lin2.apply(h_encoder)
    return mu, log_sigma

def sampler(mu, log_sigma, deterministic=False, use_noise=True, input_log=False):
    log_sigma = debug_print(log_sigma, 'log_sigma', False)
    logger.info('deterministic: %s --- use noise: %s', deterministic, use_noise)
    if deterministic and use_noise:
        return mu
    if deterministic:
        #return mu + T.exp(0.5 * log_sigma)
        return mu + log_sigma
    eps = srng.normal(size=mu.shape, std=1)
    # Reparametrize
    if use_noise:
        if input_log:
            return mu + T.exp(0.5 * log_sigma) * eps
        else: 
            return mu + log_sigma * eps
    else:
        #return mu + T.exp(0.5 * log_sigma)
        return mu + log_sigma

def generation(z_list, n_latent, hu_decoder, n_out, y):
    logger.info('in generation: n_latent: %d, hu_decoder: %d', n_latent, hu_decoder)
    if hu_decoder == 0:
        return generation_simple(z_list, n_latent, n_out, y)
    mlp1 = MLP(activations=[Rectifier()], dims=[n_latent, hu_decoder], name='latent_to_hidDecoder')
    initialize([mlp1])
    hid_to_out = Linear(name='hidDecoder_to_output', input_dim=hu_decoder, output_dim=n_out)
    initialize([hid_to_out])
    mysigmoid = Logistic(name='y_hat_vae')
    agg_logpy_xz = 0.
    agg_y_hat = 0.
    for i, z in enumerate(z_list):
        y_hat = mysigmoid.apply(hid_to_out.apply(mlp1.apply(z))) #reconstructed x
        agg_logpy_xz += cross_entropy_loss(y_hat, y)
        agg_y_hat += y_hat
    
    agg_logpy_xz /= len(z_list)
    agg_y_hat /= len(z_list)
    return agg_y_hat, agg_logpy_xz

def generation_simple(z_list, n_latent, n_out, y):
    logger.info('generate output without MLP')
    hid_to_out = Linear(name='hidDecoder_to_output', input_dim=n_latent, output_dim=n_out)
    initialize([hid_to_out])
    mysigmoid = Logistic(name='y_hat_vae')
    agg_logpy_xz = 0.
    agg_y_hat = 0.
    for z in z_list:
        lin_out = hid_to_out.apply(z)
        y_hat = mysigmoid.apply(lin_out) #reconstructed x
        logpy_xz = -cross_entropy_loss(y_hat, y)
        agg_logpy_xz += logpy_xz
        agg_y_hat += y_hat
    agg_logpy_xz /= len(z_list)
    agg_y_hat /= len(z_list)
    return agg_y_hat, agg_logpy_xz

def compute_KLD_old(qmu, qsigma, pmu, psigma):
    KLdiv=0.
    KLdiv += T.sum(psigma, axis=1)
    KLdiv -= T.sum(qsigma, axis=1)
    qsigma = T.exp(qsigma)
    psigma = T.exp(psigma)
    diff = pmu - qmu
    kl = KLdiv - psigma.shape[1] + T.sum(qsigma/psigma, axis=1) + T.sum(diff**2/psigma, axis=1)
    kl *= 0.5
    return kl

def compute_KLD(qmu, qsigma, pmu, psigma):
    psigma = psigma ** 2
    qsigma = qsigma ** 2
    log_p_sigma = T.log(psigma)
    log_q_sigma = T.log(qsigma)
    KLdiv = T.sum(log_p_sigma - log_q_sigma, axis=1)
    diff = pmu - qmu
    KLdiv = KLdiv - psigma.shape[1] + T.sum(qsigma/psigma, axis=1) + T.sum(diff**2/psigma, axis=1)
    KLdiv *= 0.5
    return KLdiv


def build_feature_vec(fea2obj, config):
    feature_vec = None; feature_vec_len = 0
    for fea in fea2obj:
        print fea, fea2obj[fea]
        if fea == 'targets' : continue
        if fea in seq_features:
            x = T.matrix(fea, dtype='int32')
        else:
            x = T.matrix(fea, dtype='float32')
            
        fv, fv_len = fea2obj[fea].build_model(x, config)
        if feature_vec != None:
            feature_vec = T.concatenate([feature_vec, fv], axis=1)
            feature_vec_len += fv_len
        else:
            feature_vec = fv
            feature_vec_len = fv_len
    return feature_vec, feature_vec_len

    #KLD =  kld_weight * -0.5 * T.sum(1 + log_sigma_prior - mu_prior**2 - T.exp(log_sigma_prior), axis=1)
def build_vae_basic(kl_weight, feature_vec, feature_vec_len, config, y, test=False, deterministic=False, num_targets=102, n_latent_z=50, hidden_size=400, hu_decoder=200):
    logger.info('build VAE recognition network using basic prior: p(z)')
    y_as_float = T.cast(y, 'float32')
    drop_prob = float(config['dropprob']) if 'dropprob' in config else 0
    mask = T.cast(srng.binomial(n=1, p=1-drop_prob, size=feature_vec.shape), 'float32')
    KLD=0
    if test:
        gen_inp = []
        for _ in range(10):
            z_sampled = srng.normal([feature_vec.shape[0], n_latent_z])
            #z_sampled = T.cast(srng.binomial(n=1, p=0, size=[feature_vec.shape[0], n_latent_z]), 'float32')
            gen_inp.append(T.concatenate([z_sampled, feature_vec], axis=1))
    else:
        recog_input = T.concatenate([feature_vec*mask, y_as_float], axis=1)
        mu_recog, log_sigma_recog = recognition_network(x=recog_input, n_input=feature_vec_len+num_targets, hu_encoder=hidden_size, n_latent=n_latent_z)
        z_sampled = sampler(mu_recog, log_sigma_recog, deterministic=deterministic,use_noise=True, input_log=True)
        gen_inp = [T.concatenate([z_sampled, feature_vec], axis=1)]
        KLD =  kl_weight * -0.5 * T.sum(1 + log_sigma_recog - mu_recog**2 - T.exp(log_sigma_recog), axis=1)
    y_hat, logpy_z = generation(gen_inp, n_latent=n_latent_z+feature_vec_len, hu_decoder=hu_decoder, n_out=num_targets, y=y)
    #logpy_z *= 1 - T.nnet.sigmoid(kl_weight)
    return y_hat, logpy_z, KLD

def build_vae_conditoinal(kl_weight, entropy_weight, y_hat_init, feature_vec, feature_vec_len, config, y,
        test=False, deterministic=False, num_targets=102, n_latent_z=50, hidden_size=400, hu_decoder=200):
    logger.info('build VAE recognition network using conditional modeling: q(z|x,y)')
    y_as_float = T.cast(y, 'float32')
    drop_prob = float(config['dropprob']) if 'dropprob' in config else 0
    logger.info('drop out probability: %d', drop_prob)
    if test == False or True:
        mask = T.cast(srng.binomial(n=1, p=1-drop_prob, size=feature_vec.shape), 'float32')
#         feature_vec *= mask
    recog_input = T.concatenate([feature_vec * mask, y_as_float], axis=1)
    logpy_xz_init = cross_entropy_loss(y_hat_init, y)
    # recognition network q(z|x,y) #sampling z from recognition
    mu_recog, log_sigma_recog = recognition_network(x=recog_input, n_input=feature_vec_len+num_targets, hu_encoder=hidden_size, n_latent=n_latent_z)
    z_recog = sampler(mu_recog, log_sigma_recog, deterministic=deterministic, input_log=True)
    
    prior_input = T.concatenate([feature_vec, y_hat_init], axis=1)
    prinlen = feature_vec_len + num_targets
    mu_prior, log_sigma_prior = prior_network(x=prior_input, n_input=prinlen, hu_encoder=hidden_size, n_latent=n_latent_z)
    z_prior = sampler(mu_prior, log_sigma_prior, deterministic=deterministic, use_noise=True, input_log=True)
    
    if test:
        geninputs = [T.concatenate([z_prior, feature_vec], axis=1)]
        if deterministic == False:
            for _ in range(500):
                geninputs.append(T.concatenate([sampler(mu_prior, log_sigma_prior, deterministic=False, use_noise=True), feature_vec], axis=1))
        y_hat, logpy_z = generation(geninputs, n_latent=n_latent_z+feature_vec_len, hu_decoder=hu_decoder, n_out=num_targets, y=y)
        y_hat_init = 0.5 * (y_hat + y_hat_init)
#         y_hat_init = y_hat 
    else:
        gen_inp = [T.concatenate([z_recog, feature_vec], axis=1)]
        y_hat, logpy_z = generation(gen_inp, n_latent=n_latent_z+feature_vec_len, hu_decoder=hu_decoder, n_out=num_targets, y=y)
    logpy_z = (logpy_xz_init + logpy_z) / 2.
    KLD = kl_weight * compute_KLD_old(mu_recog, log_sigma_recog, mu_prior, log_sigma_prior)
    entropy_weight = T.nnet.sigmoid(-kl_weight)
    entropy_weight = T.switch(entropy_weight > 0, entropy_weight, 1)
#     logpy_z *= 1 - T.nnet.sigmoid(kl_weight)
    return y_hat_init, logpy_z, KLD, y_hat

def build_model_new(fea2obj, num_targets, config, kl_weight, entropy_weight, deterministic=False, test=False ):
    hidden_size = config['hidden_units'].split()
    use_highway = str_to_bool(config['use_highway']) if 'use_highway' in config else False
    use_gaus = str_to_bool(config['use_gaus']) if 'use_gaus' in config else False 
    use_rec = str_to_bool(config['use_rec']) if 'use_rec' in config else True
    n_latent_z = int(config['n_latent']) if 'use_gaus' in config else 0
    use_noise = str_to_bool(config['use_noise']) if 'use_noise' in config else False
    use_vae=str_to_bool(config['use_vae']) if 'use_vae' in config else False
    hu_decoder = int(config['hu_decoder']) if 'hu_decoder' in config else hidden_size
    logger.info('use_gaus: %s, use_rec: %s, use_noise: %s, use_vae: %s, hidden_size: %s, n_latent_z: %d, hu_decoder: %s, hu_encoder: %s', use_gaus, use_rec, use_noise, use_vae, hidden_size, n_latent_z, hu_decoder, hidden_size)
    init_with_type = str_to_bool(config['init_with_type']) if 'init_with_type' in config else False
    y = T.matrix('targets', dtype='int32')
    
    drop_prob = float(config['dropout']) if 'dropout' in config else 0
    
    #build the feature vector with one model, e.g., with cnn or mean or lstm
    feature_vec, feature_vec_len = build_feature_vec(fea2obj, config)
    
    #drop out
    if drop_prob > 0:
        mask = T.cast(srng.binomial(n=1, p=1-drop_prob, size=feature_vec.shape), 'float32')
        if test:
            feature_vec *= (1 - drop_prob)
        else:
            feature_vec *= mask
            

    #Highway network
    if use_highway:
        g_mlp = MLP(activations=[Rectifier()], dims=[feature_vec_len, feature_vec_len], name='g_mlp')
        t_mlp = MLP(activations=[Logistic()], dims=[feature_vec_len, feature_vec_len], name='t_mlp')
        initialize([g_mlp, t_mlp])
        t = t_mlp.apply(feature_vec)
        z = t * g_mlp.apply(feature_vec) + (1. - t) * feature_vec
        feature_vec = z
        
    #MLP(s)         
    logger.info('feature vec length = %s and hidden layer units = %s', feature_vec_len, ' '.join(hidden_size))
    if len(hidden_size) > 1:
        #2 MLP on feature fector    
        mlp = MLP(activations=[Rectifier(), Rectifier()], dims=[feature_vec_len, int(hidden_size[0]), int(hidden_size[1])], name='joint_mlp')
        initialize([mlp])
        before_out = mlp.apply(feature_vec)
        last_hidden_size = int(hidden_size[1])
    else:
        hidden_size = int(hidden_size[0])
        mlp = MLP(activations=[Rectifier()], dims=[feature_vec_len, hidden_size], name='joint_mlp')
        initialize([mlp])
        before_out = mlp.apply(feature_vec)
        last_hidden_size = hidden_size

        
    #compute y_hat initial guess
    hidden_to_output = Linear(name='hidden_to_output', input_dim=last_hidden_size, output_dim=num_targets)
    
    typemfile = None
    if init_with_type:
        typemfile = config['dsdir'] + '/_typematrix.npy'
        #typemfile = config['dsdir'] + '/_typeCooccurrMatrix.npy'
        
    initialize_lasthid(hidden_to_output, typemfile)
#         initialize([hidden_to_output])
    
    y_hat_init = Logistic().apply(hidden_to_output.apply(before_out))
    y_hat_init.name='y_hat_init'
    y_hat_init = debug_print(y_hat_init, 'yhat_init', False)
    logpy_xz_init = cross_entropy_loss(y_hat_init, y)
    logpy_xz = logpy_xz_init  
    y_hat_recog = y_hat_init
    y_hat = y_hat_init
    KLD = 0
    
    if use_gaus:     
        if use_vae:
            logger.info('using VAE')
            vae_conditional=str_to_bool(config['vae_cond']) 
            if vae_conditional:
                y_hat, logpy_xz, KLD, y_hat_recog = build_vae_conditoinal(kl_weight, entropy_weight, y_hat_init, feature_vec, feature_vec_len, config, y,
                    test=test, deterministic=deterministic, num_targets=num_targets, n_latent_z=n_latent_z, hidden_size=hidden_size, hu_decoder=hu_decoder)
            else:
                y_hat, logpy_xz, KLD = build_vae_basic(kl_weight, feature_vec, feature_vec_len, config, y, 
                    test=test, deterministic=deterministic, num_targets=num_targets, n_latent_z=n_latent_z, hidden_size=hidden_size, hu_decoder=hu_decoder)
                y_hat_recog = y_hat
        else:
            if use_rec:
                logger.info('Not using VAE... but using recursion')
                prior_in = T.concatenate([feature_vec, y_hat_init], axis=1)
                mu_prior, log_sigma_prior = prior_network(x=prior_in, n_input=feature_vec_len+num_targets, hu_encoder=hidden_size, n_latent=n_latent_z)
                z_prior = sampler(mu_prior, log_sigma_prior, deterministic=deterministic, use_noise=use_noise)
                zl = [T.concatenate([z_prior, feature_vec], axis=1)]
                y_hat, logpy_xz = generation(zl, n_latent=n_latent_z+feature_vec_len, hu_decoder=hu_decoder, n_out=num_targets, y=y)
                y_hat = (y_hat + y_hat_init) / 2. 
                logpy_xz = (logpy_xz + logpy_xz_init) / 2.
            else:
                prior_in = T.concatenate([feature_vec], axis=1)
                mu_prior, log_sigma_prior = prior_network(x=prior_in, n_input=feature_vec_len, hu_encoder=hidden_size, n_latent=n_latent_z)
                z_prior = sampler(mu_prior, log_sigma_prior, deterministic=deterministic, use_noise=use_noise)
                zl = [T.concatenate([z_prior, feature_vec], axis=1)]
                y_hat, logpy_xz = generation(zl, n_latent=n_latent_z+feature_vec_len, hu_decoder=hu_decoder, n_out=num_targets, y=y)
            
            y_hat_recog = y_hat
                

    y_hat = debug_print(y_hat, 'y_hat', False)

    pat1 = T.mean(y[T.arange(y.shape[0]), T.argmax(y_hat, axis=1)])
    max_type = debug_print(T.argmax(y_hat_recog, axis=1), 'max_type', False)
    pat1_recog = T.mean(y[T.arange(y.shape[0]), max_type])
    mean_cross = T.mean(logpy_xz)
    mean_kld = T.mean(KLD)
    cost = mean_kld + mean_cross 
    cost.name = 'cost'; mean_kld.name = 'kld'; mean_cross.name = 'cross_entropy_loss'; pat1.name = 'p@1'; pat1_recog.name = 'p@1_recog'
    misclassify_rate = MultiMisclassificationRate().apply(y, T.ge(y_hat, 0.5))
    misclassify_rate.name = 'error_rate'

    return cost, pat1, y_hat, mean_kld, mean_cross, pat1_recog, misclassify_rate

def build_input_objs(cnf_features, config):
    fea2obj = {}
    for fea in cnf_features:
        if fea in seq_features:
            obj = SequentialInput(fea, config)
        elif fea == 'targets':
            obj = Target(fea, config)
        else:
            obj = InputFixed(fea, config)
        fea2obj[fea] = obj
    return fea2obj

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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')
rng = numpy.random.RandomState(23455)
from theano.tensor import shared_randomstreams

seq_features = ['letters', 'words', 'ngrams2', 'ngrams3', 'ngrams4', 'ngrams5', 'subwords', 'desc_features']

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

