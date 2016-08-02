'''
Created on Oct 27, 2015

@author: yadollah
'''
import sys, os, string
import theano,numpy, codecs, h5py, yaml, logging
from src.common.myutils import convertTargetsToBinVec,\
    get_char_seq, build_char_vocab
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('makefueldataset')
import src.common.myutils as cmn
from fuel.datasets import H5PYDataset
import numpy as np
from src.classification.nn.gm.nsl.featurizer import build_features,\
    build_feature2id, make_input_matrix

def make_ngram_features(etrain2names, edev2names, etest2names, etrain2types, edev2types, etest2types, featuresToUse, word2cluster=None, npdir='', maxngram=5, t2idx=None):
    upto = -1
    etrain2rawFea = build_features(etrain2names, etrain2types, word2cluster, featuresToUse, maxnamenum=3, maxngram=maxngram, upto=upto)
    edev2rawFea = build_features(edev2names, edev2types, word2cluster, featuresToUse, maxnamenum=1, maxngram=maxngram, upto=upto)
    etest2rawFea = build_features(etest2names, etest2types, word2cluster, featuresToUse, maxnamenum=1, maxngram=maxngram, upto=upto)
    print '.. building raw features finished!'
    allfeatures = [v for mye in etrain2rawFea for name in etrain2rawFea[mye] for v in etrain2rawFea[mye][name]]
    allfeatures.extend([v for mye in edev2rawFea for name in edev2rawFea[mye] for v in edev2rawFea[mye][name]])
    allfeatures.extend([v for mye in etest2rawFea for name in etest2rawFea[mye] for v in etest2rawFea[mye][name]])
    fea2id = build_feature2id(allfeatures, npdir+'feature2id')
    print 'feature vocabsize = ', len(fea2id)
    (in_trn_matrix, _, _) = make_input_matrix(etrain2rawFea, etrain2types, fea2id, 'train', maxname=3, npdir=npdir, t2idx=t2idx)
    (in_tst_matrix, _, _) = make_input_matrix(etest2rawFea, etest2types, fea2id, 'test', npdir=npdir, t2idx=t2idx)
    (in_dev_matrix, _, _) = make_input_matrix(edev2rawFea, edev2types, fea2id, 'dev', npdir=npdir, t2idx=t2idx)
    return in_trn_matrix, in_tst_matrix, in_dev_matrix

print 'loading config file', sys.argv[1]
config = cmn.loadConfig(sys.argv[1])
hdf5_file = config['fuelfile'] 

trainfile=config['Etrain']
devfile=config['Edev']
testfile=config['Etest']
targetTypesFile=config['typefile']
max_len_name= int(config['max_name_length'])
vectorFile=config['ent_vectors']
if 'typecosine' in config:
    usetypecosine = cmn.str_to_bool(config['typecosine'])
brownMappingFile=config['brownclusters']
maxngram = int(config['maxngram'])
featuresToUse= [fea for fea in config['features'].split(' ')]
npdir = config['npdir']
if not os.path.exists(npdir): os.makedirs(npdir)

upto = -1
(t2idx, idx2t) = cmn.loadtypes(targetTypesFile)
numtargets = len(t2idx)
# load entity to type datasets
(etrain2types, etrain2names,_) = cmn.load_entname_ds(trainfile, t2idx)
print "number of train examples:" + str(len(etrain2names))
(etest2types, etest2names,_) = cmn.load_entname_ds(testfile, t2idx)
print "number of test examples:" + str(len(etest2names))
(edev2types, edev2names,_) = cmn.load_entname_ds(devfile, t2idx)
print "number of dev examples:" + str(len(edev2names))
#building character vocabulary using train names
char_vocab = build_char_vocab(etrain2names)
vocab_size = len(char_vocab)
print 'distinct char_vocab', char_vocab, ' and #=', len(char_vocab)
char_to_ix = {ch: i for i, ch in enumerate(char_vocab)}
ix_to_char = {i: ch for i, ch in enumerate(char_vocab)}

maxngram = int(config['maxngram'])
featuresToUse= [fea for fea in config['features'].split(' ')]
npdir = config['npdir']
if not os.path.exists(npdir): os.makedirs(npdir)

#loading word2vec embeddings of words, entities, and types

(input_nsl_trn, input_nsl_dev, input_nsl_tst) = make_ngram_features(etrain2names, edev2names, etest2names, etrain2types, edev2types, etest2types, featuresToUse, npdir=npdir, maxngram=maxngram, t2idx=t2idx) 

(ttt, n_targets, wordvectors, vectorsize, typefreq_traindev) = cmn.loadTypesAndVectors(targetTypesFile, vectorFile, upto=1000)

(in_let_trn, in_ent_vec_trn, in_mention_words_vec_trn, enttrn, outputs_trn, in_tycos_trn) = fill_inout(wordvectors, vectorsize, char_to_ix, max_len_name, etrain2types, etrain2names, t2idx, maxnamenum=1, maxwords=4, usetypecos=usetypecosine)
(in_let_dv, in_ent_vec_dv, in_mention_words_vec_dv, entdv, outputs_dv, in_tycos_dev) = fill_inout(wordvectors, vectorsize, char_to_ix, max_len_name, edev2types, edev2names, t2idx, maxnamenum=1, usetypecos=usetypecosine)
(in_let_tst, in_ent_vec_tst, in_mention_words_vec_tst, enttst, outputs_tst, in_tycos_tst) = fill_inout(wordvectors, vectorsize, char_to_ix, max_len_name, etest2types, etest2names, t2idx, maxnamenum=1, usetypecos=usetypecosine)

nsamples_test = len(in_let_tst); nsamples_dev = len(in_let_dv); nsamples_train = len(in_let_trn)
nsamples = nsamples_train + nsamples_dev + nsamples_test
logger.info('nsamples_train = %d -- dev = %s -- test = %d', nsamples_train, nsamples_dev, nsamples_test)

input_letters = np.vstack((in_let_trn, in_let_dv, in_let_tst))
input_ent_vec = np.vstack((in_ent_vec_trn, in_ent_vec_dv, in_ent_vec_tst))
input_mention_words_vec = np.vstack((in_mention_words_vec_trn, in_mention_words_vec_dv, in_mention_words_vec_tst))
input_ent_type_cos = np.vstack((in_tycos_trn, in_tycos_dev, in_tycos_tst))
input_nsl = np.vstack((input_nsl_trn, input_nsl_dev, input_nsl_tst))
outputs = np.vstack((outputs_trn, outputs_dv, outputs_tst))

logger.info('building input and output matrixes finished!')

f = h5py.File(hdf5_file, mode='w')
features_letters = f.create_dataset('letters', input_letters.shape, dtype='uint8')  # @UndefinedVariable
features_entvec = f.create_dataset('w2v_entvec', input_ent_vec.shape, dtype='float32')  # @UndefinedVariable
features_mentionvec = f.create_dataset('w2v_mention', input_mention_words_vec.shape, dtype='float32')  # @UndefinedVariable
features_typecos = f.create_dataset('w2v_typecos', input_ent_type_cos.shape, dtype='float32')  # @UndefinedVariable
features_nsl = f.create_dataset('nsl_features', input_nsl.shape, dtype='uint8')  # @UndefinedVariable
targets = f.create_dataset('targets', outputs.shape, dtype='uint8')

features_letters.attrs['train_ents'] = yaml.dump(enttrn, default_flow_style=False)
features_letters.attrs['dev_ents'] = yaml.dump(entdv, default_flow_style=False)
features_letters.attrs['test_ents'] = yaml.dump(enttst, default_flow_style=False)
features_entvec.attrs['vector_size'] = yaml.dump(vectorsize)
features_mentionvec.attrs['vector_size'] = yaml.dump(vectorsize)
features_letters.attrs['num_features'] = yaml.dump(vocab_size) # which is 100 now - distinct letters + digits + punctuations
targets.attrs['type_to_ix'] = yaml.dump(t2idx)
targets.attrs['ix_to_type'] = yaml.dump(idx2t)
print input_letters.shape, outputs.shape

features_letters[...] = input_letters
features_entvec[...] = input_ent_vec
features_mentionvec[...] = input_mention_words_vec
features_typecos[...] = input_ent_type_cos
features_nsl[...] = input_nsl
targets[...] = outputs

features_letters.dims[0].label = 'entity'
features_letters.dims[1].label = 'name_char_seq'
features_entvec.dims[0].label = 'entity'
features_entvec.dims[1].label = 'w2v_embedding'
features_mentionvec.dims[0].label = 'entity'
features_mentionvec.dims[1].label = 'w2v_mentionwords_embedding'
targets.dims[0].label = 'entity'
targets.dims[1].label = 'types-binvec'

split_dict = {
    'train': {'letters': (0, nsamples_train), 
              'w2v_entvec': (0, nsamples_train), 
              'w2v_mention': (0, nsamples_train), 
              'w2v_typecos': (0, nsamples_train), 
              'nsl_features': (0, nsamples_train), 
              'targets': (0, nsamples_train)},
    'dev': {'letters': (nsamples_train, nsamples_train + nsamples_dev), 
            'w2v_entvec': (nsamples_train, nsamples_train + nsamples_dev), 
            'w2v_mention': (nsamples_train, nsamples_train + nsamples_dev), 
            'w2v_typecos': (nsamples_train, nsamples_train + nsamples_dev), 
            'nsl_features': (nsamples_train, nsamples_train + nsamples_dev), 
            'targets': (nsamples_train, nsamples_train + nsamples_dev)},
    'test': {'letters': (nsamples_train + nsamples_dev, nsamples), 
             'w2v_entvec': (nsamples_train + nsamples_dev, nsamples), 
             'w2v_mention': (nsamples_train + nsamples_dev, nsamples), 
             'w2v_typecos': (nsamples_train + nsamples_dev, nsamples), 
             'nsl_features': (nsamples_train + nsamples_dev, nsamples), 
             'targets': (nsamples_train + nsamples_dev, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
    


