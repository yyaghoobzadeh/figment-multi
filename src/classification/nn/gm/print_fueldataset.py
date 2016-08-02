
'''
Created on Jan 22, 2016

@author: yadollah
'''
import sys, h5py, yaml
from fuel.datasets import H5PYDataset

def get_vocab_meta(fuelfile, srcname):
        with h5py.File(fuelfile) as f:
            voc2idx = yaml.load(f[srcname].attrs['voc2idx'])
            idx2voc = yaml.load(f[srcname].attrs['idx2voc'])
        return voc2idx, idx2voc
    
def load_w2v_vocab(vocfile):
    print 'loading word2vec vocab file: ', vocfile
    voc2freq = {}
    with open(vocfile) as fp:
        for myl in fp:
            parts = myl.split()
            voc2freq[parts[0]] = int(parts[1])
    return voc2freq

def find_w2v_fuel_map(fuelvoc, w2vvoc):
    print 'words in fuel vocab that are not in word2vec vocab:'
    for v in fuelvoc:
        if v not in w2vvoc:
            print v
            
def print_ds(idx2voc, data):
    for d in data:
        for j in d:
            name = ''
            for k in j: 
                name +=idx2voc[k]
            print name
    
    
if __name__ == '__main__':
    hdf5_file = sys.argv[1]
    dataset = H5PYDataset(hdf5_file, which_sets=('dev',), load_in_memory=True)
    data = dataset.data_sources
    print '# examples:', dataset.num_examples
    
    voc2idx, idx2voc = get_vocab_meta(hdf5_file, 'words')
    print idx2voc
    w2v_vocabfile = '/nfs/datm/cluewebwork/nlu/merged/12/00/00-02_w.ext.myVoc-cout.txt'
    w2v_voc2freq = load_w2v_vocab(w2v_vocabfile)
    
    find_w2v_fuel_map(voc2idx, w2v_voc2freq)
    
    
