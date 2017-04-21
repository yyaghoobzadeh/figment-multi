import sys,os
import numpy as np
import numpy

def save_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            print >>f, w

def read_embeddings(fname, num=None):
    """ Read word embeddings from file

    :param embedding_file:  Path to embedding file
    :type embedding_file:   str/unicode
    :param num:             Restrict number of embeddings to load
    :type num:              int
    :returns:               Mapping from words to their embedding
    :rtype:                 str -> numpy.array dict
    """
    # NOTE: This should be pretty efficient, since we're only reading the file
    #       line by line and are reading directly into memory-efficient numpy
    #       arrays.
    print "loading word2vec format embeddings from: ", fname
    with open(fname) as fp:
        num_vecs, vec_size = (int(x) for x in fp.readline().strip().split())
        num_vecs += 2  # For <UNK>
        embeddings = np.zeros((num_vecs, vec_size), dtype='float32')
        embeddings[0,:] = 0.001
        word_to_idx = {'<UNK>': 0, '<PAD>': 1}
        idx2word = {0: '<UNK>', 1: '<PAD>'}
        for idx, line in enumerate(fp, start=2):
            if num is not None and idx > num:
                break
            parts = line.strip().split()
            embeddings[idx,:] = [float(v) for v in parts[1:]]
            word_to_idx[parts[0]] = idx
            idx2word[idx] = parts[0]
    return embeddings, word_to_idx, idx2word

if __name__ == '__main__':
    embpath = sys.argv[1]
    if os.path.exists(embpath+'.npy'):
        sys.exit()
    embmatrix, w2idx, idx2w = read_embeddings(embpath)
    numpy.save(embpath, embmatrix)
    vocab = [idx2w[i] for i in range(len(idx2w))]
    save_vocabulary(embpath + '.vocab', vocab)
    print "saving numpy matrix and vocab of embedding in: ", embpath
