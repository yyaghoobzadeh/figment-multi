'''
Created on Nov 3, 2015

@author: yadollah
'''
from src.data_prep.fbclueweb.entname import *


def load_ent2name2freq(inpath):
    print "loading ent2name2freq from: ", inpath
    e2name2freq = defaultdict(dict)
    with open(inpath) as fp:
        for l in fp:
            parts = l.strip().split('\t')
            name2freq = {} 
            for i in range(1, len(parts[1:]) - 1, 2):
                name2freq[parts[i]] = int(parts[i + 1])
            e2name2freq[parts[0]] = name2freq
    return e2name2freq

def save_entitynames(mye2name2freq, outfile, max_name=10):
    with open(outfile, 'w') as fp:
        for mye in mye2name2freq:
            name2freq = mye2name2freq[mye]
            sorted_by_freq = sorted(name2freq.items(), key=operator.itemgetter(1), reverse=True)
            c = 0
            outstr = mye + '\t'
            for name, freq in sorted_by_freq:
                outstr += '\t'.join([name, str(freq)])
                outstr += '\t'
                if c == max_name:
                    break
                c += 1
            fp.write(outstr + '\n')
                
if __name__ == '__main__':
    ds_dir = sys.argv[1]
    Etrain = ds_dir + '/Etrain'
    Etest = ds_dir + '/Etest'
    Edev = ds_dir + '/Edev'
    faccpath = '/nfs/data1/proj/cluewebwork/nlu/FACC1/12/ClueWeb12_00/'
    linespath = '/nfs/data1/proj/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/NYT_expr/dsFromNYT/in_figer_types/test_lines'
    fromFACC = True
    if fromFACC:
        if True:
            mye2name2freq = readFaccs(faccpath)
            save_entitynames(mye2name2freq, 'ent2names.txt')
        else:
            mye2name2freq = load_ent2name2freq('ent2names.txt')
        write_ds_names(Etrain, mye2name2freq, ds_dir + 'Etrain.names') 
        write_ds_names(Edev, mye2name2freq, ds_dir + 'Edev.names') 
        write_ds_names(Etest, mye2name2freq, ds_dir + 'Etest.names') 
    
    else:
        mye2name2freq = fillUsingLines(linespath)
        write_ds_names(Etest, mye2name2freq, ds_dir + 'Etest.names') 
        

    