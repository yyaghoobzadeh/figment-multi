# extract names of Etrain, Edev, Etest entities from FACC1 .tsv files and write them into Etrain.names Edev.names Etest.names 
from src.common.myutils import load_dataset
import sys
import random
from os.path import isfile, join, isdir
from os import listdir
from sets import Set
from _collections import defaultdict
import operator
from src.common.myutils import getentparts



def readFaccs(mypath):
    e2name2freq = defaultdict(dict)
    onlydirs = [ join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
    for mydir in onlydirs:
        print mydir
#         onlyfiles = [ join(mydir,f) for f in listdir(mydir) if isfile(join(mydir,f)) ]
        for f in listdir(mydir):
#             print f
            if isfile(join(mydir, f)):
                myfile = join(mydir, f)
                if '.hist' in myfile:
                    continue
                print 'loading entities from ', myfile
                reader = open(myfile)
                for line in reader:
                    myparts = line.split('\t')
                    name = myparts[2]
                    entmid = myparts[7].strip()
                    if entmid not in e2name2freq:
                        e2name2freq[entmid] = defaultdict(lambda: 0)
                    e2name2freq[entmid][name] += 1
                reader.close()
    print len(e2name2freq)
    return e2name2freq



def write_ds_names(dsfile, mye2name2freq, nameoutfile, max_name=10, fb_ent2name=None):
    (e2types, t2ents, e2freq) = load_dataset(dsfile)
    f = open(nameoutfile, 'w')
    for mye in e2types:
        outstr = '\t'.join([mye, e2types[mye][0], ' '.join(e2types[mye]), str(e2freq[mye])])
        if mye in mye2name2freq:
            name2freq = mye2name2freq[mye]
            sorted_by_freq = sorted(name2freq.items(), key=operator.itemgetter(1), reverse=True)
#             sorted_by_freq = sorted(name2freq, key=name2freq.get, reverse=True)
            c = 0
            outstr += '\t####\t'
            for name, freq in sorted_by_freq:
                outstr += '\t'.join([name, str(freq)])
                outstr += '\t'
                if c == max_name:
                    break
                c += 1
        elif fb_ent2name and mye in fb_ent2name:
            outstr += '\t####\t'
            for c in range(max_name):
                outstr += '\t'.join([name, '1000'])
                outstr += '\t'
        else:
            print mye
        f.write(outstr + '\n')
    f.close()
            
def fillUsingLines(linespath):
    e2name2freq = defaultdict(dict)
    f = open(linespath)
    for line in f:
        parts = line.split('\t')
        for w in parts[4].split():
            if '/m/' in w:
                (mid, tokens, notabletype) = getentparts(w)
                name = ' '.join(tokens)
                if mid not in e2name2freq:
                    e2name2freq[mid] = defaultdict(lambda: 0)
                e2name2freq[mid][name] += 1
    f.close()
    return e2name2freq 
    