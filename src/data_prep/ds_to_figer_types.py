'''
Created on Jun 11, 2015

@author: yadollah
'''
import sys
import os
import logging
from _collections import defaultdict
from src.common.myutils import load_dataset, write_ds

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('lines-gen')

highleveltypes = {'-location', '-building', '-person', '-event', '-product', '-organization', '-art'}

def load_parents(myfile):
    t2parents = {}
    f = open(myfile)
    for l in f:
        parts = l.split(' ')
        if len(parts) > 1:
            t2parents[parts[0]] = [parts[i].strip() for i in range(1,len(parts))]
    return t2parents


def laod_figermapping(fil):
    freebase2figer = {}
    f = open(fil)
    for l in f:
        parts = l.split('\t')
        freebase2figer[parts[0]] = parts[1].replace('/', '-').strip()
    return freebase2figer
        

def load_type_names(fil):
    mid2name = {}
    f = open(fil)
    for l in f:
        parts = l.split('\t')
        mid2name[parts[0]] = parts[2].strip()
    return mid2name

def parse_ent_tok(ent_token):
    myparts = ent_token.split('/')
    mid = '/m/' + myparts[2]
    name = myparts[3]
    nt = myparts[5]
    return (mid, name, nt)


def addhighleveltypes(inittypes, fig2parents):
    extended_types = inittypes
    for myt in inittypes:
        if myt in fig2parents:
            for parent in fig2parents[myt]:
                if parent not in inittypes:
                    extended_types.append(parent)
    assert len(extended_types) >= len(inittypes)
    return extended_types

def filter_write_ds(ds, fbname2figer, mid2name, outdir, fig2parents):
    f = open(outdir + 'Eds_figer', 'w')
    newDs = defaultdict(lambda: [])
    for mye in ds:
        types = ds[mye]
        converted_types = []
        nt = types[0]
            
        if nt in mid2name and mid2name[nt] in fbname2figer:
            converted_types.append(fbname2figer[mid2name[nt]])
            for t in types[1:]:
                if t in mid2name and mid2name[t] in fbname2figer:
                    if fbname2figer[mid2name[t]] != fbname2figer[mid2name[nt]]:
                        converted_types.append(fbname2figer[mid2name[t]])
#             f.write(mye + '\t' + converted_types[0] + '\t' + ' '.join(converted_types[1:]))
#             f.write('\n')
            ext_types = addhighleveltypes(converted_types, fig2parents)
            newDs[mye] = ext_types
    f.close() 
    return newDs


def filter_ds_lines(ds_lines, newConvertedDs, outdir):
    new_lines = []
    f = open(ds_lines)
    e2freq = defaultdict(lambda: 0)
    for l in f:
        parts = l.split('\t')
        if parts[1] not in newConvertedDs:
            continue
        newsent = ''
        emid = parts[1]
        e2freq[emid] += 1
        figernt = newConvertedDs[emid][0]
        for w in parts[3].split():
            if '/m/' in w:
                (mid, name, nt) = parse_ent_tok(w)
                if mid in newConvertedDs:
                    nt = newConvertedDs[mid][0]
                newsent += mid + '/' + name + '##' + nt + ' '
            else:
                newsent += w + ' '
        new_lines.append(parts[0] + '\t' + emid + '\t' + figernt + '\t' + ' '.join(newConvertedDs[emid][1:]) + '\t' + newsent)
    return (new_lines, e2freq) 
                
def write_lines(lines, e2freq, outdir):                
    f = open(outdir + '/ds_lines_figer', 'w')
    logger.info('write test lines in %s', outdir + 'ds_lines_figer')
    for myline in lines:
        f.write(myline.strip())
        f.write('\n')
    f.close()
        

if __name__ == '__main__':
    fbname2figer = laod_figermapping('/nfs/data3/yadollah/nlptools_resources/figer/config/types.map')
    fig2parents = load_parents('/nfs/data3/yadollah/nlptools_resources/figer/config/yy_type2parents')
    mid2name = load_type_names('/nfs/data1/proj/yadollah/cluewebwork/nlu/dataForImport/type.name')
    logger.info('size types with names: %d', len(mid2name))
    dsfile = sys.argv[1]
    ds_linesfile = sys.argv[2]
    outdir = sys.argv[3]
    
    (e2types, t2ents, e2freq) = load_dataset(dsfile, logger)
    logger.info(len(e2types))
    
    newConvertedDs = filter_write_ds(e2types, fbname2figer, mid2name, outdir, fig2parents)
    logger.info('size of dataset after filtering to figer type: %d', len(newConvertedDs))
    
    (new_dslines, e2freq) = filter_ds_lines(ds_linesfile, newConvertedDs, outdir)
    logger.info('#lines after filtering to figer type: %d', len(new_dslines))
    write_ds(newConvertedDs, e2freq, outdir + 'Eds_figer')
    write_lines(new_dslines, e2freq, outdir)
    
    
    
    