import sys
import re
import os
import math

from src.common.myutils import has_ent, getentparts, getsentences



#import nltk
'''
Created on May 10, 2015
converts sample files from clueweb+facc to figer system files. 
One text file and one segment file is required. 
@author: yy1
'''

def convert_text2figer_format(sent, sent_counter, ent_mid):

    new_lined_formatted = ''
    ent_inds = []
    tokens = sent.strip().split(' ')
    token_counter = 0
    first_occr = True
    for token in tokens: 
        if has_ent(token, ent_mid) and first_occr == True:
            (mid, ent_tokens, notabletype) = getentparts(token)
            new_lined_formatted += ent_tokens[0] + '\tB-E\n'
            if len(ent_tokens) > 1:
                for i in range(1, len(ent_tokens)):
                    new_lined_formatted += ent_tokens[i] + '\tI-E\n'
            ent_ind = str(sent_counter) + '\t' + str(token_counter) + '\t' + mid + '\t' + str(ent_tokens) + '\t' + notabletype
            ent_inds.append(ent_ind)
            first_occr = False
        elif '/m/' in token:
            (mid, ent_tokens, notabletype) = getentparts(token)
            for t in ent_tokens:
                new_lined_formatted += t + '\tO\n'
        else:
            new_lined_formatted += token + '\tO\n'
        token_counter += 1
    return (new_lined_formatted, ent_inds)
            
def getrawsent(sent):
    new_lined_formatted = ''
    tokens = sent.strip().split(' ')
    token_counter = 0
    for token in tokens: 
        if '/m/' in token:
            (mid, ent_tokens, notabletype) = getentparts(token)
            for t in ent_tokens:
                new_lined_formatted += t + ' '
        else:
            new_lined_formatted += token + ' '
        token_counter += 1
    return (new_lined_formatted.strip())

#94      /m/03qsc/  /m/not /m/t1,/m/t2    We ve got to battle back sin with the /m/03qsc/Holy_Spirit##-god , actively engaging
def load_lines(mylineformat_file):
    sents = []
    all_ent_ins = []
    sent_counter = 0
    raw_sents = []
    f = open(mylineformat_file)
    for line in f:
        parts = line.split('\t')
        if len(parts) != 5:
            print line
        assert  len(parts) == 5
        ent_mid = parts[1].strip()#parseents(parts[1])
#         myent = parts[1]
#         myent_mid = myent.split('/')[2]
        text = parts[4]
        subsents = getsentences(text)
        for i, sent in enumerate(subsents):
            if has_ent(sent, ent_mid):
                if len(subsents) > 1: sent = sent.strip() + ' .'
                (formatted_sent, ent_inds) = convert_text2figer_format(sent, sent_counter, ent_mid)
                sents.append(formatted_sent + '\n')
                all_ent_ins.append(ent_inds)
                raw_sents.append(getrawsent(sent))
                sent_counter += 1
                break # only the first sentence of the line -- 
    return (sents, all_ent_ins, raw_sents)
 

def getsubfiles_start_end(total_len, n, each_len):
    strt = n * each_len
    end = strt + each_len
    if strt + each_len > total_len:
        end = total_len
    return (strt, end)


def write_files(sents, sents_inds, raw_sents, outdir, numsplit=30):
    if numsplit > 1:
        assert len(sents) == len(sents_inds) == len(raw_sents)
        small_line_numbers = int(math.ceil(len(sents) / float(numsplit)))
        for i in range(numsplit):  
            mydir = outdir + str(i) + "/"
            if not os.path.exists(mydir): os.makedirs(mydir)
            (s_ind, end_ind) = getsubfiles_start_end(len(sents), i, small_line_numbers)
            sentw = open(mydir + 'lines.segment', 'w')     
            for sent in sents[s_ind: end_ind]:
                sentw.write(sent)
            sentw.close()
            entw = open(mydir + 'lines_entindex', 'w')
            for sent_inds in sents_inds[s_ind: end_ind]:
                for ind in sent_inds:
                    entw.write(ind + '\n')
            entw.close()
            sentw = open(mydir + 'lines.txt', 'w')     
            for sent in raw_sents[s_ind:end_ind]:
                sentw.write(sent + '\n')
            sentw.close()
        return
    
    sentw = open(outdir + 'lines.segment', 'w')     
    for sent in sents:
        sentw.write(sent)
    sentw.close()
    entw = open(outdir + 'lines_entindex', 'w')
    for sent_inds in sents_inds:
        for ind in sent_inds:
            entw.write(ind + '\n')
    entw.close()
    sentw = open(outdir + 'lines.txt', 'w')     
    for sent in raw_sents:
        sentw.write(sent + '\n')
    sentw.close()
     

        
if __name__ == '__main__':
    mylineformat_file = sys.argv[1]#'small_test_line'
    outdir = sys.argv[2]
    num_split = int(sys.argv[3])
    (sents, all_ent_ins, raw_sents) = load_lines(mylineformat_file)
    write_files(sents, all_ent_ins, raw_sents, outdir, num_split)
    
