# for debugging: upto = 10000 will only look at the first 10000 lines

#finding thresholds from summarized scores for each entity2type
Etestfreq = [5, 10, 30, 100]
from src.common.myutils import * 
import string,collections, sys

donorm = False
print '** norm is ', donorm
config = loadConfig(sys.argv[1])#/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/experiments/nonsbjfiger/exp2-0.0/configFIGER_2_nonsbj.txt")
print 'loading cofing ',config
numtype = int (config['numtype'])
onlynt = config['onlynt']
Etestfile = config['Etest'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Etest
Edevfile = config['Edev'] #'/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Edev'

matrixdev = config['matrixdev'] #'enttypescores_dev' + str(numtype)
matrixtest = config['matrixtest']#'enttypescores_test' + str(numtype)
typefilename = config['typefile'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/experiments/807types/cis/context-datasets/rndTypes_trndevcontextFreq'

upto = 10000
topk = 50
def precisionAt(unsortedlist, topnum=20):
    mylist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1] #mypair[1] is one for labeled entity and type
    if total < topnum:
        topnum = total
    good = 0.0; bad = 0.0; fn = 0.0; tn = 0.0

    for i in range(0, topnum):
        if mylist[i][1] == 1:
            good += 1
    return good / topnum

etest2f = filltest2freq(Etestfile)

print matrixdev, matrixtest
(t2i,t2f) = fillt2i(typefilename, numtype)
e2i_test = readdsfile(Etestfile, t2i)
(bigtest, numScorePerType) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm)

overalPrec = [0.0 for x in range(numScorePerType)]
prec = 0.0 
for i in range(numtype):
    print 'type: ', i, '-----------'
    thelist_test = []
    for j in range(numScorePerType):
        thelist_test.append([])
        for mye in bigtest[i]:
            correct = 0
            if onlynt == 'False' and i in e2i_test[mye]:
                correct = 1
            elif onlynt == 'True' and i == e2i_test[mye]:
                correct = 1
            if numScorePerType == 1:
                thelist_test[j].append((bigtest[i][mye], correct))    
            else:
                thelist_test[j].append((bigtest[i][mye][j], correct))
    ind = -1
    for sublist in thelist_test:
        ind += 1
        precAtk = precisionAt(sublist, topk)
        print precAtk
        overalPrec[ind] += precAtk

print 'Average Prec@ ', topk, ':'
for i in range(len(overalPrec)):
    print i, " ", ff.format(overalPrec[i] / numtype)
         
