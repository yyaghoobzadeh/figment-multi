# for debugging: upto = 10000 will only look at the first 10000 lines

#finding thresholds from summarized scores for each entity2type
Etestfreq = [5, 10, 30, 100]
import string,collections, sys, numpy, os
from src.common.myutils import * 

# Thresholds are found and put into typethresholdMatrix--- Now we should apply thresholds and calc performance
def fillE2t(e2i_test, t2i):
    i2t = {}
    e2t = {}
    for t in t2i:
        i2t[t2i[t]] = t
    for e in e2i_test:
        types = []
        for ti in e2i_test[e]:
            types.append(i2t[ti])
        e2t[e] = types
    return e2t
    
def find_thresholds(numScorePerType, numtype, e2i_dev, smalldev, onlynt=False):
    typethresholdMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)] 
    firstrun = True
    for i in range(numtype):
        thelist = []
        for j in range(numScorePerType):
            thelist.append([])
            for mye in smalldev[i]:
                entsnum = 0
                if onlynt == False and i in e2i_dev[mye]:
                    entsnum = 1
                elif onlynt == True and i == e2i_dev[mye]:
                    entsnum = 1
                thelist[j].append((smalldev[i][mye][j], entsnum))
        ind = -1
        for sublist in thelist:
            ind += 1
            best = findbesttheta(sublist)
            typethresholdMatrix[i][ind] = best[1]
    return typethresholdMatrix

def apply_thre_test(typethresholdMatrix, smalltest, e2i_test, t2i, onlynt=False):
    uperThetaEnts = {}; typeEtestFreq = {}
    for tv in t2i:
        i = t2i[tv]
        print tv, typethresholdMatrix[i][0]
        thelist_test = []
        mylist = []
        entsnum = 0
        for mye in smalltest[i]:
            if onlynt == False and i in e2i_test[mye]:
                entsnum += 1
            elif onlynt == True and i == e2i_test[mye]:
                entsnum = 1
            if smalltest[i][mye][0] > typethresholdMatrix[i][0]:
               mylist.append((mye, smalltest[i][mye][0]))
        sortedlist = sorted(mylist, key=lambda tuple: tuple[1], reverse=True)
        uperThetaEnts[tv] = sortedlist
        typeEtestFreq[tv] = entsnum
    return uperThetaEnts, typeEtestFreq
    
def write_t2upents(t2i, type2uperThreshEnts, typeEtestFreq, numtype, etest2f, etest, outfile='type2uperThreshEnts', etest2names=None):
    print 'writing entities uppen thresholds for each type in: ', outfile
    confmatrix = [[0 for x in range(numtype)] for x in range(numtype)]
    f = open(outfile, 'w')
    for tv in type2uperThreshEnts:
        numgood = 0.0
        mylist = type2uperThreshEnts[tv]
        for (mye, score) in mylist:
            if etest2names != None:
                thestr = '\t'.join([tv, mye, etest2names[mye][0], str(ff.format(score)), str(etest2f[mye])]) + '\t'
            else:
                thestr = '\t'.join([tv, mye, str(ff.format(score)), str(etest2f[mye])]) + '\t'
            thestr += str(etest[mye]) + '\n'
            f.write(thestr)
            
            if tv in etest[mye]:
                numgood += 1
                confmatrix[t2i[tv]][t2i[tv]] += 1
            else:
                for goldt in etest[mye]:
                    confmatrix[t2i[goldt]][t2i[tv]] += 1
                
        if len(type2uperThreshEnts[tv]) == 0 or typeEtestFreq[tv] == 0: print tv + "\t" + str(len(type2uperThreshEnts[tv])) + "\t" + str(typeEtestFreq[tv]) + "\t0.00\t0.00\t0.00"
        else: 
            prec = (numgood / len(type2uperThreshEnts[tv]))
            rec = (numgood / typeEtestFreq[tv])
            if prec + rec == 0.0 : f1 = ff.format(0.00) 
            else: f1 = ff.format(2.0 / (1 / prec + 1 / rec))
            print tv + "\t" + str(len(type2uperThreshEnts[tv])) + "\t" + str(typeEtestFreq[tv]) + "\t" + ff.format(prec) + "\t" + ff.format(rec) + "\t" + f1  
    f.close()
    return confmatrix

def write_conf_matrix(confmatrix, numtype, outfile):
    fout = open(outfile, 'w')
    for i in range(numtype):
        numgold = sum(confmatrix[i]) + 0.00000001
        for j in range(numtype):
            fout.write(ff.format(confmatrix[i][j] / numgold) + '\t')
        fout.write('\n')
    fout.close()
    
def main(args):
    upto = -1
    config = loadConfig(args[1])
    outdir = args[2]
    if not os.path.exists(outdir): os.makedirs(outdir)
    donorm = str_to_bool(config['norm'])
    print '** norm is ', donorm
    print 'loading cofing ',config
    numtype = int(config['numtype'])
    onlynt = False if 'onlynt' not in config else str_to_bool(config['onlynt'])
    Etestfile = config['Etest'] 
    Edevfile = config['Edev'] 
    matrixdev = config['matrixdev']
    matrixtest = config['matrixtest']
    typefilename = config['typefile']
    useName = False
    if 'name' in Etestfile:
        useName = True
    print matrixdev, matrixtest
    etest2f = filltest2freq(Etestfile)
    (t2i,t2f) = fillt2i(typefilename)
    etest2names = None
    if useName:
        e2i_dev, edev2names,_ = load_entname_ds(Edevfile, t2i, use_ix=True)
        e2i_test, etest2names,_ = load_entname_ds(Etestfile, t2i, use_ix=True)
    else:
        e2i_dev = readdsfile(Edevfile, t2i)
        e2i_test = readdsfile(Etestfile, t2i)
    etest = fillE2t(e2i_test, t2i)

    (smalldev, numScorePerType, e2freq) = loadEnt2ScoresFile(matrixdev, upto, numtype, donorm)
    (smalltest, numScorePerTypetest, e2freq) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm)
    assert numScorePerTypetest == numScorePerType
    
    typethresholdMatrix = find_thresholds(numScorePerType, numtype, e2i_dev, smalldev)
    type2uperThreshEnts,typeEtestFreq = apply_thre_test(typethresholdMatrix, smalltest, e2i_test, t2i, onlynt=onlynt)

    confmatrix = write_t2upents(t2i, type2uperThreshEnts, typeEtestFreq, numtype, etest2f, etest, args[2]+'type2upents.csv', etest2names=etest2names)
    write_conf_matrix(confmatrix, numtype, 'confusionmatrix-relative')
    
if __name__ == '__main__':
    main(sys.argv)         


        


