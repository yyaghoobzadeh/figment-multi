import string,collections, sys, os
from src.common.myutils import * 

donorm = False
config = loadConfig(sys.argv[1])
print 'loading cofing ',config
numtype = int (config['numtype'])
onlynt = config['onlynt']
Etestfile = config['Etest'] 
Edevfile = config['Edev'] 
matrixdev = config['matrixdev'] 
matrixtest = config['matrixtest']
typefilename = config['typefile']

upto = -1

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


(t2i,t2f) = fillt2i(typefilename)

e2i_dev = readdsfile(Edevfile, t2i)
(bigdev, numScorePerType,e2freq) = loadEnt2ScoresFile(matrixdev, upto, numtype, donorm)
    
# assert len(e2i_test) == len(bigtest[0])
typethresholdMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)] 
firstrun = True
for i in range(numtype):
    thelist = []
#     print 'calc theta for type: ', i
    for j in range(numScorePerType):
        thelist.append([])
        for mye in bigdev[i]:
            correct = 0
            if onlynt == 'False' and i in e2i_dev[mye]:
                correct = 1
            elif onlynt == 'True' and i == e2i_dev[mye]:
                correct = 1
            thelist[j].append((bigdev[i][mye][j], correct))
    ind = -1
    for sublist in thelist:
        ind += 1
        best = findbesttheta(sublist)
        typethresholdMatrix[i][ind] = best[1]



# Thresholds are found and put into typethresholdMatrix--- Now we should apply thresholds and calc performance
def calcPrintMeasures(myetest, findGoodEnts=False):
    ######
    e2predictedTypes = {}
    e2toptype = {}
    for mye in bigtest[0]:
        predTypes = []
        etypes = myetest[mye]
        e2tscores = []
        for tv in t2i:
            i = t2i[tv]
            if mye not in bigtest[i]:
                continue
            e2tscores.append((tv, bigtest[i][mye][0]))
            if bigtest[i][mye][0] > typethresholdMatrix[i][0]:
                predTypes.append(tv)
        e2tscores = sorted(e2tscores, key=lambda tuple: tuple[1], reverse=True)
        e2toptype[mye] = e2tscores[0]
        e2predictedTypes[mye] = predTypes
    
    outf = open(sys.argv[2] + 'ent2freq-pred-goldtypes-sbj.csv', 'w')
    for e in e2predictedTypes:
        strout = e + '\t' + str(etest2f[e]) + '\t' + str(etest[e]) + '\t' + e2toptype[e][0]+ '\t' + str(e2predictedTypes[e]) + '\t'
        correct = 0
        for goldt in etest[e]:
            if goldt in e2predictedTypes[e]:
                correct += 1
        if correct == len(etest[e]) and len(etest[e]) == len(e2predictedTypes[e]): strout += 'Strict\t'
        else: strout += '\t'
        if e2toptype[e][0] in etest[e]:
            strout += 'P@1'
        strout += '\t' + ff.format(e2toptype[e][1])
#         print strout
        outf.write(strout + '\n')
	if correct == 0:
		print strout
    outf.close()
             
e2i_test = readdsfile(Etestfile, t2i)
(bigtest, numScorePerTypetest,e2freq) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm)
etest = fillE2t(e2i_test, t2i)
freqtype = 'mention'
if 'rel' in freqtype:
    etest2f = filltest2relfreq(config['ent2relfreq'])
#     e2i_test_list = divideEtestByFreq(e2i_test, etest2f, EtestRelFreq)
else:    
    etest2f = filltest2freq(Etestfile)
#     e2i_test_list = divideEtestByFreq(e2i_test, etest2f, Etestfreq)

print '-------\nresult for All Etest entities'
print 'num of entities:', len(e2i_test)
calcPrintMeasures(e2i_test, findGoodEnts=True)

