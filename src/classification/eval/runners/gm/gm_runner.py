
## this is for configs with different number of negatives

import numpy as np
import math, string
from threading import Thread
from multiprocessing.process import Process
from multiprocessing.pool import Pool
import glob
import shutil
import os
import cmd
import multiprocessing
import sys
import src.common.myutils as cmn

def oneConfTrainer(myconfig, outdir, scriptfile):
#     train_nn_2h_file = '/nfs/data3/yadollah/cis_git/kbp/cnn/codeJan2015/cnn/entityonly/train_nn_entonly-adagrad-multilabel.py'
    print 'training a model using config', myconfig
    #os.system('python ' + train_nn_2h_file + ' ' + myconfig + ' > ' + outdir + 'train.log')
    os.system('python ' + scriptfile + ' ' + myconfig)
    
def runcmdfortest(k, cmd):
    os.system(cmd)        
            
def oneConfTester(dirName, confdir, scriptfile, edev, etest):
#     test_nn_2h_file = '/nfs/data3/yadollah/cis_git/kbp/cnn/codeJan2015/cnn/entityonly/test_nn_entonly-multilabel.py'
    mydir = confdir + '/' + str(dirName) + '/'
    myconfig = mydir + "config"
    print 'testing with config', myconfig 
    mypool = cmn.MyPool(2)
    processes = []
    cmdstr = 'python ' + scriptfile + ' ' + myconfig + ' ' + edev + ' '+ mydir + 'dev.probs > ' + mydir + 'devres.log'
    processes.append(mypool.apply_async(runcmdfortest, args=(1, cmdstr)))
    cmdstr = 'python ' + scriptfile + ' ' + myconfig + ' ' + etest + ' ' + mydir + 'test.probs > ' + mydir + 'testres.log'
    processes.append(mypool.apply_async(runcmdfortest, args=(1, cmdstr)))
    #dev.sampled.predType has context dependent type and target type
    for p in processes:
        p.get()
     
    
def oneMeasure(outtype, confdir):
    mydir = confdir + '/' + str(outtype) + '/'
    myconfig = mydir + "config"
#     big2smallfile = '/nfs/data3/yadollah/cis_git/kbp/cnn/codeJan2015/hs_threshold/14jan/big2smallmatrix.py'
#     big2smallcmd = 'python ' + big2smallfile + ' ' + myconfig
#     os.system(big2smallcmd)
    print 'measuring with config', myconfig
#     scriptFile = '/nfs/data3/yadollah/cis_git/kbp/cnn/codeJan2015/hs_threshold/14jan/matrix2NNLBmeasures-headtailall.py'
#     measurecmd = 'python ' + scriptFile + ' ' + myconfig + ' > ' + mydir + 'nnplbmeasures.txt'
#     os.system(measurecmd)
    scriptFile = '/mounts/Users/student/yadollah/new_ws/phdworks/src/classification/eval/matrix2measures-headtail-ents.py'
    measurecmd = 'python ' + scriptFile + ' ' + myconfig + ' > ' + mydir + 'thetaMeasures-ents.txt'
    os.system(measurecmd)
    scriptFile = '/mounts/Users/student/yadollah/new_ws/phdworks/src/classification/eval/matrix2measures-headtail-types.py'
    measurecmd = 'python ' + scriptFile + ' ' + myconfig + ' > ' + mydir + 'thetaMeasures-types.txt'
    os.system(measurecmd)
    

def parsecommand(arg):
    iftrain = True; iftest = True; ifmeasure = True
    if arg == '-measure':
        iftrain = False
        iftest = False
    elif arg == '-test':
        iftrain = False
    elif arg == '-train':
        iftest = False
        ifmeasure = False
        
    return (iftrain, iftest, ifmeasure)  
        

if __name__ == '__main__':
#     dfconfig=loadConfig('/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/cis1/cnn/config4nn')
    dfconfig = cmn.loadConfig(sys.argv[1])
    outputdir = sys.argv[2]
    (iftrain, iftest, ifmeasure) = parsecommand(sys.argv[3])
    outtypes = sys.argv[4].split(',')
    mypool = cmn.MyPool(2)
    processes = []
#     testexampledir = '/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov/figertypes/'
    testscript = dfconfig['test_script']
    trainscript = dfconfig['train_script']
    etest = dfconfig['Etest']
    edev = dfconfig['Edev']
    if iftrain:
        for outtype in outtypes:
#             dfconfig['hidden_units'] = 200 
            mydir = outputdir + '/' + outtype + "/"
            if not os.path.exists(mydir): os.makedirs(mydir)
            myconfig = mydir + "config"
            f = open(myconfig, 'w')
            for name in dfconfig:
                val = dfconfig[name]
                if name == 'matrixtest':
                    f.write(name + '=' + mydir + 'test.probs')
                elif name == 'matrixdev':
                    f.write(name + '=' + mydir + 'dev.probs')
                elif name == 'net':
                    f.write(name + '=' + mydir + 'model')
                else:
                    f.write(name + '=' + str(val))
                f.write('\n')
            f.close()
            processes.append(mypool.apply_async(oneConfTrainer, args=(myconfig, mydir, trainscript)))
        for p in processes:
            p.get()    
    
    processes = []            
    if iftest:
        for outtype in outtypes:
            processes.append(mypool.apply_async(oneConfTester, args=(outtype, outputdir, testscript, edev, etest)))
        for p in processes:
            p.get()    
    
    processes = []            
    if ifmeasure:
        for outtype in outtypes:
            processes.append(mypool.apply_async(oneMeasure, args=(outtype, outputdir)))
        for p in processes:
            p.get()    

