'''
Created on May 11, 2015

@author: yadollah
'''
import sys
import multiprocessing
import os
from multiprocessing.process import Process
from multiprocessing.pool import Pool


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
def loadConfig(configFile):
    config = {}
# read config file
    f = open(configFile, 'r')
    for line in f:
        if line.strip() == '':
            continue
        if "#" == line[0]:
            continue  # skip commentars
        line = line.strip()
        parts = line.split('=')
        name = parts[0]
        value = parts[1]
        config[name] = value
    f.close()
    return config


def one_figer_runner(myconfig, alaki):
    figerscript = '/nfs/data1/proj/yadollah/nlptools_resources/figer/run.sh'
    os.system('nice -n 5 sh ' +  figerscript + ' ' + myconfig)
    

if __name__ == '__main__':
    srcdir = sys.argv[1]
    baseconfig = sys.argv[2]
    num_runs = int(sys.argv[3])
    mypool = MyPool(num_runs)
    processes = []
    dfconfig = loadConfig(baseconfig)
    for i in range(num_runs):
        mydir = srcdir +  str(i) + '/'
        newconfigfile = mydir + 'figer.config'
        segmentfile = mydir + 'lines.segment'
        testfile = mydir + 'lines.txt'
        outfile = mydir + 'lines.out'
        f = open(newconfigfile, 'w')
        for name in dfconfig:
            val = dfconfig[name]
            if name == 'inputSegments':
                f.write(name + '=' + segmentfile)
            elif name == 'testFile':
                f.write(name + '=' + testfile)
            elif name == 'outputFile':
                f.write(name + '=' + outfile)
            else:
                f.write(name + '=' + str(val))
            f.write('\n')
        f.close()
        processes.append(mypool.apply_async(one_figer_runner, args=(newconfigfile, 1)))
    for p in processes:
        p.get() 
    