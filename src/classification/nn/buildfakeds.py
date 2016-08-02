'''
Created on Nov 16, 2015

@author: yadollah
'''
import sys

postr1 = 'abccc'
negstr = 'bc'
# posstr_test = 'cccabda'
posstr_trn = 'dddab'
posstr_test = 'ab'

def writefake(filename, numline=1000):
    dsfile = open(filename, 'w')
    for i in range(numline):
        entid = 'ABC' + str(i)
        type = '-pos'
        freq = '10'
        el = '####'
        name = postr1
        if 'train' not in filename and i % 2 == 1:
            name = posstr_test
            entid = 'CAB' + str(i)
        if 'train' in filename and i % 2 == 1:
            name = posstr_trn
            entid = 'DAB' + str(i)
        namefreq = '1'
        line = '\t'.join([entid, type, type, freq, el, name, namefreq])
        dsfile.write(line + '\n')
        entid = 'ACB' + str(i)
        line = '\t'.join([entid, '-neg', '-neg', freq, el, negstr, namefreq])
        dsfile.write(line + '\n')
    dsfile.close()
    
def main(args):
    fakedir = 'fakeexp/'
    dev = fakedir + 'Edev.name.fake'
    train = fakedir + 'Etrain.name.fake'
    test = fakedir + 'Etest.name.fake'
    #/m/05phv    -chemistry    -chemistry    275    ####    Ozone    894    O3    466    oZone    10    OZONE    1    OZone    1
    writefake(train, 100)
    writefake(dev, 100)
    writefake(test, 1000)
        
        
if __name__ == '__main__':
    main(sys.argv)