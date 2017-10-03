from __future__ import print_function
import numpy as np
import healpy as hp
import subprocess
import os
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def compare_maps(component,i_check) :
    for nu in ['30p0','100p0','353p0'] :
        mp1=hp.read_map('test/benchmark/check%d'%i_check+component+'_'+nu+'_64.fits',field=[0,1,2],verbose=False)
        mp2=hp.read_map('test/Output/check%d'%i_check+component+'_'+nu+'_64.fits',field=[0,1,2],verbose=False)
        for i in [0,1,2] :
            norm=np.std(mp1[i])
            if norm<=0 : norm=1.
            diff=np.std((mp1[i]-mp2[i]))/norm
            if diff>1E-6 :
                return 0
    return 1

def run_check(i_check,component) :
    print("Running check %d..."%i_check)
    os.system('python main.py test/check%d_config.ini > log_checks 2>&1'%i_check)
    passed=compare_maps(component,i_check)
    if passed :
        print(bcolors.OKGREEN+"   PASSED"+bcolors.ENDC)
    else :
        print(bcolors.FAIL+"   FAILED"+bcolors.ENDC)
    subprocess.call(['rm', '-r', 'test/Output'])
    return passed

n_passed=0; n_total=0
for i_check,component in zip([1,2,3,4,5,6,7,8,9,10,11],['therm','synch','spinn','freef','cmb', 'therm', 'synch', 'spinn', 'therm', 'synch', 'therm']) :
    n_passed+=run_check(i_check,component)
    n_total+=1
print("%d tests passed "%n_passed+"out of %d"%n_total)
