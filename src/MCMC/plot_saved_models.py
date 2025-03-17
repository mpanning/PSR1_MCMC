# Load up all saved models in SAVE directory and make plots
import numpy as np
import pickle
import os
import glob
import copy
import sys

from MCMC_functions import (MODEL,modfig,mintwo,vdispfig)

MAIN = os.getcwd()
if (len(sys.argv) == 1):
    raise RuntimeError('Please specify folder with stored models')
else:
    foldername = sys.argv[1]


SAVEMs = MAIN + '/' + foldername
mod_wildcard = '*.swm.pkl'

model_list = glob.glob(SAVEMs + '/' + mod_wildcard)

CHMODS = []

for file in model_list:
    with open(file, 'rb') as input:
        CHMODS.append(pickle.load(input))

nummods = len(CHMODS)
print('Read ' + str(nummods) + ' models')

# Loop over models and gather info
keptPHI = []
INTF = []
NM = []
SIGH = []
ii = 0
for model in CHMODS:
    curPHI = model.PHI
    keptPHI = np.append(keptPHI, curPHI)
    newcol = np.array(model.mantleR)
    INTF = np.append(INTF, newcol)
    newnm = model.nmantle
    NM = np.append(NM, newnm)
    newsighyp = model.sighyp
    if (ii == 0):
        SIGH = copy.deepcopy(newsighyp)
    else:
        SIGH = np.vstack((SIGH, newsighyp))
    ii = ii + 1

PHIind = np.argsort(keptPHI)
Ult_ind = PHIind[0]
revPHIind = PHIind[::-1]
# Specify colormap for plots
chosenmap='brg_r'
# vmin = 4.0
# vmax = 7.5
rep_cnt = 1
repeat = 1
maxz_m = 2891.0
planet_radius = CHMODS[0].radius
vrad = np.array([planet_radius-maxz_m, planet_radius])
vmin = np.array([4.0, 4.0])
vmax = np.array([7.5, 7.5])
abc = 'a'
run = 0
weight_opt = 'ON'
	
# Read in data
#Dispersion group arrival picks masterfile
masterfile = MAIN + '/gtimes.dat'
f = open(masterfile)
line = f.readline()
nevts = int(line.split()[0])
dispfiles = []
for i in range(0,nevts):
	dispfiles.append(f.readline().split()[0])
f.close()

# Read in group arrival time data into a list of arrays
gdisp = []
for i in range(0,nevts):
	gdisp.append(np.genfromtxt(dispfiles[i], 
				   dtype={'names': ('period', 'gtime'), 
					  'formats': ('f', 'f')}))

# make array of center frequencies (cf) and center periods (cp)
# as well as data
cp = []
cf = []
dobs_sw = []
cpemin = np.zeros(nevts)
cpemax = np.zeros(nevts)
fnum = np.zeros(nevts)
tmin_sw = np.zeros(nevts)
for i in range(0,nevts):
	cp.append(gdisp[i]['period'])
	cpemin[i] = cp[i].min()
	cpemax[i] = cp[i].max()
	cf.append(1/cp[i])
	fnum[i] = len(cf[i])
	dobs_sw.append(gdisp[i]['gtime'])
	tmin_sw[i] = dobs_sw[i].min()

cpmin = cpemin.min()
cpmax = cpemax.max()
#cf = vsts['freq']
#cp = 1/cf
#fnum = len(cf)
N = copy.deepcopy(fnum)

# Get body wave travel times
# Body wave picks masterfile
masterfile = MAIN + '/bwtimes.dat'
f = open(masterfile)
line = f.readline()
nevts_bw = int(line.split()[0])
if (not(nevts_bw == nevts)):
	print('Inconsistent events for body wave and surface waves')
	print(nevts,nevts_bw)
	raise ValueError("nevts do not match")
bwfiles = []
for i in range(0,nevts):
	bwfiles.append(f.readline().split()[0])
f.close()

# Read in group arrival time data into a list of arrays
bwtimes = []
phases = []
dobs_bw = []
tmin_bw = np.zeros(nevts)
tmin = np.zeros(nevts)
pnum = np.zeros(nevts)
for i in range(0,nevts):
	bwtimes.append(np.genfromtxt(bwfiles[i], 
				     dtype={'names': ('phase', 'time'), 
					    'formats': ('|S10', 'f')}))
	phases.append(bwtimes[i]['phase'])
	dobs_bw.append(bwtimes[i]['time'])
	tmin_bw[i] = dobs_bw[i].min()
	tmin[i] = mintwo(tmin_sw[i], tmin_bw[i])
	pnum[i] = len(dobs_bw[i])

# make single data array
dobs = np.concatenate([np.concatenate(dobs_sw), np.concatenate(dobs_bw)])
ndata = len(dobs)
ndsub = np.zeros(2, dtype=np.int64)
ndsub[0] = len(np.concatenate(dobs_sw))
ndsub[1] = len(np.concatenate(dobs_bw))
	
# data error estimates
gtime_relsig = 0.02
bwtime_sig = 0.5

wsig_sw = []
bsig_sw = []
for i in range(0,nevts):
	wsig_sw.append(np.zeros(int(fnum[i])))
	wsig_sw[i] = cp[i]*gtime_relsig #relative to center period
	bsig_sw.append(np.zeros(len(phases[i])))
	bsig_sw[i][:] = bwtime_sig

# Make single vector of wsig
wsig = np.concatenate([np.concatenate(wsig_sw), np.concatenate(bsig_sw)])
if (not(len(wsig) == ndata)):
	print('Problem with wsig')
	raise ValueError('Inconsistent ndata')

# ==================== PLOT [1] ==================== 
# ================ Velocity Models =================
CS3,scalarMap=modfig(rep_cnt,repeat,keptPHI,vrad,vmin,vmax,chosenmap,
                     nummods,revPHIind,CHMODS,Ult_ind,maxz_m,abc,run,
                     MAIN)


# ==================== PLOT [2] ==================== 
# ============== Group Times Vertical ==============
vdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,
         dobs_sw,cp,Ult_ind,weight_opt,wsig,cpmin,cpmax,vrad,vmin,vmax,CS3,
         maxz_m,abc,run,MAIN)
