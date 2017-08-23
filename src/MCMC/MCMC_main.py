#!/opt/local/bin/python2.7

# This program is a TRANS-DIMENSIONAL version of MCMC_VS_interface.py in which the 
# number of model layers is an unknown parameter in the MCMC algorithm. 
#
# This program uses Markov Chain Monte Carlo (MCMC) algorithm to explore the model
# space and find the family of models that have minimum misfit between observed 
# dispersion values and calculated dispersion values
#
# The number of model layers can be changed
#
# NETWORK AVERAGE
#
# NEXT STEP: READ IN P AND S TRAVEL TIME DATA AND MERGE DATA INTO SINGLE DOBS AND DPRE ARRAY
# -----------------------------------------------------------------------------

from numpy import inf, log, cos, array
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cmx
from matplotlib import rc
import math
import os
import operator
import datetime
import sys
import copy
import subprocess
import random
from random import randint
import shutil
import pylab as P
import string
from pprint import pprint
from scipy.interpolate import interp1d
import cPickle as pickle

from MCMC_functions import (startmodel,MODEL,startchain,runmodel,finderror,
			    randINTF,accept_reject,mintwo,runmodel_bw,errorfig,
			    accratefig,nlhist,sighhist,modfig,vdispfig)

# ----------------------------------------------------------------------------
# ****************************************************************************
# --------------------------Set up INPUT information--------------------------
# directory where working
MAIN = '/home/mpanning/MCMC/test_run'
os.chdir(MAIN)
# MAIN = os.getcwd()

now = datetime.datetime.now()
foldername = now.strftime("%m_%d_%Y,%H:%M")
os.mkdir(foldername)

SAVEMs = MAIN + '/' + foldername

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
	print 'Inconsistent events for body wave and surface waves'
	print nevts,nevts_bw
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
ndsub = np.zeros(2, dtype=np.int)
ndsub[0] = len(np.concatenate(dobs_sw))
ndsub[1] = len(np.concatenate(dobs_bw))

# create boss matrix to control all combinations of starting number of layers 
#layopt = ([2, 3, 4, 5])
layopt = ([3])
botopt = ([2891.0]) #core-mantle boundary depth
#botopt = ([0.2, 0.4, 0.6])
repeat = 1
all_letters = list(string.ascii_lowercase)
letters = all_letters[0:repeat]
abc=[]

DRAW = ['CHANGE EPICENTRAL DISTANCE: Perturb an epicentral distance <->',
	'CHANGE ORIGIN TIME: Perturb an origin time of event <+>',
	'CHANGE VELOCITY: Perturb the velocity of a layer -->',
	'MOVE: Perturb an interface depth ^^^',
       	'BIRTH: Create a new layer ++', 
        'DEATH: Remove a layer X', 
       	'CHANGE HYPER-PARAMETER: Perturb the estimation of data error ~~~']
# ---------------------------------------------------------------------------------------
# ----------------
# totch = 10			# Total number of chains
# numm = 1000
totch = 1
numm = 10000			# Number of iterations per chain
# ----------------
# ---------------------------------------------------------------------------------------
# ----------------
BURN = 2000
# BURN = 2000		# Number of models designated BURN-IN, gets discarded
# M = 10			# Interval to keep models (e.g. keep every 100th model, M=100)
M = 10
MMM = np.arange(BURN-1,numm,M)
# ----------------
# ---------------------------------------------------------------------------------------
#########Option to weight data points######### 
######### by variance or stand. dev. ######### 
weight_opt = 'ON'
#weight_opt = 'OFF'
# --------------------------------------------
########## Options for pdf figure  ########### 
########## 'on' to count interfaces ##########
##########   (connect the layers)   ##########
###########'off' to only count the ###########
########layers and not the jumps between######
#pdf_connect = 'on'
pdf_connect = 'off'
# --------------------------------------------
########## Options for making half- ########### 
######### space velocity a parameter ##########
#vhs_opt = 'on'
#vhs_opt = 'off'
# --------------------------------------------
########## Options for controlling ############
######## minimum thickness of layers ##########
#THICK_FLAG = "ON"
#THICK_FLAG = "OFF"
# --------------------------------------------
########## Options for weighting ############
##########   sigd   or   sigd_n  ############
#if weight_opt == 'ON':
#	wsig = sigd
#	weight_sig = "sigd"
	#wsig = sigd_n
	#weight_sig = "sigd_n"
#else:
#	wsig = np.zeros(fnum)
######### Option for number of layers ########
#########   to output in surface wave file ###
swm_nlay = 300

######### Options for sw code rayleigh #######
######### Tune for desired frequency band ####
eps = 1.0e-9
dt = 10.0
npow = 7
fnyquist = 0
nbran = 0
cmin = 2.5
cmax = 250.0
maxlyr = 1
######### Option for setting model ###########
######### planet radius (Earth or Mars) ######
planet_radius = 6371.0 #Earth
#planet_radius = 3389.5 #Mars
######### Option for initial guess of ########
######### relative surface group time error ##
######### and body wave absolute error #######
gtime_relsig = 0.02
bwtime_sig = 0.5
######### Option for assumed source depth ####
source_depth = 10.0

wsig_sw = []
bsig_sw = []
for i in range(0,nevts):
	wsig_sw.append(np.zeros(fnum[i]))
	wsig_sw[i] = cp[i]*gtime_relsig #relative to center period
	bsig_sw.append(np.zeros(len(phases[i])))
	bsig_sw[i][:] = bwtime_sig

# Make single vector of wsig
wsig = np.concatenate([np.concatenate(wsig_sw), np.concatenate(bsig_sw)])
if (not(len(wsig) == ndata)):
	print 'Problem with wsig'
	raise ValueError('Inconsistent ndata')

# --------------------------------------------
loptnum = len(layopt)
boptnum = len(botopt)
numrun = loptnum*boptnum*repeat

boss = np.zeros((numrun, 2))
k=0
for i in layopt:
	for j in botopt:
		print 'starting layer option: ' + str(i)
		print 'bottom option: ' + str(j)
		r = 0
		while r < repeat:
			abc.append(letters[r])
			r = r + 1
		boss[k:k+repeat,0]=i
		boss[k:k+repeat,1]=j
		k=k+repeat

rep_cnt = 0
run = 0
nlay_init = int(boss[run,0])
#bounds on number of layers in mantle
nmin = 1
nmax = 10
deln = nmax - nmin
maxz = boss[run,1]
maxz_m = 1000*(maxz)
#crust layer plus nmantle + 1 mantle layers
nl_init = nlay_init + 2 
#need one extra velocity to specify linear gradients
nv_init = nl_init + 1 
nintf_init= nl_init - 1
hmin = 10.0      # minimum layer thickness of 10 km
#What is N?
#N = np.linspace(hmin, (maxz-hmin), ((maxz*500)+1))
reprunsPHI = []
reprunsdiagCE = []
reprunsHIST = []

# Standard deviations for the Gaussian proposal distributions
thetaI = 10.0	# Interface perturbations, qc(c'i|ci)
thetaV1 = 0.15		# Velocity perturbations, qv1(v'i|vi)
thetaV2 = 0.05		# New layer velocity, qv2(v'n+1|vi)
thetaHYP = 0.1		# Hyper parameter, qh(h'j|hj)
thetaEPI = 40.0    # Epicentral distance (km)
thetaOT = 10.0      # Origin time (s)
# ---------------------------------------------------------------------------------------

savefname = 'saved_initial_'+str(nlay_init)+'_lay_'+str(maxz)+'_m'
SAVEF = SAVEMs + '/' + savefname
os.mkdir(SAVEF)

bossfile = open(SAVEMs+'/boss.txt', 'w')
bossfile.write(now.strftime("%m_%d_%Y, %H:%M")+'\n')
bossfile.write('   ' + '\n')
bossfile.write('# OF INITIAL LAYERS       MAX DEPTH (m)'+'\n')
k=0
while (k < numrun):
	maxdepth = boss[k,1]
	writestring = ('         ' + str(boss[k,0]) + '                ' 
		       + str(boss[k,1])+'\n')
	bossfile.write(writestring)
	k = k + 1
bossfile.write('   ' + '\n')
bossfile.write('TOTAL # OF CHAINS: '+str(totch)+'    ITERATIONS: '+str(numm)+
	       '\n')
bossfile.write('   ' + '\n')
bossfile.write('SURFACE WAVE CODE PARAMETERS:\n')
bossfile.write('{:e} {:f} {:d} {:f} {:d} {:f} {:f} {:d}\n'.format(eps,dt,npow,
								  fnyquist,
								  nbran,cmin,
								  cmax,maxlyr))
bossfile.write('   ' + '\n')
bossfile.write('PLANET RADIUS: '+str(planet_radius)+'\n')
bossfile.write('   ' + '\n')
bossfile.write('RELATIVE GTIME ERROR: '+str(gtime_relsig)+'\n')
bossfile.write('   ' + '\n')
bossfile.write('MINIMUM BODY WAVE TIME ERROR: '+str(bwtime_sig)+'\n')
bossfile.write('   ' + '\n')
bossfile.write('ASSUMED SOURCE DEPTH: '+str(source_depth)+' km\n')
bossfile.close()

Elog = open(SAVEMs+'/'+'errorlog.txt','w')

#### Select the colormaps desired for the output pdf figures
#### options are GREY, GREY_rev, HOT, HOT_rev
pdfcmap=('GREYS_rev','HOT')

RUNMODS = []
runPHI = []
rnummods = len(RUNMODS)		
runINTF=[]	
runNM=[]	
runSIGH=[]

while (run < numrun):
	
	# -------------------------------------------------------------------
	# --------------------------Set up MODEL information-----------------
	boss0 = boss[run,0]
	boss1 = boss[run,1]

	# nmantle = number of layer interfaces in mantle
	nmantle_init = int(boss0)
	
	# maxz = maximum depth for full model
	maxz = boss1
	#maxz_m=maxz*1000
	
	# number of perturbable velocities (crust + nmantle + 2 to define all 
	# linear gradients in mantle)
	nv_init = nmantle_init + 3
	
	# the number of interfaces will be the number of mantle
	# interfaces + 1 (the crust-mantle interface). 
	nintf_init= nmantle_init + 1
	
	# Crust bounds on velocity
	cvmin = 2.0
	cvmax = 5.0
	cavgV = (cvmax + cvmin)/2.0

	# Crust thickness bounds
	chmin = 5.0
	chmax = 70.0

	# Expected group velocity bounds used for epimin/max calculation
	gvmin = 2.5
	gvmax = 5.5

	# Set epicentral bounds
	otmin = np.zeros(nevts)
	otmax = np.zeros(nevts)
	epimin = np.zeros(nevts)
	epimax = np.zeros(nevts)
	for i in range(0,nevts):
		# Check if we can get an S-P time and scale ot bounds to that
		if ((len(phases[i][phases[i] == 'P']) > 0) and 
		    (len(phases[i][phases[i] == 'S']) > 0)):
			TSP = (dobs_bw[i][phases[i] == 'S'][0] -
			       dobs_bw[i][phases[i] == 'P'][0])
			otmin[i] = dobs_bw[i][phases[i] == 'P'][0] - TSP/0.6
			otmax[i] = dobs_bw[i][phases[i] == 'P'][0] - TSP
			epimin[i] = gvmin * (tmin_sw[i] - otmax[i])
			epimax[i] = gvmax * (tmin_sw[i] - otmin[i])
		else:
			# Epicentral distance bounds (0-180 degrees)
			epimin[i] = 0.0
			epimax[i] = math.pi*planet_radius
			otmax[i] = tmin[i]
			otmin[i] = tmin[i] - (epimax[i]/gvmin) 

	if rep_cnt == repeat:
		rep_cnt = 0
		RUNMODS = []
		reprunsPHI = []
		reprunsHIST = []
		BEST_RUNMODS = []
		BESTreprunPHI = []
		BESTreprunNL = []
		BESTrerunsSIGH = []
		savefname = 'saved_initial_'+str(nl_init)+'_lay'
		SAVEF = SAVEMs + '/' + savefname
		os.mkdir(SAVEF)
		
	# Bounds on mantle velocities:
	vmin = 4.0
	vmax = 7.5
	delv = vmax - vmin
	
	# Bounds on hyper-parameters 
	hypswmin = 0.0
	hypswmax = 20.0
	hypbwmin = 0.0
	hypbwmax = 10.0
	hypmin = np.array([hypswmin, hypbwmin])
	hypmax = np.array([hypswmax, hypbwmax])
	
	# Vp/Vs ratio:
	vpvs = math.sqrt(3.0)

	# rho/Vs ratio:
	rhovs = 0.75

	CHMODS = []
	BEST_CHMODS = []

	# Set up totch chains:------------------------	
	stz,stvel,stepi,stotime,sthyp = startmodel(hmin,nintf_init,totch,
						   nv_init,maxz,vmin,vmax,
						   cvmin,cvmax,cavgV,chmin,
						   chmax,epimin,epimax,otmin,
						   otmax,hypmin,hypmax,nevts)
	
	errorflag1='off'
	#errorflag2='off'
	
	acc_rate = np.zeros(totch)
	draw_acc_rate = np.zeros((len(DRAW),totch))
	
	"""      ------  Loop through multiple chains:  -------------     """
	chain=0
	while (chain < totch):
		
		# Create new MODEL instance
		x = MODEL()
		x.nmantle = nmantle_init
		x.number = 0
		x.radius = planet_radius #Earth model
		x.PS_scale = vpvs
		x.RS_scale = rhovs
		x.nevts = nevts
		x.hypDepth = np.zeros(nevts)
		x.hypDepth[:] = source_depth
		# Establish interface depths and velocity as random starting 
		# model for the current chain:
		# VS[rows,columns] rows = layers, columns = model iteration
		# INTF[rows,columns rows = interface depth, columns = model 
		# iteration
		x,errorflag1=startchain(x,errorflag1,stz,stvel,stepi,stotime,
					sthyp,vmin,vmax,cvmin,cvmax,cavgV,
					chmin,chmax,epimin,epimax,otmin,otmax,
					hypmin,hypmax,hmin,maxz,chain)		
		
		
		# Create swm and nd input files
		try:
			x.create_swm_file(swm_nlay, create_nd_file=True)
		except ValueError:
			print ('WARNING: failed to create swm file on ' +
			       'initial model')
			print 'Restart chain'
			errorflag1 = 'on'
			continue
			

		# Run sw model
		(modearray,nmodes)=runmodel(x,eps,npow,dt,fnyquist,nbran,cmin,
					    cmax,maxlyr)

		# Confirm that rayleigh run covers desired frequency band
		# print modearray[2,:nmodes]
		parray = 1./modearray[2,:nmodes]
		gvarray = modearray[4,:nmodes]
		pmin = parray.min()
		pmax = parray.max()
		if (pmin > cpmin or pmax < cpmax):
			print pmin, cpmin, pmax, cpmax
			print 'Initial model did not calculate correctly'
			print 'Restart chain'
			errorflag1 = 'on'
			continue
		
		# Interpolate predicted gv to cp of data
		fgv = interp1d(parray, gvarray)
		gv_pre = []
		tnum = np.zeros(nevts)
		dpre_sw = []

		for i in range(0,nevts):
			gv_pre.append(fgv(cp[i]))
			tnum[i] = len(dobs_sw[i])
			dpre_sw.append(np.zeros(tnum[i]))
			dpre_sw[i][:] =  (x.epiTime[i] + 
					  (x.epiDistkm[i]/gv_pre[i]))	

		# Run bw model
		try:
			dpre_bw = runmodel_bw(x, phases) 
		except UserWarning:
			print 'Body wave phases not found in initial model'
			print 'Restart chain'
			errorflag1 = 'on'
			continue
		except AttributeError as e:
			print 'No taup model file was set'
			raise e
		# except:
		#	print 'taup threw an error'
		#	print 'Restart chain'
		#	errorflag1 = 'on'
		#	continue

		# Merge into single array
		dpre = np.zeros((ndata,numm))
		dpre[:,0] = np.concatenate([np.concatenate(dpre_sw), 
				       np.concatenate(dpre_bw)])
		x.dpre = dpre[:,0]

		# CALCULATE MISFIT BETWEEN D_PRE AND D_OBS:
		# CONTINUE BODY WAVE MOD FROM HERE
		misfit = np.zeros(ndata)
		newmis = np.zeros(ndata)
		diagCE = np.zeros((ndata,numm))
		PHI = np.zeros(numm)
		misfit,newmis,PHI,x,diagCE = finderror((-1),x,ndsub,dpre,dobs,
						       misfit,newmis,wsig,PHI,
						       diagCE,weight_opt)
								   	   
		ITMODS = []
		ITMODS.append(x)

		numreject = 0.0
		numaccept = 0.0
		drawnumreject = np.zeros(len(DRAW))
		drawnumaccept = np.zeros(len(DRAW))

		keep_cnt = 0

		# =============================================================
		k = 0
		while (k < (numm-1)):
				
			print "================================================"
			print ("                   CHAIN # [" + str(chain)+
			       "]    ITERATION # ["+str(k)+"]" )
			print " "
					
			# Set previous model object as "oldx" so can call on 
			# it's attributes when needed
			oldx = copy.deepcopy(ITMODS[k])
					
			# Copy old model to new model and update number
			# Perturbation steps then only need to change
			# elements that are perturbed
			x = copy.deepcopy(oldx)
			x.number = k+1
					
			curNM = copy.deepcopy(oldx.nmantle)
			WARN_BOUNDS = 'OFF'
				
			# save original velocities
			vsIN = np.zeros(curNM+3)
			vsIN[0] = copy.deepcopy(oldx.crustVs)
			vsIN[1:] = copy.deepcopy(oldx.mantleVs)

		        ########### Draw a new model ########################
			# Choose random integer between 0 and 6 such that each 
			# of the 7 options
			# (Change epidist, Change otime, Change Velocity, Move,
			# Birth, Death, Change Hyper-parameter) have
			#  a 1/7 chance of being selected
		
			# pDRAW = randint(0,6)
			# Change odds so that 50% change of changing epicentral
			# parameters
			epichange = randint(0,1)
			if epichange == 0:
				pDRAW = randint(0,1)
			else:
				pDRAW = randint(2,6)
					
			# Change epicentral distance
			if pDRAW == 0:
				
				print DRAW[pDRAW]

				wEPI = np.random.normal(0,thetaEPI)
				ievt = randint(0,nevts-1)

				newdist = x.epiDistkm[ievt] + wEPI
				print ('Perturb epicentral distance of evt[' +
				       str(ievt) + '] by ' + 
				       str(wEPI) + ' km')

				if ((newdist < epimin[ievt]) or 
				    (newdist > epimax[ievt])):
					print ("!!! Outside epicentral " +
					       "distance range")
					print "Automatically REJECT model"
					WARN_BOUNDS = 'ON'
				else:
					x.epiDistkm[ievt] = newdist
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			# Change origin time
			if pDRAW == 1:
				
				print DRAW[pDRAW]

				wOT = np.random.normal(0,thetaOT)
				ievt = randint(0,nevts-1)
				
				print ('Perturb origin time of evt[' + 
				       str(ievt) + '] by ' + str(wOT)
				       + 's')

				newtime = x.epiTime[ievt] + wOT

				if ((newtime < otmin[ievt]) or 
				    (newtime > otmax[ievt])):
					print ("!!! Outside origin time " +
					       "range\n")
					print "Automatically REJECT model"
					WARN_BOUNDS = 'ON'
				else:
					x.epiTime[ievt] = newtime
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			# Change velocity of a layer (vi)
			if pDRAW == 2:
						
				print DRAW[pDRAW]
						
				# Randomly select a perturbable velocity
				pV = randint(0,curNM+2)
						
				# Generate random perturbation value
				wV1 = np.random.normal(0,thetaV1)
						
						
		
				# initialize all velocities as the same as 
				# previously
				vsOUT = copy.deepcopy(vsIN)

				# target layer being perturbed and add in 
				# random wV
				vsOUT[pV] = vsOUT[pV] + wV1
	
				if pV == 0:
					print ('Perturb crust VS[' + str(pV) +
					       ']\n')
					print 'wV1 = ' + str(wV1)
							
					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < cvmin) or 
					    (vsOUT[pV] > cvmax) or
					    (vsOUT[pV] > vsOUT[pV+1])):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				elif pV == curNM+2:
					print ('Perturb base VS[' + str(pV) +
					       ']\n')
					print 'wV1 = ' + str(wV1) + '\n'

					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < vsOUT[pV-1]) or 
					    (vsOUT[pV] > vmax)):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				else:
					print 'Perturb VS[' + str(pV) +']'
					print 'wV1 = ' + str(wV1) 
		
					# if the new value is outside of 
					# velocity range or creates a negative
					# velocity gradient - REJECT
					if ((vsOUT[pV] < vsOUT[pV-1]) or 
					    (vsOUT[pV] > vsOUT[pV+1]) or
					    (vsOUT[pV] < vmin) or 
					    (vsOUT[pV] > vmax)):
						print ('!!! Outside velocity ' +
						       'range allowed!!')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
		
				BDi = 0
				delv2 = 0
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
									
			# Change an interface depth	
			elif pDRAW == 3:
						
				# MOVE an interface!!!
				nintf = curNM+1
				print DRAW[pDRAW]
						
				# initialize all velocities and interfaces the 
				# same as previously
				vsOUT = copy.deepcopy(vsIN)
				intfIN = np.zeros(curNM+1)
				intfIN = oldx.radius - oldx.mantleR
				intfOUT = copy.deepcopy(intfIN)
						
				# Choose an interface at random to perturb
				pI = randint(0,(nintf-1))
				print 'Perturbing INTF['+str(pI)+']'
						
				# Generate random perturbation value
				wI = np.random.normal(0,thetaI)
				print 'wI = ' + str(wI)
									
				# select the interface being perturbed and add 
				# in random wI and then resort
				intfOUT[pI] = intfIN[pI] + wI
				# sorting avoids interfaces overtaking each 
				# other
				tmpint = np.array(sorted(intfOUT))
				intfOUT = tmpint

				# Check if crustal thickness has changed
				if not (intfOUT[0] == oldx.crustThick):
					x.crustThick = intfOUT[0]
					if ((x.crustThick < chmin) or 
					    (x.crustThick > chmax)):
						print ('!!! Crustal thickness '
						       + 'outside of depth ' + 
						       'bounds')
						print ('Automatically REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'

				# if the new value is outside of depth bounds 
				# - REJECT
				if ((intfOUT[pI] < x.crustThick) or 
				    (intfOUT[pI] > maxz)):
					print ('!!! Outside of depths ' +
					       'allowed!!!')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
	

				# if any layers are now thinner than hmin - 
				# REJECT
				for i in range(1,curNM+1):
					if ((intfOUT[i] - intfOUT[i-1]) < hmin):
						print '!!! Layer too thin!!!\n'
						print ('Automatialy REJECT ' +
						       'model')
						WARN_BOUNDS = 'ON'
				if (x.radius - x.cmbR - intfOUT[curNM]) < hmin:
					print '!!! Layer too thin!!!'
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'

				x.crustThick = intfOUT[0]
				x.mantleR = x.radius - intfOUT
				BDi = 0
				delv2 = 0

			# Create a new layer
			elif pDRAW == 4:
						
				# ADD a layer in!!!
				print DRAW[pDRAW]
						
				# initialize all velocities as the same as 
				# previously
				vsOUT = copy.deepcopy(vsIN)
				   						
				newNM = curNM + 1
				x.nmantle = newNM
						
				if newNM > nmax:
					print ('!!! exceeded the maximum ' +
					       'number of layers allowed!!!\n')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
						
				# Get the current model's interface depths
				intfIN = np.zeros(curNM+2)
				intfIN = oldx.radius - oldx.mantleR
				#F = np.zeros(curNL-1)
				#F[:] = copy.deepcopy(oldx.intf[:])
						
				# Generate an interface at a random depth, but 
				# have check measures in place so that the 
				# interface cannot be within hmin of any of 
				# the existing interfaces
				(addI,intfOUT,vsOUT,WARN_BOUNDS,
				 BDi,delv2) = randINTF(vmin,vmax,chmin,hmin,
						       maxz,intfIN,vsIN,thetaV2)
			        
				x.mantleR = x.radius - np.array(intfOUT)    
				x.crustThick = intfOUT[0]
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
				
			# Delete a layer
			elif pDRAW == 5:
						
				# REMOVE a layer!!!
				print DRAW[pDRAW]
						
				newNM = curNM - 1
				x.nmantle = newNM
						
				if newNM < nmin:
					print ('!!! dropped below the ' + 
					       'minimum number of layers ' +
					       'allowed!!!')
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'

				# Get the current model's interface depths
				intfIN = np.zeros(curNM+2)
				intfIN = oldx.radius - oldx.mantleR
						
				# Randomly select an interface to remove
				# except crust or cmb
				kill_I = randint(1,(curNM))
				BDi = kill_I
				intfOUT = np.delete(intfIN, kill_I)
				print "removing interface["+str(kill_I)+"]"
				delv2 = vsIN[kill_I+1]-vsIN[kill_I-1]
						
				# Transfer over velocity values.  We remove the
				# velocity associated with the removed 
				# interface, meaning the new velocity profile
				# simply linearly interpolates over the region
				# of the removed interface
				VSlen = len(vsIN)
				vsOUT = np.zeros(VSlen-1)
				vsOUT[0:kill_I] = vsIN[0:kill_I]
				vsOUT[kill_I:(VSlen-1)] = vsIN[kill_I+1:(VSlen)]
						
				x.mantleR = []
				x.mantleR = x.radius - np.array(intfOUT)    
				x.crustThick = intfOUT[0]
				x.crustVs = vsOUT[0]
				x.mantleVs = []
				x.mantleVs = vsOUT[1:]
						
			# Change Hyper-parameter
			elif pDRAW == 6:	
						
				# Change the estimate of data error
				print DRAW[pDRAW]
						
				# Determine which hyperparameter to change
				ihyp = randint(0,len(oldx.sighyp)-1)

				# Generate random perturbation value
				wHYP = np.random.normal(0,thetaHYP)
						
				# Change the hyper-parameter value
				curhyp = copy.deepcopy(oldx.sighyp)
				newHYP = copy.deepcopy(curhyp)
				newHYP[ihyp] = curhyp[ihyp] + wHYP
				print ('Changing hyperparameter ' + str(ihyp) +
				       ' from ' + str(curhyp[ihyp]) + ' to ' +
				       str(newHYP[ihyp]))
				x.sighyp = newHYP
						
				# if new hyperparameter is outside of range - 
				# REJECT
				if ((newHYP[ihyp] < hypmin[ihyp]) or 
				    (newHYP[ihyp] > hypmax[ihyp])):
					print '!!! Outside the range allowed!!!'
					print 'Automatically REJECT model'
					WARN_BOUNDS = 'ON'
						
				vsOUT = copy.deepcopy(vsIN)
				BDi = 0
				delv2 = 0

			x.crustVp = x.PS_scale * x.crustVs
			x.crustRho = x.RS_scale * x.crustVs
			x.mantleVp = x.PS_scale * x.mantleVs
			x.mantleRho = x.RS_scale * x.mantleVs
				
			newflag =  any(value<0 for value in x.mantleVs)
			if newflag == True:
				WARN_BOUNDS == 'ON'
						
			# Continue as long as proposed model is not out of 
			# bounds:
			if WARN_BOUNDS == 'ON':
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				numreject = numreject + 1
				drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
				continue
			        # re-do iteration -- DO NOT increment (k)

			print 'test3'
			print x.epiDistkm
			print x.epiTime
			print x.nmantle
			print x.mantleR
			print x.crustVs, x.mantleVs
                        # Create swm input file
			try:
				x.create_swm_file(swm_nlay, 
						  create_nd_file=True)
			except ValueError:
				print 'WARNING: Unable to create model files'
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				continue

                        # Run sw model
			(modearray,nmodes)=runmodel(x,eps,npow,dt,fnyquist,
						    nbran,cmin,cmax,maxlyr)

		        # Confirm that rayleigh run covers desired frequency 
			# band
			parray = 1./modearray[2,:nmodes]
			gvarray = modearray[4,:nmodes]
			pmin = parray.min()
			pmax = parray.max()
			if (pmin > cpmin or pmax < cpmax):
				print pmin, cpmin, pmax, cpmax
				print ('Model did not calculate ' +
				       'correctly')
				print 'Redo iteration'
				del x
				continue
		
		        # Interpolate predicted gv to cp of data
			fgv = interp1d(parray, gvarray)
			gv_pre = []

			for i in range(0,nevts):
				gv_pre.append(fgv(cp[i]))
				dpre_sw[i][:] =  (x.epiTime[i] + 
						  (x.epiDistkm[i]/gv_pre[i]))	

			# Run bw model
			try:
				dpre_bw = runmodel_bw(x, phases) 
			except UserWarning:
				print ('Body wave phases not calculated ' +
				       'correctly')
				print 'Redo iteration'
				del x
				continue
			except:
				print 'taup threw an exception'
				print 'Redo iteration'
				del x
				continue

			# Merge into single array
			dpre[:,k+1] = np.concatenate([np.concatenate(dpre_sw), 
					       np.concatenate(dpre_bw)])
			if (not (len(dpre) == ndata)):
				print 'Problem with dpre'
				raise ValueError('Inconsistent ndata')
			x.dpre = copy.deepcopy(dpre[:,k+1])

			# Calculate error of the new model:
			misfit,newmis,PHI,x,diagCE = finderror(k,x,ndsub,dpre,
							       dobs,misfit,
							       newmis,wsig,PHI,
							       diagCE,
							       weight_opt)
							
			pac,q = accept_reject(PHI,k,pDRAW,WARN_BOUNDS,delv,
					      delv2,thetaV2,diagCE,vsIN,vsOUT,
					      BDi)
	
			if pac < q:
				print (' == ! == ! == !  REJECT NEW MODEL  ' +
				       '! == ! == ! ==  ')
				del x
				PHI[k+1] = PHI[k]
									
				numreject = numreject + 1
				drawnumreject[pDRAW] = drawnumreject[pDRAW] + 1
				# re-do iteration -- DO NOT increment (k)
			else: 
				print ('******   ******  ACCEPT NEW MODEL  ' +
				       '******   ******')
									
				# Calculate the depths and velocities for model 
				# (for plotting purposes to be used later)
				#npts = ((x.nl)*2)-1
				#F = copy.deepcopy(x.intf)
				#VS = copy.deepcopy(x.vs)
				#depth = np.zeros(npts+1)
				#depth[npts]=maxz
				#vels = np.zeros(npts+1)
				#adj = 0.001
				#ii=1
				#jj=0
				#ll=0
									
				#ii=1
				#jj=0
				#while (ii < npts):
				#	depth[ii]=F[jj]-adj
				#	depth[ii+1]=F[jj]+adj
				#	jj = jj + 1
				#	ii = ii + 2
				#ii=0
			       	#jj=0
				#while (ii < npts):
				#	vels[ii]=VS[jj]
				#	vels[ii+1]=VS[jj]
				#	jj = jj + 1
				#	ii = ii + 2
				#x.depths = depth
				#x.vels = vels
								
				# Retain MODEL 
				ITMODS.append(x)
					
				numaccept = numaccept + 1
				drawnumaccept[pDRAW] = drawnumaccept[pDRAW] + 1

				if k == MMM[keep_cnt]:
					print ('Adding model #' + str(k + 1) +
					       ' to the CHAIN ensemble')
					CHMODS.append(x)
					modl = x.filename
					curlocation = MAIN + '/' + modl
					newfilename = ('M' + str(ii) + '_' + 
						       abc[run] + '_' + 
						       str(chain)+ '_' + modl)
					newlocation = SAVEF + '/' + newfilename
					shutil.copy2(curlocation, newlocation)
					# Try to also save a binary version of the
					# MODEL class object
					classOutName = ('M' + str(ii) + '_' + 
							abc[run] + '_' + 
							str(chain) + '_' + modl +
							'.pkl')
					newlocation = SAVEF + '/' + classOutName
					with open(newlocation, 'wb') as output:
						pickle.dump(sample, output, -1)
					keep_cnt = keep_cnt + 1

			        #### Remove all models from current chain ####
				for filename in glob(MAIN+"/*.swm"):
					os.remove(filename) 
				for filename in glob(MAIN+"/*.tvel"):
					os.remove(filename) 
				for filename in glob(MAIN+"/*.npz"):
					os.remove(filename) 
				
			        # move to next iteration
				k = k + 1
	
		# Calculate the acceptance rate for the chain:
		numgen = numreject + numaccept
		drawnumgen = drawnumreject + drawnumaccept
		acc_rate[chain] = (numaccept/numgen)*100
		draw_acc_rate[:,chain] = (drawnumaccept[:]/drawnumgen[:])*100.0
		print draw_acc_rate

		if errorflag1 == 'on':
			print " "
			print " error occurred on first model generation, "
			print " no disp.out file found, try to re-do start of chain "
			print " "
			errorflag1 = 'off'
		else:
			print PHI
			inumm = numm + 0.0
			cc = ([plt.cm.brg(columnno/inumm) 
			       for columnno in range(numm)])
			plt.close('all')
				
		       	# # Ignore first BURN # of models as burn-in period
			# keepM = ITMODS[BURN:numm]
			# numremain = len(keepM)
				
	       		# # Keep every Mth model from the chain
       			# ii = BURN - 1
			# while (ii < numm):
			# 	sample = copy.deepcopy(ITMODS[ii])
			# 	print ('Adding model #  [ '+str(sample.number)+
			# 	       ' ]  to the CHAIN ensemble')
			# 	CHMODS.append(sample)
			# 	modl = sample.filename
			# 	curlocation = MAIN + '/' + modl
			# 	newfilename = ('M'+str(ii)+'_'+abc[run]+ '_' 
			# 		       + str(chain)+ '_' + modl)
			# 	newlocation = SAVEF + '/' + newfilename
			# 	shutil.copy2(curlocation, newlocation)
			# 	# Try to also save a binary version of the
			# 	# MODEL class object
			# 	classOutName = ('M' + str(ii) + '_' + abc[run] +
			# 			'_' + str(chain) + '_' + modl +
			# 			'.pkl')
			# 	newlocation = SAVEF + '/' + classOutName
			# 	with open(newlocation, 'wb') as output:
			# 		pickle.dump(sample, output, -1)
			# 	ii = ii + M
					
			# #### Remove all models from current chain ####
		       	# for filename in glob(MAIN+"/*.swm"):
			# 	os.remove(filename) 
		       	# for filename in glob(MAIN+"/*.tvel"):
			# 	os.remove(filename) 
		       	# for filename in glob(MAIN+"/*.npz"):
			# 	os.remove(filename) 
				
			#### Plot the error at each iterations ####
			errorfig(PHI, BURN, chain, abc, run, maxz, SAVEF)
								
			# # Keep lowest error models from posterior distribution
			# realmin = np.argsort(PHI)
			# jj=0
			# while (jj < keep):				
			# 	ind=realmin[jj]
			# 	sample = copy.deepcopy(ITMODS[ind])
			# 	BEST_CHMODS.append(sample)
			# 	jj = jj + 1	

		       	#### Advance to next chain ####
			chain = chain + 1
	
	#### Plot acceptance rate ####
	accratefig(totch, acc_rate, draw_acc_rate, abc, run, SAVEF)
	
	keptPHI = []
	nummods = len(CHMODS)		
	INTF=[]	
	NM=[]	
	SIGH=[]
	jj = 0
	while (jj < nummods):
		sample = copy.deepcopy(CHMODS[jj])
		RUNMODS.append(sample)
		
		curPHI = sample.PHI
		keptPHI = np.append(keptPHI, curPHI)
		
		#newcol = 1000*(sample.intf)
		newcol = np.array(sample.mantleR)
		INTF = np.append(INTF, newcol)	
		newnm = sample.nmantle
		NM = np.append(NM, newnm)	
		newsighyp = sample.sighyp
		if (jj == 0):
			SIGH = copy.deepcopy(newsighyp)
		else:
			SIGH=np.vstack((SIGH, newsighyp))
		jj = jj + 1
		
	runINTF = np.append(runINTF, INTF)
	runNM = np.append(runNM, NM)
	if (run == 0):
		runSIGH = copy.deepcopy(SIGH)
	else:
		runSIGH = np.vstack((runSIGH, SIGH))
	runPHI = np.append(runPHI, keptPHI)
	
	PHIind = np.argsort(keptPHI)
	Ult_ind = PHIind[0]
	revPHIind = PHIind[::-1]
				
	#### Plot histogram of the number of layers ####		
	nlhist(rep_cnt,repeat, NM, nmin, nmax, maxz_m, abc, run, SAVEF)
	
	#### Plot histogram of the hyperparameter SIGH ####
	sighhist(rep_cnt,repeat, SIGH, hypmin, hypmax, maxz_m, abc, run, SAVEF)
	
	# Specify colormap for plots
	chosenmap='brg_r'	
	
	# ==================== PLOT [1] ==================== 
	# ================ Velocity Models =================
	CS3,scalarMap=modfig(rep_cnt,repeat,keptPHI,vmin,vmax,chosenmap,
			     nummods,revPHIind,CHMODS,Ult_ind,maxz_m,abc,run,
			     SAVEF)
	
	# ==================== PLOT [2] ==================== 
	# =========== Dispersion Curve Vertical ============
	vdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,
		 dobs_sw,cp,Ult_ind,weight_opt,wsig,cpmin,cpmax,vmin,vmax,CS3,
		 maxz_m,abc,run,SAVEF)
	
	# ==================== PLOT [3] ==================== 
	# ========== Dispersion Curve Horizontal ===========
	# hdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,
	# 	 dobs,instpd,Ult_ind,weight_opt,wsig,pmin,pmax,vmin,vmax,CS3,
	# 	 maxz_m,abc,run,SAVEF)
	
	#### Plot histogram of ALL interface depths ####
	# intffig(rep_cnt,repeat,INTF,maxz_m,abc,run,SAVEF)

	# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
	# = # = # = # = # = # PROBABILITY DENSITY FUNCTIONS # = # = # = # = # = # = # 
	# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
	#vh,maxline,maxlineDISP,newpin,newzin,newvin,cutpin,newvinD,normvh,normph=pdfdiscrtze(maxz_m,vmax,instpd,nummods,CHMODS,pdf_connect,pmin,pmax)
	
	#### Create the pdf figures for disperions curves and models ####
	#setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
	
	if rep_cnt == (repeat - 1):
		rnummods = len(RUNMODS)		
				
		#### Plot histogram of the number of layers ####		
		nlhist(rep_cnt,repeat,runNM, nmin, nmax, maxz_m, abc, run, 
		       SAVEF)
	
		#### Plot histogram of the hyperparameter SIGH ####
		sighhist(rep_cnt,repeat,runSIGH, hypmin, hypmax, maxz_m, abc, 
			 run, SAVEF)
	
		runPHIind = np.argsort(runPHI)
		rUlt_ind = runPHIind[0]
		revrunind = runPHIind[::-1]
	
		# ==================== PLOT [1] ==================== 
		# ================ Velocity Models =================
		CS3,scalarMap=modfig(rep_cnt,repeat,runPHI,vmin,vmax,chosenmap,
				     rnummods,revrunind,RUNMODS,rUlt_ind,
				     maxz_m,abc,run,SAVEF)
                     
		# ==================== PLOT [2] ==================== 
		# =========== Dispersion Curve Vertical ============
		vdispfig(rep_cnt,repeat,rnummods,revrunind,runPHI,RUNMODS,
			 scalarMap,dobs_sw,cp,rUlt_ind,weight_opt,wsig,cpmin,
			 cpmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF)

		# ==================== PLOT [3] ==================== 
		# ========== Dispersion Curve Horizontal ===========
		# hdispfig(rep_cnt,repeat,rnummods,revrunind,runPHI,RUNMODS,
		# 	 scalarMap,dobs,instpd,rUlt_ind,weight_opt,wsig,pmin,
		# 	 pmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF)

		#### Plot histogram of ALL interface depths ####
		# intffig(rep_cnt,repeat,runINTF,maxz_m,abc,run,SAVEF)

		# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
		# = # = # = # = # = # PROBABILITY DENSITY FUNCTIONS # = # = # = # = # = # = # 
		# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # 
		# vh,maxline,maxlineDISP,newpin,newzin,newvin,cutpin,newvinD,normvh,normph=pdfdiscrtze(maxz_m,vmax,instpd,rnummods,RUNMODS,pdf_connect,pmin,pmax)

		#### Create the pdf figures for disperions curves and models ####
		# setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
					
	rep_cnt = rep_cnt + 1
	run = run + 1

Elog.close()



