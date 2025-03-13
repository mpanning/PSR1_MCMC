#!/opt/local/bin/python2.7

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
import rayleigh_python
from obspy.taup import TauPyModel,taup_create

#Updated version of MODEL class for use with mantle models
#Linear interpolated velocity in mantle, constant crust, inner core, and
#outer core.  Flexible number of point in mantle for transdimensional
#inversion.
#Includes function for writing out model files for use in surface wave 
#dispersion calculations and taup travel time calculations
#includes default PREM-like values for values not varied
class MODEL:
    def __init__(self):
        self.number = []
        self.radius = 6371.0
        self.crustThick = []
        self.crustVp = []
        self.crustVs = []
        self.crustRho = []
        self.crustQ = 300
        self.nmantle = []
        self.mantleRho = []
        self.mantleVp = []
        self.mantleVs = []
        self.mantleR = []
        self.mantleQ = 300
        self.cmbR = 3480.0
        self.outercoreRho = 11.0
        self.outercoreVp = 9.1
        self.outercoreVs = 0.0
        self.outercoreQ = 100000.0
        self.innercoreR = 1221.5
        self.innercoreRho = 12.9
        self.innercoreVp = 11.1
        self.innercoreVs = 3.6
        self.innercoreQ = 84.6
        self.bulkQ=100000.0
        self.eta=1.0
        self.PS_scale = math.sqrt(3.0)
        self.RS_scale = 0.75
        self.filename = []
        self.sighyp = []
        self.nevts = []
        self.epiDistkm = []
        self.epiTime = []
        self.hypDepth = []
    def create_swm_file(self, nlayers, create_tvel_file=False, 
                        create_nd_file=False):
        try:
            self.filename = "{:06d}.swm".format(self.number)
        except ValueError:
            print("MODEL object number not yet defined.  Cannot make file.")
            return -1
        layerthick = self.radius/(nlayers - 4)
        nclay = int(math.ceil(self.crustThick/layerthick)) + 1
        crustRads = np.linspace(self.radius - self.crustThick, self.radius, 
                                nclay)
        mantleThick = self.radius - self.crustThick - self.cmbR
        nmlay = int(math.ceil(mantleThick/layerthick)) + 1
        nmsublayers = np.zeros(self.nmantle+1, dtype=np.int_)
        for i in range(0, self.nmantle):
            nmsublayers[-1-i] = int(math.ceil(nmlay*(self.mantleR[-2-i]
                                                     -self.mantleR[-1-i])/
                                              mantleThick))
        nmsublayers[0] = nmlay - nmsublayers[1:].sum()
        indx = 0
        mantleRads = np.zeros(nmlay)
        mantleVs = np.zeros(nmlay)
        mantleVp = np.zeros(nmlay)
        mantleRho = np.zeros(nmlay)
        for i in range(0, self.nmantle):
            mantleRads[indx:indx+nmsublayers[-1-i]] = np.linspace(self.mantleR[-1-i], self.mantleR[-2-i], nmsublayers[-1-i], endpoint = False)
            mantleVs[indx:indx+nmsublayers[-1-i]] = np.linspace(self.mantleVs[-1-i], self.mantleVs[-2-i], nmsublayers[-1-i], endpoint = False)
            indx = int(indx + nmsublayers[-1-i])
        mantleRads[indx:] = np.linspace(self.mantleR[1], self.mantleR[0], 
                                        nmsublayers[0])
        mantleVs[indx:] = np.linspace(self.mantleVs[1], self.mantleVs[0],
                                      nmsublayers[0])
        mantleVp = mantleVs * self.PS_scale
        mantleRho = mantleVs * self.RS_scale
        ocThick = self.cmbR - self.innercoreR
        nolay = int(math.ceil(ocThick/layerthick)) + 1
        ocRads = np.linspace(self.innercoreR, self.cmbR, nolay)
        nilay = nlayers - nclay - nmlay - nolay
        if (not (nilay > 1)):
            raise ValueError("Layer calculation error")
        icRads = np.linspace(0, self.innercoreR, nilay)
        icbnum = nilay
        cmbnum = nilay + nolay
        
        modelout = open(self.filename,'w')
        modelout.write(' ' + self.filename + "\n")
        modelout.write("           1   1.000000               1\n")
        modelout.write(" {:11d} {:11d} {:11d}\n".format(nlayers,icbnum,cmbnum))
        #write inner core layers
        for i in range(0, nilay):
            modelout.write("{:7d}. {:8.2f} {:8.2f} {:8.2f} {:8.1f} {:8.1f} "
                           "{:8.2f} {:8.2f} {:8.5f}\n"
                           .format(int(icRads[i]*1.e3), self.innercoreRho*1.e3, 
                                   self.innercoreVp*1.e3, self.innercoreVs*1.e3,
                                   self.bulkQ, self.innercoreQ, 
                                   self.innercoreVp*1.e3, 
                                   self.innercoreVs*1.e3, self.eta))
        #write outer core layers
        for i in range(0, nolay):
            modelout.write("{:7d}. {:8.2f} {:8.2f} {:8.2f} {:8.1f} {:8.1f} "
                           "{:8.2f} {:8.2f} {:8.5f}\n"
                           .format(int(ocRads[i]*1.e3), self.outercoreRho*1.e3, 
                                   self.outercoreVp*1.e3, self.outercoreVs*1.e3,
                                   self.bulkQ, self.outercoreQ, 
                                   self.outercoreVp*1.e3, 
                                   self.outercoreVs*1.e3, self.eta))
        #write mantle layers
        for i in range(0, nmlay):
            modelout.write("{:7d}. {:8.2f} {:8.2f} {:8.2f} {:8.1f} {:8.1f} "
                           "{:8.2f} {:8.2f} {:8.5f}\n"
                           .format(int(mantleRads[i]*1.e3), 
                                   mantleRho[i]*1.e3, mantleVp[i]*1.e3,
                                   mantleVs[i]*1.e3, self.bulkQ, self.mantleQ, 
                                   mantleVp[i]*1.e3, mantleVs[i]*1.e3, 
                                   self.eta))
        #write crust layers
        for i in range(0, nclay):
            modelout.write("{:7d}. {:8.2f} {:8.2f} {:8.2f} {:8.1f} {:8.1f} "
                           "{:8.2f} {:8.2f} {:8.5f}\n"
                           .format(int(crustRads[i]*1.e3), self.crustRho*1.e3,
                                   self.crustVp*1.e3, self.crustVs*1.e3,
                                   self.bulkQ, self.crustQ, self.crustVp*1.e3,
                                   self.crustVs*1.e3, self.eta))
            
        modelout.close()
        if (create_tvel_file):
            filein = open(self.filename, 'r')
            lines = [x.strip('\n') for x in filein.readlines()]
            filein.close()
            
            #Remove header lines and reverse order (increasing depth rather 
            #than radius)
            lines = lines[-1:-len(lines)+2:-1]
            modelrad = float(lines[0].split()[0])

            self.tvelfilename = self.filename + '.tvel'
            fileout = open(self.tvelfilename,'w')
            
            #output header lines
            fileout.write("Header line 1\nHeader line 2\n")

            for line in lines:
                values = line.split()
                depth = 1.e-3*(modelrad-float(values[0]))
                Vp = 1.e-3*(float(values[2]))
                Vs = 1.e-3*(float(values[3]))
                rho = 1.e-3*(float(values[1]))
                fileout.write("{:f} {:f} {:f} {:f}\n".format(depth, Vp, Vs, 
                                                             rho))

            fileout.close()
        if (create_nd_file):
            tol = 0.1
            filein = open(self.filename, 'r')
            lines = [x.strip('\n') for x in filein.readlines()]
            filein.close()
            
            #Remove header lines and reverse order (increasing depth rather 
            #than radius)
            lines = lines[-1:-len(lines)+2:-1]
            modelrad = float(lines[0].split()[0])

            self.ndfilename = self.filename + '.nd'
            fileout = open(self.ndfilename,'w')
            
            mohoFind = False
            mohoWrite = False
            cmbFind = False
            cmbWrite = False
            iocbFind = False
            iocbWrite = False
            for line in lines:
                values = line.split()
                depth = 1.e-3 * (modelrad - float(values[0]))
                rad = 1.e-3 * (float(values[0]))
                if (not mohoFind and (depth > self.crustThick - tol) and 
                    (depth < self.crustThick + tol)):
                    mohoFind = True
                elif (mohoFind and (depth > self.crustThick - tol) and 
                    (depth < self.crustThick + tol)):
                    fileout.write("mantle\n")
                    mohoWrite = True
                elif (not cmbFind and (rad > self.cmbR - tol) and 
                    (rad < self.cmbR + tol)):
                    cmbFind = True
                elif (cmbFind and (rad > self.cmbR - tol) and 
                    (rad < self.cmbR + tol)):
                    fileout.write("outer-core\n")
                    cmbWrite = True
                elif (not iocbFind and (rad > self.innercoreR - tol) and 
                    (rad < self.innercoreR + tol)):
                    iocbFind = True
                elif (iocbFind and (rad > self.innercoreR - tol) and 
                    (rad < self.innercoreR + tol)):
                    fileout.write("inner-core\n")
                    iocbWrite = True
                    
                Vp = 1.e-3*(float(values[2]))
                Vs = 1.e-3*(float(values[3]))
                rho = 1.e-3*(float(values[1]))
                fileout.write("{:f} {:f} {:f} {:f}\n".format(depth, Vp, Vs, 
                                                             rho))
            
            fileout.close()
            if not mohoWrite or not cmbWrite or not iocbWrite:
                raise ValueError("create nd file did not specify all disc")

# ***************************** DEFINE FUNCTIONS *****************************
def mintwo (number1, number2):
        if number1 < number2:
                comparmin = number1
        else:
                comparmin = number2
        return comparmin
# ----------------------------------------------------------------------------
def maxtwo (number1, number2):
        if number1 < number2:
                comparmax = number2
        else:
                comparmax = number1
        return comparmax
# ----------------------------------------------------------------------------
def startmodel (hmin,nintf,totch,nl,maxz,prior_vrad,prior_vmin,prior_vmax,
                cvmin,cvmax,cavgV,chmin,chmax,epimin,epimax,otmin,otmax,
                hypmin,hypmax,nevts,max_neg_grad,planet_radius):
        # stz = starting interface depths
        stz = np.zeros((nintf,totch))
        
        # stvel = starting layer velocities
        stvel = np.zeros((nl,totch))

        # stepi = starting epicentral distance in km
        stepi = np.zeros((nevts,totch))

        # stotime = starting origin time in s
        stotime = np.zeros((nevts,totch))
        
        # sthyp = starting hyper parameter for data noise std dev, one per 
        # starting model and datatype
        nhyp = len(hypmin)
        sthyp = np.zeros((nhyp,totch))

        # scale degree of variance based on total depth of full model
        cnt = 0
        while (cnt < totch):
            i = 0
            while (i < nhyp):
                sthyp[i,cnt] = random.uniform(hypmin[i],hypmax[i])
                i = i + 1

            i = 0
            while (i < nevts):
                stepi[i,cnt] = random.uniform(epimin[i],epimax[i])
                stotime[i,cnt] = random.uniform(otmin[i],otmax[i])
                i = i + 1

            i = 0
            stz[i,cnt] = random.uniform(chmin,chmax)
            i = 1
            while (i < nintf - 1):
                stz[i,cnt] = random.uniform((stz[0,cnt]+hmin), 
                                            (maxz-hmin))
                i = i + 1
            stz[nintf - 1, cnt] = maxz
            tmp = stz[:,cnt]
            tmp = sorted(tmp)
            stz[:,cnt] = tmp
            i = 0
            stvel[i,cnt] = random.uniform(cvmin,cvmax)
            i = 1
            # Rework to have depth dependent vmin/vmax
            # allow negative gradients (will lead to taup failures)?
            while (i < nl):
                depth = stz[i-1,cnt]
                radius = planet_radius - depth
                rad1 = prior_vrad[prior_vrad >= radius][-1]
                rad2 = prior_vrad[prior_vrad <= radius][0]
                vmin1 = prior_vmin[prior_vrad >= radius][-1]
                vmin2 = prior_vmin[prior_vrad <= radius][0]
                vmax1 = prior_vmax[prior_vrad >= radius][-1]
                vmax2 = prior_vmax[prior_vrad <= radius][0]
                
                vmin = vmin1 + (vmin2 - vmin1)*(radius - rad1)/(rad2 - rad1)
                vmax = vmax1 + (vmax2 - vmax1)*(radius - rad1)/(rad2 - rad1)
                # stvel[i,cnt] = random.uniform(maxtwo(stvel[i-1,cnt], 
                #                                     vmin), vmax)
                # first mantle velocity must not be slower than crust and no
                # no negative gradient allowed in base layer
                if (i == 1 or i == nl -1):
                    stvel[i,cnt] = random.uniform(maxtwo(stvel[i-1,cnt], 
                                                         vmin), vmax)
                else:
                    dint = stz[i-1,cnt] - stz[i-2,cnt]
                    gradvmin = stvel[i-1,cnt] + (max_neg_grad * dint)
                    print("grad test 2 ",i,stz[:,cnt],stvel[:,cnt],dint,
                          gradvmin)
                    stvel[i,cnt] = random.uniform(maxtwo(gradvmin,vmin), vmax)
                # stvel[i,cnt] = random.uniform(vmin, vmax) # allow negative
                i = i + 1
            cnt = cnt + 1

        return (stz,stvel,stepi,stotime,sthyp)
# ----------------------------------------------------------------------------
#def sobs_set (k):
#        nextmodel = k+1
#        modl = 'modl%s.in' % nextmodel
#        try:
#                os.remove('sobs.d')
#                os.remove('log.txt')
#                os.remove('disp.out')
#        except:
#                pass
#        sobs = open('sobs.d','w')
#        sobs.write('0.005 0.005 0.0 0.005 0.0'+'\n')
#        sobs.write('1 0 0 0 0 0 1 0 1 0'+'\n')
#        sobs.write(modl+'\n')
#        sobs.write('disp.d')
#        sobs.close()
#        return (modl)
# ----------------------------------------------------------------------------
def finderror (k,x,ndsub,dpre,dobs,misfit,newmis,wsig,PHI,diagCE,weight_opt):
        curhyp = copy.deepcopy(x.sighyp)
        # Find misfit
        # count1 = 0
        #for ievt in range(0,x.nevts):
        #    count = 0
        #    print 'Calculating misfit for event '+str(ievt)
        #    while (count < tnum[ievt]):
        #        misfit[ievt][count,k+1] = (dpre[ievt][count,k+1] - 
        #                                   dobs[ievt][count])
        #        sigEST = math.sqrt(((wsig[ievt][count]**2)+(curhyp**2)))
        #        diagCE[count1,k+1] = sigEST**2
        #        newmis[ievt][count,k+1] = misfit[ievt][count,k+1]*(1/sigEST)
        #       formatstring = ('Period: {:7.2f} misfit: {:10g} newmis: ' +
        #                        '{:10g} dobs: {:8.2f} dpre: {:8.2f}')
        #        print formatstring.format(instpd[ievt][count], 
        #                                  misfit[ievt][count,k+1], 
        #                                  newmis[ievt][count,k+1], 
        #                                  dobs[ievt][count], 
        #                                  dpre[ievt][count,k+1])
        #        count = count + 1
        #        count1 = count1 + 1
        nsubsets = len(ndsub)
        if (not (nsubsets == len(curhyp))):
            print('Problem with hyp dimensions')
            raise ValueError('inconsistent number of subsets')
        count = 0
        for i in range(0,nsubsets):
            for j in range(0,ndsub[i]):
                misfit[count] = dpre[count,k+1] - dobs[count]
                sigEST = math.sqrt(((wsig[count]**2)+(curhyp[i]**2)))
                diagCE[count] = sigEST**2
                newmis[count] = misfit[count]*(1.0/sigEST)
                count = count + 1

        print(" ")
        print('iteration: ' + str(k))
        #x.misfit = []
        #x.w_misfit = []
        #for ievt in range(0,x.nevts):
        #    x.misfit.append(misfit[ievt][:,k+1])
        #    x.w_misfit.append(newmis[ievt][:,k+1])

        x.misfit = misfit
        x.w_misfit = newmis

        # Make e_sqd as list of arrays
        #e_sqd = []
        #if weight_opt == 'OFF':
        #    for ievt in range(0,x.nevts):
        #        e_sqd.append((x.misfit[ievt][:])**2)
        #elif weight_opt == 'ON':
        #    for ievt in range(0,x.nevts):
        #        e_sqd.append((x.w_misfit[ievt][:])**2)
        # Convert to single np array
        #e_sqd = np.concatenate(e_sqd)
        e_sqd = []
        if weight_opt == 'OFF':
            e_sqd = (x.misfit[:])**2
        elif weight_opt == 'ON':
            e_sqd = (x.w_misfit[:])**2

        if (k == -1):
                PHI[0] = (sum(e_sqd))
                PHIold = 'nan'
                PHInew = PHI[0]
        else:
                PHI[k+1] = (sum(e_sqd))
                PHIold = PHI[k]
                PHInew = PHI[k+1]
        x.PHI = sum(e_sqd)
        x.diagCE = diagCE[:]
        print('PHIold = ' + ' ' + str(PHIold) + '    ' + 'PHInew = ' + ' '
              + str(PHInew))
        return (misfit,newmis,PHI,x,diagCE)        
# ----------------------------------------------------------------------------
def accept_reject (PHI,k,pDRAW,WARN_BOUNDS,delv,delv2,thetaV2,diagCE,vsIN,
                   vsOUT,BDi):
        pi = math.pi
        if WARN_BOUNDS == 'ON':
                pac = 0
        else:
                # If changing source parameters, moving interface, or changing 
                # a velocity:
                if (pDRAW >= 0) and (pDRAW <= 3):
                        try:
                                misck = math.exp(-(PHI[k+1]-PHI[k])/2)
                        except OverflowError:
                                misck = 1
                        print('PHIs: '+str(PHI[k])+', '+str(PHI[k+1]))
                        print('misck is:   '+str(misck))
                        pac = mintwo(1,misck)
                        
                # If changing dimension of model: Birth
                elif pDRAW == 4:
                        try:
                            term1 = ((thetaV2*math.sqrt(2*pi))/delv)
                            term2 = (((vsOUT[BDi]-vsIN[BDi])**2)/
                                     (2*(thetaV2**2)))
                            term3 = ((PHI[k+1]-PHI[k])/2)
                            print('term 1: ' +str(term1))
                            print('term 2: ' +str(term2))
                            print('term 3: ' +str(term3))
                            misck = term1*math.exp(term2-term3)
                            # term1 = delv2/delv
                            # term2 = (-(PHI[k+1]-PHI[k])/2.0)
                            # misck = term1*math.exp(term2)
                        except OverflowError:
                            misck = 1
                        print('PHIs: '+str(PHI[k])+', '+str(PHI[k+1]))
                        print('delvs: '+str(delv)+', '+str(delv2))
                        print('misck is:   '+str(misck))
                        pac = mintwo(1,misck)
                        
                # If changing dimension of model: Death
                elif pDRAW == 5:
                        try:
                            term1 = (delv/(thetaV2*math.sqrt(2*pi)))
                            term2 = (-((vsOUT[BDi]-vsIN[BDi])**2)/
                                      (2*(thetaV2**2)))
                            term3 = ((PHI[k+1]-PHI[k])/2)
                            print('term 1: ' +str(term1))
                            print('term 2: ' +str(term2))
                            print('term 3: ' +str(term3))
                            misck = term1*math.exp(term2-term3)
                            # term1 = delv/delv2
                            # term2 = (-(PHI[k+1]-PHI[k])/2.0)
                            # misck = term1*math.exp(term2)
                        except OverflowError:
                            misck = 1
                        print('PHIs: '+str(PHI[k])+', '+str(PHI[k+1]))
                        print('delvs: '+str(delv)+', '+str(delv2))
                        print('misck is:   '+str(misck))
                        pac = mintwo(1,misck)
                
                # If changing hyperparameter for data error estimate:
                elif pDRAW == 6:
                        try:
                                olddet = np.prod(diagCE[:,k])
                                newdet = np.prod(diagCE[:,k+1])
                                term1 = (olddet/newdet)
                                term2 = math.exp(-(PHI[k+1]-PHI[k])/2)
                                misck = term1*term2 
                        except OverflowError:
                                misck = 1
                        print('PHIs: '+str(PHI[k])+', '+str(PHI[k+1]))
                        print('misck is:   '+str(misck))
                        pac = mintwo(1,misck)
                        
        print(' ')                
        print('pac = min[ 1 , prior ratio x likelihood ratio x proposal ' +
              'ratio ] :   ' + str(pac))
        print(' ')
        q = random.uniform(0,1)
        print('random q:   '+str(q))
        return (pac,q)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def runmodel (x,eps,npow,dt,fnyquist,nbran,cmin,cmax,maxlyr):
    modelfile = x.filename
    (modearray, nmodes) = rayleigh_python.rayleigh(eps,npow,dt,fnyquist,nbran,
                                                   cmin,cmax,maxlyr,modelfile)
    return (modearray,nmodes)
# ----------------------------------------------------------------------------
def runmodel_bw (x,phases):
    try:
        modelfile = x.ndfilename
    except AttributeError:
        try:
            modelfile = x.tvelfilename
        except AttributeError:
            raise AttributeError("No nd or tvel file name set")

    #modelfile = x.filename + '.tvel'
    try:
        taup_create.build_taup_model(modelfile,os.getcwd())
    except:
        print('taup could not build model')
        raise UserWarning("taup build model error")
    taupfile = "{}/{}.npz".format(os.getcwd(), x.filename)
    model = TauPyModel(model=taupfile)

    if (not(len(phases) == x.nevts)):
        raise ValueError('Inconsistent nevts')
    dpre_bw = []
    for i in range(0,x.nevts):
        print("Event {:d}\n".format(i))
        dpre_bw.append(np.zeros(len(phases[i])))
        distdeg = x.epiDistkm[i] * 180.0/(x.radius*math.pi)
        #print('test arrivals',x.hypDepth[i],distdeg,phases[i])
        arrivals = np.array(model.get_travel_times(source_depth_in_km=x.hypDepth[i], 
                                                   distance_in_degree=distdeg,
                                                   phase_list=phases[i]))
        # Extract arrival times 
        names = np.array([arrivals[j].name for j in range(len(arrivals))])
        print(names)
        # Need to catch errors if phase is not found FIX THIS
        for j in range(0,len(phases[i])):
            print(i,j,phases[i][j])
            try:
                dpre_bw[i][j] = (arrivals[names == phases[i][j]][0].time + 
                                 x.epiTime[i])
            except IndexError:
                print('Phases not found')
                raise UserWarning("Phases not found")

    return (dpre_bw)
# ----------------------------------------------------------------------------
def startchain (x,errorflag1,stz,stvel,stepi,stotime,sthyp,prior_vrad,
                prior_vmin,prior_vmax,cvmin,cvmax,cavgV,chmin,chmax,epimin,
                epimax,otmin,otmax,hypmin,hypmax,hmin,maxz,chain,max_neg_grad,
                planet_radius):
        nmantle = copy.deepcopy(x.nmantle)
        if errorflag1 == 'on':
                INTF = np.zeros((nmantle+2))
                VS = np.zeros((nmantle+3))
                EPI = np.zeros(x.nevts)
                OT = np.zeros(x.nevts)
                i = 0
                while (i < x.nevts):
                    EPI[i] = random.uniform(epimin[i],epimax[i])
                    OT[i] = random.uniform(otmin[i],otmax[i])
                    i = i+1

                nhyp = len(hypmin)
                HYP = np.zeros(nhyp)
                i = 0
                while (i < nhyp):
                    HYP[i] = random.uniform(hypmin[i],hypmax[i])
                    i = i + 1
                i = 0
                INTF[i] = random.uniform(chmin, chmax)
                i = 1
                while (i < nmantle + 1):
                        INTF[i] = random.uniform((INTF[0]+hmin), (maxz-hmin))
                        i = i + 1
                INTF[nmantle + 1] = maxz
                INTF[:] = sorted(INTF[:])
                i = 0
                VS[i] = random.uniform(cvmin, cvmax)
                i = 1
                while (i < nmantle + 3):
                    depth = INTF[i-1]
                    radius = planet_radius - depth
                    rad1 = prior_vrad[prior_vrad >= radius][-1]
                    rad2 = prior_vrad[prior_vrad <= radius][0]
                    vmin1 = prior_vmin[prior_vrad >= radius][-1]
                    vmin2 = prior_vmin[prior_vrad <= radius][0]
                    vmax1 = prior_vmax[prior_vrad >= radius][-1]
                    vmax2 = prior_vmax[prior_vrad <= radius][0]
                
                    vmin = vmin1 + (vmin2 - vmin1)*(radius - rad1)/(rad2 - rad1)
                    vmax = vmax1 + (vmax2 - vmax1)*(radius - rad1)/(rad2 - rad1)
                    if (i == 1 or i == nmantle + 2):
                        VS[i] = random.uniform(maxtwo(VS[i-1],vmin), vmax)
                    else:
                        dint = INTF[i-1] - INTF[i-2]
                        gradvmin = VS[i-1] + (max_neg_grad * dint)
                        VS[i] = random.uniform(maxtwo(gradvmin,vmin), vmax)
                    # VS[i] = random.uniform(vmin, vmax) # allow negative
                    # VS[i] = random.uniform(maxtwo(VS[i-1],vmin), vmax)
                    i = i + 1
                errorflag1 = 'off'
        else:
                # Hyper-parameter for data error estimation 
                # nhyp = len(hypmin)
                # HYP = np.zeros(nhyp)
                HYP = sthyp[:,chain]

                # Epicentral distance
                EPI = stepi[:,chain]

                # Origin time
                OT = stotime[:,chain]

                # Depth to interfaces between layers 
                INTF = np.zeros((nmantle+2))
                INTF[:] = stz[:,chain]

                # S-wave velocity in layers
                VS = np.zeros((nmantle+3))
                VS[:] = stvel[:,chain]

        x.crustThick = INTF[0]
        x.crustVs = VS[0]
        x.crustVp = VS[0] * x.PS_scale
        x.crustRho = VS[0] * x.RS_scale
        x.cmbR = x.radius - maxz
        x.mantleR = np.zeros(nmantle+2)
        x.mantleR = x.radius - INTF
        # x.mantleR[:nmantle+1] = x.radius - INTF
        # x.mantleR[nmantle+1] = x.cmbR
        x.mantleVs = VS[1:]
        x.mantleVp = x.mantleVs * x.PS_scale
        x.mantleRho = x.mantleVs * x.RS_scale
        x.epiDistkm = EPI
        x.epiTime = OT
        x.sighyp = HYP
        return (x,errorflag1)
# ----------------------------------------------------------------------------
def randINTF (prior_vrad,prior_vmin,prior_vmax,prior_delv,chmin,hmin,maxz,F,
              vsIN,thetaV2,max_neg_grad,planet_radius):
        nintf = len(F)
        # Have a flag so if proper conditions are not met the process will 
        # repeat

        itmax = 10
        niter = 0
        WARN_BOUNDS = "OFF"
        badvelflag = True
        while badvelflag:
            niter = niter + 1
            redoflag="True"
            while redoflag == "True":
                # empty list to be added to with check flag for each interface
                intfCHK = []
                # shallowest interface at min crust thickness
                r = random.uniform(chmin, (maxz-hmin))
                rr = 0
                while (rr < nintf):
                    curINT = F[rr]
                    if (((curINT-hmin) <= r <= curINT) or 
                        (curINT <= r <= (curINT+hmin))):
                        intfCHK.append("ON")
                    else:
                        intfCHK.append("OFF")
                    rr = rr + 1
                redoflag = "ON" in intfCHK
            print("adding interface in at:   "+str(r)+" km depth\n") 
            newF = np.append(F, r)
            newF = sorted(newF)
        
            # Find index of new interface within the model so know which layer was 
            # split and can adjust velocities appropriately
            matching = [i for i,item in enumerate(newF) if item not in F]
            matching = matching[0]
            VSlen = len(vsIN)
            vsOUT = np.zeros(VSlen+1)        
        
            # Copy velocity values over, the lower half of the split layer stays 
            # the same velocity as before. The new layer (upper half of split) 
            # gets assigned a new random velocity uniform between the velocities
            # above and below (to avoid negative gradients)
            wV2 = np.random.normal(0,thetaV2)
        
            if matching == 0: #crustal case perturb bottom half instead
                vsOUT[0] = vsIN[0]
                # Calculate expected "old" velocity value by extrapolating old
                # sub-moho velocity gradient to new moho depth
                vsOld = vsIN[1] + (vsIN[2] - vsIN[1])*(r - F[0])/(F[1] - F[0])
                vsOUT[1] = vsOld + wV2
                # vsOUT[1] = random.uniform(maxtwo(vmin,vsIN[0]),vsIN[1])
                delv2 = vsIN[1]-vsIN[0] #Is this right?
                vsOUT[2:] = vsIN[1:]
            else:
                vsOUT[0:matching] = vsIN[0:matching]
                # determine the old velocity at the new interface from the 
                # linear interpolation between the points above and below
                # and then add in random perturbation
                vsOld = (vsIN[matching - 1] + (vsIN[matching] - vsIN[matching - 1])*
                         (r - F[matching - 1])/(F[matching] - F[matching - 1]))
                vsOUT[matching] = vsOld + wV2
                # vsOUT[matching] = random.uniform(maxtwo(vmin,vsIN[matching-1]), 
                #                                  vsIN[matching])
                delv2 = vsIN[matching]-vsIN[matching-1] #Is this right?
                print('vsIN[matching] ' + str(vsIN[matching]))
                print('vsOUT[matching] ' + str(vsOUT[matching]))
                vsOUT[matching+1:(VSlen+1)] = vsIN[matching:VSlen]
                # TEMP TEST
                if vsOUT[matching] > 1000:
                    print('What is happening in randINTF?')
                    print(vsIN)
                    print(vsOUT)
                    print(vsOLD)
                    print(wV2)
        
            # If the velocity of the newly added layer is outside bounds or 
            # creates a negative gradient, set flag
            if (matching == 0):
                depth = newF[0]
                radius = planet_radius - depth
                rad1 = prior_vrad[prior_vrad >= radius][-1]
                rad2 = prior_vrad[prior_vrad <= radius][0]
                vmin1 = prior_vmin[prior_vrad >= radius][-1]
                vmin2 = prior_vmin[prior_vrad <= radius][0]
                vmax1 = prior_vmax[prior_vrad >= radius][-1]
                vmax2 = prior_vmax[prior_vrad <= radius][0]
                
                vmin = (vmin1 + (vmin2 - vmin1)*(radius - rad1)/(rad2 - rad1))
                vmax = (vmax1 + (vmax2 - vmax1)*(radius - rad1)/(rad2 - rad1))
                # if ((vsOUT[1] > vmax) or (vsOUT[1] < vmin) or 
                #    (vsOUT[1] < vsOUT[0]) or (vsOUT[1] > vsOUT[2])):
                # if ((vsOUT[1] > vmax) or (vsOUT[1] < vmin)):
                grad2 = (vsOUT[2] - vsOUT[1])/(newF[1] - newF[0])
                if ((vsOUT[1] > vmax) or (vsOUT[1] < vmin) or
                    (grad2 < max_neg_grad)):
                    print('!!! Velocity of new layer outside bounds !!!')
                    print('Try a new interface and velocity')
                    if niter > itmax:
                        print('WARNING, exceeded itmax')
                        WARN_BOUNDS = 'ON'
                        badvelflag = False
                        delv = 0.0
                else:
                    badvelflag = False
                    delv11 = prior_delv[prior_vrad >= radius][-1]
                    delv22 = prior_delv[prior_vrad <= radius][0]
                    delv = (delv11 + (delv22 - delv11)*(radius-rad1)/(rad2-rad1))
            else:
                depth = newF[matching - 1]
                radius = planet_radius - depth
                rad1 = prior_vrad[prior_vrad >= radius][-1]
                rad2 = prior_vrad[prior_vrad <= radius][0]
                vmin1 = prior_vmin[prior_vrad >= radius][-1]
                vmin2 = prior_vmin[prior_vrad <= radius][0]
                vmax1 = prior_vmax[prior_vrad >= radius][-1]
                vmax2 = prior_vmax[prior_vrad <= radius][0]
                
                vmin = (vmin1 + (vmin2 - vmin1)*(radius - rad1)/(rad2 - rad1))
                vmax = (vmax1 + (vmax2 - vmax1)*(radius - rad1)/(rad2 - rad1))
                # if ((vsOUT[matching] > vmax) or (vsOUT[matching] < vmin) or 
                #    (vsOUT[matching] > vsOUT[matching + 1]) or 
                #    (vsOUT[matching] < vsOUT[matching - 1])):
                # if ((vsOUT[matching] > vmax) or (vsOUT[matching] < vmin)):
                grad1 = ((vsOUT[matching] - vsOUT[matching-1])/
                         (newF[matching-1] - newF[matching-2]))
                grad2 = ((vsOUT[matching+1] - vsOUT[matching])/
                         (newF[matching] - newF[matching-1]))
                print('test grad 1 ',grad1,grad2)
                # Set grad2 to fail check if final gradient is negative
                if (matching == nintf - 1) and (grad2 < 0.0):
                    grad2 = max_neg_grad - 0.1
                if ((vsOUT[matching] > vmax) or (vsOUT[matching] < vmin) or
                    (grad1 < max_neg_grad) or (grad2 < max_neg_grad)):
                    print('!!! Velocity of new layer outside bounds !!!')
                    print('Try a new interface and velocity')
                    if niter > itmax:
                        print('WARNING, exceeded itmax')
                        WARN_BOUNDS = 'ON'
                        badvelflag = False
                        delv = 0.0
                else:
                    badvelflag = False
                    delv11 = prior_delv[prior_vrad >= radius][-1]
                    delv22 = prior_delv[prior_vrad <= radius][0]
                    delv = (delv11 + (delv22 - delv11)*(radius-rad1)/
                            (rad2-rad1))
        
        return (r, newF, vsOUT, WARN_BOUNDS, matching, delv, delv2)
        
# ----------------------------------------------------------------------------
def errorfig(PHI, BURN, chain, abc, run, maxz, SAVEF):
        plt.close('all')
        fig, ax1 = plt.subplots()

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.63, 0.6, 0.25, 0.25]
        ax2 = fig.add_axes([left, bottom, width, height])

        ax1.plot(PHI[BURN:], color='red')
        
        #ax1.set_ylabel('E(m)', fontweight='bold', fontsize=14)
        ax1.set_ylabel(r'$\phi$(m)', fontsize=20)
        ax1.set_xlabel('Iteration',  fontweight='bold', fontsize=14)
        figtitle = 'Evolution of model error '+ r'$\phi$(m)'+ '\n TRANS-D layers, '+str(1000*maxz)+' m total depth'
        ax1.set_title(figtitle, fontweight='bold', fontsize=14)

        ax2.plot(PHI[0:BURN], color='red')
        ax2.set_ylabel(r'$\phi$(m)')
        ax2.set_xlabel('Iteration')
        ax2.set_title('Burn - in Period', fontsize=12)

        Efig = SAVEF + '/' + 'Error_chain_' + str(chain)+'_'+abc[run]+'.png'
        P.savefig(Efig)
# ----------------------------------------------------------------------------
def accratefig(totch, acc_rate, draw_acc_rate, abc, run, SAVEF):
        plt.close('all')
        pltx = range(totch)
        plt.plot(pltx, acc_rate, 'ko')
        plt.plot(pltx, acc_rate, 'k:')
        plt.xlabel('Chain')
        plt.ylabel('Acceptance Percentage (%)')
        plt.axis([0, (totch-1), 0, 100])
        plt.title('Generated Model Acceptance Rate', fontweight='bold', fontsize=14)
        picname='Acceptance_rate_'+abc[run]+'.png'
        accfig = SAVEF + '/' + picname
        P.savefig(accfig)
        plt.close('all')
        for i in range(0, draw_acc_rate.shape[0]):
            plt.plot(pltx, draw_acc_rate[i], 'ko')
            plt.plot(pltx, draw_acc_rate[i], 'k:')
            plt.xlabel('Chain')
            plt.ylabel('Acceptance Percentage (%)')
            plt.axis([0, (totch-1), 0, 100])
            plt.title('Generated Model Acceptance Rate ' + str(i), 
                      fontweight='bold', fontsize=14)
            picname='Acceptance_rate_'+abc[run]+'_'+str(i)+'.png'
            accfig = SAVEF + '/' + picname
            P.savefig(accfig)
            plt.close('all')

            
# ----------------------------------------------------------------------------
def nlhist(rep_cnt,repeat, NL, nmin, nmax, maxz_m, abc, run, SAVEF):
        plt.close('all')
        P.figure
        P.hist(NL, bins=np.linspace(nmin-0.5, nmax+0.5, (nmax-nmin)+2))
        #plt.show()
        plt.ylabel('Count')
        plt.xlabel('Number of Layers')
        if rep_cnt == (repeat - 1):
                figname = ('REP_NL_hist_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]
                           +'.png')
        else:
                figname = 'NL_hist_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        figtitle = 'Number of Model Layers \n '+str(maxz_m)+' m total depth'
        plt.title(figtitle, fontweight='bold', fontsize=14)
        Hfig = SAVEF + '/' + figname
        P.savefig(Hfig)
# ----------------------------------------------------------------------------
def sighhist(rep_cnt, repeat, SIGH, hypmin, hypmax, maxz_m, abc, run, SAVEF):
        plt.close('all')
        P.figure
        nsigh = SIGH.shape[1]
        for i in range(0, nsigh):
            P.subplot(1, nsigh, i+1)
            P.hist(SIGH[:,i], bins=np.linspace(hypmin[i], hypmax[i], 21))
            if (i == 0):
                plt.ylabel('Count')
            labelstr = 'Hyper-parameter '+str(i)
            plt.xlabel(labelstr)
        if rep_cnt == (repeat - 1):
            figname = ('REP_SIGHYP_hist_TRANSD_lay_'+str(maxz_m)+'m_'+
                       abc[run]+'.png')
        else:
            figname = ('SIGHYP_hist_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]
                       +'.png')
        figname = 'SIGHYP_hist_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        figtitle = ('Hyper-parameter for data error \n '+str(maxz_m)+
                    ' m total depth')
        plt.suptitle(figtitle, fontweight='bold', fontsize=14)
        Hfig = SAVEF + '/' + figname
        P.savefig(Hfig)
# ----------------------------------------------------------------------------
def modfig(rep_cnt,repeat,keptPHI,vmin,vmax,chosenmap,nummods,revPHIind,CHMODS,
           Ult_ind,maxz_m,abc,run,SAVEF):
        plt.close('all')
        minCBE=keptPHI.min()
        maxCBE=keptPHI.max()
        cNorm  = colors.Normalize(vmin=minCBE, vmax=maxCBE)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=chosenmap)
        
        # loop through all kept models plotting one at a time
        jj=0
        while (jj < nummods):
                # pull out index of current BEST model
                ind = revPHIind[jj]
                
                # error of current BEST model
                moderror = keptPHI[ind]
                
                # retrieve color value specific to this model's error
                colorVal= scalarMap.to_rgba(moderror)
                
                curmod = copy.deepcopy(CHMODS[ind])
                
                #pltvels = curmod.vels
                #pltzs = 1000*(curmod.depths)
                pltvels = np.repeat(curmod.crustVs, 2)
                pltzs = np.array([curmod.radius, curmod.mantleR[0]])
                pltvels = np.append(pltvels, curmod.mantleVs)
                pltzs = np.append(pltzs, curmod.mantleR)
                print(len(pltvels), len(pltzs))
                
                linebest = plt.plot(pltvels, pltzs, c=colorVal, linewidth=2.0, 
                                    zorder=3)
                jj = jj + 1
        
        # Plot ultimate best model of all chains
        curmod = copy.deepcopy(CHMODS[Ult_ind])
        #pltvels = curmod.vels
        #pltzs = 1000*(curmod.depths)
        pltvels = np.repeat(curmod.crustVs, 2)
        pltzs = np.array([curmod.radius, curmod.mantleR[0]])
        pltvels = np.append(pltvels, curmod.mantleVs)
        pltzs = np.append(pltzs, curmod.mantleR)

        ULTl, = plt.plot(pltvels, pltzs, 'k--', linewidth=4.0, 
                         zorder=3, 
                         label='model with lowest \n prediction error')
        Z = [[0,0],[0,0],[0,0]]
        levels = np.linspace(minCBE, maxCBE, 100)
        CS3 = plt.contourf(Z, levels, cmap=chosenmap)
        cb=plt.colorbar(CS3)
        cb.set_label(r'$\phi$(m)', labelpad=-65)
        #plt.axis([0, vmax, 0, maxz_m])
        plt.axis([0, vmax, curmod.mantleR[-1], curmod.radius])
        ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        ax.grid(True,linestyle='-',color='0.75',zorder=0)
        plt.ylabel('depth (m)')
        plt.xlabel('VS (km/s)')
        plt.legend(handles=[ULTl], fontsize=10)
        if rep_cnt == (repeat-1):
                bestname = 'REP_M_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        else:
                bestname = 'M_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        figtitle = ('Velocity Models \n Transdimensional, '+str(maxz_m)
                    +' m total depth')
        plt.title(figtitle, fontweight='bold', fontsize=14)
        Mfig = SAVEF + '/' + bestname
        P.savefig(Mfig)
        print("SHOULD BE SAVING M FIGURE RIGHT NOW!!!!")
        plt.close('all')
        
        return (CS3, scalarMap)
# ----------------------------------------------------------------------------
def vdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,dobs_sw,
             instpd,Ult_ind,weight_opt,wsig,pmin,pmax,vmin,vmax,CS3,maxz_m,abc,
             run,SAVEF):
        plt.close('all')
        plt.figure
        # Figure out how many events we have
        nevts = len(dobs_sw)
        # loop through all kept models plotting one at a time
        # separate panel for each event
        jj=0
        while (jj < nummods):
                # pull out index of current BEST model
                ind = revPHIind[jj]
                
                # error of current BEST model
                moderror = keptPHI[ind]
                
                curmod = copy.deepcopy(CHMODS[ind])
                
                # assemble dpre into dpre_sw
                dpre_sw = []
                dindx = 0
                for ii in range(nevts):
                    npts = len(dobs_sw[ii])
                    dpre_sw.append(curmod.dpre[dindx:dindx + npts])
                    dindx = dindx + npts

                    pltdpre = dpre_sw[ii]
                
                    # retrieve color value specific to this model's error
                    colorVal= scalarMap.to_rgba(moderror)
                
                    plt.subplot(1, nevts, ii + 1)
                    linebest = plt.plot(pltdpre, instpd[ii], c=colorVal, 
                                        linewidth=2.0, zorder=3)

                jj = jj + 1
        
        # Plot ultimate best model of all chains
        curmod = copy.deepcopy(CHMODS[Ult_ind])
        dpre_sw = []
        wsig_sw = []
        dindx = 0
        for ii in range(nevts):
            npts = len(dobs_sw[ii])
            dpre_sw.append(curmod.dpre[dindx:dindx + npts])
            if (weight_opt == 'ON'):
                wsig_sw.append(wsig[dindx:dindx + npts])
            dindx = dindx + npts

            pltdpre = dpre_sw[ii]

            plt.subplot(1, nevts, ii + 1)
            ULTl, = plt.plot(pltdpre, instpd[ii], 'k--', linewidth=4.0, 
                             zorder=3, label=('best fit'))
            if (ii == nevts - 1):
                cb=plt.colorbar(CS3)
                cb.set_label(r'$\phi$(m)', labelpad=-65)
            plt.xlabel('Group time (s)')
            if (ii == 0):
                plt.ylabel('Period (s)')
            ymin = 0.85*dobs_sw[ii].min()
            ymax = 1.15*dobs_sw[ii].max()
            plt.axis([ymin, ymax, pmin, pmax])
            ax = plt.gca()
            # ax.set_ylim(ax.get_ylim()[::-1])
            plt.grid(True,linestyle='-',color='0.75',zorder=0)
            obs, = plt.plot(dobs_sw[ii], instpd[ii], 'ko', zorder=10, 
                            markersize=7, 
                            label='obs')
            if weight_opt == 'ON':                
                obserr = plt.errorbar(dobs_sw[ii], instpd[ii], 
                                      xerr=wsig_sw[ii], zorder=20, 
                                      linestyle="None", linewidth=2.0, 
                                      ecolor='k')

        plt.legend(handles=[ULTl, obs], fontsize=10, loc='upper right')
        if rep_cnt == (repeat-1):        
                bestname = 'REP_D_Vert_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        else:
                bestname = 'D_Vert_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        figtitle = 'Group Velocity Dispersion \n Transdimensional, '+str(maxz_m)+' m total depth'
        plt.suptitle(figtitle, fontweight='bold', fontsize=14)
        Dfig = SAVEF + '/' + bestname
        P.savefig(Dfig)
        print("SHOULD BE SAVING D Vert FIGURE RIGHT NOW!!!!")

# ----------------------------------------------------------------------------
def hdispfig(rep_cnt,repeat,nummods,revPHIind,keptPHI,CHMODS,scalarMap,dobs,instpd,Ult_ind,weight_opt,wsig,pmin,pmax,vmin,vmax,CS3,maxz_m,abc,run,SAVEF):
        plt.close('all')
        # loop through all kept models plotting one at a time
        jj=0
        while (jj < nummods):
                # pull out index of current BEST model
                ind = revPHIind[jj]
                
                # error of current BEST model
                moderror = keptPHI[ind]
                
                curmod = copy.deepcopy(CHMODS[ind])
                
                pltdpre = curmod.dpre
                
                # retrieve color value specific to this model's error
                colorVal= scalarMap.to_rgba(moderror)
                
                linebest = plt.plot(instpd, pltdpre, c=colorVal, linewidth=2.0, zorder=3)
                jj = jj + 1
        
        # Plot ultimate best model of all chains
        curmod = copy.deepcopy(CHMODS[Ult_ind])
        pltdpre = curmod.dpre
        ULTl, = plt.plot(instpd, pltdpre, 'k--', linewidth=4.0, 
                                        zorder=3, label='model with lowest \n prediction error')
        cb=plt.colorbar(CS3)
        cb.set_label(r'$\phi$(m)', labelpad=-65)
        plt.ylabel('VS (km/s)')
        plt.xlabel('Period (s)')
        plt.axis([pmin, pmax, vmin, vmax])
        plt.grid(True,linestyle='-',color='0.75',zorder=0)
        obs, = plt.plot(instpd, dobs, 'ko', zorder=10, markersize=7, 
                                        label='measured group velocities')
        if weight_opt == 'ON':
                obserr = plt.errorbar(instpd, dobs, yerr=wsig, zorder=20, linestyle="None", linewidth=2.0, ecolor='k')
        plt.legend(handles=[ULTl, obs], fontsize=10)
        if rep_cnt == (repeat-1):        
                bestname = 'REP_D_Horz_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        else:
                bestname = 'D_Horz_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        figtitle = 'Group Velocity Dispersion \n Transdimensional, '+str(maxz_m)+' m total depth'
        plt.title(figtitle, fontweight='bold', fontsize=14)
        Dfig = SAVEF + '/' + bestname
        P.savefig(Dfig)
        print("SHOULD BE SAVING D Horz FIGURE RIGHT NOW!!!!")
# ----------------------------------------------------------------------------
def intffig(rep_cnt,repeat,INTF,maxz_m,abc,run,SAVEF):
        plt.close('all')
        # Determine number of bins based on total depth
        tmp_I=1000*INTF
        nbins = (maxz_m)/10.0
        histI, bin_edges = np.histogram(tmp_I, bins=np.linspace(0, maxz_m, (nbins+1)), density=True)
        temp=len(histI)
        pltzs = np.linspace(5, (maxz_m-5), temp)
        plt.plot(histI,pltzs)
        P.gca().invert_yaxis()
        P.ylabel('Depth (m)')
        P.xlabel('Interface Probability')
        P.title('Histogram of Interface Depths')
        if rep_cnt == (repeat - 1):
                bestname = SAVEF +'/'+'REP_HIST_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        else:
                bestname = SAVEF +'/'+'HIST_TRANSD_lay_'+str(maxz_m)+'m_'+abc[run]+'.png'
        P.savefig(bestname)
# ----------------------------------------------------------------------------
def pdfdiscrtze(maxz_m,vmax,instpd,nummods,CHMODS,pdf_connect,pmin,pmax):
        # Discretize depth range, velocity range, and period range
        #zints=maxz/0.001
        zints=maxz_m/5
        zin=np.linspace(0,maxz_m,zints+1)
        newzin=zin[0:zints]
        
        # Discretize velocities more coarsely for the dispersion pdf
        vints=vmax/0.05
        vintsD=vmax/0.01
        vin=np.linspace(0,vmax,vints+1)
        vinD=np.linspace(0,vmax,vintsD+1)
        newvin=vin[0:vints]
        newvinD=vinD[0:vintsD]
        
        pints=200
        pin=np.linspace(pmin,pmax,pints+1)
        newpin=pin[0:pints]
        
        # Find the indices within newpin that correspond to the discretized value closest
        # to the highest and lowest instantaneous period of the measurements
        pstart_ind, pstart_value = min(enumerate(newpin), key=lambda x: abs(x[1]-instpd[0]))
        pend_ind, pend_value = min(enumerate(newpin), key=lambda x: abs(x[1]-instpd[-1]))
        
        # Cut the discretized array of periods down to only those occupied by the data
        cutpin=newpin[pstart_ind:(pend_ind+1)]
        intpnum = len(cutpin)
        
        # The vh array will be the 2D array of hit counts for velocities with depth
        vh=np.zeros((zints,vints))

        # The ph array will be the 2D array of hit counts for velocities with period
        # The ph array will still cover the full range of periods, but only the cut
        # period range occupied by the data will be interpolated and counted toward
        # the pdf
        ph=np.zeros((vintsD,pints))

        # Loop through all kept models, dealing with one model at a time
        cmod = 0
        while (cmod < nummods):
                sample = copy.deepcopy(CHMODS[cmod])
                cdpre = sample.dpre
                        
                # ---------- DISPERSION CURVE INTERPOLATION + COUNTS -------------
                intpDISP = np.interp(cutpin, instpd, cdpre)
                
                # Loop through all discretized period values
                i = pstart_ind
                ii = 0
                while (i < (pend_ind+1)):
                        curvel = intpDISP[ii]
                        # Loop through velocity indices until the proper one is found
                        g = 0
                        while (g < vintsD):
                                # get values of current and next velocity increment
                                upvel = vinD[g+1]
                                lovel = vinD[g]
                                #print 'upvel = ' + str(upvel) + '   lovel = ' + str(lovel)
                
                                # if the velocity in the current layer falls between the current
                                # velocity increment and the next one, add a count to the current
                                # velocity increment
                                if (curvel < upvel) and (curvel > lovel):
                                        #print str(lovel) + ' < ' + str(curvel) + ' < ' + str(upvel)
                                        ph[g,i] = ph[g,i] + 1
                                        
                                        # The velocity bin was found so no need going through rest
                                        g = vintsD
                                
                                # Move to next velocity bin
                                g = g + 1
                                
                        # Move to next period bin
                        ii = ii + 1
                        i = i + 1
                
                # ---------- VELOCITY MODEL COUNTS -------------

                cnl = sample.nl
                
                # starting at the surface so flag for bottom layer is initially off
                bottomflag = 'off'
        
                # Loop through all layers of current model, starting at the surface boundary
                i = 0
                prevbound = 0
                while (i < cnl):
                        # if current layer is the bottom layer, set the lower boundary to be the 
                        # max depth of the model space and turn on the flag
                        if (i == (cnl-1)):
                                bound = maxz_m
                                bottomflag = 'on'
                        # if current layer is NOT the bottom layer, then make the boundary as the 
                        # next interface between this layer and the next one. Also pull out what 
                        # the velocity value is in the next layer
                        else:
                                bound = 1000*(sample.intf[i])
                                nextvel = sample.vs[i+1]
                        # Pull out the velocity of the current layer
                        curvel = sample.vs[i]
                        #curvel = reprunsVS[i,ind]
                
                        # If we are no longer at the surface working our way down, then we only 
                        # want to loop through the depth range or the current layer, so find the
                        # index value of the previous interface/boundary to use as starting point
                        if (prevbound != 0):
                                ii=0
                                while (ii < zints):
                                        testii = zin[ii]
                                        if (testii > prevbound):
                                                zcnt = ii
                                                ii = zints
                                        ii = ii + 1
                        else:
                                zcnt = 0
                
                        # starting at the depth for the top of the current layer, loop through
                        # depth increments counting up model hit points until we reach the current
                        # boundary/interface
                        k = zin[zcnt] 
                
                        # once the index for the current layer's velocity is found, no sense in
                        # finding it in every loop through so use a flag:
                        vindexflag = 'off'
                        while (k < bound):
                                # First we need to find the index value to place the counts for the 
                                # current layer's velocity
                                if (vindexflag == 'off'):
                                        # Loop through velocity indices until the proper one is found
                                        g = 0
                                        while (g < vints):
                                                # get values of current and next velocity increment
                                                upvel = vin[g+1]
                                                lovel = vin[g]
                                                #print 'upvel = ' + str(upvel) + '   lovel = ' + str(lovel)
                                        
                                                # if the velocity in the current layer falls between the current
                                                # velocity increment and the next one, add a count to the current
                                                # velocity increment and store the index as a variable
                                                if (curvel < upvel) and (curvel > lovel):
                                                        #print str(lovel) + ' < ' + str(curvel) + ' < ' + str(upvel)
                                                        vh[zcnt,g] = vh[zcnt,g] + 1
                                                        vindex = g
                                                
                                                        # the index was found so turn on the flag and force break
                                                        # out of this loop by raising the counter (g) to the limit
                                                        vindexflag = 'on'
                                                        g = vints
                                                g = g + 1
                                # while looping through depth increments within the current layer, add
                                # counts to each at the velocity of the layer using stored index 
                                else:
                                        vh[zcnt,vindex] = vh[zcnt,vindex] + 1
                        
                                # store the depth increment for use when reach end of current layer
                                zindex = zcnt        
                        
                                zcnt = zcnt + 1
                                k = zin[zcnt]        
                
                        if (pdf_connect == 'on'):
                                # As long as we are not in the bottom layer, then can add "dummy" points
                                # to connect current layer and the next one        at the interface
                                if (bottomflag == 'off'):
                                        # loop through velocity increments, starting at the index of the 
                                        # current layer's velocity. But first need to establish if have 
                                        # velocity DROP or velocity JUMP:
                        
                                        # For velocity jump:
                                        if (curvel < nextvel):
                                                gg = vindex + 1
                                                while (gg < vints):
                                                        curvel2 = vin[gg]
                                                        # as long as the value of the current velocity index is belo
                                                        if (curvel2 < nextvel):
                                                                vh[zindex,gg] = vh[zindex,gg] + 1
                                                        gg = gg + 1        
                                        
                                        # For velocity drop:
                                        elif (curvel > nextvel):
                                                gg = vindex - 1
                                                while (gg > 0):
                                                        curvel2 = vin[gg]
                                                        # as long as the value of the current velocity index is belo
                                                        if (curvel2 > nextvel):
                                                                vh[zindex,gg] = vh[zindex,gg] + 1
                                                        gg = gg - 1        
                
                        # before moving to the next layer, set the previous boundary variable 
                        # equal to the current boundary
                        prevbound = bound
                
                        i = i + 1
                cmod = cmod + 1

        normvh=np.zeros((zints,vints))
        maxline=np.zeros(zints)
        i=0
        while (i < zints):
                sum_nonnorm=sum(vh[i,:])
                normvh[i,:]=vh[i,:]/sum_nonnorm
                max_index, max_value = max(enumerate(normvh[i,:]), key=operator.itemgetter(1))
                maxline[i] = newvin[max_index]
                i = i + 1
        normph=np.zeros((vintsD,pints))
        maxlineDISP=np.zeros(intpnum)
        i=pstart_ind
        ii=0
        while (i < (pend_ind+1)):
                sum_nonnorm=sum(ph[:,i])
                normph[:,i]=ph[:,i]/sum_nonnorm
                max_index, max_value = max(enumerate(normph[:,i]), key=operator.itemgetter(1))
                maxlineDISP[ii]=newvinD[max_index]
                ii=ii+1
                i = i + 1
        return (vh, maxline, maxlineDISP, newpin, newzin, newvin, cutpin, newvinD, normvh, normph)
# ----------------------------------------------------------------------------
def setpdfcmaps(pdfcmap,rep_cnt,repeat,weight_opt,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax):
        
        totmaxhist=50
        itotmaxhist = totmaxhist + 0.0
        minH=0
        maxH=normvh.max()
        levels2 = np.linspace(minH, maxH, totmaxhist+1)
        Z2 = [[0,0],[0,0],[0,0]]
        maxDH=normph.max()
        levels22 = np.linspace(minH, maxDH, totmaxhist+1)
        
        ncmap=len(pdfcmap)
        i=0
        while (i < ncmap):
                curcmap=pdfcmap[i]
                
                if curcmap == 'GREYS':
                        CCC = [plt.cm.Greys(chosen/itotmaxhist) for chosen in range(totmaxhist)]
                        cmap2=LinearSegmentedColormap.from_list('Greys', CCC, N=totmaxhist, gamma=1.0)
                        CS33 = plt.contourf(Z2, levels2, cmap=cmap2)
                        CS22 = plt.contourf(Z2, levels22, cmap=cmap2)
                        linopt=['r-.','r--','black','ko']
                        
                elif curcmap == 'GREYS_rev':
                        CCC1 = [plt.cm.Greys(chosen/itotmaxhist) for chosen in range(totmaxhist)]
                        CCC=CCC1[::-1]
                        cmap2=LinearSegmentedColormap.from_list('Greys', CCC, N=totmaxhist, gamma=1.0)
                        CS33 = plt.contourf(Z2, levels2, cmap=cmap2)
                        CS22 = plt.contourf(Z2, levels22, cmap=cmap2)
                        linopt=['r-.','r--','black','wo']
                        
                elif curcmap == 'HOT':
                        CCC = [plt.cm.hot(chosen/itotmaxhist) for chosen in range(totmaxhist)]
                        cmap2=LinearSegmentedColormap.from_list('HOT', CCC, N=totmaxhist, gamma=1.0)
                        CS33 = plt.contourf(Z2, levels2, cmap=cmap2)
                        CS22 = plt.contourf(Z2, levels22, cmap=cmap2)
                        linopt=['w-.','w--','black','wo']
                        
                elif curcmap == 'HOT_rev':
                        CCC1=[plt.cm.hot(chosen/itotmaxhist) for chosen in range(totmaxhist)]
                        CCC=CCC1[::-1]
                        cmap2=LinearSegmentedColormap.from_list('HOT', CCC, N=totmaxhist, gamma=1.0)
                        CS33 = plt.contourf(Z2, levels2, cmap=cmap2)
                        CS22 = plt.contourf(Z2, levels22, cmap=cmap2)
                        linopt=['k-.','k--','black','ko']
                
                mkpdffigs(rep_cnt,repeat,weight_opt,curcmap,linopt,cmap2,CS33,CS22,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax)
                
                i = i + 1

# ----------------------------------------------------------------------------
def mkpdffigs(rep_cnt,repeat,weight_opt,curcmap,linopt,cmap2,CS33,CS22,newvin,newpin,newzin,newvinD,normvh,normph,vmin,vmax,instpd,dobs,wsig,maxz_m,abc,run,SAVEF,maxlineDISP,maxline,cutpin,pmin,pmax):

        # Dispersion Curve PDFs
        plt.close("all")
        plt.contourf(newpin, newvinD, normph, cmap=cmap2)
        cb=plt.colorbar(CS22)        
        plt.ylabel('VS (km/s)')
        plt.xlabel('Period (s)')
        plt.axis([pmin, pmax, 0, vmax,])
        obs, = plt.plot(instpd, dobs, linopt[3], zorder=100, markersize=7, 
                                        label='measured group velocities')
        if weight_opt == 'ON':
                obserr = plt.errorbar(instpd, dobs, yerr=wsig, zorder=20, linestyle="None", linewidth=2.0, ecolor='k')
        plt.legend(handles=[obs], fontsize=10)
        if rep_cnt == (repeat-1):
                figname = 'PDF_REP_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
        else:
                figname = 'PDF_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
        figtitle = 'Probability Density Function \n Transdimensional, '+str(maxz_m)+' m total depth'
        plt.title(figtitle, fontweight='bold', fontsize=14)
        Dfig = SAVEF + '/' + figname
        P.savefig(Dfig)
        meanD, = plt.plot(cutpin, maxlineDISP, linopt[0], linewidth=4.0, zorder=3, label='MEAN')
        meanD.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='black'),
                                   PathEffects.Normal()])
        plt.legend(handles=[meanD, obs], fontsize=10)
        figname = 'PDF_DISP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_meanline_'+abc[run]+'.png'
        Dfig = SAVEF + '/' + figname
        P.savefig(Dfig)
        
        
        # Velocity Model PDFs
        plt.close("all")
        plt.contourf(newvin, newzin, normvh, cmap=cmap2)
        cb=plt.colorbar(CS33)        
        plt.axis([vmin, vmax, 0, maxz_m])
        plt.gca().invert_yaxis()
        plt.xlabel('VS (km/s)')
        plt.ylabel('Depth (m)')
        if rep_cnt == (repeat-1):
                figname = 'PDF_REP_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
        else:
                figname = 'PDF_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_'+abc[run]+'.png'
        figtitle = 'Probability Density Function \n Transdimensional, '+str(maxz_m)+' m total depth'
        plt.title(figtitle, fontweight='bold', fontsize=14)
        Mfig = SAVEF + '/' + figname
        P.savefig(Mfig)
        meanline = plt.plot(maxline, newzin, linopt[1], linewidth=4.0)
        figname = 'PDF_TRANSD_lay_'+str(maxz_m)+'m_'+curcmap+'_meanline_'+abc[run]+'meanline.png'
        Mfig = SAVEF + '/' + figname
        P.savefig(Mfig)        

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------









