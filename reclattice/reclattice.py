import chetax.montecarlo as ct
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm
import bisect

np.random.seed(0)
palete = list(matplotlib.cm.tab10.colors)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def methyl_level(m) :
    if p.eta == 0 :
        mL = m
    else :
        mL = m&1;
        for i in range(1,M) :
            mL += (m>>i) & 1
    return mL*np.sign(m)
    
def neigh_act() :
    neigh = np.concatenate((a[  1:2,:], a[  2:,:]+a[  :-2,:], a[  -2:-1,:])) \
          + np.concatenate((a[:,1:2],   a[:,2:]  +a[:,:-2],   a[:,-2:-1]),axis=1)
    neigh = neigh.astype(np.float)
    neigh[::N-1,::N-1] = neigh[::N-1,::N-1]/2
    neigh[::N-1, 1:-1] = neigh[::N-1, 1:-1]/3
    neigh[ 1:-1,::N-1] = neigh[ 1:-1,::N-1]/3
    neigh[ 1:-1, 1:-1] = neigh[ 1:-1, 1:-1]/4
    return neigh   

def energy(evalChange=False) :
    '''Calculate the energy of the current receptors state.
        Arguments:
        evalChange: if true, calculate the energy associanted with reversing
            the activity or the ligand binding.
    '''
    global neigha,neighb,neighc,neigh
    neigha = (a[:-1,:]-0.5)*(a[1:,:]-0.5)
    neighb = (a[:,:-1]-0.5)*(a[:,1:]-0.5)
    neigh = neigha.sum() + neighb.sum()
    neighc = (neigh_act()-0.5)*(a-0.5)
    neigh = neighc[::N-1,::N-1].sum() \
          + 1.5*(neighc[::N-1,1:-1].sum() + neighc[1:-1,::N-1].sum()) +\
          + 2*neighc[1:-1,1:-1].sum()
    if p.symmetry==6 : 
        neigh += ((a[1:  :2,1:]-0.5)*(a[ ::2,:-1]-0.5)).sum()+\
                 ((a[1:-1:2,1:]-0.5)*(a[2::2,:-1]-0.5)).sum()
    neighC = neigh*p.C 
    mlevel = methyl_level(m)
    #If evalChange, calculate the energy change of each of the four changes
    #else calulate just one energy of the system at the state.
    changes = [False,True] if evalChange else [False]
    E = pd.DataFrame(columns=changes,index=changes)
    for changel in changes :
        for changea in changes :
             E.loc[changel,changea] = np.where(1-l if changel else l,
                    np.log(p[['K0','K1']]/p.L)[1-a if changea else a],0).sum() \
                + ((1-a if changea else a)*(mlevel-p.m0)).sum()*p.alpha \
                + (mlevel-p.m0).sum()*p.alpha0 \
                + neighC
    return E if evalChange else E.loc[False,False]
    
def entropy() :
    '''Calculate the entropy of the methylation distribution.
    '''
    f = np.bincount((4*m+l*2+a).astype(int).flatten())
    f = f[f.nonzero()]
    p = f/sum(f)
    return -sum(p*np.log(p))

def calculateFlows() :
    global Jsat, Tst, pm, pam, fam, flam, psat, pst
    #Counting of recptors in each state, the rows are the activity
    #and the columns are the m-state
    flam=np.histogramdd(np.array((l.flatten(),a.flatten(), m.flatten())).T,
          ((-0.5,0.5,1.5), (-0.5,0.5,1.5), 
            np.arange(-0.5,np.max(transitions[:,:2])+1)))[0]
    #fam = flam.sum(axis=0)

    fam=np.histogram2d(a.flatten(), m.flatten(), ((-0.5,0.5,1.5),
            np.arange(-0.5,np.max(transitions[:,:2])+1)))[0].astype(int)
    #Probability of each combination of a and m
    pam = fam/fam.sum()
    pm = pam.sum(axis=0)
    #Probability initial and final states of each transition
    #indexes: 0-initial or final states, 1-activity, 2-transition
    psat = np.array([pam[:,transitions[:,i]] for i in [0,1]])
    pst = psat.sum(axis=1)
    #Flows to in the direcition of adding or removing methyl groups
    Jsat = psat*Tsat
    #Transition rates between the states of the transitions
#    if (pst==0).any() :
#        print(Jsat.sum(axis=1))
#        print(pst)
#    else :
#        print(rn)
#    try :
    with np.warnings.catch_warnings() :
        np.warnings.filterwarnings('ignore')
        Tst = np.where(np.abs(pst)>1e-30, Jsat.sum(axis=1)/pst,Tsat.mean(axis=1))
#    except RuntimeWarning :
#        sys.exit(0)

def powerConsumption() :
    '''Calculate the energy comsuption by the flow between m-states.
    '''
    calculateFlows()
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((Jsat==0).any(axis=0),0,
                    (Jsat[0]-Jsat[1])*np.log(Jsat[0]/Jsat[1])).sum()

nRatiosSSErrors=0
nRatiosSS=0

def checkSteadyState(reset=False) :
    global nRatiosSS, nRatiosSSErrors 
    if reset :
        nRatiosSSErrors=nRatiosSS=0
        return
    ratioSS =  eVec[:,0].real*pm.mean()/np.where(pm!=0,pm,1)/eVec[:,0].real.mean()
    trustful = (pm>pm.mean()/2) & (pm*p.N**2>50)
    nRatiosSS += sum(trustful)
    nerrors0 = sum(((ratioSS>3/2) | (ratioSS<2/3)) & trustful)
    nerrors1 = sum(((ratioSS>3) | (ratioSS<1/3)) & trustful)
    if False : # nerrors0 or nerrors1 :
        nRatiosSSErrors += nerrors0
        if True :
            print ('***',nRatiosSSErrors, nRatiosSS,'***')
        if (nRatiosSSErrors>5 and nRatiosSSErrors>nRatiosSS/10) or nerrors1>0 :
            print(ratioSS)
            print((pm*p.N**2).astype(int))
            print(pm/pm.mean())
            print(eVec[:,0]/eVec[:,0].mean())
            print(eVec[:,1]/np.abs(eVec[:,1]).mean())
            print(pam)
            print(methyl_level(met_states))
            print(-1/eVal[1].real)
            print (nRatiosSSErrors, nRatiosSS)
            raise RuntimeError('Inconsistent m steady-states\nTau:'
                +str(-1/eVal[1].real)+'\n'+str(ratioSS))

def mRelaxationTime() :
    global T, eVal, eVec
    calculateFlows()
    T=np.zeros([len(met_states),len(met_states)])
    #Transition rate matrix: rows-to, colums-from
    T[transitions[:,1],transitions[:,0]] = Tst[0]
    T[transitions[:,0],transitions[:,1]] = Tst[1]
    np.fill_diagonal(T, -T.sum(0))
    eVal,eVec = np.linalg.eig(T)
    eVec /= abs(eVec).sum(axis=0)
    order=eVal.argsort()[::-1]
    eVal = eVal[order]
    eVec = eVec[:,order]
    if abs(eVal[0])>1e-10 or any(eVal[1:].real>0) :
        raise RuntimeError('Invalid eigenvalue:\n'+str(eVal))
    if any(eVec[:,0].mean()*eVec[:,0]<-1e-10) :
        raise RuntimeError('Neg. comp. on the 0 eigenVector:\n'+str(eVec[:,0]))
    if any(abs(eVec.sum(0)[1:]>1e-10)) :
        RuntimeError('Non-zero fluc. eigenvect:\n'+str(abs(eVec.sum(0)[1:]>1e-10)))
    if any(abs(eVec[:,0].imag)>1e-10) : 
        raise RuntimeError('Complex eigenvector:\n'+str(eVec[:,0]))
    return -1/eVal[1].real

#Read the simulation parameters from file
def readParamFile(filename) :
    global parF, parFileOrder
    parF = {}
    parFileOrder = pd.Series(['N','M','symmetry'])
    newParNames = {'n':'N', 'numSitM':'M', 'alfa':'alpha', 'maxGama': 'maxXi', 
        'nMetroPassos':'nMetroSteps', 'nRepMut':'nScans', 'mutante':'mutant'}
    with open(filename, 'r') as file :
        for line in file :
            line = line.replace('[ ','').replace(' ]','').strip().split()
            if len(line)>1 and line[0][0]=='#' and line[0]!='#Time':
                parName = line[0][1:].strip()
                if parName in newParNames : #Make compatible with old par names
                    parName = newParNames[parName]
                if not (parName==parFileOrder).any(): 
                    #Store the order of the parameters in the file
                    parFileOrder.loc[len(parFileOrder)] = parName
                for i in range(1,len(line)) :
                    parNameN = parName+str('' if len(line)==2 else i-1)
                    if parNameN not in parF :
                        li = line[i].strip()
                        parF[parNameN] = eval(li) if is_number(li) else li
    if len({'N','M'}.intersection(parF)) != 2 :
        raise Exception('N or M not found in parameter file')
    if 'symmetry' not in parF :
        parF['symmetry'] = 4

def writeParamFile(file) :
    print('#Time',datetime.datetime.fromtimestamp(time.time()).strftime(
            '%Y-%m-%d %H:%M:%S'), file=file)
    for par in parFileOrder :
        if par!='Time' :
            print('#'+par, parF[par] if par in parF else 
                 pd.Series({p:v for p,v in parF.items() 
                 if p.startswith(par)}).sort_index().values, file=file) 

#These parameters must be the same as in the c-function set_parameters
par_names = ["M", "fixedM0", "fixedM1", "eta", "g", "kr","kb", "K0", "K1", 
        "C", "alpha","alpha0", "m0", "L", "dt"]
def updateParameters(newParams={}) :
    global parF, p, par
    for name in newParams :
        parF[name] = newParams[name]
        if not (name==parFileOrder).any() :
            parFileOrder.loc[len(parFileOrder)] = name
    p = pd.Series({n:parF[n] for n in parF if is_number(parF[n])})
    #Get the paramet from parF and set 0 for the ones not present
    par = {name:(parF[name] if name in parF else 0) for name in par_names}
    not_numbers = {k:v for k,v in par.items() if not is_number(v)}
    if len(not_numbers)!=0 :
        raise Exception('Problem with the parameters'+str(not_numbers))
    ct.set_parameters(**par)

def prepStateMatrices() :
    #Create and define the matrices used in the simulation
    global parF, a, l, m, N, M, fixedM0, fixedM1, met_states, seq_states,\
           add_1met, transitions, Tsat, mStatesColumns, siteMethylationColumns
    N = parF['N']
    M = parF['M']
    a=np.random.randint(0,2,N*N,np.uint8).reshape(N,N)
    l=np.random.randint(0,2,N*N,np.uint8).reshape(N,N)
    m = np.full((N,N),parF['initialMState'],np.uint64) if 'initialMState' in parF else \
        np.random.randint(0,M+1 if p.eta==0 else 1<<M,(N,N),np.uint64)
    fixedM0 = np.uint64(par['fixedM0'])
    fixedM1 = np.uint64(par['fixedM1'])
    if p.eta!=0 :
        m &= ~fixedM0
        m |= fixedM1
    ct.define_state_variables(N, parF['symmetry'], a, l, m)
    #The add_1met conect states that have one extra methyl group
    if p.eta==0 :
        met_states = np.arange(M+1)
        seq_states = met_states
        add_1met = met_states-np.transpose([met_states]*len(met_states))==1
    else :
        met_states = np.arange(2**M)
        seq_states = 2**np.arange(M+1)-1
        arrS = np.array([met_states]*(2**M)).T
        add_1met = methyl_level(met_states-arrS)*methyl_level(met_states^arrS)==1
    mStatesColumns = ['Pm'+str(m) for m in met_states]
    siteMethylationColumns = ['Sm'+str(m) for m in range(M)]
    add_1met = add_1met.astype(int)
    #Allowed transitions: beween states from the columns 0 to 1
    #Forward and reverse sequential transistions marked in columns 2 and 3
    transitions = np.argwhere(add_1met)
    if p.eta==0 :
        transitions = np.concatenate((transitions,[[1,1]]*len(transitions)),axis=1)
        trans_seq_m = transitions
    else :
        delta=transitions[:,0]^transitions[:,1]
        transitions = np.concatenate((transitions,np.array([
            (transitions[:,0]&(delta>>1)!=0) | (delta==1),
            (transitions[:,1]&(delta<<1)==0) | (delta==2**(M-1))]).T),axis=1)
    #Transition rates between the states
    krEta = np.where(transitions[:,2],p.kr,p.kr*p.eta)
    kbEta = np.where(transitions[:,3],p.kb,p.kb*p.eta)
    #Tsa are the rates that of the transtions m -> m+-1
    #Indices: 0-(0 add m, 1 subtract m), 1-activity, 2-transition
    Tsat = np.array([[krEta, kbEta * p.g * np.exp(-p.alpha-p.alpha0)],
                     [krEta * p.g * np.exp(p.alpha0), kbEta]])
    #TaPlus/Minus are the rates that add or subract 1 to m
    #The rows are the activity state and the columns are the mStates
    #TaPlus = np.array([krEta, kbEta * p.g * np.exp(-p.alpha-p.alpha0)])
    #TaMinus= np.array([krEta * p.g * np.exp(p.alpha0), kbEta])

def probChange() :
    eChange = energy(True) / N**2
    eChange -= eChange.loc[False,False]
    probEChange = np.exp(-max(eChange.loc[True,False],eChange.loc[False,True]))
    probMChange = (np.where(m<M,1-a,0)*p.kr+np.where(m>0,a,0)*p.kb).mean()
    print("\n",probEChange,probMChange)
    
def plotDyn() :
    if "fig" not in plotDyn.__dict__ :
        plotDyn.fig, plotDyn.plot = plt.subplots(2,2)
        #plotDyn.met_mat=plotDyn.plot[0][0].matshow(m,cmap='cool')
        plotDyn.plot[0][0].set_title('Met')
        plotDyn.act_mat=plotDyn.plot[0][1].matshow(a,cmap='binary_r')
        plotDyn.plot[0][1].set_title('Act')
        plotDyn.plot3=plotDyn.plot[1][0]
        plotDyn.ax2 = plotDyn.plot3.twinx()
        plotDyn.plot4=plotDyn.plot[1][1]
        plt.pause(0.0001)
        plotDyn.data=pd.DataFrame(columns=['t','m','a','power'])
    plotDyn.fig.suptitle('t: %.1f  [L]:%f'%(t,par['L']) )
    #plotDyn.met_mat.set_data(mlevel)
    rgb = np.array([mlevel/mlevel.max(),1-mlevel/mlevel.max(),np.zeros(a.shape)])
    rgb = rgb.transpose([1,2,0])
    #rgb = (rgb+2)/(rgb.max()+2)
    plotDyn.plot[0][0].imshow(rgb)
    plotDyn.act_mat.set_data(a)
    if t>0 :
        plotDyn.data = plotDyn.data.append({'t':t,'m':mlevel.mean(),'a':a.mean(),
                'power':powerConsumption()},ignore_index=True)
    plotDyn.plot3.clear()
    plotDyn.plot3.plot(plotDyn.data.t, plotDyn.data.m,'-b',label='m')
    plotDyn.ax2.clear()
    plotDyn.ax2.plot(plotDyn.data.t, plotDyn.data.a,'-r',label='a')
    plotDyn.plot4.clear()
    plotDyn.plot4.plot(plotDyn.data.t, plotDyn.data.power,'-c',label='power')
    plt.pause(0.00001)

def performLScan(showDynamics=True) :
    global previous, mlevel, t
    '''Simulate varing the value of L at speciefic times, defined at the matrix steps.
    '''
    updateParameters({'L':steps.L[0]})
    checkSteadyState(reset=True)
    #print(p)
    E = energy()
    si = 0; di = 0 #index of steps and dyn
    t = steps.t[0]
    steps.loc[1:,'Lpre'] = steps.L[:-1].values     
    for c in ['m','m_std','m_cov','apre','lpre','apremean','apre2mean','apos',
              'lpos','mpos','tpos','fracSeq','entropy','powCons','mTau','eneCons',
              'taupeak','apeak','taucross','aFinal']+mStatesColumns+siteMethylationColumns : 
        steps[c]=0
    dyn_columns = ['L','a','l','m','m_std','power']+mStatesColumns+siteMethylationColumns
    for c in dyn_columns : dyn[c]=0
    while (di < len(dyn)) or (si < len(steps))  :
        print("\b\b\b\b\b\b\b\b\b%.1f "%t,end='')
        ts = steps.t[si] if si<len(steps) else 1e10 
        td = dyn.t[di] if di<len(dyn) else 1e10
        dt,dE = ct.MCsimulation(parF['nMetroSteps'], min(td,ts)-t)
        E += dE
        mlevel = methyl_level(m) 
        t += dt
        #print("\ntdtstdisi",td,ts,t,di,si,mlevel.mean().mean())
        #if td<ts :
        if td <= t+dt/2 :
            hist = np.histogram(m,list(met_states)+[met_states[-1]+1])
            siteMethylation = [hist[0][hist[1][:-1]&(2**site)!=0].sum()/N/N 
                    for site in range(M)] 
            dyn.loc[di,dyn_columns] = \
                [p.L, a.mean(), l.mean(), mlevel.mean(), mlevel.std(), powerConsumption()] + \
                list(hist[0])+siteMethylation
            #print(dyn.t[di], t)
            di += 1
        #else :
        if ts<=t+dt/2 :
            E1 = energy()
            if abs(E-E1) > 1e-7*abs(E+E1) :
                raise Exception('Error in the energy calculation '+str(E1)+' '+
                            str(E)+' t='+str(t))
            ##################################################
            #Save the state before the L change (pre-values).#
            ##################################################
            recentAct = dyn.a[max(0,di-6):max(0,di-1)]
            meanAct = recentAct.mean()
            N1 = N*(N-1)
            covNeig = (
                np.cov(mlevel[1:,:].reshape(N1), mlevel[:-1,:].reshape(N1))+
                np.cov(mlevel[:,1:].reshape(N1), mlevel[:,:-1].reshape(N1)))[0,1]
            N2 = N1//2-(N-1)
            covNeig = covNeig / 2 if parF['symmetry']==4 else \
                ((np.cov(mlevel[1:  :2,1:].reshape(N1//2), mlevel[ ::2,:-1].reshape(N1//2))+
                 np.cov(mlevel[1:-1:2,1:].reshape(N2)  , mlevel[2::2,:-1].reshape(N2))
                 )[0,1]/2 + covNeig)/3
            powCons = powerConsumption()
            relaxTime = mRelaxationTime()
            if si>0 and (steps.t[si]-steps.t[si-1])>10:
                checkSteadyState()
            steps.loc[si,['m','m_std','m_cov','apre','lpre','apremean',
                'apre2mean','fracSeq','entropy','powCons','mTau','eneCons']] =\
                [mlevel.mean(),mlevel.std(),covNeig,a.mean(),l.mean(),
                        meanAct,(recentAct**2).mean(),
                        np.in1d(m.reshape(N*N),seq_states).sum()/N/N,entropy(),
                        powCons, relaxTime,  powCons*relaxTime]
            hist = np.histogram(m,list(met_states)+[met_states[-1]+1])
            steps.loc[si,mStatesColumns] = list(hist[0])
            steps.loc[si,siteMethylationColumns] = siteMethylation
            ######################################################
            #Change [L], evolve the time and save the pos values.#
            ######################################################
            alm = pd.DataFrame(columns=['a','l','m'])
            pdt = par['dt']
            updateParameters({'L': steps.L[si], 'dt':4/p['nMetroSteps']})
            if par['dt']>pdt :
                raise RuntimeError('nMetroSteps must be at least: %.0f'%
                    (4/pdt))
            a0=a1=a2 = a.mean()
            m1=m2 = mlevel.mean()
            l1=l2 = l.mean()
            t0=t
            #Save the first peak
            while (a0-a1)*(a1-a2)>=0 :
                dt,dE = ct.MCsimulation(parF['nMetroSteps'], p['dt'])
                (a0, a1, a2) = (a1, a2, a.mean())
                (m1, m2) = (m2, mlevel.mean())
                (l1, l2) = (l2, l.mean())
                E += dE
                t += dt
                #print("%.5f %.2f %.2f %.2f"%(t,a0, a1, a2))
            #print(t-t0)
            #Save the data that has biggest, negative or positive, variation
            steps.loc[si,['apos','lpos','mpos']] = [a1,l1,m1]
            steps.loc[si,'tpos'] = t-t0
            updateParameters({'dt':pdt})
            E = energy()
            #####################################################
            # Save the information regarding the previous step. #
            #####################################################
            if si>0 :
                delL = steps.L[si-1]-steps.Lpre[si-1]
                #Find the first activity peak afert change in [L]
                for iPeak in range(steps.idyn[si-1], steps.idyn[si]-2) :
                    if (dyn.a[iPeak]-dyn.a[iPeak+1])*delL<0 and \
                       (dyn.a[iPeak]-dyn.a[iPeak+2])*delL<0 :
                       break
                #Find the first time activity cross the stead state value
                for ix in range(iPeak, steps.idyn[si]-1) :
                    if (dyn.a[ix]-meanAct)*(dyn.a[ix+1]-meanAct)<0 :
                        break
                steps.loc[si-1,['taupeak','apeak','taucross','aFinal']] = \
                    [dyn.t[iPeak]-steps.t[si-1], dyn.a[iPeak],
                     dyn.t[ix:ix+2].mean()-dyn.t[iPeak], meanAct]
            #Freezes methylation and evolves a and l to the steady-state
            #krb = {'kr':p.kr, 'kb':p.kb}
            #updateParameters({'L': steps.L[si], 'kr':0, 'kb':0})
            #dt,dE = ct.MCsimulation(parF['nMetroSteps'], 10*p['dt'])
            #mlevel = methyl_level(m) 
            #steps.loc[si,['apos','lpos','mpos']] = [a.mean(),l.mean(),mlevel.mean()]
            #updateParameters(krb)
            #E = energy()
            si += 1
        if showDynamics and di%10==0:
            plotDyn()
    #sys.exit()

def prepareLScan(scantype='GammaXi', bidirection=True, tau=None) :
    '''Prepare the matrices steps and dyn, used by performLScan.
       
       Create the following global matrices:
            steps: will contain the values of the simulation data before and 
                after each step.
            dyn: will contain the values of the simulation data at 
                equaly spaced instants.
       Parameters:
       ----------
            scantype: 
                '0Response': change L from 0 to certain successifully higher 
                            values and back to 0;
                'GammaXi': change L by 1/20, and then by 2 and wait for 20 s 
                        and change again.
    '''
    global steps, dyn
    globals()['scantype'] = scantype 
    GammaXi = scantype=='GammaXi'
    Response = scantype=='0Response'
    if not GammaXi and not Response :
        raise ValueError('Invalid value of GammaXi')
    #Number of L steps
    nL = int(np.log10(p.Lfim/p.Lini)/np.log10(p.fatorL)+1.5)
    #t and L are build as a [2,nL] 2D-arrays, to be reshaped to [2*nL] 1D-arrays
    #Instants in which the concentrationg change.
    t = [[(i+1/(20 if GammaXi else 2))*p.LdeltaT,(i+1)*p.LdeltaT] for i in range(nL)]
    #Ascending values of L, attributed at the instants above
    L = [([L,L*(1+p.deltaLResp)] if GammaXi else [0,L])
            for L in np.logspace(np.log10(p.Lini), np.log10(p.Lfim),nL)]
    #Reshape and zip the t and L; also produce the descending L sequence
    steps=[np.concatenate([[-p.LdeltaT]] + t + 
            list(max(max(t))+np.array(t) if bidirection and GammaXi else [])),
           np.concatenate([[p.Lini]] + L + 
            list(p.Lfim*p.Lini/np.array(L) if bidirection and GammaXi else []))]
    steps = pd.DataFrame(steps, index=['t','L']).transpose()
    if tau is None :
        tau = 0.25/(par['kb']*(1+par['g']*np.exp(-par['alpha0']-par['alpha']))+
              par['kr']*(1+par['g']*np.exp(+par['alpha0'])))
    dyn = pd.DataFrame({'t': np.arange(min(steps.t), max(steps.t), tau)})
    steps['idyn'] = [bisect.bisect_left(dyn.t,t) for t in steps.t]

def averageLScans(nScans, showDynamics=False) :
    global pSteps, pResp, mDyn, pSteps1, pResp1, steps
    #pSteps = {}
    pSteps = pd.DataFrame()
    pResp = {}
    t_exec = time.time()
    initialParameters = parF.copy()
    for i in range(nScans) :
        updateParameters(initialParameters)
        prepStateMatrices()
        print(' ',i,'\b\b\b\b\b\b\b',end='')
        performLScan(showDynamics)
        #pSteps[i] = steps.copy()
        steps['dlogL'] = np.log10(steps.L/steps.Lpre)
        steps['isPulse'] = (steps.dlogL.abs()<1.1*np.log10(1+p.deltaLResp)) & \
                           (steps.dlogL.abs()>1e-3)
        steps.loc[:len(steps)-2,'dlogL']+=steps.dlogL[1:].values
        steps['g'] = (-(steps.apos-steps.apre)*steps.Lpre/steps.apremean/
                    (steps.L-steps.Lpre)).where(steps.isPulse,0)
        steps[:] = np.where(steps<-1e10,-1e10,np.where(steps>1e10,1e10,steps)) 
        pSteps = pSteps.append(steps.assign(ii=i).set_index('ii',append=True).swaplevel(0,1))
        pResp[i] = steps.copy()
        #gSNR_noise(pResp[i])
        #pResp[i] = pResp[i][~pResp[i].gSNR.isnull()]
        if i==0 :
            mDyn = dyn.copy()
        else :
            mDyn += dyn
    print(" %.3f"%(time.time()-t_exec), end=' ')
    mDyn /= nScans
    #pSteps = pd.concat(pSteps.values(), keys=pSteps.keys())
    #pSteps = pd.Panel(pSteps)
    pResp = pd.concat(pResp.values(), keys=pResp.keys())
    #pResp = pd.Panel(pResp)

#def gSNR_noise(pSteps,i0=None) : 
#    '''Calculate the adapation error and the gSNR. (deprecated definitions)

#    pSteps is a dataframe if i0=None otherwise it is a multindex dataframe
#    Works better with the averages of several simulations than calculating
#    over simulation and them taking the average.
#    '''
#    #Select the steps to be used to calculate the reponse
#    steps = pSteps if i0 is None else pSteps.loc[i0]
#    fl = steps.L/steps.Lpre
#    iResp = fl[(0.5<fl) & (fl<2) & (fl!=1)].index.values
#    piResp = iResp if i0 is None else [(i0,i1) for i1 in iResp]
#    #Find the noise before each step. Averaging between simulation corrrect
#    #small noise sampling within each simulation
#    #steps['noisepre'] = np.NAN
#    s = steps.loc[iResp]
#    pSteps.loc[piResp,'noisepre'] = np.sqrt((s.apre2mean-s.apremean**2))
#    #Calculate the gSNR of each step
#    #steps['gSNR'] = np.NAN
#    s = steps.loc[iResp]
#    pSteps.loc[piResp,'gSNR'] = -(s.apos-s.apre)*s.Lpre/s.noisepre/(s.L-s.Lpre)
#    pSteps.loc[piResp,'g'] = -(s.apos-s.apre)*s.Lpre/s.apremean/(s.L-s.Lpre)
#    #Caculate the delta log10 L of each response
#    #steps['dlogL'] = np.NAN
#    if len(iResp) > 0 :
#        pSteps.loc[piResp[:-1],'dlogL'] = np.log10(s.L[iResp[1:]].values/s.L[iResp[:-1]])
#        pSteps.loc[piResp[0] ,'dlogL']  = steps.dlogL[iResp[0]]/2
#        pSteps.loc[piResp[-1], 'dlogL'] = np.log10(s.L[iResp[-1]]/s.L[iResp[-2]])/2
#        s = steps.loc[iResp]
#        return s.apremean.std(), sum(s.gSNR*abs(s.dlogL))
#    else :
#        return s.apremean.std(), np.NAN

def calc_Gamma_xi(pSteps,i0=None) :
    s = pSteps if i0 is None else pSteps.loc[i0]
    s = s.loc[s.isPulse]
    if len(s)>0  :
        return sum(s.g*abs(s.dlogL)),\
            (s.apremean.max()-s.apremean.min())/(s.apremean.max()+s.apremean.min())
    else :
        return np.NAN,\
            (steps.apremean.max()-steps.apremean.min())/(steps.apremean.max()+steps.apremean.min())

def plot_mDist(data, selectCol=None,  selectVals=None, maxY=[1,1], maxX=[None,None], 
            power10Leg=False, title=None, xticks=[None,None],paneLabels=None, 
            paneTitle=None, sidePane=True, outName='mDist.pdf',
            paneSpace=0.25, topSpace=0.35, bottomSpace=0.45,
            legFormat=None) :
    '''Bar plots of  the methylation level of mSites and the distribution of mLevels.

       Parameters:
       data: DataFrame with columns containing simulation variables and with
            different time at each row. Must contain the columns on
            ct.mStatesColumns.
       selectCol, selectVals: graphics are done only for rows whose value of 
            selectCol are present in selectVals.
       maxY: maximun value of y axis, respectively, in the plots of mSite
            methylation and mLevel distribution. Plot are not shown if the
            corresponding maxY is None.
       xTicks: labels of the xticks for [mSite, mLevel]. If one of the ticks is
            is None, the corresponding bar and xvalue are supressed.
       power10Leg: the legend at the left of each plot must be in log10 format?
    '''
    global distMLevels, ocupMSites
    dataPlot = data[np.round(data[selectCol],5).isin(selectVals)] \
            if selectCol is not None else data
    mStatesColumns = [c for c in data.columns if c[:2]=='Pm']
    siteMethylationColumns = [c for c in data.columns if c[:2]=='Sm']
    M = len(siteMethylationColumns)
    mData = dataPlot[mStatesColumns]
    nRows,nCols = mData.shape
    cols = np.arange(nCols)
    rows = np.arange(nRows)
    mSites = np.arange(M)+1
    mLevels = np.arange(M+1)
    mLevelCols = methyl_level(cols)
    #ocupMSites = pd.DataFrame([mData.iloc[:,cols&(2**(site-1))!=0].sum(axis=1) 
    #             for site in mSites]).T / ct.N**2
    ocupMSites = dataPlot[siteMethylationColumns]
    distMLevels= pd.DataFrame([mData.iloc[:,mLevelCols==mLevel].sum(axis=1) 
                 for mLevel in mLevels]).T / mData.sum(1).mean()#ct.N**2
    print()
    print(ocupMSites)
    print(distMLevels)
    nPlotCols = (maxY[0] is not None) + (maxY[1] is not None)
    plt.figure(figsize=(paneSpace/nPlotCols+nPlotCols,nRows+0.5))
    totalSpace=topSpace+bottomSpace+nRows
    plt.subplots_adjust(left=.25/nPlotCols, right=1-paneSpace/nPlotCols, 
            top=1-topSpace/totalSpace, bottom=bottomSpace/totalSpace)
    if title is not None :
        plt.suptitle(title,fontsize=10)
    plotCol=1
    #Transform the selectVals in integer if round==0 for better plotting
    vals = np.round(dataPlot[selectCol],5)
    if (np.modf(vals)[0]==0).all() :
        vals = vals.astype(int)
    for ptype,x,data,xlabel in ([[0,mSites,ocupMSites,'mSite']] if maxY[0] is not None else []) + \
                         ([[1,mLevels,distMLevels,'mLevel']] if maxY[1] is not None else []) :
        for row in rows :
            plt.subplot(nRows, nPlotCols, row*nPlotCols+plotCol)
            ix = x if xticks[ptype] is None else np.isin(x,xticks[ptype][0])
       
            plt.bar(x[ix], data.iloc[row][ix], width=0.9, color=[palete[i] for i in x])
            mX = M if maxX[ptype] is None else maxX[ptype]
            mY = maxY[ptype] 
            if mY<0 :
                #Find the y scale multiple to 0.1
                mY=np.ceil(data.values.max()*10)/10
                if np.modf(mY)[0]==0 :
                    mY = int(mY)
            plt.xlim(0.5-ptype,mX+0.5)
            plt.ylim(0,mY)
            plt.xticks([])
            plt.yticks([mY])
            for border in ['top','right'] :
                plt.gca().spines[border].set_visible(False)
            if plotCol== nPlotCols and sidePane :
                posX = (mX + 3*len(x)/nCols)*0.7
                if legFormat is None :
                    legFormat='$10^{%.0f}$' if power10Leg else '%f'
                plt.text(posX, 0.7*mY, legFormat%(np.log10(vals.iloc[row]) 
                    if power10Leg else vals.iloc[row]) if paneLabels is None else 
                    paneLabels[row], horizontalalignment='left')
                if row==0 :
                    plt.text(posX,0.9*mY,selectCol if paneTitle is None else paneTitle,
                                 horizontalalignment='left')
        if xticks[ptype] is None :
            plt.xticks(x[x<=mX])
        else :
            plt.xticks(*xticks[ptype])
        plt.xlabel(xlabel)
        plotCol+=1
    #plt.tight_layout()
    plt.savefig(outName)
    plt.close('all')
           

