#import chetax.chetax as ct
import reclattice.reclattice as ct
import numpy as np
import pandas as pd
import sys

def isMutantFitter(Gamma, xi, act, parFittest) :
    allowedXi   = xi  <= ct.parF['maxXi']
    allowedAct  = act >  1/3
    betterXi    = xi    <= parFittest['xi']
    betterAct   = act   >= parFittest['act']
    betterGamma = Gamma >= parFittest['Gamma']
    badFitXi  = parFittest['xi']  > ct.parF['maxXi']
    badFitAct = parFittest['act'] <  1/3
    #If the fittest mutant is bad, Gamma doesn't need to improve.
    #Xi and act, can either be allowed or improve if the fittest is bad
    return (betterGamma or badFitXi   or  badFitAct) and \
           ((betterXi  and badFitXi ) or allowedXi ) and \
           ((betterAct and badFitAct) or allowedAct)

def mutantsEvolution(prefixArq, mutants) :
    '''Explore the parameter space, looking for the fittest mutant.

    Arguments:
    prefixArq -- the begining of the files that will be read.
    mutants -- if integer, the number of evolving mutants. If a list,
        each element must be a dictionary with the parameters that are
        different from the one read from file.
    '''
    global resp, steps
    ct.readParamFile(prefixArq+'_param.dat')
    ct.parF['mutant'] = 0
    ct.parF['L'] = ct.parF['Lini']
    if 'Fittest' in ct.parF :
        del ct.parF['Fittest']
    parFromFile = ct.parF
    ct.updateParameters()
    ct.prepStateMatrices()
    ct.prepareLScan('GammaXi')
    #File containing several measures, averaged over the scans of each mutant
    fileMutants = open(prefixArq+'_mutants.dat', 'w')
    ct.writeParamFile(fileMutants)
    print('mutant\txi\txi_err\tGamma\tGamma_err',file=fileMutants)
    #File containng the results of each scan
    fileScans = open(prefixArq+'_scans.dat','w')
    ct.writeParamFile(fileScans)
    print('mutant\tscan\tK0\tK1\tC\tm0\tkr\talpha\talpha0\teta\tg\txi\tGamma'
        #+ '\txi0\tGamma0\tnoise
        + '\tmeanA\tmeanM\trangeM\tsigmaM'
        + '\tcovarMViz\tfracSeq\tentropy\teneCons\tmTau\tpowCons\ttaupeak'
        + '\tapeak\ttaucross\tafinal',file=fileScans)
    #File containing the response to the stimulus change, averaged over the 
    #scans of each mutant
    fileResp = open(prefixArq+'_resp.dat', 'w')
    #Change the mutants parameters and perform nScans scans for each mutant
    searchCandidate = False
    for imut in range(mutants if isinstance(mutants,int) else len( mutants)) :
        nCandidate = 0
        if isinstance(mutants, int) :
            while searchCandidate  :
                newPar = parFittest.copy()
                newPar['mutant'] = imut
                changedPar = False
                evolvingParameters = ['C','alpha','m0','kr']
                #if parFittest['alpha0']!=0 :
                #    evolvingParameters += ['alpha0']
                while not changedPar :
                    for p in evolvingParameters :
                        if np.random.rand() < 0.5 :
                            newPar[p]=(parFittest[p]*1.2**np.random.normal(0,1)
                                    if p!='m0' and p!='alpha0' else 
                                       parFittest[p]+0.2*np.random.normal(0,1))
                            changedPar = True
                if newPar['kr']>2 :
                    newPar['kr'] = 2
                newPar['alpha0'] = -newPar['alpha']/2
                ct.updateParameters(newPar)
                #ct.writeParamFile(fileScans)
                #fileScans.flush()
                ct.performLScan(False)
                #xi0,Gamma0 = ct.gSNR_noise(ct.steps.copy())
                Gamma,xi = ct.calc_Gamma_xi(ct.steps.copy())
                act = ct.steps.apremean.mean()
                print(nCandidate,"x%.4f g%.1f a%.2f"%(xi,Gamma,act),
                        "c%(C).1f a%(alpha).1f m0%(m0).2f kr%(kr).2f"%ct.parF)
                #Is this mutant a good candidate?
                if isMutantFitter(Gamma, xi, act, parFittest) :
                   searchCandidate=False
                nCandidate += 1
            searchCandidate=True
        else :
            newPar = parFromFile.copy()
            newPar.update(mutants[imut])
            newPar['alpha0'] = -newPar['alpha']/2
            ct.updateParameters(newPar)
            ct.prepStateMatrices()
        nScans = ct.parF['nScans'] 
        ct.averageLScans(nScans)
        #Data regarding the average of all scans of that mutant
        #meanResp = ct.pResp.mean(axis=0)
        meanResp = ct.pResp.mean(level=[1,1])
        #sdevResp = ct.pResp.std(axis=0)
        sdevResp = ct.pResp.astype(float).std(level=[1,1])
        #mXi0,mGamma0 = ct.gSNR_noise(meanResp)
        mGamma,mXi = ct.calc_Gamma_xi(meanResp)
        #Data regarding the mean response, among the several stimlus changes,
        #within each individual scan
        scans = pd.DataFrame(columns=meanResp.columns)
        #for i,resp in ct.pResp.iteritems() :
        #for i,resp in ct.pResp.groupby(level=0):
        for i in ct.pResp.index.levels[0] :
            #xi0,Gamma0 = ct.gSNR_noise(ct.pResp,i)
            Gamma,xi = ct.calc_Gamma_xi(ct.pResp,i)
            resp = ct.pResp.loc[i]
            r = resp.mean()
            scans.loc[i] = r
            scans.loc[i,'xi'] = xi
            scans.loc[i,'Gamma'] = Gamma
            #scans.loc[i,'xi0'] = xi0
            #scans.loc[i,'Gamma0'] = Gamma0
            print(imut, i, ct.parF['K0'], ct.parF['K1'], ct.parF['C'],
                ct.parF['m0'], ct.parF['kr'], ct.parF['alpha'], 
                ct.parF['alpha0'], ct.parF['eta'], ct.parF['g'], xi, Gamma, 
                #xi0, Gamma0, r.noisepre, 
                r.apremean, r.m,resp.m.max()-resp.m.min(),
                r.m_std, r.m_cov, r.fracSeq, r.entropy, r.eneCons, r.mTau,r.powCons,
                r.taupeak, r. apeak, r.taucross, r. aFinal,
                sep='\t', file=fileScans)
        fileScans.flush()
        print(imut,mXi,scans.xi.std()/np.sqrt(nScans),
              mGamma,scans.Gamma.std()/np.sqrt(nScans), sep='\t',file=fileMutants)
        fileMutants.flush()
        ct.updateParameters({'Gamma':mGamma, 'xi':mXi})
        ct.writeParamFile(fileResp)
        print('L0\tL1\tg\tg_err',file=fileResp)
        meanResp.loc[:,'g_err'] = sdevResp['g']/np.sqrt(nScans)
        meanResp.loc[:,['Lpre','L','g','g_err']].to_csv(fileResp,
            header=False,index=False,sep='\t')
        fileResp.flush()
        #print(mut, end=' ')
        mAct = meanResp.apremean.mean()
        #The variable mutant is one of the function parameters
        if isinstance(mutants,int) :
            if 'parFittest' in locals() :
                print("%.3f %.1f"%(parFittest['xi'], parFittest['Gamma']),end=' ') 
            #Check if the last mutant is the fittest up to now
            if ('parFittest' not in locals()) or \
                    isMutantFitter(mGamma, mXi, mAct, parFittest) :
                parFittest = ct.parF.copy()
                parFittest['Gamma'] = mGamma
                parFittest['xi'] = mXi
                parFittest['act'] = mAct
                print('* ', end='')
            #Search for a candidate of fittest mutant
        print(imut, "x%.3f g%.1f a%.3f\n"%(mXi, mGamma, mAct))
    if isinstance(mutants,int) :
        print('\n#Fittest_mutant', file=fileMutants)
        ct.updateParameters(parFittest)
        ct.writeParamFile(fileMutants)
    fileMutants.close()
    fileResp.close()
    fileScans.close()

etaScan = [{'eta':eta} for eta in np.linspace(1e-10,1,11)]
alphaScan = [{'alpha':alpha} for alpha in np.linspace(-1.2,-2.7,6)]


if len(sys.argv)==2 :
    fileName = sys.argv[1].replace("_param.dat","")
    ct.readParamFile(fileName+"_param.dat")
    scanType = ct.parF['scanType'].lower()
    if scanType=='fittestsearch' :
        mutantsEvolution(fileName,3)
    elif scanType=='etascan':
        mutantsEvolution(fileName,etaScan)
    elif scanType=='alphascan' :
        mutantsEvolution(fileName,alphaScan)
    else :
        print('scanType',ct.parF['scanType'],'not known.')
else :
    print("The command line must contain the name of the parameters file")



