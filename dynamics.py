import reclattice.reclattice as ct
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
import time
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
#plt.rc('font', **{'family':'serif','serif':['palatino'],'size':10})#'family':'normal', 'weight': 'normal'
plt.rcParams.update({'text.latex.preamble' : [r'\usepackage{amsmath}']})
palete = list(matplotlib.cm.tab10.colors)
#palete[1],palete[9] = palete[9],palete[1]

def sigmoid(x,a,b,x0,X) :
    return a + (b-a)*(1+np.tanh((x-x0)/X))/2

def invSigmoid(y,a,b,x0,X) :
    return x0 + X*np.arctanh(2*(y-a)/(b-a)-1)

#Acho que essa função foi movida para outro lugar e pode ser apagada 11/7/18
#def countArrows(transitions, eta=0) :
#    a = pd.DataFrame(0,columns=['in_s','in_r','out_s','out_r'],
#                          index=ct.met_states)
#    for r in transitions : 
#        for direction in [0,1] :
#            for inout in [0,1] :
#                a.loc[r[direction ^ inout],
#                    ['out_','in_'][inout]+['r','s'][r[2++direction]]] += 1
#    a['mlevel'] = ct.methyl_level(a.index.values)
#    a['unbalance'] = a.in_s-a.out_s + eta*(a.in_r-a.out_r)
#    return a.sort('mlevel')

def plotTimeEvolution(msteps, mDyn, dynamics=None) :
    '''Gera os graficos da evoução tempora para um determinado mutante,
       ou da média de varias simulacões de um mesmo mutante. No esquema de
       variacao de L usado para o cálculo de gamma e xi.
    '''
    fig = plt.figure()
    fl = msteps.L/msteps.Lpre
    iResp =(0.5<fl) & (fl<2) & (fl!=1)
    #iChan = fl[(0.5>fl) | (fl>2)]
    #print(iResp,iChan)
    #print(msteps[iResp])
    if dynamics is not None :
        plt.plot(dynamics.t, dynamics.a, 'm', label='$\\langle a\\rangle_1$')
    tD = mDyn.t
    #Plot the continuous lines
    plt.plot(tD, mDyn.a, 'r', label='$\\langle a\\rangle_{%i}$'%nRepetition)
    plt.plot(tD, mDyn.m/ct.parF['M'], 'b', 
            label='$\\langle m\\rangle_{%i}/%i$'%(nRepetition,ct.parF['M']))
    plt.plot(tD, mDyn.l, 'y',label='$l$')
    #plt.plot(tD, mDyn.power/mDyn.power.max(), 'c', label='Power')
    #plt.plot(tD, mDyn['m%i'%m]/ct.N/ct.N,label='m%i'%m)
    #Plot the symbols of the activity pre and pos stimulus
    for resp in [True,False] :
        ms = msteps[iResp if resp else ~iResp]
        for (mark,color,column) in [['D','k','apre'],['o','m','apos'],
                ['s','g','apeak'],['p','b','aFinal']] :
            ti=ms.t + (ms.taupeak if len(column)!=4 else 0) \
                    + (ms.taucross if column=='aFinal' else 0)
            plt.scatter(ti, ms[column], marker=mark, edgecolors=color,
                 facecolors=color if resp else 'none',label='')
    ti = msteps.t[iResp]#, msteps.t[~iResp]]
    #plt.plot(ti, 0.8*msteps.apremean[iResp], 'b:',label='')
    #plt.plot(ti, (2-1/1.2)*msteps.apremean[iResp], 'b:',label='')
    plt.plot(mDyn.t.iloc[[0,-1]], [0.333,0.333], 'g')
    plt.ylim(0,1)
    if 'tadapt' in msteps :
        for i,s in msteps.iterrows() :
            dadapt = mDyn[mDyn.t==s.tadapt]
            plt.plot(dadapt.t, dadapt.a, 'go',label='')
            plt.plot(s.t+s.t05, sigmoid(s.t+s.t05, s.sgA, s.sgB, s.sgT0, s.sgT),'go',label='')
            plt.plot(s.t+s.t01, sigmoid(s.t+s.t01, s.sgA, s.sgB, s.sgT0, s.sgT),'go',label='')
            t = np.arange(s.t,s.t1,0.1)
            plt.plot(t,sigmoid(t, s.sgA, s.sgB, s.sgT0, s.sgT),'-g')
    #plt.legend(loc='upper left')
    plt.gca().set_xlabel('Time')
    plt.gca().set_ylabel('$a$, $m/%i$'%ct.parF['M'])
    plt.legend(loc='upper left')
    ay1=plt.twinx()
    plt.xlim(0,mDyn.t.max())
    plt.plot(mDyn.t, mDyn.L, 'g', label='$[L]$')
    plt.legend(loc='lower right')
    ay1.set_yscale('log')
    ay1.set_ylabel('$[L]$')
    #plt.title('$\\eta=%.1f$, $M=%d$'%(ct.par['eta'],ct.par['M']))
    fig.show()
    #plt.savefig('dynamics.pdf')
    #plt.close('all')

def plotResponse(steps) :
    '''Faz o grafico da resposta aos estimulo que partem de L=0 e retornam.
    '''
    ir = steps[(steps.L>0) & (steps.Lpre==0)].index.values
    s = steps.loc[ir]
    plt.plot(s.L,-s.resp,'-o',label='-Response')
    plt.plot(s.L,1-(s.apre-s.aadpt)/(-s.resp).max(),'-s',label='Precision')
    #plt.plot(s.L,(s.apos-s.aadpt)/s.resp,'-p',label='Precision')
    #ay1=plt.twinx()
    #plt.plot(s.L,s.t05,'-^',label="t_05")
    #plt.plot(s.L,s.t01,'-^',label="t_01")
    plt.gca().set_xscale('log')
    plt.legend(loc="lower left", numpoints=1)
    plt.savefig('precision.pdf')
    plt.close('all')

def plotPop(popData) :
    '''Faz o histograma empilhando em cada colunas niveis de meitilação idênticos.
       As corese são diferentes para os vários estados de um mesmo nível.
       Parameters:
       popData: Line of the dyn DataFrame.
    '''
    global pop
    pop=pd.DataFrame(popData[ct.pop_states].values,columns=['n'])
    pop['mlevel'] = ct.methyl_level(ct.met_states)
    pop['sublevel'] = 0
    for g in pop.groupby('mlevel') :
        pop.loc[g[1].index,'sublevel'] = list(range(len(g[1])))
    stack=pd.DataFrame()
    for i,l in pop.iterrows() :
        stack.loc[int(l.mlevel), int(l.sublevel)] = l.n
    stack = stack.fillna(0)
    stack.plot(kind='bar', stacked=True, legend=False, rot=0)
    plt.show()
    return pop,stack

def plot_g_logL(stepsDict, legTitle='$\Delta[L]/[L]$') :
    '''Faz o grafico dos valores usados para calcular gammma.
       A função calc_gamma_xi, deve ser chamada antes dessa, 
       para calcular a coluna g.
       Parameters:
       -----------
       stepsDict: Dictionary, contained the size of the jump as index and
            datarframes of the dynamics as values.
    '''
    fig = plt.figure(figsize=(4, 3), dpi=300) 
    plt.xlabel("log$_{10}$([L])")
    plt.ylabel("$g$")
    ampy =  np.max([np.max(np.abs(s.g)) for s in stepsDict.values()])*1.1
    plt.xlim(-1,6)
    plt.ylim(-ampy,ampy)
    plt.plot([-1,6],[0,0],':k',lw=0.5)
    plt.plot([np.log10(18),np.log10(18)],[-ampy,ampy],':k',lw=0.5)
    plt.plot([np.log10(3000),np.log10(3000)],[-ampy,ampy],':k',lw=0.5)
    for signal in [-1,1] :
        color = 'r' if signal>0 else 'b'
        plt.text(-0.4,-1.2*signal, va='center', ha='left', color=color,
            s="$\\Delta[L]"+("<" if signal<0 else ">") +
              "0$\n $\\Delta a"+(">0$"if signal<0 else "<0$"))
        marker = 'o'
        for (deltaLResp,steps) in stepsDict.items() :
            s = steps[~steps.g.isnull()]
            s = s[s.dlogL*signal>0]
            s = s[s.Lpre<2e6]
            #symbol = 'o' if deltaLResp==0.1 else 's'
            plt.plot(np.log10(s.Lpre),-s.g*np.sign(s.dlogL),'-'+marker+color)
            if signal==1 :
                plt.plot([],[],marker+'k',label='%.2f'%deltaLResp)
            marker = 's'
    plt.tight_layout()
    plt.legend(title=legTitle,loc='lower right')
    plt.savefig('g_logL.pdf')
    plt.close('all')    

def analizeResponse(steps, dynamics) :
    '''Use the step and dynamics data to characterize the response to stimulus.
       The new data are stored as new columns of steps.
    '''
    #Response to the stimulus change, adapted activity, end of the stimulus level    
    steps['resp'] = steps.apos - steps.apre
    steps['aadpt'] = list(steps.apremean.iloc[1:])+[np.NAN] 
    steps['t1'] = list(steps.t[1:])+[np.NAN]
    for iS,s in steps.iloc[:-1].iterrows() :
        #Time interval of the current stimulus level
        i = dynamics[(dynamics.t>s.t) & (dynamics.t<s.t1)].index.values
        #Instant when the activty first reaches the adapted value
        iadpt = (s.aadpt-dynamics.a[i].values)*(s.aadpt-dynamics.a[i+1].values)<0
        tadapt = dynamics.t[i[iadpt][0]] if iadpt.any() else np.NaN
        steps.loc[iS,'tadapt']=tadapt
        #Time interval of while activity is reaching the adpated value
        i1 = dynamics[(dynamics.t>s.t) & (dynamics.t<tadapt)].index.values
        try :
            #Use the whole stimulus level duration to fit 4 parameters 
            popt, pcov = curve_fit(sigmoid, dynamics.t[i], dynamics.a[i], 
                                  [s.apos, s.aadpt, (s.t+tadapt)/2,1])
            #Refine the fitting of 3 parameters by doing the fitting only
            #on while the activity first reaches the adaption value
            if len(i1)>3 :
                popt1, pcov = curve_fit(lambda x, A, T0, T : sigmoid(x, A, popt[1], T0, T), 
                    dynamics.t[i1], dynamics.a[i1], [s.apos, (s.t+tadapt)/2,1])
                popt = [popt1[0], popt[1], popt1[1], popt1[2]]
        except (RuntimeError, TypeError) :
            popt = [np.NAN, np.NAN, np.NAN, np.NAN]
        if len(i1) <1 or popt[3]<0 or abs(popt[3])>max(dynamics.t[i1])-min(dynamics.t[i1]) :
            popt = [np.NAN, np.NAN, np.NAN, np.NAN]
        #Parameters of the simoidal curve
        steps.loc[iS,'sgA'] = popt[0]
        steps.loc[iS,'sgB'] = popt[1]
        steps.loc[iS,'sgT0'] = popt[2]
        steps.loc[iS,'sgT']  = popt[3]
        #Time need to reach 10% or 50% of the final adapted value
        steps.loc[iS,'t01'] =  invSigmoid(s.apos+(popt[1]-s.apos)*0.1, *popt) - s.t
        steps.loc[iS,'t05'] =  invSigmoid(s.apos+(popt[1]-s.apos)*0.5, *popt) - s.t

def plot_stim_adap(mDyn, letter='a') :
    '''Similar to plotTimeEvolution but restrict to activity and [L] to
       prepare plots for the paper.
    '''
    fig = plt.figure(figsize=(4, 2), dpi=300) 
    plt.xlim(0,30)
    if letter == 'a' :
        plt.ylim(0.07,1000000)
        plt.semilogy()
        plt.ylabel("$[L]$")
    else :
        plt.ylim(0.0,0.72)
        plt.text(0.12,0.05,'%s\n$\eta=%i$'%{'d':('Sequential',0),'e':('Random',1)}[letter],
                 transform=plt.gca().transAxes, horizontalalignment='center')
        plt.ylabel("$\langle a\\rangle$")
    #plt.xticks(range(10))
    #plt.yticks([0,1])
    #if letter=='c' :
    plt.xlabel("Time")
    #else :
    #    plt.gca().tick_params(labelbottom='off')  
    plt.text(0.01,0.91 if letter!='c' else 0.8,'(%s)'%letter, transform=plt.gca().transAxes)
    plt.plot(mDyn.t-1, mDyn.L if letter=='a' else mDyn.a, '-b')
    plt.savefig('stim_adap_%s.eps'%letter, bbox_inches='tight')
    plt.close('all')

#def plot_mDist(data, selectCol=None,  selectVals=None, maxY=[1,1], maxX=[None,None], 
#            power10Leg=False, title=None, xticks=[None,None]) :
#    '''Bar plots of  the methylation level of mSites and the distribution of mLevels.

#       Parameters:
#       data: DataFrame with columns containing simulation variables and with
#            different time at each row. Must contain the columns on
#            ct.mStatesColumns.
#       selectCol, selectVals: graphics are done only for rows whose value of 
#            selectCol are present in selectVals.
#       maxY: maximun value of y axis, respectively, in the plots of mSite
#            methylation and mLevel distribution. Plot are not shown if the
#            corresponding maxY is None.
#       power10Leg: the legend at the left of each plot must be in log10 format?
#    '''
#    global distMLevels, ocupMSites
#    dataPlot = data[np.round(data[selectCol],5).isin(selectVals)] \
#            if selectCol is not None else data
#    mStatesColumns = [c for c in data.columns if c[:2]=='Pm']
#    siteMethylationColumns = [c for c in data.columns if c[:2]=='Sm']
#    M = len(siteMethylationColumns)
#    mData = dataPlot[mStatesColumns]
#    nRows,nCols = mData.shape
#    cols = np.arange(nCols)
#    rows = np.arange(nRows)
#    mSites = np.arange(M)+1
#    mLevels = np.arange(M+1)
#    mLevelCols = ct.methyl_level(cols)
#    #ocupMSites = pd.DataFrame([mData.iloc[:,cols&(2**(site-1))!=0].sum(axis=1) 
#    #             for site in mSites]).T / ct.N**2
#    ocupMSites = dataPlot[siteMethylationColumns]
#    distMLevels= pd.DataFrame([mData.iloc[:,mLevelCols==mLevel].sum(axis=1) 
#                 for mLevel in mLevels]).T / mData.sum(1).mean()#ct.N**2
#    #print(ocupMSites)
#    nPlotCols = (maxY[0] is not None) + (maxY[1] is not None)
#    if title is not None :
#        plt.title(title)
#    plt.figure(figsize=(0.2+nPlotCols,nRows))
#    plt.subplots_adjust(left=.25/nPlotCols, top=1-0.1/nRows, bottom=0.5/nRows,
#                right=1-0.25/nPlotCols)
#    plotCol=1
#    #Transform the selectVals in integer if round==0 for better plotting
#    vals = np.round(dataPlot[selectCol],5)
#    if (np.modf(vals)[0]==0).all() :
#        vals = vals.astype(int)
#    for ptype,x,data,xlabel in ([[0,mSites,ocupMSites,'mSite']] if maxY[0] is not None else []) + \
#                         ([[1,mLevels,distMLevels,'mLevel']] if maxY[1] is not None else []) :
#        for row in rows :
#            plt.subplot(nRows, nPlotCols, row*nPlotCols+plotCol)
#            plt.bar(x, data.iloc[row], width=0.9, color=[palete[i] for i in x])
#            mX = ct.M if maxX[ptype] is None else maxX[ptype]
#            mY = maxY[ptype] 
#            if mY<0 :
#                #Find the y scale multiple to 0.1
#                mY=np.ceil(data.values.max()*10)/10
#                if np.modf(mY)[0]==0 :
#                    mY = int(mY)
#            plt.xlim(0.5-ptype,mX+0.5)
#            plt.ylim(0,mY)
#            plt.xticks([])
#            plt.yticks([mY])
#            for border in ['top','right'] :
#                plt.gca().spines[border].set_visible(False)
#            if plotCol== nPlotCols :
#                posX = mX + 3*len(x)/nCols
#                plt.text(posX, 0.4*mY, '$10^{%.0f}$'%np.log10(vals.iloc[row]) 
#                    if power10Leg else vals.iloc[row], horizontalalignment='left')
#                if row==0 :
#                    plt.text(posX,0.8*mY,selectCol+' ',horizontalalignment='left')
#        if xticks[ptype] is None :
#            plt.xticks(x[x<=mX])
#        else :
#            plt.xticks(*xticks[ptype])
#        plt.xlabel(xlabel)
#        plotCol+=1
#    plt.savefig('mDist.pdf')
#    plt.close('all')


#ct.readParamFile('Fittest_search/wild_type/wt_a33_M01_param.dat')
#ct.readParamFile('Fittest_search/nseq_a033/m4/new_x300_param.dat')
#ct.readParamFile('Fittest_search/seq_a033/m4/new_x040_param.dat')
#ct.readParamFile('Fittest_search/varEta_fixPars/varEta_fixPars_00.dat')
#ct.readParamFile('Fittest_search/varEta_a33/eta00_param.dat')
#ct.readParamFile('met_residue_mutants/gammaXi_eColi_M6.dat')
#ct.readParamFile('met_residue_mutants/steady_2met.dat')
ct.readParamFile('met_residue_mutants/dyn_mutants_1.dat')
#ct.readParamFile('Fittest_search/varAlpha_pFix_a33/m4_seq_alpha_1_8.dat')
#ct.readParamFile('Fittest_search/seq_a033/wild_type_error_param.dat')
#ct.readParamFile('Fittest_search/eta20_g01_param.dat')
ct.updateParameters()
ct.prepStateMatrices()

nRepetition = 50
#ct.prepareLScan('0Response')
ct.prepareLScan('GammaXi',bidirection=True, tau=ct.parF['dt'])
t0 = time.time()
ct.averageLScans(nRepetition, showDynamics=False)
print('Runtime: ',time.time()-t0)
#mSteps = ct.pSteps.mean(axis=0)
#sSteps = ct.pSteps.std(axis=0)
mSteps =  ct.pSteps.mean(level=[1,1])
mSteps.index = mSteps.index.droplevel() 
#sSteps =  ct.pSteps.std(level=[1,1])
#sSteps.index = sSteps.index.droplevel() 
#mCols = mSteps.columns[mSteps.columns.str.startswith('Pm')]
mSteady = mSteps[mSteps.index%2==1]
mDyn = ct.mDyn
Gamma,xi = ct.calc_Gamma_xi(mSteps.copy())
#plot_stim_adap(mDyn,'e')

#Plot the distribution of the methylation sites

#Perform another scan, to be used by the plot_g_logL
#ct.updateParameters({'deltaLResp':0.1})
#ct.updateParameters({'alpha':ct.p.alpha*2})
#ct.prepStateMatrices()
#ct.prepareLScan('GammaXi')
#nRepetition = 3
#ct.averageLScans(nRepetition)
#mSteps1 = ct.pSteps.mean(axis=0)
#ct.gSNR_noise(mSteps1)

#mSteps['t']-=mDyn.t.min() ; mDyn['t']-=mDyn.t.min()
if ct.M==6 :
    ct.plot_mDist(mDyn,'t',[0.,2,4,6,8,10],[-1,-1], maxX=[6,4],
        xticks=([[1,2,3,4,5,6],[4,'$*_3$',3,2,1,'$*_1$']],None))
else :
    ct.plot_mDist(mDyn,'t',[0.,2,4,6],[-1,-1])
#plot_mDist(mSteady,'L',[1,10,100],power10Leg=True)
plotTimeEvolution(mSteps, mDyn)
        
#if 'mSteps' in locals() :
#    if ct.scantype=='0Response' :
#        analizeResponse(mSteps, ct.mDyn)
#        plotResponse(mSteps)
#    else :
#        print(ct.gSNR_noise(mSteps))
#        print(ct.calc_Gamma_xi(mSteps))
#        plotTimeEvolution(mSteps, mDyn)
#        #plot_g_logL({0.2:mSteps},{0.1:mSteps1})
#        plot_g_logL({ct.p.alpha/2:mSteps,ct.p.alpha:mSteps1},r'$\alpha$')



