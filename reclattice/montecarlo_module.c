#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <Python.h>
#include <stdio.h>
#include<numpy/arrayobject.h>
#include<math.h>

typedef unsigned char uint8;
typedef unsigned long uint64;
typedef unsigned long size_t;
typedef int bool;

static char *simParameters[] = {"M", "fixedM0", "fixedM1", "eta", "g", "kr", "kb", "K0", "K1", "C", 
        "alpha","alpha0","m0", "L", "dt", NULL};

size_t N, M, fixedM0, fixedM1, symmetry;
double eta, g, kr, kb, K0, K1, C, alpha, alpha0, m0, L, dt;

uint8 **a=NULL, **l=NULL;
uint64 **m=NULL;

unsigned long vNR=4101842887655102017LL;
void seedNR(unsigned long j) {
    vNR = j ^ 4101842887655102017LL;
    vNR ^= vNR >> 21; vNR ^= vNR << 35; vNR ^= vNR >> 4;
}
inline double randNR() { //Ranq1 from Numerical recipes 3.0 
    vNR ^= vNR >> 21; vNR ^= vNR << 35; vNR ^= vNR >> 4;
    return (vNR * 2685821657736338717LL) * 5.421010862427521E-20 ; //0<r<1
}

#define max(a,b) ((a) > (b) ? (a) : (b))

//Must be fixxed to return the simulation parameters. Does not work!
static PyObject* montecarlo_module_parameters(PyObject *self, PyObject *args) {
    int n;
    for (n=0; simParameters[n]!=NULL; n++) ;
    PyArrayObject *x;// = PyArray_SimpleNewFromData(1, &n, NPY_STRING, simParameters )
    return PyArray_Return(x);
}

size_t methil_level(uint64 m) {
    //Return the number of occupied methylation sites of the state m
    if (eta==0.0) {
        return m;
    } else {
        uint64 sum=0;
        do {
            sum += m & 1;
        } while ((m>>=1)!=0);
        return sum;
    }
}

//Precalculated some values used in the MC simulation
double krdt, krdtEta, kbdt, kbdtEta; 
double logKL0, logKL1;
double probCheB, probCheR, gA0, gAA0;
uint64 S, maskFixedM0, fixedM01; //Number of methylation states
double *alphamm0 = NULL;
static PyObject* montecarlo_module_set_parameters(PyObject *self, PyObject *args, PyObject *keywds) {
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "kkkdddddddddddd", simParameters,
        &M, &fixedM0, &fixedM1, 
        &eta, &g, &kr, &kb, &K0, &K1, &C, &alpha, &alpha0, &m0, &L, &dt))
        return NULL;
    fixedM01 = fixedM0 | fixedM1;
    maskFixedM0 = ~fixedM0;
    krdt = kr*dt; krdtEta = krdt*eta; kbdt = kb*dt; kbdtEta = kbdt*eta;
    //printf("krdt: %2f %2f\n",krdt, krdtEta);
    logKL0 = log(K0/L); logKL1 = log(K1/L);
    S = eta==0 ? M+1 : 1<<M ; //Number of methylation states
    probCheB=kbdt, probCheR =krdt;
    gA0 = g*exp(alpha0), gAA0 = g*exp(-alpha-alpha0);
    if (alphamm0 != NULL) {
        free(alphamm0);
    }
    alphamm0 = (double *)malloc(S*sizeof(double));
    uint64 m;
    for (m=0; m != S; m++) {
        alphamm0[m] = alpha* (methil_level(m)-m0);
    }
    Py_RETURN_NONE;
}

static PyObject* montecarlo_module_define_state_variables(PyObject *self, PyObject *args) {
    PyArrayObject *a_obj, *l_obj, *m_obj;

    if (!PyArg_ParseTuple(args, "kkOOO", &N, &symmetry, &a_obj, &l_obj, &m_obj))
        return NULL;

    uint8 *aData = (uint8 *) (PyArray_DATA(a_obj));
    uint8 *lData = (uint8 *) (PyArray_DATA(l_obj));
    uint64 *mData = (uint64 *) (PyArray_DATA(m_obj));

    if (a!=NULL) {
        free(a);
        free(l);
        free(m);
    }
    //Define the pointers to the data received from python
    a = (uint8 **)malloc(N*sizeof(uint8 *)); 
    l = (uint8 **)malloc(N*sizeof(uint8 *)); 
    m = (uint64 **)malloc(N*sizeof(uint64 *)); 
    uint64 i;
    for (i=0; i!=N; i++) {
        a[i] = aData + i*N;
        l[i] = lData + i*N;
        m[i] = mData + i*N;
    } 
    Py_RETURN_NONE;
}

void calcProbMet(uint8 a, uint64 m,bool seq,uint64 *deltaMet, double *probMet, 
                 double *probDemet) {
    bool canMet, canDemet;
    if (seq) {
        canMet   = m<M; 
        canDemet = m>0;
    } else {
        *deltaMet =  1 << (uint8)(M*randNR()); //Site to be (de)methylated
        canDemet = (m&*deltaMet)!=0; 
        canMet   = !canDemet;
        //Selecti the proper rate depending if the change is sequential or not
        probCheB = (*deltaMet<<1)&m ? kbdtEta : kbdt;
        probCheR   = *deltaMet&((m<<1)|1) ? krdt : krdtEta;
        //if (canMet) 
        //    printf("%2lu %2lu %i -\n", m, m+*deltaMet, (*deltaMet&((m<<1)|1))!=0);
        //if (canDemet)
        //    printf("%2lu %2lu - %i\n", m-*deltaMet, m, ((*deltaMet<<1)&m)==0);
    }
    if (a) {
        *probMet   = canMet   ? probCheB*gAA0 : 0;
        *probDemet = canDemet ? probCheB      : 0; 
    } else {
        *probMet   = canMet   ? probCheR      : 0;
        *probDemet = canDemet ? probCheR*gA0  : 0;
    }
}

static PyObject* montecarlo_module_MCsimulation(PyObject *self, PyObject *args) {
    //Receive the parameters nMetroSteps and the askedDeltaT
    double deltaT=0, deltaE=0, askedDeltaT;
    size_t nMetroSteps;
    size_t k;

    //nMetroSteps: number of metropolis steps per site per time unity
    if (!PyArg_ParseTuple(args, "kd", &nMetroSteps, &askedDeltaT))
        return NULL;
    double dtNN = dt/N/N; 
    size_t N1 = N-1;
    double E00 = 0,            eE00 = exp(-E00),
           E01 = E00 + logKL0, eE01 = exp(-E01), 
           eE00E01 = eE00+eE01, elogKL1 = exp(-logKL1);
    double maxProbability = max(kbdt*max(1,gAA0),krdt*max(1,gA0));
    if (maxProbability>0.5) {
        //Transtion probability<<1 not really necessary, because the real time
        //evolution related these transitions are much smaller.
        //We should only impose probability<1, that the average number of
        //transitions per time unity would be the same.
        printf("\nTransition probability %f too big in montecarlo_module_MCsimulation\n", maxProbability);
        printf("It must be smaller than 0.5.\n");
        printf("dt (%f) should be smaller than %f\n", dt, dt*0.5/maxProbability);
        exit(0);
    }
    bool seq = eta==0;//, canMet, canDemet;
    uint64 deltaMet = 1;
    //printf("MCSim: %.2i %2f %2f\n",nMetroSteps, askedDeltaT, dt);
    while (deltaT < askedDeltaT) {
        //nMSteps is the number of methylation states modified
        //size_t nMSteps = kb!=0 && kr!=0 ? 0 : 1;
        //Perform the methylation dynamics
        for ( k= 0; k!=(seq ? 1 : M); k++) {
            size_t i = randNR() * N, j = randNR() * N;
            int aij = a[i][j];
            uint64 *mij = m[i]+j;
            double probMet, probDemet;
            calcProbMet(aij, *mij, seq, &deltaMet, &probMet, &probDemet);
            if (seq || !(deltaMet & fixedM01)) {
                double rMet = randNR();
                int dm = (rMet-=probMet)<0 ? 1 : rMet<probDemet ? -1 : 0;
                //deltaE -= aij*alphamm0[*mij];
                *mij += dm*deltaMet;
                deltaE += aij*dm*alpha + dm*alpha0;
                //deltaE += aij*alphamm0[*mij] + dm*alpha0;
            }
        }
        //Activity and ligand binding simulation
        //Perform nMetroSteps Metropolis steps per time unity per site
        //size_t nMCSteps = 1+nMSteps*nMetroSteps;
        size_t nMCSteps = nMetroSteps*dt;
        nMCSteps += randNR() < (nMetroSteps*dt-nMCSteps);
        //printf("%i %.4f %.4f ",nMCSteps, deltaT, dt);
        for(k = 0; k < nMCSteps; k++) {
            size_t i = randNR() * N;
            size_t j = randNR() * N;
            double neigh = (i!=0  ? a[i-1][j] : 0.5) +
                           (i!=N1 ? a[i+1][j] : 0.5) +
                           (j!=0  ? a[i][j-1] : 0.5) +
                           (j!=N1 ? a[i][j+1] : 0.5) - 2;
            if (symmetry==6) 
                neigh += (i&1
                     ?   (i!=0  && j!=0  ? a[i-1][j-1] -0.5 : 0) +
                         (i!=N1 && j!=0  ? a[i+1][j-1] -0.5 : 0)
                     :   (i!=0  && j!=N1 ? a[i-1][j+1] -0.5 : 0) +
                         (i!=N1 && j!=N1 ? a[i+1][j+1] -0.5 : 0) );
            //Calculate the energy of the 4 combinations of l and a
            double E10 = alphamm0[m[i][j]] + C*neigh, eE10 = exp(-E10),
                   E11 = E10 + logKL1, eE11 = eE10*elogKL1;
            deltaE -= a[i][j] ? (l[i][j] ? E11 : E10) : (l[i][j] ? E01 : E00);
            double rTrans = randNR()*(eE00E01 + eE10 + eE11);
            //Use boltzman distribution to choose l and a
            if ((rTrans-=eE01)<0) {
                a[i][j] = 0;
                l[i][j] = 1;
                deltaE += E01;
            } else if ((rTrans-=eE10)<0) {
                a[i][j] = 1;
                l[i][j] = 0;
                deltaE += E10;
            } else if ((rTrans-=eE00)<0) {
                a[i][j] = 0;
                l[i][j] = 0;
                deltaE += E00;
            } else {
                a[i][j] = 1;
                l[i][j] = 1;
                deltaE += E11;
            }
        }
        deltaT += dtNN;
    }
    //printf("MCSim: %4f\n",deltaT);
    return Py_BuildValue("dd", deltaT, deltaE);
}

static PyMethodDef montecarlo_module_methods[] = { 
    {   
        "parameters",
        montecarlo_module_parameters,
        METH_VARARGS,
        "Returns a matrix with the name of the simulation parameters."
    },  
    {   
        "define_state_variables",
        montecarlo_module_define_state_variables,
        METH_VARARGS,
        "Define the matrices a(uint8), l(uint8), and m(ulong)."
    },  
    {   
        "set_parameters",
        montecarlo_module_set_parameters,
        METH_VARARGS | METH_KEYWORDS,
        "Set the values of the simulation parameters."
    },  
    {   
        "MCsimulation",
        montecarlo_module_MCsimulation,
        METH_VARARGS,
        "Monte-Carlo simulation of the receptor dynamics."
    },  
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef montecarlo_module_definition = { 
    PyModuleDef_HEAD_INIT,
    "montecarlo_module",
    "Module for simulating bacterial chemotaxis.",
    -1, 
    montecarlo_module_methods
};

PyMODINIT_FUNC PyInit_montecarlo(void)
{
    Py_Initialize();

    return PyModule_Create(&montecarlo_module_definition);
}
