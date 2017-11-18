"""
Generates a Permutation invariant matrix for analysing systems with Dicke states
"""
from math import factorial
from decimal import Decimal

import numpy as np

from scipy import constants
from scipy.sparse import csr_matrix, dok_matrix
from scipy.integrate import odeint
from scipy import constants

import matplotlib.pyplot as plt

from qutip import *

#def state_degeneracy(N, j)
#def m_degeneracy(N, m)
#def energy_degeneracy(N, m)
#def j_min(N)
#def partition_function(N, omega_0, temperature)
#def thermal_state(N, omega_0, temperature)    
#def num_dicke_states(N)
#def num_two_level(nds)
#def num_dicke_ladders(N)
#def _check_jm(N, j0, m0)
#def dicke_state(N, j0, m0)
#def tau_column(tau, k, j)
#def rhs_generate(rho, t, M)
#def jzn_t(p, n)
#def j2n_t(p, n)
#def jpn_jzq_jmn_t(p, n, q)
#def mean_light_field(j, m)


## Brute Force Modules 

#   def su2_algebra(N)
#   def collective_algebra(N)
#   def make_cops(N, emission=1., dephasing=0., loss=0., pumping=0., collective_pumping=0.)
#   def excited_state_brute(N)
#   def ground_state_brute(N)
#   def identity_brute(N)
#   def ghz_brute(N)
#   def partition_function_brute(N, omega_0, temperature)
#   def thermal_state_brute(N, omega_0, temperature)
#   def diagonal_matrix(matrix)
#   def is_diagonal(matrix)

## Permutational Invariance Modules (jm only)

#class Pim(object)
 #    def isdicke(self, dicke_row, dicke_col)
 #    def error_jm(self, j, m)
 #    def true_jm(self, j, m)
 #    def tau_valid(self, dicke_row, dicke_col)
 #    def calculate_j_m(self, dicke_row, dicke_col)
 #    def calculate_k(self, dicke_row, dicke_col)
 #    def generate_k(self, dicke_row, dicke_col)
 #    def generate_dicke_space(self)
 #    def generate_matrix(self)
 #    def generate_sparse(self)
 #    def tau1(self, j, m)... tau9(self, j, m)

def state_degeneracy(N, j):
    """
    Calculates the degeneracy of the Dicke state |j, m>.
    Each state |j, m> includes D(N,j) irreducible representations |j, m, alpha>.
    Uses Decimals to calculate higher numerator and denominators numbers.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    j: float
        Total spin eigenvalue (cooperativity)
    
    Returns
    -------
    degeneracy: int
        The state degeneracy
    """
    numerator = Decimal(factorial(N)) * Decimal(2 * j + 1) 
    denominator_1 = Decimal(factorial(N/2 + j + 1))
    denominator_2 = Decimal(factorial(N/2 - j ))

    degeneracy = numerator/(denominator_1 * denominator_2 )
    degeneracy = int(np.round(float(degeneracy)))
    
    if degeneracy < 0 :
        raise ValueError("m-degeneracy must be >=0, but degeneracy = {}".format(degeneracy))
    
    return degeneracy

def m_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m> have the same energy (hbar * omega_0 * m) given N two-level systems. 
    
    Parameters
    ----------
    N: int
        The number of two level systems
    m: float
        Total spin z-axis projection eigenvalue (proportional to the total energy)
    
    Returns
    -------
    degeneracy: int
        The m-degeneracy
    """
    degeneracy = N/2 + 1 - abs(m)
    if degeneracy % 1 != 0 or degeneracy <= 0 :
        raise ValueError("m-degeneracy must be integer >=0, but degeneracy = {}".format(degeneracy))

    return int(degeneracy)

def energy_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m, alpha> have the same energy (hbar * omega_0 * m) given N two-level systems. 
    This definition allow to explore also N > 1020, unlike the built-in function 'scipy.special.binom(N, N/2 + m)'
    
    Parameters
    ----------
    N: int
        The number of two level systems
    m: float
        Total spin z-axis projection eigenvalue (proportional to the total energy)
    
    Returns
    -------
    degeneracy: int
        The energy degeneracy
    """
    numerator = Decimal(factorial(N))
    denominator_1 = Decimal(factorial(N/2 + m))
    denominator_2 = Decimal(factorial(N/2 - m))

    degeneracy = numerator/(denominator_1 * denominator_2 )
    
    if degeneracy % 1 != 0 or degeneracy <= 0 :
        raise ValueError("m-degeneracy must be integer >=0, but degeneracy = {}".format(degeneracy))

    #the scipy built-in function is limited to N < N0=1020 for m = 0, but can be used as test for N < N0.
    #degeneracy_check = scipy.special.binom(N, N/2 + m)

    return int(degeneracy)

def j_min(N):
    """
    Gives 0 for N even or 1/2 for N odd.
    
    Parameters
    ----------
    N: int
        The number of two level systems
  
    Returns
    -------
    jmin: float
        The minimum value of the j eigenvalue (total spin)
    """
    if (N % 2) == 0 :
        jmin = 0
    elif ((N + 1) % 2) == 0 :
        jmin = 0.5
    else :
        raise ValueError("N must be integer, but N = {}".format(N))

    return jmin


def partition_function(N, omega_0, temperature):
    """
    Gives the partition function for a collection of N two-level systems at absolute temperature T and with resonance omega_0.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    zeta: float
        The partition function for the thermal state
    """
    nds = num_dicke_states(N)
    
    num_ladders = num_dicke_ladders(N)
    
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
    
    zeta = 0
    s = 0
    
    for k in range(1, int(num_ladders + 1)):
        
        j = 0.5 * N + 1 - k
        
        mmax = (2 * j + 1)
        
        for i in range(1, int(mmax + 1)):
            
            m = j + 1 - i
            
            zeta = zeta + np.exp( - x * m ) * state_degeneracy(N, j)
            
            s = s + 1
            
    if zeta <= 0:
        raise ValueError("Z, the partition function, must be positive, but zeta = {}".format(zeta))
        
    return float(zeta)

def thermal_state(N, omega_0, temperature):
    """
    Gives the thermal state for a collection of N two-level systems at absolute temperature T and with resonance omega_0.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    rho_thermal: numpy array 
        The vector with the initial values for a state of dimension (num_dicke_states(N)) to use in pim
    """
        
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
    
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)

    rho_thermal = np.zeros(nds)
    
    s = 0
    for k in range(1, int(num_ladders + 1)):
        j = 0.5 * N + 1 - k
        mmax = (2 * j + 1)
        for i in range(1, int(mmax + 1)):
            m = j + 1 - i
            rho_thermal[s] = np.exp( - x * m ) * state_degeneracy(N, j)
            s = s + 1

    zeta = partition_function(N, omega_0, temperature)
    
    rho_thermal = rho_thermal/zeta
    
    return rho_thermal

def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles with odd number of systems.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    nds = (N/2 + 1)**2 - (N % 2)/4
    if (nds % 1) != 0:
        raise ValueError("Incorrect number of Dicke states calculated for N = {}".format(N))
    return int(nds)

def num_two_level(nds):
    """
    The number of two level systems (TLS), given the number of Dicke states. 
    Inverse function of num_dicke_states(N)
    
    Parameters
    ----------
    nds: int
         The number of Dicke states
    
    Returns
    -------
    N: int
        The number of two level systems
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2 * (np.sqrt(nds) - 1)
    else: 
        # N is odd
        N = 2 * (np.sqrt(nds + 1/4) - 1)
    
    return int(N)

def num_dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space indexed by (j,m), 
    for a collection of N two-level systems. It counts how many different "j" exist.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5
    
    if np.mod(Nj, 1) != 0 :
        raise ValueError("Incorrect number of Dicke ladders Nj = {}".format(Nj))
    
    return int(Nj)


#>>===================================================
# Check if this can be replaced by the isdicke function
#========================================================

def _check_jm(N, j0, m0):
    """
    Checks the validity of j0 and m0 for N two level systems
    """
    if (j0 > N/2):
        raise ValueError("In the initial state |j0, m0>, j0 cannot be greater than N/2 = {}".format(N/2))
        
    if (abs(m0) > j0 ):
        raise ValueError("In the initial state |j0, m0>, |m0| cannot be greater than j0")

    if N % 2 == 0:
        if ((j0 % 1 != 0) or (m0%1 != 0)):
            raise ValueError("Since N = {} is even, j0 and m0 must be whole numbers, but now j0 = {}, m0 = {}".format(N, j0, m0))

    elif ((j0 % 1 == 0) or (m0 % 1 == 0)):
            raise ValueError("Since N = {} is odd, j0 and m0 must not be whole numbers, but now j0 = {}, m0 = {}".format(N, j0, m0))


def dicke_state(N, j0, m0):
    """
    Generates a initial dicke state vector given the number of two-level systems, from specified |j0, m0>

    Parameters
    ----------
    N: int
        The number of two level systems

    j0: int
        ...

    m0: int
        ...

    Returns
    -------
    rho: array
        The ...
    """
    N = int(N)
    nds = num_dicke_states(N)
    
    _check_jm(N, j0, m0)

    rho_t0 = np.zeros(nds)
    
    ladder_rungs = j0 - m0
    delta_j = int(N/2 - j0)
    previous_ladders = 0

    for i in range(0, delta_j):
        previous_ladders = previous_ladders + N + 1 - (2 * i)
    
    k = int(previous_ladders + ladder_rungs)

#>>=============================
# same as generate k
#>>=============================
    rho_t0[k] = 1
    return rho_t0

def tau_column(tau, k, j):
    """
    Determine the column index for the non-zero elements of the matrix for a particular
    row k and the value of j from the dicke space

    Parameters
    ----------
    tau: str
        The tau function to check for this k and j

    k: int
        The row of the matrix M for which the non zero elements have
        to be calculated

    j: float
        The value of j for this row
    """
    # In the notes, we indexed from k = 1, here we do it from k = 0
    k = k + 1
    mapping = {"tau3": k - (2 * j + 3),
               "tau2": k - 1,
               "tau4": k + (2 * j - 1),
               "tau5": k - (2 * j + 2),
               "tau1": k,
               "tau6": k + (2 * j),
               "tau7": k - (2 * j + 1),
               "tau8": k + 1,
               "tau9": k + (2 * j + 1)}

    # we need to decrement k again as indexing is from 0
    return int(mapping[tau] - 1)

def rhs_generate(rho, t, M):
    """
    Get right-hand side (RHS) of the ordinary differential equation (ODE) in time. 
    
    Parameters
    ----------
    M: scipy.sparse
        A sparse matrix capturing the dynamics of the system

    Returns
    -------
    M.dot(rho): array
        The state vector at cuurent time
    """
    return M.dot(rho)

def jzn_t(p, n):
    """
    Calculates <Jz^n(t)> given the density matrix time evolution, p. 
     
    Parameters
    ----------
    p: matrix 
        The time evolution of the density matrix
    
    Returns
    -------
    jz_n: array
        The time evolution of the n-th moment of the Jz operator, <Jz^n(t)>.
    """
    nt = np.shape(p)[0]
    nds = np.shape(p)[1]
    N = num_two_level(nds)
    num_ladders = num_dicke_ladders(N)
    
    jz_n = np.zeros(nt)
    
    ll = 0
    for kk in range(1, int(num_ladders + 1)):
        jj = 0.5 * N + 1 - kk
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            jz_n = jz_n + (mm ** n) * p[:, ll]
            ll = ll + 1
            
    return jz_n


def j2n_t(p, n):
    """
    Calculates n-th moment of the total spin operator J2, that is <(J2)^n(t)> given the density matrix time evolution, p. 
     
    Parameters
    ----------
    p: matrix 
        The time evolution of the density matrix
    
    Returns
    -------
    j2_n: array
        The time evolution of the n-th moment of the total spin operator, <J2^n(t)>.
    """
    nt = np.shape(p)[0]
    nds = np.shape(p)[1]
    N = num_two_level(nds)
    num_ladders = num_dicke_ladders(N)
    
    j2_n = np.zeros(nt)
    
    ll = 0
    for kk in range(1, int(num_ladders + 1)):
        jj = 0.5 * N + 1 - kk
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            j2_n = j2_n + ((jj + 1) * jj) ** n * p[:, ll]
            ll = ll + 1
            
    return j2_n

def jpn_jzq_jmn_t(p, n, q):
    """
    Calculates <J_{+}^n J_{z}^q J_{-}^n>(t) given the density matrix's time evolution. 
     
    Parameters
    ----------
    p: matrix 
        The time evolution of the density matrix
    n: int 
        Exponent of J_{+} and J_{-}
    q: int 
        Exponent of J_{z}
            
    Returns
    -------
    jpn_jzq_jmn: array
        The time evolution of the symmetric product J_{+}^n J_{z}^q J_{-}^n 
    """
    nt = np.shape(p)[0]
    nds = np.shape(p)[1]
    N = num_two_level(nds)
    num_ladders = num_dicke_ladders(N)
    
    jpn_jzq_jmn = np.zeros(nt)
    
    if n > N:
        None
    else:
        ll = 0
        for kk in range(1, int(num_ladders + 1)):
            jj = 0.5 * N + 1 - kk
            mmax = (2 * jj + 1)
            for ii in range(1, int(mmax + 1)):
                mm = jj + 1 - ii
                if (mm + jj) < n:
                    None
                else:
                    jpn_jzq_jmn = jpn_jzq_jmn + ((jj + mm) * (jj - mm + 1)) ** n * (mm - n) ** q * p[:, ll]
                ll = ll + 1
            
    return jpn_jzq_jmn

def mean_light_field(j, m):
    """
    The coefficient is defined by <j, m|J^+ J^-|j, m>

    Parameters
    ----------
    j: float
        The total spin z-component m for the Dicke state |j, m>

    m: float
        The total spin j for the Dicke state |j, m>
    
    Returns
    -------
    y: float
        The light field average value
    """
    y = (j + m) * (j - m + 1)    
    return y

# brute force modules

def su2_algebra(N):
    """
    Creates the vector (sx, sy, sz, sm, sp) with the spin operators of a collection of N two-level 
    systems (TLSs). Each element of the vector, i.e., sx, is a vector of Qobs objects (spin matrices),
    as it cointains the list of the SU(2) Pauli matrices for the N TLSs. 
    Each TLS operator sx[i], with i = 0, ..., (N-1), is placed in a 2^N-dimensional Hilbert space.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    su2_operators: list
        A list of Qobs matrices (Qutip objects) - [sx, sy, sz, sm, sp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    sx = [0 for i in range(N)]
    sy = [0 for i in range(N)]
    sz = [0 for i in range(N)]
    sm = [0 for i in range(N)]    
    sp = [0 for i in range(N)]
    sx[0] =  0.5 * sigmax()
    sy[0] =  0.5 * sigmay()
    sz[0] =  0.5 * sigmaz()
    sm[0] =  sigmam()
    sp[0] =  sigmap()

    # 2. Place each TLS operator in total Hilbert space
    for k in range(N - 1):
        sx[0] = tensor(sx[0], identity(2))
        sy[0] = tensor(sy[0], identity(2))
        sz[0] = tensor(sz[0], identity(2))
        sm[0] = tensor(sm[0], identity(2))
        sp[0] = tensor(sp[0], identity(2))

    #3. Cyclic sequence to create all N operators
    a = [i for i in range(N)]
    b = [[a[i  -  i2] for i in range(N)] for i2 in range(N)]

    #4. Create N operators
    for i in range(1,N):
        sx[i] = sx[0].permute(b[i])
        sy[i] = sy[0].permute(b[i])
        sz[i] = sz[0].permute(b[i])
        sm[i] = sm[0].permute(b[i])
        sp[i] = sp[0].permute(b[i])
    
    su2_operators = [sx, sy, sz, sm, sp]
    
    return su2_operators

def collective_algebra(N):
    """
    Uses the module su2_algebra to create the collective spin algebra Jx, Jy, Jz, Jm, Jp.
    It uses the basis of the sinlosse two-level system (TLS) SU(2) Pauli matrices. 
    Each collective operator is placed in a Hilbert space of dimension 2^N.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    collective_operators: vector of Qobs matrices (Qutip objects)
        collective_operators = [Jx, Jy, Jz, Jm, Jp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    
    si_TLS = su2_algebra(N)
    
    sx = si_TLS[0]
    sy = si_TLS[1]
    sz = si_TLS[2]
    sm = si_TLS[3]
    sp = si_TLS[4]
        
    jx = sum(sx)
    jy = sum(sy)
    jz = sum(sz)
    jm = sum(sm)
    jp = sum(sp)
    
    collective_operators = [jx, jy, jz, jm, jp]
    
    return collective_operators

def make_cops(N = 2, emission = 1., loss = 0., dephasing = 0., pumping = 0., collective_pumping = 0., collective_dephasing = 0.):
    """
    Create the collapse operators (c_ops) of the Lindblad master equation. 
    The collapse operators oare created to be given to the Qutip algorithm 'mesolve'.
    'mesolve' is used in the main file to calculate the time evolution for N two-level systems (TLSs). 
    Notice that the operators are placed in a Hilbert space of dimension 2^N. 
    Thus the method is suitable only for small N.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    emission: float
        default = 2
        Spontaneous emission coefficient
    loss: float
        default = 0
        Losses coefficient (i.e. nonradiative emission)
    dephasing: float
        default = 0
        Dephasing coefficient
    pumping: float
        default = 0
        Incoherent pumping coefficient
    collective_pumping: float
        default = 0
        Collective pumping coefficient 
    collective_dephasing: float
        default = 0
        Collective dephasing coefficient 
        
    Returns
    -------
    c_ops: c_ops vector of matrices
        c_ops contains the collapse operators for the Lindbladian
    
    """
    N = int(N) 
    
    if N > 10:
        print("Warning! N > 10. dim(H) = 2^N. Use only the permutational invariant methods for large N. ")
    
    [sx, sy, sz, sm, sp] = su2_algebra(N)
    [jx, jy, jz, jm, jp] = collective_algebra(N)
    
    c_ops = []    
    
    if emission != 0 :
        c_ops.append(np.sqrt(emission) * jm)

    if dephasing != 0 :    
        for i in range(0, N):
            c_ops.append(np.sqrt(dephasing) * sz[i])
    
    if loss != 0 :
        for i in range(0, N):
            c_ops.append(np.sqrt(loss) * sm[i])
    
    if pumping != 0 :
        for i in range(0, N):
            c_ops.append(np.sqrt(pumping) * sp[i])
    
    if collective_pumping != 0 :
        c_ops.append(np.sqrt(collective_pumping) * jp)
    
    if collective_dephasing != 0 :
        c_ops.append(np.sqrt(collective_dephasing) * jz)
    
    return c_ops

# brute force functions

def excited_state_brute(N):
    """
    Generates a initial dicke state |N/2, N/2 > as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()

    psi0 = vn[2**N - 1]
        
    return psi0

def superradiant_state_brute(N):
    """
    Generates a initial dicke state |N/2, 0 > (N even) or |N/2, 0.5 > (N odd) as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()

    psi0 = vn[2**N - N]
        
    return psi0    

def ground_state_brute(N):
    """
    Generates a initial dicke state |N/2, - N/2 > as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()
    
    psi0 = vn[0]
        
    return psi0

def identity_brute(N):
    """
    Generates the identity in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    identity: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)
    
    rho = np.zeros((2**N,2**N))
    
    for i in range(0, 2**N) :
        rho[i, i] = 1
    
    spin_dim = [2 for i in range(0,N)]
    spins_dims = list((spin_dim, spin_dim ))

    identity = Qobj(rho, dims = spins_dims)
    
    return identity

def ghz_brute(N):
    """
    Generates the GHZ density matrix in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    ghz: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)
    
    rho = np.zeros((2**N,2**N))
    rho[0, 0] = 1/2
    rho[ 2**N - 1, 0] = 1/2
    rho[0,  2**N - 1] = 1/2
    rho[2**N - 1, 2**N - 1] = 1/2
    
    spin_dim = [2 for i in range(0,N)]
    spins_dims = list((spin_dim, spin_dim ))

    rho = Qobj(rho, dims = spins_dims)
    
    ghz = rho        
    
    return ghz

def css_brute(N):
    """
    Generates the CSS density matrix in a 2**N dimensional Hilbert space.
    The CSS state, also called 'plus state' is, |+>_i = 1/np.sqrt(2) * (|0>_i + |1>_i ).

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    ghz: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)

    # 1. Define i_th factorized density matrix in the uncoupled basis
    rho = [0 for i in range(N)]
    rho[0] = 0.5 * (qeye(2) + sigmax())

    # 2. Place single-two-level-system denisty matrices in total Hilbert space
    for k in range(N - 1):
        rho[0] = tensor(rho[0], identity(2))

    #3. Cyclic sequence to create all N factorized density matrices |+><+|_i
    a = [i for i in range(N)]
    b = [[a[i  -  i2] for i in range(N)] for i2 in range(N)]

    #4. Create all other N-1 factorized density matrices |+><+| = Prod_(i=1)^N |+><+|_i
    for i in range(1,N):
        rho[i] = rho[0].permute(b[i])
    
    identity_i = Qobj(np.eye(2**N), dims = rho[0].dims, shape = rho[0].shape)
    rho_tot = identity_i

    for i in range(0,N):
        rho_tot = rho_tot * rho[i]
    
    return rho_tot

def partition_function_brute(N, omega_0, temperature) :
    """
    Gives the partition function for a collection of N two-level systems with H = omega_0 * j_z.
    It is calculated in the full 2**N Hilbert state, using the eigenstates of H in the uncoupled basis, not the Dicke basis.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    zeta: float
        The partition function for the thermal state of H calculated summing over all 2**N states
    """
    
    N = int(N)
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
    
    jz = collective_algebra(N)[2]
    m_list = jz.eigenstates()[0]
    
    zeta = 0
    
    for m in m_list :
        zeta = zeta + np.exp( - x * m)
            
    return zeta

def thermal_state_brute(N, omega_0, temperature) :
    """
    Gives the thermal state for a collection of N two-level systems with H = omega_0 * j_z.
    It is calculated in the full 2**N Hilbert state on the eigenstates of H in the uncoupled basis, not the Dicke basis. 
    
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    rho_thermal: Qobj operator
        The thermal state calculated in the full Hilbert space 2**N
    """
    
    N = int(N)   
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
       
    jz = collective_algebra(N)[2]  
    m_list = jz.eigenstates()[0]
    m_list = np.flip(m_list,0)

    rho_thermal = np.zeros(jz.shape)

    for i in range(jz.shape[0]):
        rho_thermal[i, i] = np.exp( - x * m_list[i])
    rho_thermal = Qobj(rho_thermal, dims = jz.dims, shape = jz.shape)
    
    zeta = partition_function_brute(N, omega_0, temperature)
    
    rho_thermal = rho_thermal / zeta
    
    return rho_thermal

def diagonal_matrix(matrix):
    """
    Generates a diagonal matrix with the elements on the main diagonal of 'matrix', a Qobj. 
    This module is used by is_diagonal 
    Parameters
    ----------
    matrix: Qobj matrix (square matrix)
    
    Returns
    -------
    diag_mat: Qobj matrix
    """
    matrix_numpy = matrix.full()
    diagonal_array = np.diagonal(matrix_numpy)
    matrix_shape = matrix.shape[0]
    diag_mat = Qobj(diagonal_array * np.eye(matrix_shape), dims = matrix.dims, shape = matrix.shape)
    
    return diag_mat
    
def is_diagonal(matrix):
    """
    Returns True or False whether 'matrix' is a diagonal matrix or not. 
    The module is thought to check the properties of an initial density matrix state rho(j,m,m1) used in Pim.
    The Qobj reshaped matrix has dims = [[2,...2],[2, ...2]] (2 is repeated N times) and shape = (2**N, 2**N).
    Parameters
    ----------
    matrix: ndarray or Qobj (square matrix)
    
    Returns
    -------
    True or False
    """

    nds = matrix.shape[0]
    NN = num_two_level(nds)
    spin_dim = [2 for i in range(0,NN)]
    spins_dims = list((spin_dim, spin_dim ))
    
    matrix_qobj = Qobj(matrix, dims = spins_dims)
    
    diag_mat = diagonal_matrix(matrix_qobj)
    return matrix_qobj == diag_mat 

class Pim(object):
    """
    The permutation invariant matrix class. Initialize the class with the
    parameters for generating a permutation invariant density matrix.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
        
    emission : float
        Collective loss emmission coefficient
        default: 1.0
    
    loss : float
        Incoherent loss coefficient
        default: 0.0
        
    dephasing : float
        Local dephasing coefficient
        default: 0.0
        
    pumping : float
        Incoherent pumping coefficient
        default: 0.0
    
    collective_pumping : float
        Collective pumping coefficient
        default: 0.0

    M: dict
        A nested dictionary of the structure {row: {col: val}} which holds
        non zero elements of the matrix M

    sparse_M: scipy.sparse.csr_matrix
        A sparse representation of the matrix M for efficient vector multiplication
    """
    def __init__(self, N = 2, resonance = 1, exponent = 1, emission = 1, loss = 0, dephasing = 0, pumping = 0, collective_pumping = 0, collective_dephasing = 0):
        self.N = N
        self.resonance = resonance
        self.exponent = exponent
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        self.M = {}
        self.sparse_M = None

    def isdicke(self, dicke_row, dicke_col):
        """
        Check if an element in a matrix is a valid element in the Dicke space.
        Dicke row: j value index. Dicke column: m value index. 
        The function returns True if the element exists in the Dicke space and
        False otherwise.

        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked
        """
        rows = self.N + 1
        cols = 0
        
        if (self.N % 2) == 0:
            cols = int(self.N/2 + 1)
        else:
            cols = int(self.N/2 + 1/2)

        if (dicke_row > rows) or (dicke_row < 0):
            return (False)

        if (dicke_col > cols) or (dicke_col < 0):
            return (False)

        if (dicke_row < int(rows/2)) and (dicke_col > dicke_row):
            return False

        if (dicke_row >= int(rows/2)) and (rows - dicke_row <= dicke_col):
            return False
        
        else:
            return True

    def error_jm(self, j, m):
        """
        Checks if (j,m) is a valid Dicke state |j,m> for N two-level systems
        """
        N = self.N
        
        if (j > N/2):
            raise ValueError("In |j, m>, it must be j <= N/2 = {}".format(N/2))

        if (abs(m) > j ):
            raise ValueError("In |j, m>, it must be |m| <= j")

        if N % 2 == 0:
            if ((j % 1 != 0) or (m % 1 != 0)):
                raise ValueError("For N = {} even, (j,m) shall be integers, but j = {}, m = {}".format(N, j, m))

        elif ((j % 1 == 0) or (m % 1 == 0)):
                raise ValueError("For N = {} odd, (j,m) shall be half-integers, but j = {}, m = {}".format(N, j, m))

    def true_jm(self, j, m):
        """
        Returns True if (j,m) is a valid, False otherwise. Checks for a Dicke state |j,m> for N two-level systems.
        """
        N = self.N
               
        if (N % 1 != 0) or N <= 0:
#            print("error: N not valid")
            return False
        
        if (j > N/2):
#            print("error: j>N")
            return False
            
        if (abs(m) > j ):
#            print("error: |m|>j")
            return False

        if N % 2 == 0:
            if ((j % 1 != 0) or (m % 1 != 0)):
#                print("error: N even, m or j half integer")
                return False
        if (N + 1) % 2 == 0:
#            print("N odd")
            if (((j + 0.5) % 1 != 0) or ((m + 0.5) % 1 != 0)):
#                print("error: N odd, m or j not half integer")
                return False
#        print("all good")       
        return True

    def tau_valid(self, dicke_row, dicke_col):
        """
        Find the Tau functions which are valid for this value of (dicke_row, dicke_col) given
        the number of TLS. This calculates the valid tau values and reurns a dictionary
        specifying the tau function name and the value.

        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked

        Returns
        -------
        taus: dict
            A dictionary of key, val as {tau: value} consisting of the valid
            taus for this row and column of the Dicke space element
        """
        tau_functions = [self.tau3, self.tau2, self.tau4,
                         self.tau5, self.tau1, self.tau6,
                         self.tau7, self.tau8, self.tau9]

        N = self.N
        
        if self.isdicke(dicke_row, dicke_col) is False:
            return False

        # The 3x3 sub matrix surrounding the Dicke space element to
        # run the tau functions

        indices = [(dicke_row + x, dicke_col + y) for x in range(-1, 2) for y in range(-1, 2)]
        taus = {}
        
        for idx, tau in zip(indices, tau_functions):
            if self.isdicke(idx[0], idx[1]):
                j, m = self.calculate_j_m(idx[0], idx[1])
                taus[tau.__name__] = tau(j, m)
        
        return taus

    def calculate_j_m(self, dicke_row, dicke_col):
        """
        Get the value of j and m for the particular Dicke space element.

        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix

        Returns
        -------
        j, m: float
            The j and m values.
        """
        N = self.N
        
        j = N/2 - dicke_col
        m = N/2 - dicke_row
        
        return(j, m)

    def calculate_k(self, dicke_row, dicke_col):
        """
        Get k value from the current row and column element in the Dicke space

        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix

        Returns
        -------
        k: int
            The row index for the matrix M for given Dicke space
            element
        """
        N = self.N
        
        if dicke_row == 0:
            k = dicke_col

        else:
            k = int(((dicke_col)/2) * (2 * (N + 1) - 2 * (dicke_col - 1)) + (dicke_row - (dicke_col)))
            
        return k
       
    def generate_k(self, dicke_row, dicke_col):
        """
        Generates one row of the Matrix M based on the k value running from top to
        bottom of the Dicke space. Also update the row in M. A dictionary with {key: val}
        specifying the column index and the tau element for the given Dicke space element

        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix

        Returns
        -------
        row: dict
            A dictionary with {key: val} specifying the column index and
            the tau element for the given Dicke space element
        """
        if self.isdicke(dicke_row, dicke_col) is False:
            return False
        
        # Calculate k as the number of Dicke elements till
        
        k = int(self.calculate_k(dicke_row, dicke_col))
        
        row = {}
        
        taus = self.tau_valid(dicke_row, dicke_col)
        
        for tau in taus:
            j, m = self.calculate_j_m(dicke_row, dicke_col)
            current_col = tau_column(tau, k, j)
            self.M[(k, int(current_col))] = taus[tau]
            row[k] = {current_col: taus[tau]}

        return row

    def generate_dicke_space(self):
        """
        Generate a Dicke space if you want to play around. N is the number of TLS obtained
        from the class.
        """        
        N = self.N
        rows = N + 1
        cols = 0

        if (rows % 2) == 0:
            cols = int((rows/2))

        else:
            cols = int((rows + 1)/2)

        dicke_space = np.zeros((rows, cols), dtype = int)

        for (i, j) in np.ndindex(rows, cols):
            dicke_space[i, j] = self.isdicke(i, j)

        return (dicke_space)

    def generate_matrix(self):
        """
        Generate the matrix M
        """
        N = self.N
        rows = self.N + 1
        cols = 0
        
        if (self.N % 2) == 0:
            cols = int(self.N/2 + 1)
        else:
            cols = int(self.N/2 + 1/2)

        for (dicke_row, dicke_col) in np.ndindex(rows, cols):
            if self.isdicke(dicke_row, dicke_col):
                self.generate_k(dicke_row, dicke_col)
        
        return self.M

    def generate_sparse(self):
        """
        Generate sparse format of the matrix M
        """
        
        N = self.N  
        M = self.M

        nds = num_dicke_states(N)
        
        if not self.M.keys:
            print("Generating matrix M as a DOK to get the sparse representation")
            self.generate_matrix()

        sparse_M = dok_matrix((nds, nds), dtype=float)
        
        for (i, j) in M.keys():
            sparse_M[i, j] = M[i, j]

        self.sparse_M = sparse_M.asformat("csr")
        
        return sparse_M.asformat("csr")

    def tau1(self, j, m):
        """
        Calculate tau1 for value of j and m.
        """
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping

        N = self.N  
        N = float(N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (N/2 + m)
        pump = yP * (N/2 - m)
        collective_pump = yCP * (1 + j + m) * (j - m)
        
        if j==0:
            dephase = yD * N/4
        else :
            dephase = yD * (N/4 - m**2 * ((1 + N/2)/(2 * j *(j+1))))

        t1 = spontaneous + losses + pump + dephase + collective_pump
        
        return(-t1)

    def tau2(self, j, m):
        """
        Calculate tau2 for given j and m
        """
        yS = self.emission
        yL = self.loss

        N = self.N  
        N = float(N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (((N/2 + 1) * (j - m + 1) * (j + m))/(2 * j * (j+1)))

        t2 = spontaneous + losses

        return(t2)

    def tau3(self, j, m):
        """
        Calculate tau3 for given j and m
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        num = (j + m - 1) * (j + m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t3 = yL * (num/den)

        return (t3)

    def tau4(self, j, m):
        """
        Calculate tau4 for given j and m.
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)


        num = (j - m + 1) * (j - m + 2) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t4 = yL * (num/den)

        return (t4)

    def tau5(self, j, m):
        """
        Calculate tau5 for j and m
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)


        num = (j - m) * (j + m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t5 = yD * (num/den)

        return(t5)

    def tau6(self, j, m):
        """
        Calculate tau6 for given j and m
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)


        num = (j - m + 1) * (j + m + 1) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t6 = yD * (num/den)

        return(t6)

    def tau7(self, j, m):
        """
        Calculate tau7 for given j and m
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        num = (j - m - 1) * (j - m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t7 = yP * (float(num)/den)

        return (t7)

    def tau8(self, j, m):
        """
        Calculate self.tau8
        """
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)


        num = (1 + N/2) * (j - m) * (j + m + 1)
        den = 2 * j * (j + 1)
        pump = yP * (float(num)/den)
        collective_pump = yCP * (j - m) * (j + m + 1)
        
        t8 = pump + collective_pump

        return (t8)

    def tau9(self, j, m):
        """
        Calculate self.tau9
        """
        yP = self.pumping
        
        N = self.N 
        N = float(N)

        num = (j + m + 1) * (j + m + 2) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t9 = yP * (float(num)/den)

        return (t9)   

