"""
Generates a Permutation invariant matrix for analysing systems with Dicke states
"""
from math import factorial
from decimal import Decimal

import numpy as np

from scipy.sparse import csr_matrix, dok_matrix
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from qutip import *


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

def irreducible_dim(N, j):
    """
    Calculates the dimension of the subspace that accounts for the irreducible representations
    of a given state (j, m) when the degeneracy is removed.
    
    Parameters
    ----------
    N: int
        The number of two level systems
    j: int
        Total spin eigenvalue
    
    Returns
    -------
    djn: int
        The irreducible dimension
    """
    num = Decimal(factorial(N))
    den = Decimal(factorial(N/2 - j)) * Decimal(factorial(N/2 + j + 1))
    
    djn = float(num/den)
    djn = djn * (2*j + 1)

    return (djn)

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

    rho_t0[k] = 1
    return rho_t0

def mean_light_field(j, m) :
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


class Pim(object):
    """
    The permutation invariant matrix class. Initialize the class with the
    parameters for generating a permutation invariant density matrix.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 1
        
    emission : float
        Collective loss emmission coefficient
        default: 1.0
    
    loss : float
        Incoherent loss coefficient
        default: 1.0
        
    dephasing : float
        Local dephasing coefficient
        default: 1.0
        
    pumping : float
        Incoherent pumping coefficient
        default: 1.0
    
    collective_pumping : float
        Collective pumping coefficient
        default: 1.0

    M: dict
        A nested dictionary of the structure {row: {col: val}} which holds
        non zero elements of the matrix M

    sparse_M: scipy.sparse.csr_matrix
        A sparse representation of the matrix M for efficient vector multiplication
    """
    def __init__(self, N = 1, emission = 1, loss = 1, dephasing = 1, pumping = 1, collective_pumping = 1):
        self.N = N
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.M = {}
        self.sparse_M = None

    def isdicke(self, dicke_row, dicke_col):
        """
        Check if an element in a matrix is a valid element in the Dicke space.
        Dicke column: m value index; Dicke row: j value index. 
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
        j = self.N/2 - dicke_col
        m = self.N/2 - dicke_row
        
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
        if dicke_row == 0:
            k = dicke_col

        else:
            k = int(((dicke_col)/2) * (2 * (self.N + 1) - 2 * (dicke_col - 1)) + (dicke_row - (dicke_col)))
            
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
        
        N = self.N # Take care of integers
        M = self.M
        
        if not self.M.keys:
            print("Generating matrix M as a DOK to get the sparse representation")
            self.generate_matrix()

        sparse_M = dok_matrix((num_dicke_states(N), num_dicke_states(N)), dtype=float)
        
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

        N = self.N # Take care of integers
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

        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
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
        
        N = self.N # Take care of integers
        N = float(N)


        num = (j + m + 1) * (j + m + 2) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t9 = yP * (float(num)/den)

        return (t9)
    
def rhs_generate(M, rho):
    """
    Get right-hand side (RHS) of the ordinary differential equation (ODE) in time. 
    
    Parameters
    ----------
    M: scipy.sparse
        A sparse matrix capturing the dynamics of the system

    rho: array
        The state vector at cuurent time
    """
    return M.dot(rho)

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

    # 2. Place operators in total Hilbert space
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
    j_algebra: vector of Qobs matrices (Qutip objects)
        j_algebra = [Jx, Jy, Jz, Jm, Jp]
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


def jpn_jmn_t(p, n):
    """
    Calculate n-th moment of the J_{z}(t) operator given the time evolution of the density matrix. 
     
    Parameters
    ----------
    p: matrix 
        The time evolution of the density matrix
            
    Returns
    -------
    jpjm_n: array
        The time evolution of the n-th moment of J_{+}J_{-} operator, <J_{+}^n J_{-}^n(t)>
    """
    nt = np.shape(p)[0]
    nds = np.shape(p)[1]
    N = num_two_level(nds)
    num_ladders = num_dicke_ladders(N)
    
    jpjm_n = np.zeros(nt)
    
    ll = 0
    for kk in range(1, int(num_ladders + 1)):
        jj = 0.5 * N + 1 - kk
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            if (mm + jj) < n:
                None
            else:
                jpjm_n = jpjm_n + ((jj + mm) * (jj - mm + 1)) ** n * p[:, ll]
            ll = ll + 1
            
    return jpjm_n

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
        
    c_ops = make_cops(emission = 1, dephasing = 1, loss = 1, pumping = 1, collective_pumping = 0)


def make_cops(N, emission=1., dephasing=0., loss=0., pumping=0., collective_pumping=0.):
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
        default = 1
        Spontaneous emission coefficient
    dephasing: float
        default = 0
        Dephasing coefficient
    loss: float
        default = 0
        Nonradiative losses coefficient
    pumping: float
        default = 0
        Incoherent pumping coefficient
    collective_pumping: float
        default = 0
        Collective pumping coefficient 
        
    Returns
    -------
    c_ops: c_ops vector of matrices
        c_ops contains the collapse operators for the Lindbla
    
    """
    N = int(N) 
    
    if N > 10:
        print("Warning! N > 10. dim(H) = 2^N. Use the the permutational invariant methods in 'pim' for large N. ")
    
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
    
    return c_ops
