"""
Dynamics for dicke states exploiting permutational invariance
"""
from math import factorial
from decimal import Decimal

import numpy as np

from scipy.integrate import odeint, ode

from scipy import constants
from scipy.sparse import *
from qutip import Qobj, spre, spost
from qutip.solver import Result

from ops_h import *


def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems.

    Parameters
    -------
    N: int
        The number of two level systems.    
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    nds = (N/2 + 1)**2 - (N % 2)/4
    return int(nds)

def num_dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space for a
    collection of N two-level systems. It counts how many different "j" exist.
    Or the number of blocks in the block diagonal matrix.

    Parameters
    -------
    N: int
        The number of two level systems.    
    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5    
    return int(Nj)

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

def energy_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m, alpha> have the same energy
    (hbar * omega_0 * m) given N two-level systems. 
    The use of the Decimals class allows to explore N > 1000, 
    unlike the built-in function 'scipy.special.binom(N, N/2 + m)'

    Parameters
    ----------
    N: int
        The number of two level systems.
    m: float
        Total spin z-axis projection eigenvalue. 
        This is proportional to the total energy)

    Returns
    -------
    degeneracy: int
        The energy degeneracy
    """
    numerator = Decimal(factorial(N))
    d1 = Decimal(factorial(N/2 + m))
    d2 = Decimal(factorial(N/2 - m))

    degeneracy = numerator/(d1 * d2)

    return int(degeneracy)

def state_degeneracy(N, j):
    """
    Calculates the degeneracy of the Dicke state |j, m>.
    Each state |j, m> includes D(N,j) irreducible representations |j, m,alpha>.
    Uses Decimals to calculate higher numerator and denominators numbers.

    Parameters
    ----------
    N: int
        The number of two level systems.

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
        raise ValueError("m-degeneracy must be >=0")

    return degeneracy

def m_degeneracy(N, m):
    """
    The number of Dicke states |j, m> with same energy (hbar * omega_0 * m)
    for N two-level systems. 

    Parameters
    ----------
    N : int
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

def j_min(N):
    """
    Calculate the minimum value of j for given N
    """
    if N % 2 == 0:
        return 0
    else:
        return 0.5


def isdiagonal(matrix):
    """
    Check if a matrix is diagonal either if it is a Qobj or a ndarray
    """
    if isinstance(matrix, Qobj): 
        matrix = matrix.full()
      
    isdiag = np.all(matrix == np.diag(np.diagonal(matrix)))
    
    return isdiag 

def get_blocks(N):
        """
        A list which gets the number of cumulative elements at each block boundary.

        For N = 4

        1 1 1 1 1 
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1 
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Thus, the blocks are [5, 8, 9] denoting that after the first block 5 elements
        have been accounted for and so on. This function will later be helpful in the
        calculation of j, m, m' value for a given (row, col) index in this matrix.

        Returns
        -------
        blocks: arr
            An array with the number of cumulative elements at the boundary of each block
        """
        num_blocks = num_dicke_ladders(N)
        blocks = np.array([i * (N + 2 - i) for i in range(1, num_blocks + 1)], dtype = int)
        return blocks
    
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

class Dicke(object):
    """
    The Dicke States class.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
        
    emission : float
        Collective spontaneous emmission coefficient
        default: 1.0

    hamiltonian : Qobj matrix
        An Hamiltonian H in the reduced basis set by `reduced_algebra()`. 
        Matrix dimensions are (nds, nds), with nds = num_dicke_states.
        The hamiltonian is assumed to be with hbar = 1. 
        default: H = jz_op(N)
                
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

    collective_dephasing : float
        Collective dephasing coefficient
        default: 0.0
        
    density_dict: dict
        A nested dictionary holding the mapping (i, k): (j, m, m')}.
        This holds the values for the non zero elements of the density matrix. It
        also serves as a check for valid elements.
    """
    def __init__(self, N = 2, hamiltonian = None,
                 loss = 0., dephasing = 0., pumping = 0., emission = 1.,
                 collective_pumping = 0., collective_dephasing = 0.):
        self.N = N
        if hamiltonian == None:
            self.hamiltonian = jz_op(N)
        else :
        	self.hamiltonian = hamiltonian
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        self.blocks = get_blocks(N)
        self.nds = num_dicke_states(N)
        self.dshape = (num_dicke_states(N), num_dicke_states(N))

    def solve(self, rho0, t_list):
        """
        Solve the system dynamics for the given rho0 for the times t_list

        Parameters
        ----------
        rho0: Qobj
            The density matrix as a Qobj with dimensions (nds, nds), with nds = number of Dicke states. 
        t_list: array
            The time array.

        Returns
        -------
        result: Result (QuTiP class)
            The result as a Result class object, so that with result.states the density matrix time evolution is retrieved.          
        """
        hamiltonian = self.hamiltonian

        is_rho_diag = isdiagonal(rho0) 
      
        if hamiltonian == None:
            is_hamiltonian_diag = True
        else:
            is_hamiltonian_diag = isdiagonal(hamiltonian)        
            
        solver = Pisolve(self)

        if is_rho_diag == True and is_hamiltonian_diag == True:
            print("hamiltonian and rho0 are diagonal: fast solver")
            result = solver.fast_solve(rho0, t_list)
        else :
            print("either hamiltonian or rho0 are not diagonal: general solver")
            result = solver.general_solve(rho0, t_list)
        
        return result

    def correlation(self, rho0, t_list, a, b):
        """
        Calculate the two-time correlator <a(t)b(t+tau)>, where a, b, are operators in the dicke basis |j,m><j,m'| 
        
        Parameters
        ----------
        rho0: Qobj
            The density matrix as a Qobj with dimensions (nds, nds), with nds = number of Dicke states. 
        t_list: array
            The time array.
        a_t: Qobj
            A(0).
        b_tau: Qobj
            B(0). 
        Returns
        -------
        correl: array
            The correlation time array of the expectation value <A(t)B(t+tau)>.
        """

        rho0_a = rho0 * a 

        result = self.solve(rho0_a, t_list)

        rhot_a = result.states
        
        b_tau = []
        for i in range(0, len(t_list)):
            b_tau.append((b * rhot_a[i]).tr())

        correl = b_tau
        
        return correl

    def correlation_tau(self, rho0, t_list, a, b):
        """
        Calculate the two-time correlator <A(t+tau)B(t)>, where A, B, are operators in the dicke basis |j,m><j,m'| 
        
        Parameters
        ----------
        rho0: Qobj
            The density matrix as a Qobj with dimensions (nds, nds), with nds = number of Dicke states. 
        t_list: array
            The time array.
        a_t: Qobj
            A(0). 
        b_tau: Qobj
            B(0).
        Returns
        -------
        correl: array
            The correlation time array of the expectation value <A(t+tau)B(t)>.
        """

        rho0_b = b * rho0 

        result = self.solve(rho0_b, t_list)

        rhot_b = result.states
        
        a_tau = []
        for i in range(0, len(t_list)):
            a_tau.append((a * rhot_b[i]).tr())

        correl = a_tau
        
        return correl


    def _get_element(self, jmm):
        """
        Get the (i, k) index for given tuple (j, m, m1) from the block diagonal matrix.
        """
        j, m, m1 = jmm
        _k = j - m1
        _k_prime = j - m

        blocks = self.blocks
        block_number = int(self.N/2 - j)

        offset = 0
        if block_number > 0:
            offset = blocks[block_number - 1]

        i = _k_prime + offset
        k = _k + offset

        return (int(i), int(k))
            
    def _get_element_v(self, jmm):
        """
        Get the (k) index for given tuple (j, m, m1) from the flattened block diagonal matrix.
        """
        N = self.N
        nds = num_dicke_states(N)
        
        j, m, m1 = jmm
        
        if (j, m, m1) not in self.density_dict:
            return (N+1)
        else:
            ik = self.density_dict[(j, m, m1)]
            k = nds * (ik[0] + ik[1])
            return k

    def _get_element_flat(self, jmm):
        """
        Get the (l) index for given tuple (j, m, m1) from the flattened block diagonal matrix.
        """

        i, k = self._get_element(jmm)
        l = nds * i + k
        
        return l
 
    def jmm1_dictionary(self):
        """
        A dictionary with keys: (i,k) and values: (j, m, m1) of a block-diagonal matrix in the |j, m> <j, m1| basis. 
        """
        N = self.N
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        dict_jmm1 = {}
        
        # loop in the allowed matrix elements
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        jmm1 = (j, m, m1)
                        row_column = self._get_element(jmm1)
                        dict_jmm1['{}'.format(row_column)] = jmm1

        return dict_jmm1

    def jmm1_list(self):
        """
        A list of lists with elements (j, m, m1) of a block-diagonal matrix in the |j, m> <j, m1| basis. 
        """        
        N = self.N
        num_ladders = num_dicke_ladders(N)
        jmm1_list = []

        # loop in the allowed matrix elements
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        jmm1_list.append((j, m, m1))

        return jmm1_list

    def jmm1_flat(self):
        """
        A dictionary with keys: (l) and values: (j, m, m1) for a block-diagonal flattened matrix in the |j, m> <j, m1| basis. 
        """
        N = self.N
        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)
        
        jmm1_flat = {}
        
        # loop in the allowed matrix elements
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        jmm1 = (j, m, m1)
                        row_column = self._get_element(jmm1)
                        i,k = row_column
                        l = nds * i  + k
                        jmm1_flat['{}'.format(l)] = jmm1

        return jmm1_flat

    def css_plus(self):
        """
        Loads the coherent spin state (CSS), |+>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)

        # loop in the allowed matrix elements        
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        row_column = self._get_element((j, m, m1))
                        if mm == mm1 :
                            rho[row_column] = energy_degeneracy(N, m)/(2**N)
                        else :
                            rho[row_column] = np.sqrt(energy_degeneracy(N,m)) * np.sqrt(energy_degeneracy(N,m1)) /(2**N)
        return Qobj(rho)

    def css_minus(self):
        """
        Loads the separable spin state |->= Prod_i^N(|1>_i - |0>_i) into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)

        # loop in the allowed matrix elements        
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    sign_m = (-1)**(m + N/2)
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        row_column = self._get_element((j, m, m1))
                        sign_m1 = (-1)**(m1 + N/2)
                        if m == m1 :
                            rho[row_column] = energy_degeneracy(N, m)/(2**N)
                        else :
                            rho[row_column] = sign_m * sign_m1 * np.sqrt(energy_degeneracy(N, m)) * np.sqrt(energy_degeneracy(N, m1)) /(2**N)
        return Qobj(rho)

    def css_ab(self, a, b):
        """
        Loads the separable spin state |->= Prod_i^N(|1>_i - |0>_i) into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)

        # loop in the allowed matrix elements        
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    sign_m = (-1)**(m + N/2)
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        row_column = self._get_element((j, m, m1))
                        sign_m1 = (-1)**(m1 + N/2)
                        if m == m1 :
                            rho[row_column] = energy_degeneracy(N, m)/(2**N)
                        else :
                            rho[row_column] = sign_m * sign_m1 * np.sqrt(energy_degeneracy(N, m)) * np.sqrt(energy_degeneracy(N, m1)) /(2**N)
        return Qobj(rho)    
    
    def ghz(self):
        """
        Loads the Greenberger–Horne–Zeilinger state, |GHZ>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))

        rho[0,0] = 1/2
        rho[N,N] = 1/2
        rho[N,0] = 1/2
        rho[0,N] = 1/2
        
        return Qobj(rho)
    
    def dicke(self, j, m):
        """
        Loads the Dicke state |j, m>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))

        #row_column = self._get_element((j, m, m))
        row_column = self._get_element((j, m, m))
        rho[row_column] = 1
        
        return Qobj(rho)
    
    def thermal(self, temperature):
        """
        Gives the thermal state density matrix at the absolute temperature T.
        It is defined for N two-level systems.
        The Hamiltonian is H = hbar * omega_0 * (Jz**n_exp).
        For temperature = 0, the thermal state is the ground state. 

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin. 
        Returns
        -------
        rho_thermal: matrix array
            A square matrix of dimensions (nds, nds), with nds = num_dicke_states(N).
            The thermal populations are the matrix elements on the main diagonal
        """
        
        N = self.N        
        hamiltonian = self.hamiltonian
        
        if isdiagonal(hamiltonian) == False:
            raise ValueError("Hamiltonian is not diagonal")
            
        if temperature == 0:
            ground_state = self.dicke( N/2, - N/2)
            return ground_state
        
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        rho_thermal = np.zeros((nds,nds))

        s = 0
        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)
            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                rho_thermal[s,s] = np.exp( - x ) * state_degeneracy(N, j)
                s = s + 1
        zeta = self.partition_function(temperature)

        rho = rho_thermal/zeta

        return Qobj(rho)

    def thermal_new(self, temperature):
        """
        Gives the thermal state density matrix at the absolute temperature T.
        It is defined for N two-level systems.
        The Hamiltonian is H = hbar * omega_0 * (Jz**n_exp).
        For temperature = 0, the thermal state is the ground state. 

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin. 
        Returns
        -------
        rho_thermal: matrix array
            A square matrix of dimensions (nds, nds), with nds = num_dicke_states(N).
            The thermal populations are the matrix elements on the main diagonal
        """
        
        N = self.N        
        hamiltonian = self.hamiltonian
                    
        if temperature == 0:
            ground_state = self.dicke( N/2, - N/2)
            return ground_state
        
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        rho_thermal = np.zeros((nds,nds))

        s = 0
        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)
            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                rho_thermal[s,s] = np.exp( - x ) * state_degeneracy(N, j)
                s = s + 1
        zeta = self.partition_function(temperature)

        rho = rho_thermal/zeta

        return Qobj(rho)

    def partition_function(self, temperature):
        """
        Gives the partition function for the system at a given temperature if the Hamiltonian is diagonal.
        The Hamiltonian is assumed to be given with hbar = 1.

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin
            
        Returns
        -------
        zeta: float
            The partition function of the system, used to calculate the thermal state.
        """
        N = self.N
        hamiltonian = self.hamiltonian
        
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)
        if isdiagonal(hamiltonian) == False:
            raise ValueError("Hamiltonian is not diagonal")

        zeta = 0
        s = 0

        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)

            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                zeta = zeta + np.exp(- x) * state_degeneracy(N, j)
                s = s + 1

        if zeta <= 0:
            raise ValueError("Error, zeta <=0, zeta = {}".format(zeta))
                
        return float(zeta)


class Pisolve(object):
    """
    Permutationally Invariant Quantum Solver class. 
    """
    def __init__(self, dicke_system):
        #dicke_system is the name of the object from the Dicke class. 
        #dicke_system is used to inherit some of the properties of that object.
        #properties inherited from dicke_system:
        self.N = dicke_system.N
        self.hamiltonian = dicke_system.hamiltonian
        self.emission = dicke_system.emission
        self.loss = dicke_system.loss
        self.dephasing = dicke_system.dephasing
        self.pumping = dicke_system.pumping
        self.collective_pumping = dicke_system.collective_pumping
        self.collective_dephasing = dicke_system.collective_dephasing
        self.nds = dicke_system.nds
        self.dshape = dicke_system.dshape
        #new properties:
        self.blocks = get_blocks(self.N)
        self.density_dict = dict()
        
        self.tau_functions = [self.tau3, self.tau2, self.tau4,
                              self.tau5, self.tau1, self.tau6,
                              self.tau7, self.tau8, self.tau9]
        self.tau_dict = {x.__name__:{} for x in self.tau_functions}
        self.generate_dict()

        #properties used only by the fast diagonal solver
        self.Mdiag = {}
        self.sparse_Mdiag = None

    def j_vals(self):
        """
        Get the valid values of j for given N.
        """
        j = np.arange(j_min(self.N), self.N/2 + 1)
        return j

    def m_vals(self, j):
        """
        Get all the possible values of m or $m^\prime$ for given j.
        """
        return np.arange(-j, j+1)

    def get_index(self, j, m, m1):
        """
        Get the index in the density matrix for this j, m, m1 value.
        """
        _k = j - m1
        _k_prime = j - m

        blocks = self.blocks
        block_number = int(self.N/2 - j)

        offset = 0
        if block_number > 0:
            offset = blocks[block_number - 1]

        i = _k_prime + offset
        k = _k + offset

        return (int(i), int(k))
        
    def generate_dict(self):
        """
        Populate the density matrix and create the map from (jmm1) to (ik)
        """
        # This is the crux of the whole code
        # Need to optimize these loops
        for j in self.j_vals():
            for m in self.m_vals(j):
                for m1 in self.m_vals(j):
                    i, k = self.get_index(j, m, m1)
                    self.density_dict[(j, m, m1)] = (i, k)
                    self._generate_taus(j, m, m1)

    def _generate_taus(self, j, m, m1):
        """
        Generates a Tau mask for this j, m, m1. This is a nds x nds x 9 array
        for each Tau
        """                
        for tau in self.tau_functions:
            self.tau_dict[tau.__name__][(j, m, m1)] = tau(j, m, m1)

    def get_tau(self, tau, jmm):
        """
        Get the value of tau(j, m, m1) if it is a valid tau
        """
        j, m, m1 = jmm
        
        if (j, m, m1) not in self.density_dict.keys():
            return 0.
        else:
            return self.tau_dict[tau][jmm]

    def _get_element(self, jmm):
        """
        Get the (i, k) index for given tuple (j, m, m1) from the block diagonal matrix.
        """
        j, m, m1 = jmm
        if (j, m, m1) in self.density_dict.keys():
            return self.density_dict[(j, m, m1)]
        else:
            return (0, 0)
        
    def _get_gradient(self, rho, jmm):
        """
        The derivative for the reduced block diagonal density matrix rho which
        generates the dynamics.

        There are 9 terms which form this derivative. All the 9 terms are
        indexed by j, m, m', j +- 1, m +- 1, m' +- 1. There could be an instance
        where the value of j, m, m' leads to an element in the density matrix
        which is invalid (outside of the block diagonals). We need to check for
        this and set that term to 0

        =====================================================================
        Write the full equation here explaining the validity checks
        =====================================================================

        Parameters
        ----------
        rho: arr
            The block diagonal density matrix rho for the given time
        """
        j, m, m1 = jmm

        # change how get element works
        t1 = self.get_tau("tau1", (j, m, m1)) * rho[self._get_element((j, m, m1))]
        t2 = self.get_tau("tau2", (j, m+1, m1+1)) * rho[self._get_element((j, m+1, m1+1))]
        t3 = self.get_tau("tau3", (j+1, m+1, m1+1)) * rho[self._get_element((j+1, m+1, m1+1))]
        t4 = self.get_tau("tau4", (j-1, m+1, m1+1)) * rho[self._get_element((j-1, m+1, m1+1))]
        t5 = self.get_tau("tau5", (j+1, m, m1)) * rho[self._get_element((j+1, m, m1))]
        t6 = self.get_tau("tau6", (j-1, m, m1)) * rho[self._get_element((j-1, m, m1))]
        t7 = self.get_tau("tau7", (j+1, m-1, m1-1)) * rho[self._get_element((j+1, m-1, m1-1))]
        t8 = self.get_tau("tau8", (j, m-1, m1-1)) * rho[self._get_element((j, m-1, m1-1))]
        t9 = self.get_tau("tau9", (j-1, m-1, m1-1)) * rho[self._get_element((j-1, m-1, m1-1))]

        rdot = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
        
        return rdot

    def _get_element_flat(self, jmm):
        """
        Get the (l) index for given tuple (j, m, m1) from the flattened block diagonal matrix.
        """
        N = self.N
        nds = num_dicke_states(N)

        i, k = self._get_element(jmm)
        l = nds * i + k
        
        return l

    def jmm1_flat(self):
        """
        A dictionary with keys: (l) and values: (j, m, m1) for a block-diagonal flattened matrix in the |j, m> <j, m1| basis. 
        l is the position of the flattened matrix element. 

        """
        N = self.N
        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)
        
        jmm1_flat = {}
        
        # loop in the allowed matrix elements
        for k in range(0, num_ladders):
                j = 0.5 * N - k
                mmax = int(2 * j + 1)
                for i in range(0, mmax):
                    m = j - i
                    for i1 in range(0, mmax):
                        m1 = j - i1
                        jmm1 = (j, m, m1)
                        row_column = self._get_element(jmm1)
                        i,k = row_column
                        l = nds * i  + k
                        jmm1_flat['{}'.format(l)] = jmm1

        return jmm1_flat

    def lindblad_dense(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics.

        Returns
        ----------
        lind: Qobj superoperator
            The matrix is of size (nds**2, nds**2) where nds is the number of Dicke states.

        """
        N = self.N
        nds = num_dicke_states(N)

        num_ladders = num_dicke_ladders(N)
        llind = np.zeros((nds**2, nds**2))

        jmm1_row = self.jmm1_flat()
        jmm1_keys = [int(k) for k in jmm1_row.keys()]

        # loop in each row
        for k in jmm1_row:
            #print("k ", k)
            j, m, m1 = jmm1_row[k]
            jmm1_1 = (j, m, m1)
            jmm1_2 = (j, m+1, m1+1)
            jmm1_3 = (j+1, m+1, m1+1)
            jmm1_4 = (j-1, m+1, m1+1)
            jmm1_5 = (j+1, m, m1)
            jmm1_6 = (j-1, m, m1)
            jmm1_7 = (j+1, m-1, m1-1)
            jmm1_8 = (j, m-1, m1-1)
            jmm1_9 = (j-1, m-1, m1-1)

            t1 = self.get_tau("tau1", jmm1_1)
            l1 = self._get_element_flat(jmm1_1)
            llind[int(k), int(l1)] = t1
            #print("l1 t1 ", l1, t1)

            # generate taus in the given row 
            # checking if the taus exist
            # and load taus in the lindbladian in the correct position

            if jmm1_2 in jmm1_row.values():
                t2 = self.get_tau("tau2", jmm1_2)
                l2 = self._get_element_flat(jmm1_2)
                llind[int(k), int(l2)] = t2
                #print("l2 t2 ", l2, t2)

            if jmm1_3 in jmm1_row.values():
                l3 = self._get_element_flat(jmm1_3)
                t3 = self.get_tau("tau3", jmm1_3) 
                llind[int(k), int(l3)] = t3
                #print("l3 t3 ", l3, t3) 

            if jmm1_4 in jmm1_row.values():
                t4 = self.get_tau("tau4", jmm1_4)
                l4 = self._get_element_flat(jmm1_4)
                llind[int(k), int(l4)] = t4
                #print("l4 t4 ", l4, t4)

            if jmm1_5 in jmm1_row.values():                
                t5 = self.get_tau("tau5", jmm1_5)
                l5 = self._get_element_flat(jmm1_5)
                llind[int(k), int(l5)] = t5
                #print("l5 t5 ", l5, t5)

            if jmm1_6 in jmm1_row.values():
                t6 = self.get_tau("tau6", jmm1_6)
                l6 = self._get_element_flat(jmm1_6)
                llind[int(k), int(l6)] = t6
                #print("l6 t6 ", l6, t6)

            if jmm1_7 in jmm1_row.values():                
                t7 = self.get_tau("tau7", jmm1_7)
                l7 = self._get_element_flat(jmm1_7)
                llind[int(k), int(l7)] = t7
                #print("l7 t7 ", l7, t7)

            if jmm1_8 in jmm1_row.values():
                t8 = self.get_tau("tau8", jmm1_8)
                l8 = self._get_element_flat(jmm1_8)
                llind[int(k), int(l8)] = t8
                #print("l8 t8 ", l8, t8)

            if jmm1_9 in jmm1_row.values():
                t9 = self.get_tau("tau9", jmm1_9)
                l9 = self._get_element_flat(jmm1_9)
                llind[int(k), int(l9)] = t9
                #print("l9 t9 ", l9, t9) 
            
            #print("lind [{}, :] ".format(int(k)), llind[int(k), :])

        #make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]],[[nds], [nds]]]
        llind_qobj = Qobj(llind, dims = llind_dims)

        return llind_qobj

    def lindblad_sup(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics as a sparse matrix using COO.

        Returns
        ----------
        lindblad_qobj: Qobj superoperator (sparse)
            The matrix size is (nds**2, nds**2) where nds is the number of Dicke states.

        """
        N = self.N
        nds = num_dicke_states(N)

        num_ladders = num_dicke_ladders(N)

        jmm1_row = self.jmm1_flat()
        jmm1_keys = [int(k) for k in jmm1_row.keys()]

        # initialize lists to build the sparse COO matrix
        data = []
        row = []
        col = []

        # perform loop in each row of matrix 
        for r in jmm1_row:
            j, m, m1 = jmm1_row[r]
            jmm1_1 = (j, m, m1)
            jmm1_2 = (j, m+1, m1+1)
            jmm1_3 = (j+1, m+1, m1+1)
            jmm1_4 = (j-1, m+1, m1+1)
            jmm1_5 = (j+1, m, m1)
            jmm1_6 = (j-1, m, m1)
            jmm1_7 = (j+1, m-1, m1-1)
            jmm1_8 = (j, m-1, m1-1)
            jmm1_9 = (j-1, m-1, m1-1)
            
            t1 = self.get_tau("tau1", jmm1_1)
            c1 = self._get_element_flat(jmm1_1)
            row.append(int(r))
            col.append(int(c1))
            data.append(t1)            

            # generate taus in the given row 
            # checking if the taus exist
            # and load taus in the lindbladian in the correct position

            if jmm1_2 in jmm1_row.values():
                t2 = self.get_tau("tau2", jmm1_2)
                c2 = self._get_element_flat(jmm1_2)
                row.append(int(r))
                col.append(int(c2))                
                data.append(t2)

            if jmm1_3 in jmm1_row.values():
                t3 = self.get_tau("tau3", jmm1_3) 
                c3 = self._get_element_flat(jmm1_3)
                row.append(int(r))
                col.append(int(c3))                
                data.append(t3)

            if jmm1_4 in jmm1_row.values():
                t4 = self.get_tau("tau4", jmm1_4)
                c4 = self._get_element_flat(jmm1_4)
                row.append(int(r))
                col.append(int(c4))                
                data.append(t4)

            if jmm1_5 in jmm1_row.values():                
                t5 = self.get_tau("tau5", jmm1_5)
                c5 = self._get_element_flat(jmm1_5)
                row.append(int(r))
                col.append(int(c5))                
                data.append(t5)

            if jmm1_6 in jmm1_row.values():
                t6 = self.get_tau("tau6", jmm1_6)
                c6 = self._get_element_flat(jmm1_6)
                row.append(int(r))
                col.append(int(c6))                
                data.append(t6)

            if jmm1_7 in jmm1_row.values():                
                t7 = self.get_tau("tau7", jmm1_7)
                c7 = self._get_element_flat(jmm1_7)
                row.append(int(r))
                col.append(int(c7))                
                data.append(t7)

            if jmm1_8 in jmm1_row.values():
                t8 = self.get_tau("tau8", jmm1_8)
                c8 = self._get_element_flat(jmm1_8)
                row.append(int(r))
                col.append(int(c8))                
                data.append(t8)


            if jmm1_9 in jmm1_row.values():
                t9 = self.get_tau("tau9", jmm1_9)
                c9 = self._get_element_flat(jmm1_9)
                row.append(int(r))
                col.append(int(c9))                
                data.append(t9)

        #make Lindblad matrix as a COO sparse matrix 
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        lindblad_matrix = coo_matrix((data, (row, col)), shape=(nds**2, nds**2))

        #convert matrix into CSR sparse
        lindblad_matrix = lindblad_matrix.tocsr()

        #make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]],[[nds], [nds]]]
        lindblad_qobj = Qobj(lindblad_matrix, dims = llind_dims)

        return lindblad_qobj

    def identity_sup(self):
        """
        Build an identity superoperator.

        Returns
        ----------
        lind: Qobj superoperator
            The matrix is of size (nds**2, nds**2) where nds is the number of Dicke states.

        """
        N = self.N
        nds = num_dicke_states(N)

        num_ladders = num_dicke_ladders(N)
        llind = np.zeros((nds**2, nds**2))

        jmm1_row = self.jmm1_flat()
        jmm1_keys = [int(k) for k in jmm1_row.keys()]

        # loop in each row
        for k in jmm1_row:

            j, m, m1 = jmm1_row[k]

            # generate taus for each row
            t1 = 1

            l1 = self._get_element_flat((j, m, m1))

            # load taus in the lindbladian in the correct position
            llind[int(k), int(l1)] = t1

        #make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]],[[nds], [nds]]]
        llind_qobj = Qobj(llind, dims = llind_dims)

        return llind_qobj


    def liouvillian(self):
        """
        Gives the total liouvillian in the jmm1 basis |j, m > < j, m1|
        """ 
        hamiltonian = self.hamiltonian

        lindblad = self.lindblad_sup()
        hamiltonian_superoperator = - 1j* spre(hamiltonian) + 1j* spost(hamiltonian)
        
        liouv = lindblad + hamiltonian_superoperator 

        return liouv

    def hamiltonian_gradient(self, rho):
        """
        Get the hamiltonian gradient for all the elements in the density matrix by looping over
        j, m, m'
        """        
        h_grad = np.zeros(self.dshape, dtype=np.complex)
        hamiltonian = self.hamiltonian

        h_grad = 1j * (hamiltonian * rho - rho * hamiltonian)

        return h_grad

    def rho_dot(self, rho):
        """
        Get the gradient for all the elements in the density matrix by looping over
        j, m, m'
        """

        grad = np.zeros(self.dshape, dtype=np.complex)
        for jmm in self.density_dict.keys():
            i, k = self.density_dict[jmm]
            grad[i, k] = self._get_gradient(rho, jmm)

        hamiltonian_grad = self.hamiltonian_gradient(rho)
        total_grad = hamiltonian_grad + grad

        return total_grad.flatten()

    def f(self, t, y):
        return self.rho_dot(y.reshape(self.dshape))

    def general_solve(self, rho0, t_list):
        """
        Solve the differential equation dp/dt = Tau to evolve the density matrix.

        The density matrix is a block diagonal matrix which is sparse. Based on the
        initial density matrix, we can choose to just run evolution for the poplulation.

        If the density matrix is diagonal, we can just evolve it for the diagonal elements.

        There is a scope for parallelization as e v olution of some of the sets of elements
        are de coupled from the others.

        =============================================================================
        Check this properly and optimize
        =============================================================================

        Example: N = 4

        1 1 1 1 1 
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1 
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Parameters
        ----------
        rho: Qobj
            The intitial density matrix.
        t_list: np.array
            The time steps at which the density matrix time evolution rho_t is evaluated.
        Returns
        ----------
       result: Result (QuTiP class)
            Contains info on the solver method used (solver), the density matrix time evolution (states), the time array (times).

        """
        if isinstance(rho0, Qobj):
            rho0 = rho0.full()

        y0, t0 = rho0.flatten(), t_list[0]

        nt= np.size(t_list)

        r = ode(self.f).set_integrator("zvode")
        r.set_initial_value(y0, t0)

        t1 = t_list[-1]
        dt = (t_list[-1] - t_list[0])/len(t_list)
        
        result = Result()
        rho_t = rho0
        result.states.append(Qobj(rho_t))
        while r.successful() and r.t  < t1:
            rho_t = r.integrate(r.t + dt).reshape(self.dshape)
            result.states.append(Qobj(rho_t))

        result.states = result.states[:nt]

        #result.expect.append(Qobj(rho_t))
        
        result.solver = "general_solver"
        result.times = t_list
        
        return result

    def general_solve_liouv(self, rho0, t_list, liouv):
        """
        Solve the differential equation dp/dt = Tau to evolve the density matrix using given liouvillian.

        Parameters
        ----------
        rho: Qobj
            The intitial density matrix.
        t_list: np.array
            The time steps at which the density matrix time evolution rho_t is evaluated.
        liouvillian: Qobj
            The liouvillian as a Qobj superoperator.
        Returns
        ----------
        result: Result (QuTiP class)
            Contains info on the solver method used (solver), the density matrix time evolution (states), the time array (times).

        """
        if isinstance(rho0, Qobj):
            rho0 = rho0.full()

        y0, t0 = rho0.flatten(), t_list[0]

        nt= np.size(t_list)

        r = ode(self.f).set_integrator("zvode")
        r.set_initial_value(y0, t0)

        t1 = t_list[-1]
        dt = (t_list[-1] - t_list[0])/len(t_list)
        
        result = Result()
        rho_t = rho0
        result.states.append(Qobj(rho_t))
        while r.successful() and r.t  < t1:
            rho_t = r.integrate(r.t + dt).reshape(self.dshape)
            result.states.append(Qobj(rho_t))

        result.states = result.states[:nt]

        #result.expect.append(Qobj(rho_t))
        
        result.solver = "general_solver"
        result.times = t_list
        
        return result

    def tau1(self, j, m, m1):
        """
        Calculate tau1 for value of j, m, m'
        """
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping
        yCD = self.collective_dephasing
        
        N = self.N  
        N = float(N)

        spontaneous = yS / 2 * (2 * j * (j + 1) - m * (m - 1) - m1 * (m1 - 1))
        losses = yL/2 * (N + m + m1)
        pump = yP/2 * (N - m - m1)
        collective_pump = yCP / 2 * (2 * j * (j + 1) - m * (m + 1) - m1 * (m1 + 1))
        collective_dephase = yCD / 2 * (m - m1)**2
        
        if j <= 0:
            dephase = yD * N/4
        else :
            dephase = yD/2 * (N/2 - m * m1 * (N/2 + 1)/ j /(j + 1))

        t1 = spontaneous + losses + pump + dephase + collective_pump + collective_dephase
        
        return(-t1)
    
    def tau2(self, j, m, m1):
        """
        Calculate tau2 for given j, m, m'
        """
        yS = self.emission
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if yS == 0:
            spontaneous = 0.0
        else:            
            spontaneous = yS * np.sqrt((j + m) * (j - m + 1)* (j + m1) * (j - m1 + 1))

        if (yL == 0) or (j <= 0):
            losses = 0.0
        else:            
            losses = yL / 2 * np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1)) * (N/2 + 1) / (j * (j + 1))

        t2 = spontaneous + losses

        return (t2)
    
    def tau3(self, j, m, m1):
        """
        Calculate tau3 for given j, m, m'
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0) or (j <= 0) :
            t3 = 0.0
        else:
            t3 = yL / 2 * np.sqrt((j + m) * (j + m - 1) * (j + m1) * (j + m1 - 1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t3)
    
    def tau4(self, j, m, m1):
        """
        Calculate tau4 for given j, m, m'
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0)  or ( (j + 1) <= 0):
            t4 = 0.0
        else:
            t4 = yL / 2 * np.sqrt((j - m + 1) * (j - m + 2) * (j - m1 + 1) * (j - m1 + 2)) * (N/2 - j )/((j + 1)* (2 * j + 1))

        return (t4)
    
    def tau5(self, j, m, m1):
        """
        Calculate tau5 for given j, m, m'
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if (yD == 0)  or (j <= 0):
            t5 = 0.0
        else:                    
            t5 = yD / 2 * np.sqrt((j**2 - m**2) * (j**2 - m1**2))* (N/2 + j + 1) / (j * (2 * j + 1))

        return (t5)
    
    def tau6(self, j, m, m1):
        """
        Calculate tau6 for given j, m, m'
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if yD == 0:
            t6 = 0.0
        else:            
            t6 = yD / 2 * np.sqrt(((j + 1)**2 - m**2) * ((j + 1)**2 - m1**2)) * (N/2 - j )/((j + 1) * (2 * j + 1))

        return (t6)
    
    def tau7(self, j, m, m1):
        """
        Calculate tau7 for given j, m, m'
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            t7 = 0.0
        else:    
            t7 = yP / 2 * np.sqrt((j - m - 1) * (j - m)* (j - m1 - 1) * (j - m1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t7)
    
    def tau8(self, j, m, m1):
        """
        Calculate tau8 for given j, m, m'
        """
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            pump = 0.0
        else:    
            pump = yP / 2 * np.sqrt((j + m + 1) * (j - m) * (j + m1 + 1) * (j - m1)) * (N/2 + 1) / (j * (j + 1))

        if yCP == 0:
            collective_pump = 0.0
        else:    
            collective_pump = yCP * np.sqrt((j - m) * (j + m + 1) * (j + m1 + 1) * (j - m1))
        
        t8 = pump + collective_pump
        
        return (t8)
    
    def tau9(self, j, m, m1):
        """
        Calculate tau9 for given j, m, m'
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0):
            t9 = 0.0
        else:    
            t9 = yP / 2 * np.sqrt((j + m + 1) * (j + m + 2) *(j + m1 + 1) * (j + m1 + 2)) * (N/2 - j )/((j + 1)*(2 * j + 1))

        return (t9)

    #below: diagonal solver functions 

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
                taus[tau.__name__] = tau(j, m, m)
        
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
            self.Mdiag[(k, int(current_col))] = taus[tau]
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
        Generate the matrix Mdiag
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
        
        return self.Mdiag

    def generate_sparse(self):
        """
        Generate sparse format of the matrix M
        """
        
        N = self.N  
        Mdiag = self.Mdiag

        nds = num_dicke_states(N)
        
        if not self.Mdiag.keys:
            self.generate_matrix()

        sparse_Mdiag = dok_matrix((nds, nds), dtype=float)
        
        for (i, j) in Mdiag.keys():
            sparse_Mdiag[i, j] = Mdiag[i, j]

        return sparse_Mdiag.asformat("csr")

    def fast_solve(self, rho0, t_list):
        """
        Fast  Solver for the system which returns a QuTiP result
        """

        if isinstance(rho0, Qobj):
        #this manipulation limits the initial dm to be real-valued ones. 
        #it is inserted in case one is using the solver to calculate a correlation function.
            rho0_real = np.real(rho0.full())
            if Qobj(rho0_real) != rho0:
            	print("Warning: only real part of density matrix  considered")
            rho0 = rho0_real

        self.generate_matrix()
        sparse_Mdiag = self.generate_sparse()
        # Convert the initial full density matrix into a vector for M
        initial_state = rho0.diagonal()

        rho_t = odeint(self.generate_rhs, initial_state, t_list, args=(sparse_Mdiag,))
        result_list = [np.diag(x) for x in rho_t]
        result = Result()

        for r in result_list:
            result.states.append(Qobj(r))

        nt = np.size(t_list)

        result.states = result.states[:nt]

        result.solver = "fast_solver"
        result.times = t_list
        
        return result

    def generate_rhs(self, rho, t_list, Mdiag):
        """
        Get right-hand side (RHS) of the ordinary differential equation (ODE) in time. 
        
        Parameters
        ----------
        M: scipy.sparse
            A sparse matrix capturing the dynamics of the system

        Returns
        -------
        Mdiag.dot(rho): array
            The state vector at current time
        """
        return Mdiag.dot(rho)
