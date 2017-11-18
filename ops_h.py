"""
Dynamics for dicke states exploiting permutational invariance
"""
from math import factorial
from decimal import Decimal

import numpy as np

from scipy.integrate import odeint, ode

from scipy import constants
from scipy.sparse import csr_matrix, dok_matrix
from qutip import Qobj
from qutip.solver import Result
from scipy.linalg import block_diag

#from dicke_h import num_dicke_states, num_dicke_ladders, num_two_level

#redefined: num_dicke_states, num_dicke_ladders, num_two_level
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


def block_matrix(N):
	"""
	Gives the block diagonal matrix filled with 1 if the matrix element is allowed in the reduced basis |j,m><j,m'|.   
	Parameters
	----------
	N: int 
		Number of two-level systems
	
	Returns
	-------
	block_matr: ndarray
		A block diagonal matrix of ones with dimension (nds,nds), where nds is the number of Dicke states for N two-level systems.   
	"""
	nds = num_dicke_states(N)
	ones_submatrices = np.zeros((nds,nds))

	#create a list with the sizes of the blocks, in order
	blocks_dimensions = int(N/2 + 1 - 0.5 *(N%2))
	blocks_list = [ (2*(i+1*(N%2))+1*((N+1)%2)) for i in range(blocks_dimensions)]
	blocks_list = np.flip(blocks_list,0)

	#create a list with each block matrix as element  
	for i in range(0,nds):
		for k in range(0,nds):
			ones_submatrices[i,k] = 1
	square_blocks = []
	k = 0
	for i in blocks_list:
		square_blocks.append(np.ones((i,i)))
		k = k + 1

	#create the final block diagonal matrix  
	block_matr = block_diag(*square_blocks)

	return block_matr

def reduced_algebra(N):
	"""
	Gives the list with the collective operators of the total algebra, using the reduced basis |j,m><j,m'| in which the density matrix is expressed.    
	The list returned is [J^2, J_x, J_y, J_z, J_+, J_-]. 
	Parameters
	----------
	N: int 
		Number of two-level systems    
	Returns
	-------
	red_alg: list
		Each element of the list is a Qobj matrix (QuTiP class) of dimensions (nds,nds). nds = number of Dicke states.    
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	j2_operator = np.zeros((nds, nds))
	jz_operator = np.zeros((nds, nds))
	jp_operator = np.zeros((nds, nds))
	jm_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				jz_operator[s,s] = m
				j2_operator[s,s] = j * (j + 1)
				if (s + 1) in range(0,nds):
					jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
				if (s - 1) in range(0,nds):
					jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
				s = s + 1
	jx_operator = 1/2*(jp_operator + jm_operator)
	jy_operator = 1j/2*(jm_operator - jp_operator)

	red_alg = [Qobj(j2_operator), Qobj(jx_operator), Qobj(jy_operator), Qobj(jz_operator), Qobj(jp_operator), Qobj(jm_operator)]

	return red_alg 

def j_algebra(N):
	"""
	Gives the list with the collective operators of the total algebra, using the reduced basis |j,m><j,m'| in which the density matrix is expressed.    
	The list returned is [J^2, J_x, J_y, J_z, J_+, J_-]. 
	Parameters
	----------
	N: int 
		Number of two-level systems    
	Returns
	-------
	red_alg: list
		Each element of the list is a Qobj matrix (QuTiP class) of dimensions (nds,nds). nds = number of Dicke states.    
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	j2_operator = np.zeros((nds, nds))
	jz_operator = np.zeros((nds, nds))
	jp_operator = np.zeros((nds, nds))
	jm_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				jz_operator[s,s] = m
				j2_operator[s,s] = j * (j + 1)
				if (s + 1) in range(0,nds):
					jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
				if (s - 1) in range(0,nds):
					jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
				s = s + 1
	jx_operator = 1/2*(jp_operator + jm_operator)
	jy_operator = 1j/2*(jm_operator - jp_operator)

	j_alg = [Qobj(jx_operator), Qobj(jy_operator), Qobj(jz_operator), Qobj(jp_operator), Qobj(jm_operator)]

	return j_alg 


def jx_op(N):
	"""
	Builds the Jx operator in the same basis of the reduced density matrix rho(j,m,m').    
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	jx_operator: Qobj matrix
		The Jx operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.         
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	jp_operator = np.zeros((nds, nds))
	jm_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				if (s + 1) in range(0,nds):
					jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
				if (s - 1) in range(0,nds):
					jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
				s = s + 1
	jx_operator = 1/2*(jp_operator + jm_operator)

	return Qobj(jx_operator)

def jy_op(N):
	"""
	Builds the Jy operator in the same basis of the reduced density matrix rho(j,m,m').    
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	jy_operator: Qobj matrix
		The Jy operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	jp_operator = np.zeros((nds, nds))
	jm_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				if (s + 1) in range(0,nds):
					jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
				if (s - 1) in range(0,nds):
					jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
				s = s + 1
	jy_operator = 1j/2*(jm_operator - jp_operator)

	return Qobj(jy_operator)

def jz_op(N):
	"""
	Builds the Jz operator in the same basis of the reduced density matrix rho(j,m,m'). Jz is diagonal in this basis.   
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	jz_operator: Qobj matrix
		The Jz operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.      
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	jz_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				jz_operator[s,s] = m
				s = s + 1
	return Qobj(jz_operator)

def j2_op(N):
	"""
	Builds the J^2 operator in the same basis of the reduced density matrix rho(j,m,m'). J^2 is diagonal in this basis.   
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	j2_operator: Qobj matrix
		The J^2 operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.  
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	j2_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				j2_operator[s,s] = j * (j + 1)
				s = s + 1

	return Qobj(j2_operator) 

def jp_op(N):
	"""
	Builds the Jp operator in the same basis of the reduced density matrix rho(j,m,m').    
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	jp_operator: Qobj matrix
		The Jp operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	jp_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				if (s + 1) in range(0,nds):
					jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
				s = s + 1

	return Qobj(jp_operator)

def jm_op(N):
	"""
	Builds the Jm operator in the same basis of the reduced density matrix rho(j,m,m').    
	Parameters
	----------
	N: int 
		Number of two-level systems
	Returns
	-------
	jm_operator: Qobj matrix
		The Jm operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
	"""
	nds = num_dicke_states(N)
	num_ladders = num_dicke_ladders(N)
	block_diagonal = block_matrix(N)
	jm_operator = np.zeros((nds, nds))

	s = 0
	for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				if (s - 1) in range(0,nds):
					jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
				s = s + 1

	return Qobj(jm_operator)

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

def jzn_t(p, n):
	"""
	Calculates <Jz^n(t)> given the flattened density matrix time evolution for the populations (jm), p. 
	 
	Parameters
	----------
	p: matrix 
		The time evolution of the density matrix rho(j,m)
	
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
	Calculates n-th moment of the total spin operator J2, that is <(J2)^n(t)> given the flattened density matrix time evolution for the populations (jm), p. 
	 
	Parameters
	----------
	p: matrix 
		The time evolution of the density matrix rho(j,m)
	
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
	Calculates <J_{+}^n J_{z}^q J_{-}^n>(t) given the flattened density matrix time evolution for the populations (jm). 
	 
	Parameters
	----------
	p: matrix 
		The time evolution of the density matrix rho(j,m)
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

def qobj_to_jm(rho_t):
		"""
		Converts rho_t, the list of Qobj matrices rho(j, m, m1), into p_t, an array of rho(j, m, m).
		Parameters
		----------
		rho_t: list of Qobj matrices
			The list dimension is nt = time steps. The matrix dimension is (nds, nds). 

		Returns
		-------
		v_t: ndarray
			A matrix (nt, nds). 
		"""
		p_t = np.asarray(rho_t)
		nt = np.shape(p_t)[0]
		nds = np.shape(p_t[0])[0]
		
		N = num_two_level(nds) 
		v_t = np.zeros(( nt, nds ))
		x_t = np.zeros(( nds, nds ))
		for i in range(0, nt):
			x_t = np.zeros(( nds, nds ))
			p_t[i] = p_t[i].full()
			p_t[i] = np.real(p_t[i])
			x_t = p_t[i]
			for kk in range(0,nds):
				v_t[i,kk] = x_t[kk,kk]
		return v_t

def e_ops_pim(rho_t, algebra, exponents):
	"""
	Calculate any product of operators momenta that includes Jz, J+, J-, in any order.

	Parameters
	----------
	rho_t: list of Qobj
		The reduced density matrix rho(j,m,m') of dimension (nds,nds), evolved in time (nt steps). 
		Object dimensions: list(nt). Each list element is a Qobj (nds,nds)
	algebra: tuple of strings
		The order of the operators Jz, J-, J+ such as ("Jz","J+","J-").
	exponents: tuple of int
		The exponents (a,b,c) corresponding to the operators in the 'algebra' list. 

	Returns
	-------
	e_ops: array (dtype=np.complex)
		The time-evolved function for the given operator momentum of dimension nt.

	Example:

	algebra_order = ("J+","Jz","J-")
	algebra_exponents = (2, 0, 2)
	e_ops = <(J+)^2(J-)^2>(t)
	"""
	
	rho_t = qobj_flatten(rho_t)
	nt = np.shape(rho_t)[0] 
	nds = np.shape(rho_t[0])[0]
	N = num_two_level(nds)
	num_ladders = num_dicke_ladders(N)
	e_ops = np.zeros(nt, dtype = np.complex)


	if algebra == ("Jz","J+","J-") or algebra == ("J+","Jz","J-") or algebra == ("J+","J-","Jz"):
		order = "order1"
		print("order 1 = ('Jz','J+','J-') or ('J+','Jz','J-') or ('J+','J-','Jz')")
	elif algebra == ("Jz","J-","J+") or algebra == ("J-","Jz","J+") or algebra == ("J-","J+","Jz"):
		order = "order2"
		print("order 2 = ('Jz','J-','J+') or ('J-','Jz','J+') or ('J-','J+','Jz')")
	else:
		raise ValueError("All three algebra operators need to be defined, e.g. ('J-','J+','Jz'). Exponents can be >=0")


	plus_index = algebra.index("J+")
	z_index = algebra.index("Jz")
	minus_index = algebra.index("J-")
	[t, s, r] = [int(exponents[plus_index]), int(exponents[z_index]), int(exponents[minus_index])]
	print("J+**{}, Jz**{}, J-**{}".format(t, s, r))
	if (t < 0) or (s < 0) or (r < 0):
		raise ValueError("The exponents need to be >= 0")
	if (t != exponents[plus_index]) or (s != exponents[z_index]) or (r != exponents[minus_index]):
		raise ValueError("The exponents need to be integers, but exponents = {}".format(exponents))


	print(order)
	print(order == "order1")
	if order == "order1":
		
		print("order 1 confirmed")

		#cycle on j,m
		for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i

				print("(j, m) = ",j,m)
				
				if (j - m + r - t) >= 0 :
					if (j + m - r) >= 0 :
						print("conditions 1 ok ")

						prod_p = 1
						prod_m = 1
						prod_z = 1

						for i1 in range(0, t):
							prod_p = prod_p * ap(j, m - r + i1)
						for i2 in range(0, r):
							prod_m = prod_m * am(j, m - i2)

						if algebra == ['J+', 'Jz', 'J-']:
							prod_z = (m - r)**s
						if algebra == ['J+', 'J-', 'Jz']:
							prod_z = m**s
						if algebra == ['Jz', 'J+', 'J-']:
							prod_z = (m - r + t)**s

						prod_mpz = prod_m * prod_p * prod_z
						print("prod_mpz ",prod_mpz)
						print("m - r + t", m - r + t)
						print("self._get_element_v((j, m, m - r + t))", self._get_element((j, m, m - r + t)))
						states = rho_t[:,self._get_element((j, m, m - r + t))]
						print("states ",states)
						e_ops[:] = e_ops[:] + states * prod_mpz 
						print("e_ops[:] ", e_ops[:])

	if order == "order2":
		
		print("order 2 confirmed")
		
		#cycle on j,m
		for k in range(0, num_ladders):
			j = 0.5 * N - k
			mmax = int(2 * j + 1)
			for i in range(0, mmax):
				m = j - i
				print("(j, m) = ",j,m)
				if (j + m + t - r) >= 0 :
					if (j - m - t) >= 0 :
						
						print("conditions 2 ok ")

						prod_p = 1
						prod_m = 1
						prod_z = 1

						for i1 in range(0, t):
							prod_p = prod_p * ap(j, m + i1)
						for i2 in range(0, r):
							prod_m = prod_m * am(j, m + t - i2)

						if algebra == ['J-', 'Jz', 'J+']:
							prod_z = (m + t)**s
						if algebra == ['J-', 'J+', 'Jz']:
							prod_z = m**s
						if algebra == ['Jz', 'J-', 'J+']:
							prod_z = (m - r + t)**s

						prod_mpz = prod_m * prod_p * prod_z
						print("m - r + t", m - r + t)
						states = rho_t[:,self._get_element((j, m, m - r + t))]
						e_ops[:] = e_ops[:] + states * prod_mpz
	return e_ops

def ap( j, m):
	"""
	Calculate A_{+} for value of j, m.
	"""
	a_plus = np.sqrt((j - m) * (j + m + 1))
	
	return(a_plus)

def am(j, m):
	"""
	Calculate A_{m} for value of j, m.
	"""
	a_minus = np.sqrt((j + m) * (j - m + 1))
	
	return(a_minus)

def qobj_flatten(rho_t):
	"""
	Converts rho_t, a list of Qobj matrices, into p_t, a numpy matrix. 
	Parameters
	----------
	rho_t: list of Qobj matrices
		The list dimension is nt = time steps. The matrix dimension is (nds, nds). 
	
	Returns
	-------
	p_t: ndarray
		A matrix (nt, nds**2). 
	"""
	
	p_t = np.asarray(rho_t)

	for i in range(0,np.shape(p_t)[0]):
		p_t[i] = p_t[i].full()
		p_t[i] = p_t[i].flatten()
	p_t = p_t.tolist()
	for i in range (len(p_t)):
		p_t[i] = list(p_t[i])
	p_t = np.array(p_t)

	return p_t
