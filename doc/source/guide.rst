***************************
User Guide
***************************

The *Permutational Invariant Quantum Solver (PIQS)* is an open-source Python solver to study the exact Lindbladian dynamics of open quantum systems consisting of identical qubits. It is integrated in QuTiP and can be imported as as a model.

Using this library, the Liouvillian of an ensemble of :math:`N` qubits, or two-level systems 
(TLSs), :math:`\mathcal{D}_{TLS}(\rho)`, can be built using only polynomial – instead of exponential – resources. This has many applications for the study of realistic quantum optics models of many TLSs and in general as a tool in cavity QED [1].

Consider a system evolving according to the equation

.. math::
	\dot{\rho} = \mathcal{D}_\text{TLS}(\rho)=-\frac{i}{\hbar}\lbrack H,\rho \rbrack
	+\frac{\gamma_\text{CE}}{2}\mathcal{L}_{J_{-}}[\rho]
	+\frac{\gamma_\text{CD}}{2}\mathcal{L}_{J_{z}}[\rho]
	+\frac{\gamma_\text{CP}}{2}\mathcal{L}_{J_{+}}[\rho]

	+\sum_{n=1}^{N}\left(
	\frac{\gamma_\text{E}}{2}\mathcal{L}_{J_{-,n}}[\rho]
	+\frac{\gamma_\text{D}}{2}\mathcal{L}_{J_{z,n}}[\rho]
	+\frac{\gamma_\text{P}}{2}\mathcal{L}_{J_{+,n}}[\rho]\right) 


where :math:`J_{\alpha,n}=\frac{1}{2}\sigma_{\alpha,n}` are SU(2) Pauli spin operators, with :math:`{\alpha=x,y,z}` and :math:`J_{\pm,n}=\sigma_{\pm,n}`. The collective spin operators are :math:`J_{\alpha} = \sum_{n}J_{\alpha,n}` . The Lindblad super-operators are :math:`\mathcal{L}_{A} = 2A\rho A^\dagger - A^\dagger A \rho - \rho A^\dagger A`.

The inclusion of local processes in the dynamics lead to using a Liouvillian space of dimension :math:`4^N`. By exploiting the permutational invariance of identical particles [2-8], the Liouvillian :math:`\mathcal{D}_\text{TLS}(\rho)` can be built as a block-diagonal matrix in the basis of Dicke states :math:`|j, m \rangle`.

The system under study is defined by creating an object of the 
:code:`Dicke` class, e.g. simply named 
:code:`system`, whose first attribute is 

- :code:`system.N`, the number of TLSs of the system :math:`N`.

The rates for collective and local processes are simply defined as 

- :code:`collective_emission` defines :math:`\gamma_\text{CE}`, collective (superradiant) emission
- :code:`collective_dephasing` defines :math:`\gamma_\text{CD}`, collective dephasing 
- :code:`collective_pumping` defines :math:`\gamma_\text{CP}`, collective pumping. 
- :code:`emission` defines :math:`\gamma_\text{E}`, incoherent emission (losses) 
- :code:`dephasing` defines :math:`\gamma_\text{D}`, local dephasing 
- :code:`pumping`  defines :math:`\gamma_\text{P}`, incoherent pumping. 

Then the :code:`system.lindbladian()` creates the total TLS Linbladian superoperator matrix. Similarly, :code:`system.hamiltonian` defines the TLS hamiltonian of the system :math:`H_\text{TLS}`.

The system's Liouvillian can be built using :code:`system.liouvillian()`. The properties of a Piqs object can be visualized by simply calling 
:code:`system`. We give two basic examples on the use of *PIQS*. In the first example the incoherent emission of N driven TLSs is considered.

.. code-block: python

	from piqs import Dicke
	from qutip import steadystate

	N = 10
	system = Dicke(N, emission = 1, pumping = 3)

	L = system.liouvillian()
	steady = steadystate(L)


.. cssclass:: table-striped

+--------------------------+----------------------------+----------------------------------------+
| Operators                | Command (# means optional) | Inputs                                 |
+==========================+============================+========================================+
| Charge operator          | ``charge(N,M=-N)``         | Diagonal operator with entries         |
|                          |                            | from M..0..N.                          |
+--------------------------+----------------------------+----------------------------------------+
| Commutator               | ``commutator(A, B, kind)`` | Kind = 'normal' or 'anti'.             |
+--------------------------+----------------------------+----------------------------------------+
| Diagonals operator       | ``qdiags(N)``              | Quantum object created from arrays of  |
|                          |                            | diagonals at given offsets.            |
+--------------------------+----------------------------+----------------------------------------+
| Displacement operator    | ``displace(N,alpha)``      | N=number of levels in Hilbert space,   |
| (Single-mode)            |                            | alpha = complex displacement amplitude.|
+--------------------------+----------------------------+----------------------------------------+
| Higher spin operators    | ``jmat(j,#s)``             | j = integer or half-integer            |
|                          |                            | representing spin, s = 'x', 'y', 'z',  |
|                          |                            | '+', or '-'                            |
+--------------------------+----------------------------+----------------------------------------+
| Identity                 | ``qeye(N)``                | N = number of levels in Hilbert space. |
+--------------------------+----------------------------+----------------------------------------+
| Lowering (destruction)   | ``destroy(N)``             | same as above                          |
| operator                 |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Momentum operator        | ``momentum(N)``            | same as above                          |
+--------------------------+----------------------------+----------------------------------------+
| Number operator          | ``num(N)``                 | same as above                          |
+--------------------------+----------------------------+----------------------------------------+
| Phase operator           | ``phase(N, phi0)``         | Single-mode Pegg-Barnett phase         |
| (Single-mode)            |                            | operator with ref phase phi0.          |
+--------------------------+----------------------------+----------------------------------------+
| Position operator        | ``position(N)``            | same as above                          |
+--------------------------+----------------------------+----------------------------------------+
| Raising (creation)       | ``create(N)``              | same as above                          |
| operator                 |                            |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Squeezing operator       | ``squeeze(N, sp)``         | N=number of levels in Hilbert space,   |
| (Single-mode)            |                            | sp = squeezing parameter.              |
+--------------------------+----------------------------+----------------------------------------+
| Squeezing operator       | ``squeezing(q1, q2, sp)``  | q1,q2 = Quantum operators (Qobj)       |
| (Generalized)            |                            | sp = squeezing parameter.              |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-X                  | ``sigmax()``               |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-Y                  | ``sigmay()``               |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma-Z                  | ``sigmaz()``               |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma plus               | ``sigmap()``               |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Sigma minus              | ``sigmam()``               |                                        |
+--------------------------+----------------------------+----------------------------------------+
| Tunneling operator       | ``tunneling(N,m)``         | Tunneling operator with elements of the|
|                          |                            | form :math:`|N><N+m| + |N+m><N|`.      |
+--------------------------+----------------------------+----------------------------------------+


.. toctree::
   :maxdepth: 2

   examples/superradiant_light_emission
   examples/superradiance

