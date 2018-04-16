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
	system = Dicke(N, emission = 1, pumping = 2)

	L = system.liouvillian()
	steady = steadystate(L)


.. cssclass:: table-striped

+--------------------------+----------------------------+----------------------------------------+
| Operators                | Command (# means optional) | Inputs                                 |
+==========================+============================+========================================+
| Collective spin Jx       | ``jspin(N, "x")``          | N = number of two-level systems        |
|                          |                            | "x" = axis of the spin                 |+--------------------------+----------------------------+----------------------------------------+
| Collective spin Jy       | ``jspin(N, "y")``          | N = number of two-level systems        |
|                          |                            | "y" = axis of the spin                 |+--------------------------+----------------------------+----------------------------------------+
| Collective spin Jz       | ``jspin(N, "z")``          | N = number of two-level systems        |
|                          |                            | "z" = axis of the spin                 |+--------------------------+----------------------------+----------------------------------------+
| Dicke state,  |j, m>     | ``dikce(N, j, m)``         | N = number of two-level systems        |
|                          |                            | j = total spin, m = spin z-projection  |+--------------------------+----------------------------+----------------------------------------+
| Coherent spin state      | ``css(N, #a, #b)``         | N = number of two-level systems        |
|                          |                            | a = coefficient of |1>_i               |
|                          |                            | b = coefficient of |0>_i               |
+--------------------------+----------------------------+----------------------------------------+
| GHZ state                | ``ghz(N)``                 | N = number of two-level systems        |
+--------------------------+----------------------------+----------------------------------------+
| Number of Dicke states   | ``num_dicke_states(N)``    | N = number of two-level systems        |
+--------------------------+----------------------------+----------------------------------------+
|Number of TLSs            | ``num_tls(N)``             | nds = number of Dicke states           |
+--------------------------+----------------------------+----------------------------------------+

.. toctree::
   :maxdepth: 2

   examples/superradiant_light_emission
   examples/superradiance

