Introduction
============

Permutational Invariant Quantum Solver (PIQS)
---------------------------------------------

PIQS is an open-source Python solver to study the exact Lindbladian dynamics of open quantum systems consisting of identical qubits.

In the case where local processes are included in the model of a system's dynamics, numerical simulation requires dealing with density matrices of size :math:`2^N`. This becomes infeasible for a large number of qubits. We can simplify the calculations by exploiting the permutational invariance of indistinguishable quantum particles which allows the user to study hundreds of qubits.

Integrated with QuTiP
---------------------

A major feature of PIQS is that it allows to build the Liouvillian of the system in an optimal way. It uses Cython to optimize performance and by taking full advangtage of the sparsity of the matrix it can deal with large systems. Since it is compatible with the `quantum object` class of [QuTiP] one can take full advantage of existing features of this excellent open-source library.

A wide range of applications
----------------------------

- The time evolution of the total density matrix of quantum optics and cavity QED systems for permutationally symmetric initial states (such as the GHZ state, Dicke states, coherent spin states).
- Quantum phase transitions (QPT) of driven-dissipative out-of-equilibrium quantum systems.  
- Correlation functions of collective systems in quantum optics experiments, such as the spectral density and second-order correlation functions.
- Various quantum optics phenomena such as steady-state superradiance, superradiant light emission, superradiant phase transition, spin squeezing, boundary time crystals, resonance fluorescence.
