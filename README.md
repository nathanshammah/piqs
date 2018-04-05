# Permutational Invariant Quantum Solver (PIQS)
[![DOI](https://zenodo.org/badge/104438298.svg)](https://zenodo.org/badge/latestdoi/104438298)

PIQS is an open-source Python solver to study the exact Lindbladian dynamics of open quantum systems consisting of identical qubits.

## Exponential reduction 
In the case where local processes are included in the model of a system's dynamics, numerical simulation requires dealing with density matrices of size $2^N$. This becomes infeasible for a large number of qubits. We can simplify the calculations by exploiting the permutational invariance of indistinguishable quantum particles which allows the user to study hundreds of qubits.

## Integrated with QuTiP
A major feature of PIQS is that it allows to build the Liouvillian of the system in an optimal way. It uses Cython to optimize performance and by taking full advangtage of the sparsity of the matrix it can deal with large systems. Since it is compatible with the `quantum object` class of [QuTiP] one can take full advantage of existing features of this excellent open-source library.


## A wide range of applications

- The time evolution of the total density matrix of quantum optics and cavity QED systems for permutationally symmetric initial states (such as the GHZ state, Dicke states, coherent spin states).
- Quantum phase transitions (QPT) of driven-dissipative out-of-equilibrium quantum systems.  
- Correlation functions of collective systems in quantum optics experiments, such as the spectral density and second-order correlation functions.
- Various quantum optics phenomena such as steady-state superradiance, superradiant light emission, superradiant phase transition, spin squeezing, boundary time crystals, resonance fluorescence.

## Installation

In the terminal enter the following commands (you just need `git` and `python` installed)
```
git clone https://github.com/nathanshammah/piqs.git
cd piqs
python setup.py install
```

## Use

```
from piqs import Piqs
from qutip import steadystate

N = 10

gamma = 1.
system = Piqs(N, emission = gamma)

L = system.liouvillian()
steady_state = steadystate(L)
```
For more details and examples on the use of *PIQS* see the doc/notebook folder. 

## License

PIQS is licensed under the terms of the BSD license.


## Resources
The code and an introductory notebook can be found in Ref. [1]. A paper detailing the theoretical aspects and illustrating many applications is in Ref. [2]. Related open-source libraries for open quantum dynamics that exploit permutational invariance are *Permutations* [3] by Peter Kirton and *PsiQuaSP* by Michael Gegg [4].

[1] https://github.com/nathanshammah/notebooks

[2] N. Shammah, S. Ahmed, N. Lambert, S. De Liberato, and F. Nori, *to be submitted*

[3] https://github.com/peterkirton/permutations P. Kirton and J. Keeling *Phys. Rev. Lett.*  **118**, 123602 (2017)

[4] https://github.com/modmido/psiquasp M. Gegg and M. Richter, *Sci. Rep.* **7**, 16304 (2017)
