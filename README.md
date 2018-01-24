# Permutational Invariant Quantum Solver (PIQS)

PIQS is an open-source Python solver to study the exact Lindbladian dynamics of open quantum systems consisting of identical qubits.

## Exponential reduction 
In the case where local processes are included in the model of a system's dynamics, numerical simulation requires dealing with density matrices of size $2^N$. This becomes infeasible for a large number of qubits. We can simplify the calculations by exploiting the permutational invariance of indistinguishable quantum particles which allows the user to study hundreds of qubits.

## Integrated with QuTiP
A major feature of PIQS is that it allows to build the Liouvillian of the system in an optimal way. It uses Cython to optimize performance and by taking full advangtage of the sparsity of the matrix it can deal with large systems. Since it is compatible with the `quantum object` class of [QuTiP] one can take full advantage of existing features of this excellent open-source library.


## A wide range of applications

- The time evolution of the total density matrix $\rho(t)$ of cavity QED systems for permutationally symmetric initial states (such as the GHZ state, Dicke states, spin-squeezed states).
- Quantum phase transitions (QPT) of driven-dissipative out-of-equilibrium quantum systems.  
- Correlation functions of collective systems in quantum optics experiments, such as $S(\omega)$ and $g^{(2)}(\tau)$.
- Various quantum phenomena such as steady-state superradiance, superradiant light emission, superradiant phase transition, spin squeezing, boundary time crystals, optical bistability.

## Installation

```
git clone https://github.com/nathanshammah/piqs.git
cd piqs
python setup.py install
```

## Use

```
from piqs import Piqs
from qutip import *

N = 10

gamma = 1.
system = Dicke(N, emission = gamma)

L = system.louivillian()
steady_state = steadystate(L)
```

## License

PIQS is licensed under the terms of the BSD license.


## Resources
The code can be found in [1]. A paper detailing the theoretical aspects and illustrating many applications is in [2]. The original permutational invariant theory can be found here Chase and Geremia 2008. Some of the other existing open-source libraries for open quantum dynamics is *Permutations* [3] by Peter Kirton and *PsiQuaSP* by Michal Gegg. [4]


[1] https://github.com/nathanshammah/piqs

[2] https://arxiv.org

[3] https://github.com/peterkirton/permutations and P. Kirton and J. Keeling $Phys. Rev. Lett.$  118, 123602 (2017)

[4] https://github.com/modmido/psiquasp and M. Gegg and M. Richter, $Sci. Rep.$ 7, 16304 (2017)
