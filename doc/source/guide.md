# User Guide

```
from piqs import Dicke
from qutip import steadystate

N = 10
system = Dicke(N, emission = 1, pumping = 3)

L = system.liouvillian()
steady = steadystate(L)
```