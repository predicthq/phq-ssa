# phq-ssa
Singular Spectrum Analysis algorithm written in Python

Forked version of https://github.com/aj-cloete/pssa

## How to use
```python
import numpy as np
from phq import ssa

demands = np.array(range(100))

trajectory_matrix = ssa.embed(demands)
unitary_matrix, singular_values = ssa.decompose(trajectory_matrix)
reconstructed_demand = ssa.reconstruction(trajectory_matrix, unitary_matrix, singular_values, nsig=2)
```
