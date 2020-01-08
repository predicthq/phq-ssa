# phq-ssa
Singular Spectrum Analysis algorithm written in Python

Forked version of https://github.com/aj-cloete/pssa

## How to use
```python
import numpy as np
from phq import ssa

demands = np.array(range(100))

trajectory_matrix = ssa.embed(demands)
reconstructed_demands, rank_reconstructed_trajectory_matrix = ssa.reconstruction(trajectory_matrix, contribution_proportion=0.9)
```
