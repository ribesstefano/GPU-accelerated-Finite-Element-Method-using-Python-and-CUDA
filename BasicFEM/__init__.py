import numpy as np
from BasicFEM.grid import Grid, DofHandler
from BasicFEM.material import Elasticity
from BasicFEM.weak_form import MomentumBalance, CST