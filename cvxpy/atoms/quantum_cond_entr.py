from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.atoms.quantum_rel_entr import quantum_rel_entr

import numpy as np


# if sys == 1
#     cvx_optval = -quantum_rel_entr(rho,kron(eye(dim(1)),TrX(rho,sys,dim)));
# elseif sys == 2
#     cvx_optval = -quantum_rel_entr(rho,kron(TrX(rho,sys,dim),eye(dim(2))));
# end

def quantum_cond_entr(rho, dim: list[int], sys=0):
    if sys == 0:
        composite_arg = kron(np.eye(dim[0]),
                             partial_trace(rho, dim, sys))
        return -quantum_rel_entr(rho, composite_arg)
    elif sys == 1:
        composite_arg = kron(partial_trace(rho, dim, sys),
                             np.eye(dim[1]))
        return -quantum_rel_entr(rho, composite_arg)
