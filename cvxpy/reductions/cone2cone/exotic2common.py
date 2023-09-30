"""
Copyright 2021 the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.solution import Solution

EXOTIC_CONES = {
    PowConeND: {PowCone3D}
}
"""
^ An "exotic" cone is defined as any cone that isn't
supported by ParamConeProg. If ParamConeProg is updated
to support more cones, then it may be necessary to change
this file.
"""


def pow_nd_canon(con, args):
    """
    con : PowConeND
        We can extract metadata from this.
        For example, con.alpha and con.axis.
    args : tuple of length two
        W,z = args[0], args[1]
    """
    alpha, axis, _ = con.get_data()
    alpha = alpha.value
    W, z = args
    if axis == 1:
        W = W.T
        alpha = alpha.T
    if W.ndim == 1:
        W = reshape(W, (W.size, 1))
        alpha = np.reshape(alpha, (W.size, 1))
    n, k = W.shape
    if n == 2:
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        T = Variable(shape=(n-2, k))
        x_3d, y_3d, z_3d, alpha_3d = [], [], [], []
        for j in range(k):
            x_3d.append(W[:-1, j])
            y_3d.append(T[:, j])
            y_3d.append(W[n-1, j])
            z_3d.append(z[j])
            z_3d.append(T[:, j])
            r_nums = alpha[:, j]
            r_dens = np.cumsum(r_nums[::-1])[::-1]
            # ^ equivalent to [np.sum(alpha[i:, j]) for i in range(n)]
            r = r_nums / r_dens
            alpha_3d.append(r[:n-1])
        x_3d = hstack(x_3d)
        y_3d = hstack(y_3d)
        z_3d = hstack(z_3d)
        alpha_p3d = hstack(alpha_3d)
        # TODO: Ideally we should construct x,y,z,alpha_p3d by
        #   applying suitable sparse matrices to W,z,T, rather
        #   than using the hstack atom. (hstack will probably
        #   result in longer compile times).
        can_con = PowCone3D(x_3d, y_3d, z_3d, alpha_p3d)
    # Return a single PowCone3D constraint defined over all auxiliary
    # variables needed for the reduction to go through.
    # There are no "auxiliary constraints" beyond this one.
    return can_con, []


class Exotic2Common(Canonicalization):

    CANON_METHODS = {
        PowConeND: pow_nd_canon
    }

    def __init__(self, problem=None) -> None:
        super(Exotic2Common, self).__init__(
            problem=problem, canon_methods=Exotic2Common.CANON_METHODS)

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}

        dv = {}
        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND):
                idx = int(f'{cons_id}')
                div_size = int(dvars[idx].shape[1] / cons.args[1].shape[0])
                dv[idx] = []
                for i in range(cons.args[1].shape[0]):
                    # Iterating over the vectorized constraints
                    # idx = int(f'{cons_id}')
                    # dv[idx] = []
                    dv[idx].append([])
                    tmp_duals = dvars[idx][:, i * div_size: (i + 1) * div_size]
                    # tmp_duals = dvars[idx]
                    for j, col_dvars in enumerate(tmp_duals.T):
                        if j == len(tmp_duals.T) - 1:
                            dv[idx][-1] += [col_dvars[0], col_dvars[1]]
                        else:
                            dv[idx][-1].append(col_dvars[0])
                    dv[idx][-1].append(tmp_duals.T[0][-1]) # dual value corresponding to `z`
                dvars[idx] = np.array(dv[idx])

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
