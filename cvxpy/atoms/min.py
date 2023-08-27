"""
Copyright 2013 Steven Diamond

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
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from cvxpy.error import NotDifferentiableError

from cvxpy.utilities import scopes

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.expressions import cvxtypes


class min(AxisAtom):
    """:math:`\\min{i,j}\\{X_{i,j}\\}`.
    """

    __EXPR_AXIS_ERROR__ = """

    The second argument to "min" was a cvxpy Expression, when it should have
    been an int or None. This is probably a result of calling "cp.min" when you
    should call "cp.minimum". The difference is that cp.min represents the
    minimum entry in a single vector or matrix, while cp.minimum represents
    the entry-wise min of a sequence of arguments that all have the same shape.

    """

    def __init__(self, x, axis: Optional[int] = None, keepdims: bool = False) -> None:
        if isinstance(axis, cvxtypes.expression()):
            raise ValueError(min.__EXPR_AXIS_ERROR__)
        super(min, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the smallest entry in x.
        """
        return values[0].min(axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        # Grad: 1 for a largest index.
        if scopes.strict_differentiability_active():
            if not self._is_differentiable_at(value):
                raise NotDifferentiableError
        value = np.array(value).ravel(order='F')
        idx = np.argmin(value)
        D = np.zeros((value.size, 1))
        D[idx] = 1
        return D

    def _is_differentiable_at(self, point: cvxtypes.constant() | cvxtypes.variable() | npt.ArrayLike) -> bool:
        """Non-Differentiable in the case of a repeated maximum"""
        if isinstance(point, np.ndarray):
            point = point
        else:
            point = point.value
        smallest_value = point.flatten().min()
        second_smallest_value = cp.sum_smallest(point, 2).value - smallest_value
        if np.isclose(smallest_value, second_smallest_value, rtol=1e-8, atol=1e-4):
            return False
        else:
            return True

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Same as argument.
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return self.args[0].is_pwl()
