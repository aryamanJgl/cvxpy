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
import cvxpy as cp

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.error import NotDifferentiableError
from cvxpy.expressions import cvxtypes


class max(AxisAtom):
    """:math:`\\max_{i,j}\\{X_{i,j}\\}`.
    """

    __EXPR_AXIS_ERROR__ = """

    The second argument to "max" was a cvxpy Expression, when it should have
    been an int or None. This is probably a result of calling "cp.max" when you
    should call "cp.maximum". The difference is that cp.max represents the
    maximum entry in a single vector or matrix, while cp.maximum represents
    the entry-wise max of a sequence of arguments that all have the same shape.

    """

    def __init__(self, x, axis: Optional[int] = None, keepdims: bool = False) -> None:
        if isinstance(axis, cvxtypes.expression()):
            raise ValueError(max.__EXPR_AXIS_ERROR__)
        super(max, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest entry in x.
        """
        return values[0].max(axis=self.axis, keepdims=self.keepdims)

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
        if self._is_differentiable_at(value):
            value = np.array(value).ravel(order='F')
            idx = np.argmax(value)
            D = np.zeros((value.size, 1))
            D[idx] = 1
            return D
        else:
            raise NotDifferentiableError

    def _is_differentiable_at(self, point: cvxtypes.constant() | cvxtypes.variable()) -> bool:
        """Checks if the function is differentiable at `point`"""
        if self.axis is not None:
            largest_vector = cp.max(point, axis=self.axis)
            f = lambda x: np.allclose(largest_vector.value, x.value)
            if self.axis == 1:
                t_value = list(np.apply_along_axis(f, axis=0, arr=point.value))
            else:
                t_value = list(np.apply_along_axis(f, axis=1, arr=point.value))
            if t_value.count(True) > 1:
                return False
            else:
                return True
        else:
            largest = cp.max(point)
            second_largest = cp.sum_largest(point, 2) - largest
            if np.allclose(largest.value, second_largest.value, rtol=1e8, atol=1e4):
                return True
            else:
                return False

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Same as argument.
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

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
