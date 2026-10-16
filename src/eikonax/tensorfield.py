r"""Composable and differentiable parameter tensor fields.

This module provides callable local metric modules for differentiable parameter fields. Recall that
for the Eikonax solver, and particularly parametric derivatives, we require an input tensor field
$\mathbf{M}: \mathbb{R}^M \times \mathbb{N}_0 \to \mathbb{S}_+^{d\times d}$. This means that the
tensor field is a mapping $\mathbf{M}(\mathbf{m},s)$ that assigns, given a global parameter vector
$\mathbf{m}$, an s.p.d tensor to every simplex $s$ in the mesh. To allow for sufficient flexibility
in the choice of tensor field, each simplex uses a fixed-width row of `parameter_inds` to gather
its local parameters. An [`AbstractSimplexTensor`][eikonax.tensorfield.AbstractSimplexTensor]
callable maps those local parameters to the simplex metric. [`TensorField`][eikonax.tensorfield.TensorField]
vectorizes the callable and uses local JAX VJPs for derivative contractions.

Classes:
    AbstractSimplexTensor: ABC interface contract for local simplex metrics
    LinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    InvLinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    TensorField: Tensor field component
"""

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal

# ==================================================================================================
class AbstractSimplexTensor(eqx.Module):
    """ABC interface contract for local simplex metrics.

    `SimplexTensor` components map local parameters to a metric tensor. The relevant parameters are
    gathered from the global parameter vector using the TensorField's fixed-width
    ``parameter_inds`` array.

    !!! note
        This class provides the metric tensor as used in the inner product for the update stencil of
        the eikonal equation. This is the **inverse** of the conductivity tensor, which is the
        actual tensor field in the eikonal equation.

    Methods:
        __call__: Map local parameters to one simplex metric
    """

    @abstractmethod
    def __call__(
        self, parameters: jtFloat[jax.Array, "num_parameters_local"]
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Map local parameters to one simplex metric."""
        raise NotImplementedError

    # Equinox modules are data classes, so we have to define attributes at the class level
    dimension: int

# ==================================================================================================
@final
class LinearScalarSimplexTensor(AbstractSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the metric is $m_s \cdot \mathbf{I}$, where
    $\mathbf{I}$ is the identity matrix.

    The callable accepts a one-element local parameter vector.
    """

    # ----------------------------------------------------------------------------------------------
    def __call__(
        self, parameters: jtFloat[jax.Array, "num_parameters_local"]
    ) -> jtFloat[jax.Array, "dim dim"]:
        return parameters[0] * jnp.identity(self.dimension, dtype=jnp.float32)

# ==================================================================================================
@final
class InvLinearScalarSimplexTensor(AbstractSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the metric is
    $\frac{1}{m_s} \cdot \mathbf{I}$, where $\mathbf{I}$ is the identity matrix.

    The callable accepts a one-element local parameter vector.
    """

    # ----------------------------------------------------------------------------------------------
    def __call__(
        self, parameters: jtFloat[jax.Array, "num_parameters_local"]
    ) -> jtFloat[jax.Array, "dim dim"]:
        return jnp.identity(self.dimension, dtype=jnp.float32) / parameters[0]

# ==================================================================================================
class TensorField(eqx.Module):
    r"""Tensor field component.

    Tensor fields gather fixed-width local parameter subsets for each simplex and call a local
    metric module. They provide the full mapping $\mathbf{M}(\mathbf{m})$ and local VJPs without
    materializing a global tensor-field Jacobian.

    Methods:
        assemble_field: Assemble the tensor field for the given parameter vector
        local_vjp: Differentiate local simplex metrics with a cotangent
    """

    # Equinox modules are data classes, so we have to define attributes at the class level
    _num_simplices: int
    _parameter_inds: jtInt[jax.Array, "num_simplices num_parameters_local"]
    _simplex_tensor: AbstractSimplexTensor

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        parameter_inds: jtInt[jax.Array | npt.NDArray, "num_simplices num_parameters_local"],
        simplex_tensor: AbstractSimplexTensor,
    ) -> None:
        """Constructor.

        Takes fixed-width global parameter indices for each simplex and a local metric callable.

        Args:
            parameter_inds (jax.Array | npt.NDArray): Global parameter indices for each simplex.
            simplex_tensor (AbstractSimplexTensor): Callable local metric assembly.
        """
        parameter_inds = jnp.array(parameter_inds, dtype=jnp.int32)
        if parameter_inds.ndim != 2:
            raise ValueError("parameter_inds must have shape (num_simplices, num_parameters_local).")
        self._num_simplices = parameter_inds.shape[0]
        self._parameter_inds = parameter_inds
        self._simplex_tensor = simplex_tensor

    @property
    def num_simplices(self) -> int:
        """Number of simplices represented by the field."""
        return self._num_simplices

    @property
    def parameter_inds(self) -> jtInt[jax.Array, "num_simplices num_parameters_local"]:
        """Fixed-width global parameter indices for each simplex."""
        return self._parameter_inds

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def assemble_field(
        self, parameter_vector: jtFloat[jax.Array | npt.NDArray, "num_parameters"]
    ) -> jtFloat[jax.Array, "num_simplices dim dim"]:
        """Assemble global tensor field from global parameter vector.

        The fixed-width parameter index rows gather local parameters before vectorizing the
        simplex tensor callable over all simplices.

        Args:
            parameter_vector (jax.Array | npt.NDArray): Global parameter vector.

        Returns:
            jax.Array: Global tensor field with shape (num_simplices, dim, dim).
        """
        parameter_vector = jnp.array(parameter_vector, dtype=jnp.float32)
        local_parameters = parameter_vector[self._parameter_inds]
        return jax.vmap(self._simplex_tensor)(local_parameters)

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def local_vjp(
        self,
        simplex_inds: jtInt[jax.Array, "num_simplices_local"],
        cotangents: jtFloat[jax.Array, "num_simplices_local dim dim"],
        parameter_vector: jtFloat[jax.Array, "num_parameters"],
    ) -> jtFloat[jax.Array, "num_simplices_local num_parameters_local"]:
        """Apply local metric VJPs without materializing a TensorField Jacobian."""
        simplex_inds = jnp.maximum(simplex_inds, 0)
        local_parameters = parameter_vector[self._parameter_inds[simplex_inds]]

        def pullback(parameters, cotangent):
            _, vjp_fun = jax.vjp(self._simplex_tensor, parameters)
            return vjp_fun(cotangent)[0]

        values = jax.vmap(pullback)(local_parameters, cotangents)
        return values
