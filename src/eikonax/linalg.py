r"""Linear algebra utilities and sparse data structures for Eikonax.

This module provides specialized sparse data structures and utilities tailored to the Eikonax
solver's derivative computation workflow. Since the derivative operators are inherently sparse due
to the local nature of the Godunov update scheme, these structures efficiently store only the
non-zero entries along with their connectivity information.

The module defines three main sparse container classes:

1. [`jax.experimental.sparse.BCOO`](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
   sparse matrix representations for derivative operators.

2. [`DerivatorSparseTensor`][eikonax.linalg.DerivatorSparseTensor]: Specialized tensor for storing
   partial derivatives $\mathbf{G}_M$ computed by the
   [`PartialDerivator`][eikonax.derivator.PartialDerivator]. Pairs derivative values with simplex
   adjacency information for each vertex.

The key operation in this module is the tensor contraction
[`contract_derivative_tensors`][eikonax.linalg.contract_derivative_tensors], which efficiently
combines the derivatives from the Eikonax solver with local VJPs from the tensor field to compute
the total parametric derivative $\mathbf{G}_m = \mathbf{G}_M \frac{d\mathbf{M}}{d\mathbf{m}}$.

Classes:
    DerivatorSparseTensor: Sparse tensor for partial derivatives w.r.t. metric tensors
Functions:
    contract_derivative_tensors: Contract DerivatorSparseTensor with local tensor-field VJPs
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


class DerivatorSparseTensor(eqx.Module):
    r"""Sparse tensor for partial derivatives from the PartialDerivator.

    This container stores the partial derivatives $\mathbf{G}_M$ of the Eikonax update operator
    with respect to the metric tensor field. For each vertex, it stores derivative contributions
    from all adjacent simplices in a dense local format, along with indices identifying which
    simplices are adjacent. This structure efficiently represents the sparse global tensor while
    maintaining JAX compatibility.

    The derivative values capture how changes in the metric tensor of each adjacent simplex affect
    the update value at each vertex. The adjacency data maps these local contributions to the
    global simplex numbering.

    Attributes:
        derivative_values (jax.Array): Partial derivative tensors with shape
            (num_vertices, max_num_neighbors, dim, dim). For each vertex and adjacent simplex,
            stores the (dim x dim) derivative of the update w.r.t. that simplex's metric tensor.
        adjacent_simplex_data (jax.Array): Global simplex indices with shape
            (num_vertices, max_num_neighbors). Maps local neighbor index to global simplex index.
            Entries of -1 indicate padding for vertices with fewer than max_num_neighbors adjacent
            simplices.
    """

    derivative_values: jtReal[jax.Array, "num_vertices max_num_neighbors dim dim"]
    adjacent_simplex_data: jtInt[jax.Array, "num_vertices max_num_neighbors"]


# ==================================================================================================
@eqx.filter_jit
def contract_derivative_tensors(
    derivative_sparse_tensor: DerivatorSparseTensor,
    tensor_field,
    parameter_vector: jtReal[jax.Array, "num_parameters"],
) -> sparse.BCOO:
    r"""Contract derivative tensors to compute total parametric derivative.

    This function performs the key operation in Eikonax's gradient computation pipeline: combining
    the partial derivatives from the solver ($\mathbf{G}_M$) with the tensor field Jacobian
    ($\frac{d\mathbf{M}}{d\mathbf{m}}$) using local vector-Jacobian products to obtain the total
    parametric derivative $\mathbf{G}_m = \mathbf{G}_M \frac{d\mathbf{M}}{d\mathbf{m}}$.

    The contraction is performed efficiently by exploiting the sparse solver derivative and the
    fixed-width local parameter connectivity:

    1. For each vertex, iterate over adjacent simplices
    2. Extract the (dim x dim) derivative matrix for that vertex-simplex pair from
       `derivative_sparse_tensor`
    3. Use the metric derivative as a cotangent for the tensor field's local VJP
    4. Scatter local derivatives to the appropriate global parameter indices
    5. Assemble all contributions into a sparse matrix of shape (num_vertices, num_parameters)

    The operation is vectorized over all vertices using JAX's `vmap` and JIT-compiled for
    performance.

    Args:
        derivative_sparse_tensor (DerivatorSparseTensor): Partial derivatives $\mathbf{G}_M$
            from the [`PartialDerivator`][eikonax.derivator.PartialDerivator].
        tensor_field (TensorField): Tensor field providing the local metric callable and fixed
            global parameter indices for each simplex.
        parameter_vector (jax.Array): Global tensor-field parameter vector.

    Returns:
        sparse.BCOO: Total parametric derivative $\mathbf{G}_m$ with shape
            (num_vertices, num_parameters_global). This can be transposed and multiplied with
            the adjoint vector to compute parametric gradients.
    """
    global_contraction_function = jax.vmap(
        _contract_vertex_tensors, in_axes=(0, 0, None, None)
    )
    values, col_inds = global_contraction_function(
        derivative_sparse_tensor.derivative_values,
        derivative_sparse_tensor.adjacent_simplex_data,
        tensor_field,
        parameter_vector,
    )
    num_vertices = derivative_sparse_tensor.derivative_values.shape[0]
    num_parameters_per_simplex = tensor_field.parameter_inds.shape[1]
    max_num_adjacent_simplices = derivative_sparse_tensor.adjacent_simplex_data.shape[1]
    row_inds = jnp.repeat(
        jnp.arange(num_vertices), max_num_adjacent_simplices * num_parameters_per_simplex
    )
    eikonax_sparse_matrix = sparse.BCOO(
        (
            values.flatten(),
            jnp.stack((row_inds, col_inds.flatten()), axis=1),
        ),
        shape=(num_vertices, parameter_vector.shape[0]),
    )
    return eikonax_sparse_matrix


# --------------------------------------------------------------------------------------------------
def _contract_vertex_tensors(
    derivator_tensor: jtReal[jax.Array, "max_num_neighbors dim dim"],
    adjacent_simplex_data: jtInt[jax.Array, "max_num_neighbors"],
    tensor_field,
    parameter_vector: jtReal[jax.Array, "num_parameters"],
) -> tuple[
    jtReal[jax.Array, "max_num_neighbors num_parameters_mapped"],
    jtInt[jax.Array, "max_num_neighbors num_parameters_mapped"],
]:
    """Contract derivatives for a single vertex with tensor field Jacobian.

    This helper function performs the tensor contraction for a single vertex. For each adjacent
    simplex, it:

    1. Extracts the (dim x dim) derivative matrix from the derivator tensor
    2. Extracts the (dim x dim x num_params) Jacobian tensor for that simplex
    3. Contracts them via Einstein summation: result[k] = sum_ij derivator[i,j] * jacobian[i,j,k]
    4. Maps the result to the global parameter indices for that simplex

    Invalid simplices (indicated by simplex_ind == -1) are handled by filtering: their
    contributions are set to zero and their column indices set to -1.

    This function is called in a vectorized manner by
    [`contract_derivative_tensors`][eikonax.linalg.contract_derivative_tensors] via `jax.vmap`.

    Args:
        derivator_tensor (jax.Array): Partial derivatives for one vertex with shape
            (max_num_neighbors, dim, dim).
        adjacent_simplex_data (jax.Array): Global simplex indices for one vertex with shape
            (max_num_neighbors,). Entries of -1 indicate padding.
        tensorfield_data (jax.Array): Global tensor field Jacobian with shape
            (num_simplices, dim, dim, num_parameters_mapped).
        parameter_inds (jax.Array): Global parameter indices for all simplices with shape
            (num_simplices, num_parameters_mapped).

    Returns:
        tuple[jax.Array, jax.Array]: Contracted derivative values with shape
            (max_num_neighbors, num_parameters_mapped) and corresponding column indices with
            the same shape. Entries corresponding to invalid simplices are zeroed/set to -1.
    """
    simplex_inds = jnp.maximum(adjacent_simplex_data, 0)
    cotangents = derivator_tensor
    values = tensor_field.local_vjp(simplex_inds, cotangents, parameter_vector)
    valid = adjacent_simplex_data[:, None] != -1
    values = jnp.where(valid, values, 0.0)
    cols = jnp.broadcast_to(
        tensor_field.parameter_inds[simplex_inds],
        values.shape,
    )
    cols = jnp.where(valid, cols, -1)
    return values, cols
