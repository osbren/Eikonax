# Parametric Derivatives

In the [Forward Solver](./solve.md) tutorial, we have shown how to solve the Eikonal equation with
Eikonax for a given mesh and tensor field. In this tutorial, we build on this procedure to obtain
parametric derivatives $\frac{d\mathbf{u}(\mathbf{M})}{d\mathbf{M}}$ of the solution w.r.t.
a given input tensor field. We further assume that a tensor field is defined via some parameter vector
$\mathbf{m}\in\mathbb{R}^M$, s.th. we can define the mapping $\mathbf{M}: \mathbb{R}^M \to \mathbb{R}^{N_S\times d\times d}$,
under the constraint that $\mathbf{M}$ is pointwise s.p.d.

In the following, we consider the scenario of having some loss functional
$l: \mathbb{R}^{N_V} \to \mathbb{R},\ l = l(\mathbf{u})$, depending on the solution $\mathbf{u}(\mathbf{M})$
of the eikonal equation for a specific tensor field $\mathbf{M}(\mathbf{m})$. In various problem settings,
such as minimization of the loss, it is essential to be able to obtain the gradient $\mathbf{g}$ w.r.t. the input parameter
vector,

$$
    \mathbf{g}(\mathbf{m}) = \frac{d l(\mathbf{u})}{d\mathbf{u}}\frac{d\mathbf{u}(\mathbf{M})}{d\mathbf{M}}\frac{d\mathbf{M}(\mathbf{m})}{d\mathbf{m}}.
$$

This is the scenario we cover in this tutorial. Eikonax follows a *discretize-then-optimize* approach
to computing the gradient. Moreover, it efficiently computes discrete adjoints by exploiting the causality in the
forward solution of the eikonal equation. A detailed description of this procedure is given
[here][eikonax.derivator.DerivativeSolver].

## Test Mesh Setup

We start by setting up the same square mesh as for the [Forward Solver](./solve.md) tutorial.
However, we rely on Eikonax' built-in [create_test_mesh][eikonax.preprocessing.create_test_mesh] function,
instead of using `scipy`. We also choose a much smaller mesh with $3\times 3$ vertices, to efficiently
compare derivatives against finite differences later.

```py
from eikonax import corefunctions, preprocessing

vertices, simplices = preprocessing.create_test_mesh((0, 1), (0, 1), 3, 3)
mesh_data = preprocessing.MeshData(vertices, simplices)
```

## Tensor Field Setup

In the [Forward Solver](./solve.md) tutorial, we have constructed a specific tensor field instance as a simple
`numpy` array. To evaluate derivatives, however, we need to properly define a mapping $\mathbf{M}(\mathbf{m})$,
and its derivative. Such a mapping is provided by the [tensorfield][eikonax.tensorfield] module.
The [`SimplexTensor`][eikonax.tensorfield.AbstractSimplexTensor] describes how a local parameter
vector $\mathbf{m}_s$ is mapped to the tensor $M_s$ for a simplex. The
[`TensorField`][eikonax.tensorfield.TensorField] gathers each simplex's local parameters using its
fixed-width `parameter_inds` array and applies the local tensor's `assemble` method.

!!! info
    This structure keeps parameter connectivity explicit while allowing arbitrary differentiable
    local tensor assembly methods, which are vectorized and differentiated by JAX.

In our example, we define local tensors with the built-in [InvLinearScalarSimplexTensor][eikonax.tensorfield.InvLinearScalarSimplexTensor]. This assembles the local tensor from a scalar $m_s > 0$
simply as $\mathbf{M}_s = \frac{1}{m_s}\mathbf{I}$. The fixed parameter-index array selects
$m_s = \mathbf{m}[s]$. In total, we create our tensor field like this:
```py
from eikonax import tensorfield

tensor_on_simplex = tensorfield.InvLinearScalarSimplexTensor(dimension=vertices.shape[1])
tensor_field_object = tensorfield.TensorField(
    parameter_inds=jnp.arange(simplices.shape[0], dtype=jnp.int32)[:, None],
    simplex_tensor=tensor_on_simplex,
)
```

The `tensor_field_object` is an intelligent mapping for any valid input vector $\mathbf{m}$. For
demonstration purposes, we simply create a random input vector and build the tensor field with the
[`assemble_field`][eikonax.tensorfield.TensorField.assemble_field] method,
```py
import numpy as np

rng = np.random.default_rng(seed=0)
parameter_vector = rng.uniform(0.5, 1.5, simplices.shape[0])
tensor_field_instance = tensor_field_object.assemble_field(parameter_vector)
```


## Solver Setup and Run

We now have all components to conduct a forward solver run with Eikonax, analogously to the one
described in the [Forward Solver](./solve.md) tutorial.
```py
from eikonax import solver

solver_data = solver.SolverData(
    tolerance=1e-8,
    max_num_iterations=1000,
    loop_type="jitted_while",
    max_value=1000,
    use_soft_update=False,
    softminmax_order=10,
    softminmax_cutoff=0.01,
)
initial_site_inds = jnp.array((0,), dtype=jnp.int32)
eikonax_solver = solver.Solver(
    mesh_data,
    solver_data,
    tensor_field_object,
    initial_site_inds=initial_site_inds,
)
solution = eikonax_solver.run(parameter_vector, initial_site_values=jnp.array([0.0]))
```

## Automatic Gradients

The solver integrates the derivative machinery into its parameter path. Users can therefore apply
`jax.grad` to any scalar objective built from `Solver.run` without constructing a
`PartialDerivator` or `DerivativeSolver` explicitly:

```py
import jax

def objective(parameter_vector):
    solution = eikonax_solver.run(
        parameter_vector,
        initial_site_values=jnp.array([0.0]),
    )
    return solution.values.sum()

gradient = jax.grad(objective)(parameter_vector)
```

The custom backward pass computes the local update derivatives, contracts them with the
TensorField's local VJPs, and solves the discrete adjoint system internally. The forward result
remains a [`Solution`][eikonax.solver.Solution] object.

## Low-Level Derivatives

For advanced use cases, the individual derivative components remain available. Evaluating the
gradient $g(\mathbf{m})$ manually is a two-step procedure in Eikonax. Firstly, we evaluate, for a
given parameter vector $\mathbf{m}$ and associated solution $\mathbf{u}$, the
partial derivatives $\mathbf{G}_u$ and $\mathbf{G}_M$ of the global update operator in the
[iterative solver][eikonax.solver.Solver]. This can be done with the
[`PartialDerivator`][eikonax.derivator.PartialDerivator] object. Its configuration in 
[`PartialDerivatorData`][eikonax.derivator.PartialDerivatorData] is analogous to that of the
[forward solver](./solve.md).

!!! note
    First derivatives do not require a differentiable transformation for the update parameter $\lambda$.
    Second derivatives do, however.

```py
from eikonax import derivator

derivator_data = derivator.PartialDerivatorData(
    use_soft_update=False,
    softminmax_order=None,
    softminmax_cutoff=None,
)
eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_site_inds)
```

We can obtain sparse representations (through local dependencies) of the partial derivatives via
the [`compute_partial_derivatives`][eikonax.derivator.PartialDerivator.compute_partial_derivatives]
method,
```py
output_partial_solution, output_partial_tensor = eikonax_derivator.compute_partial_derivatives(
    solution.values, tensor_field_instance
)
```

$\mathbf{G}_u$ and $\mathbf{G}_M$ are returned using Eikonax's custom sparse data structures from the
[`linalg`][eikonax.linalg] module. These structures are designed to efficiently represent the sparse 
derivatives while maintaining JAX compatibility.

$\mathbf{G}_u$ is returned as a JAX BCOO sparse matrix with shape
$N_V \times N_V$. This is a COO-style sparse matrix representation using JAX arrays that stores row 
indices, column indices, and values separately.

$\mathbf{G}_M$ is returned as a [`DerivatorSparseTensor`][eikonax.linalg.DerivatorSparseTensor], which 
stores the partial derivatives with respect to the metric tensor field. For each vertex, it contains 
derivative contributions from adjacent simplices in a dense local format (shape 
$N_V \times \text{max\_neighbors} \times d \times d$) along with adjacency data mapping to global 
simplex indices.

The total derivative $\mathbf{G}_m$ is computed with local vector-Jacobian products. The tensor
field uses its fixed-width `parameter_inds` array to select the parameters for each simplex, and
JAX differentiates the local metric assembly method directly. No dense TensorField Jacobian is formed.
We can obtain $\mathbf{G}_m$ via the tensor contraction function provided by Eikonax,
```py
from eikonax import linalg

output_partial_parameter = linalg.contract_derivative_tensors(
    output_partial_tensor,
    tensor_field_object,
    parameter_vector,
)
```
which efficiently applies the local VJPs and returns an
JAX BCOO sparse matrix with shape $N_V \times M$


## Derivative Solver
With the partial derivatives, we can now set up a sparse, triangular equation system for computing discrete adjoints,
and subsequently the gradient $\mathbf{g}$. The rational behind this procedure is explained in more detail 
[here][eikonax.derivator.DerivativeSolver]. 

The [`DerivativeSolver`][eikonax.derivator.DerivativeSolver] accepts the partial derivative
$\mathbf{G}_u$ directly as a JAX BCOO sparse matrix,
```py
derivative_solver = derivator.DerivativeSolver(solution.values, output_partial_solution)
```

We can now evaluate the discrete adjoint from a given loss gradient $\frac{dl}{d\mathbf{u}}$,
```py
loss_grad = np.ones(solution.values.size)
adjoint = derivative_solver.solve(loss_grad)
```

We then obtain the gradient by simple multiplication of the adjoint with $\mathbf{G}_m$:

```py
total_grad = output_partial_parameter.T @ adjoint
```

!!! tip "Reusability of the gradient copmutation"
    The setup of the derivative solver and $\mathbf{G}_m$ basically constitutes a sparse representation
    of the entire parametric Jacobian $\mathbf{J}$ at a point $\mathbf{m}$. Once assembled, arbitrary portions of
    the Jacobian can be constructed with negligible cost, even for large systems.

## Comparison to Finite Differences

As a proof-of-concept, we compare the Jacobian $\mathbf{J}$ at the point $\mathbf{m}$ produced by Eikonax
to forward finite differences. With Eikonax, we simply evaluate the gradient for all unit vectors
$\mathbf{e}_i,\ i=1,\ldots,N_V$ as "loss gradient" $\frac{dl}{d\mathbf{u}}$. Each such computation
yields a row $\mathbf{J}_i$ of the Jacobian.
To this end, Eikonax provides the utility function [`compute_eikonax_jacobian`][eikonax.derivator.compute_eikonax_jacobian],
```python
eikonax_jacobian = derivator.compute_eikonax_jacobian(
    derivative_solver, output_partial_parameter
)
```

With finite difference, we evaluate a column $\mathbf{J}_j$
of the jacobian as

$$
    \mathbf{J}_j = \frac{\partial\mathbf{u}}{\partial m_j} \approx \frac{\mathbf{u}(\mathbf{m} + h\mathbf{e}_j) - \mathbf{u}(\mathbf{m})}{h},
$$

with $j=1,\ldots,N_S$, and $h$ denotes the step width of the finite difference scheme. Again,
we use an Eikonax utility module, [`finitediff`][eikonax.finitediff],

```python
step_widths = np.logspace(-5, -1, 101)
errors = []
for step_width in step_widths:
    finite_diff_jacobian = finitediff.compute_fd_jacobian(
        eikonax_solver=eikonal_solver,
        stencil=finitediff.finite_diff_1_forward,
        eval_point=parameter_vector,
        step_width=step_width,
        initial_site_values=jnp.array([0.0]),
    )
    error = np.linalg.norm(finite_diff_jacobian - eikonax_jacobian)
    errors.append(error)
```

The figure below shows the difference of the Jacobian matrices in Frobenius norm for $h\in[1\mathrm{e}{-5}, 1\mathrm{e}{-1}]$.
As can be expected, the error decreases linearly for decreasing $h$ down to the square root of the floating point precision
(32 bit in this case). Beyond this threshold, the error increases again due to round-off errors.
<figure markdown>
![samples](../images/ug_derivative_fd.png){ width="500" style="display: inline-block" }
</figure>