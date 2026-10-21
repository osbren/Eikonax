# Forward Solver

In this tutorial, we discuss how to solve the Eikonal equation with Eikonax on a simple square domain of dimension $d=2$.

## Mesh Setup

As a first step, we create a unit square mesh $\Omega = [0,1]^2$, with $n_x = n_y = 100$ discretization points in each dimension, resulting in a total of $N_V=10000$ mesh vertices.
We rely on `scipy`'s `Delaunay` triangulation function:

```py
import numpy as np
from scipy.spatial import Delaunay

mesh_points_x = np.linspace(0, 1, 100)
mesh_points_y = np.linspace(0, 1, 100)
mesh_points = np.column_stack((np.repeat(mesh_points_x, 100), np.tile(mesh_points_y, 100)))
triangulation = Delaunay(mesh_points)

vertices = triangulation.points
simplices = triangulation.simplices
```

`vertices` and `simplices` define the mesh in a canonical form. The first array of shape $N_V\times d$ contains the coordinates of all vertices, with their respective index $i$ defined via their position in the global vector. The second array is of shape $N_S \times 3$, where $N_S = 19602$. The three entries of a simplex $j$, defined by its position in the global vector, contain the indices $i_1, i_2, i_3$ of the vertices that simplex is composed of.

## Tensor Field

Besides the computational domain $\Omega$, we need a tensor field $\mathbf{M}$ characterizing the transport properties within that domain. In this example, we utilize an isotropic field of the form $\mathbf{M}^{-1}(\mathbf{x}) = \frac{1}{f(\mathbf{x})} \mathbf{I}$, where $\mathbf{I}$ is the two-dimensional identity matrix and $f(\mathbf{x})$ is given as

$$
    f(x) = 1 + 10 \exp\big( -50||\mathbf{x} -\bar{\mathbf{x}}||^2 \big) > 0,\quad \bar{\mathbf{x}} = (0.65, 0.65)^T.
$$

We generate that field with through a couple of simple `numpy` operations. The coordinate $\mathbf{x}_i$ for each mesh simples $i$ is set as the center of the respective simplex.
```py
simplex_centers = np.mean(vertices[simplices], axis=1)
inv_speed_values = \
    1 / (1 + 10 * np.exp(-50 * np.linalg.norm(simplex_centers - np.array([[0.65, 0.65]]), axis=-1) ** 2))
tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
```

The resulting field has a circular region of slow conductivity around the center point $\bar{\mathbf{x}}$:

<figure markdown="span">
  ![Tensor Field](../images/ug_solve_tensorfield.png){ width="700" }
</figure>

!!! info
    We directly generate the inverse field $\mathbf{M}^{-1}$ as input for Eikonax, because the inverse is required to actually solve the eikonal equation.

## Solver Setup

With the mesh and tensor field, we can now set up the Eikonax solver. This requires the [`preprocessing`][eikonax.preprocessing] and [`solver`][eikonax.solver] modules.

```py
from eikonax import preprocessing, solver, tensorfield
```


To begin with, we process the mesh data given by `vertices` and `simplices` to facilitate a vertex-wise update procedure. Such an update requires for each vertex $i$ information on the adjacent simplices and vertices, respectively. On a general triangulation, the number of adjacent simplices may be different for each vertex, resulting in heterogenous data structures. such data cannot be processed with JAX. Therefore, we evaluate the maximum number of adjacent simplices $n_{\text{max}}$ of any vertices in the triangulation, and build up the global array with that size. Superfluous entries are padded with a value of `-1`. For every vertex $i$, we then get an array of shape $n_{\text{max}} \times 4$, where the four entries in the last dimension contain (for non-padded data) the index $j$ of an adjacent simplex, as well as the indices of the vertices composing that triangle. This includes, for convenience, the vertex $i$ itself. 

The preprocessing is performed automatically during the initialization of the  [`MeshData`][eikonax.preprocessing.MeshData] object to be used by the solver,
```py
mesh_data = preprocessing.MeshData(vertices, simplices)
```

For a well-defined solution of the eikonal equation, we further require a set of initial sites $\Gamma\subset\Omega$. In Eikonax, the integer JAX vector `initial_site_inds` stores the indices of the respective mesh vertices. Here, we choose $i = 0$,
```py
initial_site_inds = jnp.array((0,), dtype=jnp.int32)
```

Lastly, we set up the configuration for the Eikonax solver via the [`SolverData`][eikonax.solver.SolverData] object:
```py
solver_data = solver.SolverData(
    tolerance=1e-8,
    max_num_iterations=1000,
    loop_type="jitted_while",
    max_value=1000,
    use_soft_update=False,
    softminmax_order=None,
    softminmax_cutoff=None,
)
```

The above data class requires some further elaboration, which we give in the following:

- **`tolerance`** is the absolute difference in supremum norm $||\mathbf{u}^{(k+1)} - \mathbf{u}^{(k)}||$ of two solution vector iterates, according to the global update procedure $\mathbf{u}^{(k+1)} = \mathbf{G}(\mathbf{u}^{(k)})$. For tolerance-based solvers, the iteration terminates when this tolerance is undercut.
- **`max_num_iterations`** indicates the maximum number of iterations after which a solver is terminated, regardless of a reached tolerance.
- **`loop_type`** determines the outer loop procedure. Currently implemented options are:
    1. `jitted_for`: JIT-compiles with JAX, runs a fixed number of iterations prescribed through `max_num_iterations`
    2. `jitted_while`: JIT-compiles with JAX, runs until `tolerance` is reached or maximum number of iterations has been performed.
    3. `nonjitted_while`: Like `jitted_while`, but not JIT-compiled with JAX
   
        !!! warning
            Non-jitted loops are very slow, and should only be used for small, exploratory runs. On the other hand, a [Logger][eikonax.logging.Logger] can be passed to the solver for additional output.

- **`max_value`** is the value to which all vertices not belonging to the initial active set $\Gamma$ are initialized. For the algorithm to be well defined, this value needs to be larger than any value of the actual solution.
- **`use_soft_update`** determines whether to perform a differentiable transformation on the update parameters $\lambda$.
  
    !!! info
        A soft update on the solution candidates $u$ is not performed, as this only has an effect for derivatives.

- **`softminmax_order`** is the order $\kappa$ of the differentiable transformation for $\lambda$, see [`compute_softminmax`][eikonax.corefunctions.compute_softminmax], only applies if `use_soft_update` is `True`.
- **`softminmax_cutoff`** is the offset after which the differentiable transformation for $\lambda$ is truncated, see [`compute_softminmax`][eikonax.corefunctions.compute_softminmax], only applies if `use_soft_update` is `True`.

## Initialization and Run
The tensor-field mapping is owned by the solver. For the scalar field above, use one parameter per
simplex and provide the inverse-speed values as the parameter vector:
```py
tensor_field_object = tensorfield.TensorField(
    parameter_inds=jnp.arange(simplices.shape[0], dtype=jnp.int32)[:, None],
    simplex_tensor=tensorfield.LinearScalarSimplexTensor(dimension=2),
)
eikonax_solver = solver.Solver(
    mesh_data,
    solver_data,
    tensor_field_object,
    initial_site_inds=initial_site_inds,
)
solution = eikonax_solver.run(inv_speed_values, initial_site_values=jnp.array([0.0]))
```

Initial site values are source values rather than strict boundary conditions. A lower value
propagated from another source can overwrite one of them.

The initial-site vertices are fixed when the solver is constructed, while their values are supplied
to each `run` call. This permits repeated solves with different source values without rebuilding the
solver.

A successful solver run returns a [`Solution`][eikonax.solver.Solution] data class, which contains the actual solution array, the number of performed iterations, and the reached tolerance. In our example, the solution of the eikonal equation looks like this:

<figure markdown="span">
  ![Solution](../images/ug_solve_solution.png){ width="700" }
</figure>

We can clearly observe a circular propagation in the regions of constant conductivity, as well as a "dent" for the regions of slower conductivity.
