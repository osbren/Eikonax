import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fimpy.solver import create_fim_solver

from eikonax import (
    derivator,
    finitediff,
    linalg,
    logging,
    preprocessing,
    solver,
    tensorfield,
)

pytestmark = pytest.mark.integration


# =============================== Integration Tests for Tensor Field ===============================
def test_tensor_field_assemble(
    small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor,
):
    data, SimplexObject = (  # noqa: N806
        small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor
    )
    dimension, num_simplices, parameter_vector, expected_tensor_field, _ = data
    simplex_object = SimplexObject(dimension)
    tensorfield_object = tensorfield.TensorField(
        jnp.arange(num_simplices, dtype=jnp.int32)[:, None],
        simplex_object,
    )
    field = tensorfield_object.assemble_field(parameter_vector)
    assert jnp.allclose(field, expected_tensor_field)


# ================================== Integration Tests for Solver ==================================
@pytest.mark.slow
def test_solver_loop_types(configurations_and_tensorfields_2d_uniform):
    logger_data = logging.LoggerSettings(
        log_to_console=False,
        logfile_path=None,
    )
    logger = logging.Logger(logger_data)
    config, _, parameter_vector = configurations_and_tensorfields_2d_uniform
    *_, mesh_data, solver_data, initial_site_inds, initial_site_values, tensor_field_object = config
    solver_data_jitted_while = solver_data
    solver_data_nonjitted_while = copy.deepcopy(solver_data)
    solver_data_nonjitted_while.loop_type = "nonjitted_while"
    solver_data_jitted_for = copy.deepcopy(solver_data)
    solver_data_jitted_for.loop_type = "jitted_for"
    eikonax_solver = solver.Solver(
        mesh_data, solver_data_jitted_while, tensor_field_object, initial_site_inds
    )
    solution_jitted_while = eikonax_solver.run(parameter_vector, initial_site_values)
    eikonax_solver = solver.Solver(
        mesh_data, solver_data_nonjitted_while, tensor_field_object, initial_site_inds, logger
    )
    solution_nonjitted_while = eikonax_solver.run(parameter_vector, initial_site_values)
    eikonax_solver = solver.Solver(
        mesh_data, solver_data_jitted_for, tensor_field_object, initial_site_inds
    )
    solution_jitted_for = eikonax_solver.run(parameter_vector, initial_site_values)
    assert np.allclose(solution_jitted_while.values, solution_nonjitted_while.values)
    assert np.allclose(solution_jitted_while.values, solution_jitted_for.values)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2d_uniform_tensorfield(configurations_and_tensorfields_2d_uniform):
    config, tensor_field, parameter_vector = configurations_and_tensorfields_2d_uniform
    simplices, vertices, mesh_data, solver_data, initial_site_inds, initial_site_values, tensor_field_object = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_site_inds, initial_site_values)
    eikonax_solver = solver.Solver(
        mesh_data, solver_data, tensor_field_object, initial_site_inds
    )
    eikonax_solution = eikonax_solver.run(parameter_vector, initial_site_values)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2d_random_tensorfield(configurations_and_tensorfields_2d_random):
    config, tensor_field, parameter_vector = configurations_and_tensorfields_2d_random
    simplices, vertices, mesh_data, solver_data, initial_site_inds, initial_site_values, tensor_field_object = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_site_inds, initial_site_values)
    eikonax_solver = solver.Solver(
        mesh_data, solver_data, tensor_field_object, initial_site_inds
    )
    eikonax_solution = eikonax_solver.run(parameter_vector, initial_site_values)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2d_function_tensorfield(configurations_and_tensorfields_2d_function):
    config, tensor_field, parameter_vector = configurations_and_tensorfields_2d_function
    simplices, vertices, mesh_data, solver_data, initial_site_inds, initial_site_values, tensor_field_object = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_site_inds, initial_site_values)
    eikonax_solver = solver.Solver(
        mesh_data, solver_data, tensor_field_object, initial_site_inds
    )
    eikonax_solution = eikonax_solver.run(parameter_vector, initial_site_values)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# ================================= Integration Tests for Derivator ================================
@pytest.mark.slow
def test_compute_partial_derivatives(setup_analytical_partial_derivative_tests):
    input_data, fwd_solution, expected_partial_derivatives = (
        setup_analytical_partial_derivative_tests
    )
    (
        vertices,
        simplices,
        parameter_vector,
        tensor_field,
        initial_site_inds,
        tensor_field_object,
        derivator_data,
    ) = input_data
    derivator_data = derivator.PartialDerivatorData(**derivator_data)
    mesh_data = preprocessing.MeshData(vertices=vertices, simplices=simplices)
    eikonax_derivator = derivator.PartialDerivator(
        mesh_data, derivator_data, initial_site_inds
    )
    output_partial_solution, output_partial_tensor = eikonax_derivator.compute_partial_derivatives(
        fwd_solution, tensor_field
    )
    output_partial_parameter = linalg.contract_derivative_tensors(
        output_partial_tensor, tensor_field_object, jnp.array(parameter_vector)
    )
    sparse_partial_solution = output_partial_solution.todense()
    sparse_partial_parameter = output_partial_parameter.todense()
    expected_sparse_partial_solution, expected_sparse_partial_parameter = (
        expected_partial_derivatives
    )

    assert np.allclose(sparse_partial_solution, expected_sparse_partial_solution)
    assert np.allclose(sparse_partial_parameter, expected_sparse_partial_parameter)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_constructor_viable(setup_derivative_solve_checks):
    _, solution_vector, _, _, derivative_solver, _ = setup_derivative_solve_checks
    system_matrix = derivative_solver.sparse_system_matrix.todense()
    assert system_matrix.shape == (solution_vector.size, solution_vector.size)
    assert np.allclose(system_matrix, np.triu(system_matrix))
    permutation_indices = np.asarray(derivative_solver.permutation_indices)
    assert np.array_equal(np.sort(permutation_indices), np.arange(solution_vector.size))


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_solve_viable(setup_derivative_solve_checks):
    parameter_vector, solution_vector, _, _, derivative_solver, partial_derivative_parameter = (
        setup_derivative_solve_checks
    )
    rhs_adjoint = np.ones(solution_vector.size)
    adjoint = derivative_solver.solve(rhs_adjoint)
    gradient = partial_derivative_parameter.T @ adjoint
    assert gradient.size == parameter_vector.size


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_vs_finite_differences(setup_derivative_solve_checks):
    (
        parameter_vector,
        _,
        tensor_field,
        eikonal_solver,
        derivative_solver,
        partial_derivative_parameter,
    ) = setup_derivative_solve_checks
    eikonax_jacobian = derivator.compute_eikonax_jacobian(
        derivative_solver, partial_derivative_parameter
    )
    assert isinstance(eikonax_jacobian, jax.Array)
    finite_diff_jacobian = finitediff.compute_fd_jacobian(
        eikonax_solver=eikonal_solver,
        stencil=finitediff.finite_diff_1_forward,
        eval_point=parameter_vector,
        step_width=1e-3,
        initial_site_values=np.array([0.0]),
    )
    error = np.linalg.norm(finite_diff_jacobian - eikonax_jacobian)
    assert error < 2e-3


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_supports_jax_grad(setup_derivative_solve_checks):
    (
        parameter_vector,
        solution_vector,
        _,
        eikonal_solver,
        derivative_solver,
        partial_derivative_parameter,
    ) = setup_derivative_solve_checks
    expected_gradient = partial_derivative_parameter.T @ derivative_solver.solve(
        jnp.ones_like(solution_vector)
    )
    gradient = jax.grad(lambda parameters: eikonal_solver.run(parameters, jnp.array([0.0])).values.sum())(
        parameter_vector
    )
    assert np.allclose(gradient, expected_gradient, atol=2e-3)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_supports_64_bit_precision():
    previous_x64_enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        vertices, simplices = preprocessing.create_test_mesh((0, 1), (0, 1), 3, 3)
        mesh_data = preprocessing.MeshData(vertices, simplices)
        solver_data = solver.SolverData(
            loop_type="jitted_while",
            max_value=1000,
            use_soft_update=True,
            softminmax_order=20,
            softminmax_cutoff=1,
            max_num_iterations=1000,
            tolerance=1e-8,
        )
        tensor_field = tensorfield.TensorField(
            jnp.arange(mesh_data.num_simplices)[:, None],
            tensorfield.InvLinearScalarSimplexTensor(2),
        )
        eikonax_solver = solver.Solver(mesh_data, solver_data, tensor_field, jnp.array([0]))
        parameter_vector = jnp.ones(mesh_data.num_simplices, dtype=jnp.float64)
        solution = eikonax_solver.run(parameter_vector, jnp.array([0.0], dtype=jnp.float64))
        gradient = jax.grad(
            lambda parameters: eikonax_solver.run(
                parameters, jnp.array([0.0], dtype=jnp.float64)
            ).values.sum()
        )(parameter_vector)
        assert jax.config.x64_enabled
        assert parameter_vector.dtype == jnp.float64
        assert solution.values.dtype == jnp.float64
        assert gradient.dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", previous_x64_enabled)
