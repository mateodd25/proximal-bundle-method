import ProximalBundleMethod
import LinearAlgebra

function least_squares_objective(x, A, b)
  err = A*x-b
  return err'*err
end

function least_squares_gradient(x, A, b)
  err = A*x-b
  return A'*err
end

function compute_ideal_step_size(new_objective, new_point, min_value, solution)
  distance_to_solution = LinearAlgebra.norm(new_point - solution)
  if distance_to_solution > 0.0
    return (new_objective - min_value)/distance_to_solution^2
  end
  return 0.0
end

function compute_step_size(new_objective, min_value, diameter)
  return (new_objective - min_value)/diameter^2
end

function create_bundle_method_parameters(;
                                         eps_optimal=1e-6,
                                         iteration_limit=100,
                                         contraction_factor=0.5,
                                         printing_frequency=50,
                                         verbose=false,
                                         )
  return ProximalBundleMethod.BundleMethodParams(
    eps_optimal, iteration_limit, contraction_factor, verbose, printing_frequency)
end

function main()
  n, m = 20, 10
  A = randn(n, m)
  x_opt = randn(m)
  b = A * x_opt

  x_init = randn(m)

  objective = (x -> least_squares_objective(x, A, b))
  gradient = (x -> least_squares_gradient(x, A, b))
  step_size = (x, y, _) -> compute_ideal_step_size(x, y, 0.0, x_opt)
  # step_size = (x, _, _) -> compute_step_size(x, 0.0, objective(x_init))

  params = create_bundle_method_parameters(
    iteration_limit = 100,
    verbose = true,
    printing_frequency = 1,
  )
  ProximalBundleMethod.solve(objective, gradient, params, step_size, x_init)
end

main()
