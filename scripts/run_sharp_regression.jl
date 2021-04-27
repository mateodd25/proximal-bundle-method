import ProximalBundleMethod
import LinearAlgebra
include("utils.jl")
include("csv_exporter.jl")

function least_squares_objective(x, A, b)
  err = A*x-b
  return LinearAlgebra.norm(err)
end

function least_squares_gradient(x, A, b)
  err = A*x-b
  if LinearAlgebra.norm(err) == 0.0
    return LinearAlgebra.zeros(length(err))
  end
  return  A'*err / LinearAlgebra.norm(err)
end

function main()
  n, m = 100, 90
  A = randn(n, m)/sqrt(m+n)
  x_opt = randn(m)
  b = A * x_opt
  # b += 0.01 * randn(n)/sqrt(n)
  x_init = randn(m)

  objective = (x -> least_squares_objective(x, A, b))
  gradient = (x -> least_squares_gradient(x, A, b))
  step_size = (x, y, _) -> compute_ideal_step_size(x, y, 0.0, x_opt)
  # step_size = (x, _, _) -> compute_step_size(x, 0.0, objective(x_init))

  params = create_bundle_method_parameters(
    iteration_limit = 3000,
    verbose = true,
    printing_frequency = 100,
    full_memory = false,
  )
  step_sizes = [.25 * 4^j for j in 0:19]
  println("About to solve a random least squares problem using ideal step size.")
  ProximalBundleMethod.solve(objective, gradient, params, step_size, x_init)
  println("\nAbout to solve a random least squares problem using adaptive parallel method.")
  sol, iter_info_agents = ProximalBundleMethod.solve_adaptive(
    objective, gradient, params, step_sizes, x_init)
  export_losses_dataframe(iter_info_agents, "/tmp/results_sharp.csv")
end

main()
