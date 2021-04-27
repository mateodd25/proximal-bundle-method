import ProximalBundleMethod
import LinearAlgebra
import Random

include("utils.jl")
include("csv_exporter.jl")
include("plot_utils.jl")


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
  Random.seed!(926)
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
    iteration_limit = 1200,
    verbose = true,
    printing_frequency = 100,
    full_memory = false,
  )
  step_sizes = [.1 * 10^j for j in 0:7]

  csv_path_ideal = "/tmp/rusults_ideal_step_size_sharp_regression.csv"
  output_stepsize_ideal_plot = "/tmp/rusults_ideal_step_size_sharp_regression.pdf"
  output_objective_ideal_plot = "/tmp/rusults_ideal_objectives_sharp_regression.pdf"
  println("About to solve a random least squares problem using ideal step size.")
  sol_ideal = ProximalBundleMethod.solve(objective, gradient, params, step_size, x_init)
  export_statistics(sol_ideal, csv_path_ideal)
  plot_step_sizes(csv_path_ideal, output_stepsize_ideal_plot)
  plot_objective(csv_path_ideal, output_objective_ideal_plot)
  print(
    "\nRelative distance to solution: "
    * string(LinearAlgebra.norm(sol_ideal.solution - x_opt)/LinearAlgebra.norm(x_opt)))


  println("\nAbout to solve a random least squares problem using adaptive parallel method.")
  sol, iter_info_agents = ProximalBundleMethod.solve_adaptive(
    objective, gradient, params, step_sizes, x_init)
  print(
    "\nRelative distance to solution: "
    * string(LinearAlgebra.norm(sol.solution - x_opt)/LinearAlgebra.norm(x_opt)))

  csv_path = "/tmp/results_agents_sharp_regression.csv"
  output_path = "/tmp/results_agents_sharp_regression.pdf"
  export_losses_agents(iter_info_agents, step_sizes, csv_path)
  plot_agent_losses(csv_path, output_path)

end

main()
