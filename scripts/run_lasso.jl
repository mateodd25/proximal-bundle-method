import ProximalBundleMethod
import Printf
import Convex
import SCS
using LinearAlgebra
using SparseArrays

include("utils.jl")
include("csv_exporter.jl")

function lasso_objective(x, A, b, reg_coeff)
  err = A*x-b
    return 1/2 * norm(err)^2 + reg_coeff * norm(x, 1)
end

function lasso_subgradient(x, A, b, reg_coeff)
  err = A*x-b
  return A'*err + reg_coeff * map(sign, x)
end

function compute_minimum_with_cvx(A, b, reg_coeff)
  _, n = size(A)
  x = Convex.Variable(n)

  problem = Convex.minimize((1/2) * Convex.sumsquares(A * x - b) + reg_coeff * Convex.norm(x, 1))
  Convex.solve!(problem, SCS.Optimizer) #() -> SCS.Optimizer(verbose=false), verbose=false)
  return problem
end

function main()
  m, n = 10, 12
  A = randn(m, n)
  x_opt = vec(sprandn(n, 1, .1))

  b = A * x_opt
  # b += 0.01 * randn(m)/sqrt(m)
  x_init = randn(n)
  # x_init = x_opt + 0.01 * randn(n)/sqrt(n)
  reg_coeff = 0.05
  objective = (x -> lasso_objective(x, A, b, reg_coeff))
  subgradient = (x -> lasso_subgradient(x, A, b, reg_coeff))
  step_size = (x, y, _) -> compute_ideal_step_size(x, y, objective(x_opt), x_opt)
  # step_size = (x, _, _) -> compute_step_size(x, 0.0, objective(x_init))

  println("Sloving with Convex.jl first")
  problem = compute_minimum_with_cvx(A, b, reg_coeff)
  Printf.@printf(
    "Obj=%12g\n",
    problem.optval,
  )

  println("Number of nonzeros")
  println(nnz(x_opt))
  println("Objective at signal")
  println(objective(x_opt))


  params = create_bundle_method_parameters(
    iteration_limit = 100000,
    verbose = true,
    printing_frequency = 1000,
    full_memory = false,
  )
  # println("\nAbout to solve a random least squares problem using ideal step size.")
  # ProximalBundleMethod.solve(objective, subgradient, params, step_size, x_init)
  println("\nAbout to solve a random lasso problem using adaptive parallel method short memory.")
  sol = ProximalBundleMethod.solve_adaptive(
    objective, subgradient, params, ProximalBundleMethod.AdaptiveStepSizeInterval(.001, 20), x_init)
  println(norm(sol.solution-x_opt)/norm(x_opt))

  params = create_bundle_method_parameters(
    iteration_limit = 1000,
    verbose = true,
    printing_frequency = 50,
    full_memory = true,
  )
  # println("\nAbout to solve a random least squares problem using ideal step size.")
  # ProximalBundleMethod.solve(objective, subgradient, params, step_size, x_init)
  println("\nAbout to solve a random lasso problem using adaptive parallel method full memory.")
    sol, iter_info = ProximalBundleMethod.solve_adaptive(
        objective, subgradient, params, ProximalBundleMethod.AdaptiveStepSizeInterval(.001, 20), x_init)
    println(norm(sol.solution-x_opt)/norm(x_opt))

end

main()
