import ProximalBundleMethod
import LinearAlgebra
import Convex
import SCS
import Printf

include("utils.jl")

struct Quadratic
  constant::Float64
  slope::Vector{Float64}
  matrix::Matrix{Float64}
end

function evaluate_quadratic(x::Vector{Float64}, quadratic::Quadratic)
  return 1 / 2 * LinearAlgebra.norm(quadratic.matrix * x)^2 + quadratic.slope' * x + quadratic.constant
end

function evaluate_gradient_quadratic(x::Vector{Float64}, quadratic::Quadratic)
  return quadratic.matrix * (quadratic.matrix * x) + quadratic.slope
end

function maximum_quadratic_objective(x, quadratics::Vector{Quadratic})
  return maximum([evaluate_quadratic(x, quadratic) for quadratic in quadratics])
end

function maximum_quadratic_gradient(x, quadratics)
  _, idx = findmax([evaluate_quadratic(x, quadratic) for quadratic in quadratics])
  if  sum([(quadratic == quadratics[idx] ? 1.0 : 0.0) for quadratic in quadratics]) > 1.0
    println("Here")
  end
  grad = evaluate_gradient_quadratic(x, quadratics[idx])
  # println(LinearAlgebra.norm(grad))
  return grad
end

function compute_minimum_with_cvx(quadratics::Vector{Quadratic})
  x = Convex.Variable(length(quadratics[1].slope))
  t = Convex.Variable(1)

  problem = Convex.minimize(t)
  for quadratic in quadratics
    problem.constraints += t >= 1 / 2 * Convex.sumsquares(quadratic.matrix * x) + quadratic.slope' * x + quadratic.constant
  end
  Convex.solve!(problem, SCS.Optimizer) #() -> SCS.Optimizer(verbose=false), verbose=false)
  return problem
end

function main()
  d = 100
  k = 10

  quadratics = Quadratic[]
  for i in 1:k
    Q = randn(d, d)/sqrt(d)
    Q = Q' * Q
    push!(quadratics, Quadratic(randn(), randn(d), Q))
  end

  # println("Solving with Convex.jl first")
  # problem = compute_minimum_with_cvx(quadratics)
  # Printf.@printf(
  #   "Obj=%12g\n",
  #   problem.optval,
  # )

  x_init = randn(d)

  objective = (x -> maximum_quadratic_objective(x, quadratics))
  gradient = (x -> maximum_quadratic_gradient(x, quadratics))
  # step_size = (x, y, _) -> compute_ideal_step_size(x, y, 0.0, x_opt)
  # step_size = (x, _, _) -> compute_step_size(x, 0.0, objective(x_init))

  # params = create_bundle_method_parameters(
  # iteration_limit = 100000,
  # verbose = true,
  # printing_frequency = 1000,
  # )

  step_sizes = [.000001 * 2^j for j in 0:15]
  params = create_bundle_method_parameters(
    iteration_limit = 3000,
    verbose = true,
    printing_frequency = 100,
    # full_memory = true,
  )
  # println("About to solve a random least squares problem using ideal step size.")
  # ProximalBundleMethod.solve(objective, gradient, params, step_size, x_init)
  println("\nAbout to solve a random maximum of quadratics using adaptive parallel method with short memory.")
  ProximalBundleMethod.solve_adaptive(
    objective, gradient, params, step_sizes, x_init)
  #     params = create_bundle_method_parameters(
  #         iteration_limit = 1000,
  #         verbose = true,
  #         printing_frequency = 100,
  #         full_memory = true,
  #     )
  #     println("\nAbout to solve a random maximum of quadratics using adaptive parallel method with full memory.")
  #     ProximalBundleMethod.solve_adaptive(
  #         objective, gradient, params, ProximalBundleMethod.AdaptiveStepSizeInterval(.000001, 16), x_init)
end

main()
