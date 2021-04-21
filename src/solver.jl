
@enum TerminationReason begin
  TERMINATION_REASON_UNSPECIFIED
  TERMINATION_REASON_OPTIMAL
  TERMINATION_REASON_ITERATION_LIMIT
end

struct BundleMethodParams
  eps_optimal::Float64
  iteration_limit::Int64
  contraction_factor::Float64
  verbose::Bool
  printing_frequency::Int64
end

struct Cut
    constant_term::Float64
    gradient::Vector{Float64}
end

struct IterationInformation
  iteration::Int64
  is_null::Bool
  objective::Float64
  approximation_gap::Float64
  residual::Float64
end

"""
Struct encoding an interval of the form
[left_extreme, 2^exponent * left_extreme]
"""
struct AdaptiveStepSizeInterval
    left_extreme::Float64
    exponent::Int64
end

struct BundleMethodOutput
  solution::Vector{Float64}
  iteration_stats::Vector{IterationInformation}
  termination_reason::TerminationReason
end

mutable struct SolverState
    current_objective::Float64
    current_gap::Float64
    current_residual::Float64
    latest_is_null::Bool
    current_solution::Vector{Float64}
    cuts::Vector{Cut}
end

function create_cut(point::Vector{Float64}, objective_function, subgradient_section)
    subgradient = subgradient_section(point)
    return Cut(objective_function(point) - dot(point, subgradient), subgradient)
end

function evaluate_cut(cut::Cut, point::Vector{Float64})
    return cut.constant_term + dot(cut.gradient, point)
end

function evaluate_model(cuts::Vector{Cut}, point::Vector{Float64})
  return maximum(map(cut -> evaluate_cut(cut, point), cuts))
end

function aggregate_cuts(cuts::Vector{Cut}, interpolation_coefficients::Vector{Float64})
  constant_term = cuts[1].constant_term * interpolation_coefficients[1]
  gradient = cuts[1].gradient * interpolation_coefficients[1]
  for i in 2:length(cuts)
    constant_term +=  cuts[i].constant_term * interpolation_coefficients[i]
    gradient += cuts[i].gradient * interpolation_coefficients[i]
  end
  return [Cut(constant_term, gradient)]
end

function check_termination(params::BundleMethodParams, iteration_info::IterationInformation)
  if params.verbose && mod(iteration_info.iteration, params.printing_frequency)
    iteration_log(params, iteration_log)
  end
  if iteration_info.iteration >= params.iteration_limit
    return TERMINATION_REASON_ITERATION_LIMIT
  end
  if iteration_info.approximation_gap <= eps_optimal
    return TERMINATION_REASON_OPTIMAL
  end
end

function iteration_log(params::BundleMethodParams, iteration_info::IterationInformation)
  if params.verbose
    Printf.@printf(
      " %s %5d obj=%12g gap=%12g residual=%12g\n",
        iteration_info.is_null ? "n" : "d",
        iteration_info.iteration,
        iteration_info.objective,
        iteration_info.approximation_gap,
        iteration_info.residual,
    )
  end
end

function solve_model(cuts::Vector{Cut}, point::Vector{Float64}, stepsize::Float64)
    if length(cuts) == 1
    return point - cuts[1].gradient / stepsize, [1.0]
  elseif length(cuts) == 2

    candidate = point - cuts[1].gradient /stepsize
    if evaluate_cut(cuts[1], candidate) >= evaluate_cut(cuts[2], candidate)
      return candidate, [1.0, 0.0]
    end

    candidate = point - cuts[2].gradient /stepsize
    if evaluate_cut(cuts[2], candidate) >= evaluate_cut(cuts[1], candidate)
      return candidate, [0.0, 1.0]
    end

    grad_diff = cuts[1].gradient - cuts[2].gradient
    interpolation = stepsize / dot(grad_diff, grad_diff) * (
      cuts[1].constant_term - cuts[2].constant_term + dot(grad_diff, candidate))
    return candidate - grad_diff * interpolation / stepsize, [interpolation, 1.0 - interpolation]
  else
    # TODO Implement using CVX
    return nothing
  end
end

function take_step(objective_function, subgradient_map,
                   params, step_size, solver_state,
                   iteration_stats)
    (new_point, interpolation_coefficients) = solve_model(
        solver_state.cuts, solver_state.current_solution,
        step_size,
    )
    new_objective = objective_function(new_point)
    solver_state.current_residual = norm(new_point - solver_state.current_solution)
    solver_state.latest_is_null = true

    model_value = evaluate_model(solver_state.cuts, new_point)
    solver_state.current_gap = new_objective - model_value

    if params.contraction_factor * (solver_state.current_objective - model_value) <= solver_state.current_objective - new_objective
        solver_state.latest_is_null = false
        solver_state.current_solution = new_point
        solver_state.current_objective = new_objective
  end
    solver_state.cuts = aggregate_cuts(solver_state.cuts, interpolation_coefficients)
    push!(solver_state.cuts, create_cut(new_point, objective_function, subgradient_map))
end

function solve(objective_function, subgradient_map, params, stepsize_policy, initial_point)

    iteration_stats = [IterationInformation(0, false, objective_function(initial_point), Inf, Inf)]
    solver_state = SolverState(
        objective_function(initial_point), Inf, Inf, false, initial_point,
        [create_cut(initial_point, objective_function, subgradient_map)]
    )
    for i in 1:params.iteration_limit
        take_step(
            objective_function, subgradient_map, params,
            stepsize_policy(solver_state.current_objective,
                            solver_state.current_solution, i),
            solver_state, iteration_stats,
        )
        push!(iteration_stats,
              IterationInformation(i, solver_state.latest_is_null, solver_state.current_objective,
                                   solver_state.current_gap, solver_state.current_residual))
        iteration_log(params, last(iteration_stats))
  end

    return BundleMethodOutput(solver_state.current_solution, iteration_stats, TERMINATION_REASON_ITERATION_LIMIT)
end

function combine_parallel_models(solver_states)
    min_fun, idx = minimum([state.current_objective for state in solver_states])
    for state in solver_states
        if min_fun < state.current_objective

        end
    end
end

function solve_adaptive(objective_function, subgradient_map, params,
                        step_size_interval, initial_point)
    iteration_stats = [IterationInformation(0, false, objective_function(initial_point), Inf, Inf)]
    step_sizes = [step_size_interval.left_extreme * 2^j for j in 0:step_size_interval.exponent]
    solver_states = [
        SolverState(
            objective_function(initial_point), Inf, Inf, false, initial_point,
            [create_cut(initial_point, objective_function, subgradient_map)])
        for j in 1:(step_size_interval.exponent + 1)
    ]

    for i in 1:params.iteration_limit
        take_step(
            objective_function, subgradient_map, params,
            stepsize_policy(solver_state.current_objective,
                            solver_state.current_solution, i),
            solver_state, iteration_stats,
        )
        push!(iteration_stats,
              IterationInformation(i, solver_state.latest_is_null, solver_state.current_objective,
                                   solver_state.current_gap, solver_state.current_residual))
        iteration_log(params, last(iteration_stats))
    end

    return BundleMethodOutput(solver_state.current_solution, iteration_stats, TERMINATION_REASON_ITERATION_LIMIT)
end
