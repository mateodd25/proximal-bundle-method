
struct BundleMethodParams
    eps_optimal::Float64
    num_iter::Int64
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
    is_null::bool
    objective::Float64
end

struct BundleMethodOutput
end

function create_cut(point::Vector{Float64}, objective_function, subgradient_section)
    subgradient = subgradient_section(point)
    return Cut(objective_function(point) - dot(point, subgradient), subgradient)
end

function evaluate_cut(cut::Cut, point::Vector{Float64})
    return cut.constant_term + dot(cut.gradient, point)
end

function evaluate_model(cuts::Vector{Cut}, point::Vector{Float64})
    return max(map(cut -> evaluate_cut(cut, point), cuts))
    # result = evaluate_cut(cuts[1], point)
    # for i  2:n
    #     new_result = evaluate_cut(cuts[i], point)
    #     if evaluate_cut(cuts[i], point) > result
    #             result = new_result
    #     end
    # end
    # return result
end

function check_termination(information::IterationInformation)
    if params.verbose = true && mod(i, params.printing_frequency)

    end
end

function iteration_log(params, iteration_info)
end
function solve_model(cuts::Vector{Cut}, point::Vector{Float64}, stepsize::Float64)
    if length(cuts) == 1
        return point - cuts[1].gradient / stepsize
    elseif length(cuts) == 2
        candidate = point - cuts[1].gradient /stepsize
        if evaluate_cut(cuts[1], candidate) >= evalute_cut(cuts[2], candidate)
            return candidate, [1.0, 0.0]
        end
        candidate = point - cuts[2].gradient /stepsize
        if evaluate_cut(cuts[2], candidate) >= evalute_cut(cuts[1], candidate)
            return candidate, [0.0, 1.0]
        end

        grad_diff = cuts[1].gradient - cuts[2].gradient
        interpolation = stepsize / dot(grad_diff, grad_diff) * (cuts[1].constant_term - cuts[2].constant_term + dot(grad_diff, candidate))
        return z - grad_diff * interpolation / stepsize, [interpolation, 1.0 - interpolation]
    else
        # TODO Implement using CVX
        return nothing
    end
end

function solve(objective_function, subgradient_map, params, stepsize_policy, initial_point)
    cuts = [create_cut(initial_point, objective_function, subgradient_map)]
    current_point = initial_point
    current_objective = objective_function(current_point)
    iteration_information = [IterationInformation(0, false, current_objective)]
    for i  1:params.num_iter
        new_point, interpolation_coeff = solve_model(cuts, current_point,
                                                     stepsize_policy(objective_function, current_point, i))
        new_objective = objective_function(new_point)
        is_null = true
        if params.contraction_factor * (current_objective - evaluate_model) <= current_objective - new_objective
            is_null = false
            current_point = new_point
            current_objective = objective_function(current_point)
        end
        append!(iteration_information, IterationInformation(i, is_null, new_objective))
        iteration_log(params, last(iteration_info))
    end
end
