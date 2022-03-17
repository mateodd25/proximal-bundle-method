import ProximalBundleMethod
using LinearAlgebra
using Random

include("utils.jl")
include("csv_exporter.jl")
include("plot_utils.jl")

function polyhedral_support_function(w, A)
    return maximum(A*w)
end

function polyhedral_support_function_gradient(w, A)
    _, idx = findmax(A * w)
    return A[idx, :]
end

function generate_crosspolytope_primitives(m, v)
    A = [Matrix{Float64}(I, m, m); -Matrix{Float64}(I, m, m)] .- v'
    return (x -> polyhedral_support_function(x, A)), (x -> polyhedral_support_function_gradient(x, A))
end

function generate_sphere_example_primitives(m, v)
    objective = function(x)
        if norm(x + v) > 0
            return dot((x + v)/norm(x + v) - v, x)
        else
            normv = norm(v)
            return normv^2 *(1 + 1/normv)
        end
    end

    gradient = function(x)
            if norm(x + v) > 0
                return (x + v)/norm(x + v) - v
            else
                return -v * (1 + 1/norm(v))
            end
    end
    return objective, gradient
end

function run_experiment(objective, gradient, x_init, m, v)

    distances_inf_disp_vector = []
    iterates = []
    gradient_wrapper = function (x)
        if length(distances_inf_disp_vector) > 0
            best = distances_inf_disp_vector[end]
        else
            best = 200
        end
        push!(distances_inf_disp_vector, norm((x-x_init)/norm(x-x_init) - v/norm(v)))
        if length(distances_inf_disp_vector) > 10
            distances_inf_disp_vector[end] = min(distances_inf_disp_vector[end], best)
        end
        # println("dist = " * string(distances_inf_disp_vector[end]))
        # if length(iterates) > 0
        #     x_last = iterates[end]
        #     println("dist = " * string(distances_inf_disp_vector[end]) * " and " * string(norm((x-x_last)/norm(x-x_last)-v/norm(v))))
        # else
        #     println("dist = " * string(distances_inf_disp_vector[end]))
        # end
        push!(iterates, x)
        return gradient(x)
    end
    poly_step_size = ((_, _, t) -> 1 / (1.0 * t))
    # poly_step_size = ((_, _, t) -> 1)
    params_subgradient = create_subgradient_method_parameters(
        iteration_limit = 1e7,
        verbose = true,
        printing_frequency = 1e6,
    )
    print("\n------------------------------------\n")
    println("About to solve problem using subgradient method.")
    results = ProximalBundleMethod.solve(
        objective,
        gradient_wrapper,
        params_subgradient,
        poly_step_size,
        x_init)
    return results, distances_inf_disp_vector
end

function main()
    Random.seed!(2625)
    m = 2
    x_init = randn(m)/sqrt(m)
    # Crosspolytope example with simple v
    # # v = zeros(m)
    # # v[1] = 2
    v = ones(m)
    objective, gradient = generate_crosspolytope_primitives(m, v)
    results, distances_inf_disp_vector = run_experiment(objective, gradient, x_init, m, v)
    println(results.solution)
    println(distances_inf_disp_vector[end])
    plot_objectives([distances_inf_disp_vector], "results/divergent/crosspolytope_ones_distances.pdf", [false]; lower_bound = 1.0e-10, upper_bound = 10.0)

    # Crosspolytope example with simple v
    v = zeros(m)
    v[1] = 2
    objective, gradient = generate_crosspolytope_primitives(m, v)
    results, distances_inf_disp_vector = run_experiment(objective, gradient, x_init, m, v)
    println(results.solution)
    println(distances_inf_disp_vector[end])
    plot_objectives([distances_inf_disp_vector], "results/divergent/crosspolytope_e1_distances.pdf", [false]; lower_bound = 1.0e-10, upper_bound = 10.0)

    # Sphere
    v = zeros(m)
    v[1] = 2
    # v = ones(m)
    objective, gradient = generate_sphere_example_primitives(m, v)
    results, distances_inf_disp_vector = run_experiment(objective, gradient, x_init, m, v)
    println(results.solution)
    println(distances_inf_disp_vector[end])
    plot_objectives([distances_inf_disp_vector], "results/divergent/circle_distances.pdf", [false]; lower_bound = 1.0e-10, upper_bound = 10.0)

end

main()
