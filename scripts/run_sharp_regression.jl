import ProximalBundleMethod
import LinearAlgebra
import Random
using Printf

include("utils.jl")
include("csv_exporter.jl")
include("plot_utils.jl")


function least_squares_objective(x, A, b)
    err = A * x - b
    return LinearAlgebra.norm(err)
end

function least_squares_gradient(x, A, b)
    err = A * x - b
    if LinearAlgebra.norm(err) == 0.0
        return LinearAlgebra.zeros(length(err))
    end
    return A' * err / LinearAlgebra.norm(err)
end

function main()
    Random.seed!(926)
    n, m = 100, 60
    A = randn(n, m) / sqrt(n)
    x_opt = randn(m)
    b = A * x_opt
    x_init = zeros(m)

    objective = (x -> least_squares_objective(x, A, b))
    gradient = (x -> least_squares_gradient(x, A, b))
    step_size = (x, y, _) -> compute_ideal_step_size(x, y, 0.0, x_opt)

    params = create_bundle_method_parameters(
        iteration_limit = 150,
        verbose = true,
        printing_frequency = 50,
    )
    step_sizes = [1e-8 * 10^j for j = 0:8]

    println("About to solve a random least squares problem using ideal step size.")
    sol_ideal = ProximalBundleMethod.solve(objective, gradient, params, step_size, x_init)

    println(
        "\nAbout to solve a random least squares problem using adaptive parallel method.",
    )
    sol, iter_info_agents =
        ProximalBundleMethod.solve_adaptive(objective, gradient, params, step_sizes, x_init)

    csv_path_ideal = "results/regression/rusults_ideal_step_size_sharp_regression.csv"
    output_stepsize_ideal_plot = "results/regression/rusults_ideal_step_size_sharp_regression.pdf"
    output_objective_ideal_plot = "results/regression/rusults_ideal_objectives_sharp_regression.pdf"

    # Upper and lower bounds for the stepsize log plots
    stepsize_lower_bound = 0.8
    stepsize_upper_bound = 1.0e15

    # Upper and lower bounds for the objective function log plots
    obj_lower_bound = 1.5e-16
    obj_upper_bound = 2.0

    export_statistics(sol_ideal, csv_path_ideal)
    plot_step_sizes(
        csv_path_ideal,
        output_stepsize_ideal_plot;
        lower_bound = stepsize_lower_bound,
        upper_bound = stepsize_upper_bound,
    )
    plot_objective(
        csv_path_ideal,
        output_objective_ideal_plot;
        lower_bound = obj_lower_bound,
        upper_bound = obj_upper_bound,
    )


    csv_path = "results/regression/results_agents_sharp_regression.csv"
    output_path = "results/regression/results_agents_sharp_regression.pdf"
    export_losses_agents(
        iter_info_agents,
        [(@sprintf "%.0e" 1 / mu) for mu in step_sizes],
        csv_path,
    )
    plot_agent_losses(
        csv_path,
        output_path;
        lower_bound = obj_lower_bound,
        upper_bound = obj_upper_bound,
    )

    csv_path_ideal = "results/regression/rusults_parallel_bundle_sharp_regression.csv"
    output_stepsize_plot = "results/regression/rusults_parallel_stepsize_sharp_regression.pdf"
    output_objective_plot = "results/regression/rusults_parallel_objective_sharp_regression.pdf"

    export_statistics(sol, csv_path_ideal)
    plot_step_sizes(
        csv_path_ideal,
        output_stepsize_plot;
        lower_bound = stepsize_lower_bound,
        upper_bound = stepsize_upper_bound,
    )
    plot_objective(
        csv_path_ideal,
        output_objective_plot;
        lower_bound = obj_lower_bound,
        upper_bound = obj_upper_bound,
    )

end

main()
