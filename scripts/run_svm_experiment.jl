import ProximalBundleMethod
import Printf
import Convex
import Gurobi
import SCS
using LinearAlgebra
using Random
using SparseArrays

include("utils.jl")
include("csv_exporter.jl")
include("plot_utils.jl")


"""
The following functions are used to process LIBSVM files.
"""
mutable struct LearningData
    feature_matrix::SparseMatrixCSC{Float64,Int64}
    labels::Vector{Float64}
end

function load_libsvm_file(file_name::String)
    open(file_name, "r") do io
        target = Array{Float64,1}()
        row_indicies = Array{Int64,1}()
        col_indicies = Array{Int64,1}()
        matrix_values = Array{Float64,1}()

        row_index = 0
        for line in eachline(io)
            row_index += 1
            split_line = split(line)
            label = parse(Float64, split_line[1])
            # This ensures that labels are 1 or -1. Different dataset use {-1, 1}, {0, 1}, and {1, 2}.
            if abs(label - 1.0) < 1e-05
                label = 1.0
            else
                label = -1.0
            end
            push!(target, label)
            for i = 2:length(split_line)
                push!(row_indicies, row_index)
                matrix_coef = split(split_line[i], ":")
                push!(col_indicies, parse(Int64, matrix_coef[1]))
                push!(matrix_values, parse(Float64, matrix_coef[2]))
            end
        end
        feature_matrix = sparse(row_indicies, col_indicies, matrix_values)
        return LearningData(feature_matrix, target)
    end
end

function normalize_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
    m = size(feature_matrix, 2)
    normalize_columns_by = ones(m)
    for j = 1:m
        col_vals = feature_matrix[:, j].nzval
        if length(col_vals) > 0
            normalize_columns_by[j] = 1.0 / norm(col_vals, 2)
        end
    end
    return feature_matrix * sparse(1:m, 1:m, normalize_columns_by)
end

function remove_empty_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
    keep_cols = Array{Int64,1}()
    for j = 1:size(feature_matrix, 2)
        if length(feature_matrix[:, j].nzind) > 0
            push!(keep_cols, j)
        end
    end
    return feature_matrix[:, keep_cols]
end

function add_intercept(feature_matrix::SparseMatrixCSC{Float64,Int64})
    return [sparse(ones(size(feature_matrix, 1))) feature_matrix]
end


function preprocess_learning_data(result::LearningData)
    result.feature_matrix = remove_empty_columns(result.feature_matrix)
    result.feature_matrix = add_intercept(result.feature_matrix)
    result.feature_matrix = normalize_columns(result.feature_matrix)
    return result
end


function svm_objective(w, Xp, y, coeff)
    n, _ = size(Xp)
    soft_margin = map(x -> max(0, x), ones(length(y)) - y .* (Xp * w))
    return sum(soft_margin) / n + coeff / 2 * LinearAlgebra.norm(w)^2
end

function svm_subgradient(w, Xp, y, coeff)
    n, m = size(Xp)
    soft_margin = map(x -> max(0, x), ones(length(y)) - y .* (Xp * w))
    result = zeros(length(w))
    for i = 1:length(soft_margin)
        if soft_margin[i] > 0.0
            result += -y[i] * Xp[i, :] / n
        end
    end
    return result + coeff * w
end

function compute_minimum_with_cvx(Xp, y, reg_coeff)
    n, m = size(Xp)
    w = Convex.Variable(m)
    problem = Convex.minimize(
        (reg_coeff / 2) * Convex.sumsquares(w) +
        Convex.sum(Convex.pos((1 - y .* (Xp * w)) / n)),
    )
    Convex.solve!(
        problem,
        () -> Gurobi.Optimizer(BarConvTol = 1e-10, BarQCPConvTol = 1e-10),
    ) #() -> SCS.Optimizer(verbose=false), verbose=false)
    return problem
end


function main()
    Random.seed!(2625)
    instance_names = ["colon-cancer", "duke", "leu"]
  coefficients = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0]

  step_sizes = [1e-15 * 100^j for j = 0:10]

  instances = Dict()
  for name in instance_names
        instances[name] = preprocess_learning_data(load_libsvm_file("data/" * name))
    end

    errors = DataFrame(
        instance = String[],
        coeff = Float64[],
        error_subg = Float64[],
        error_bundle = Float64[],
    )

    params_subgradient = create_subgradient_method_parameters(
        iteration_limit = 2000,
        verbose = true,
        printing_frequency = 100,
    )

    params_bundle = create_bundle_method_parameters(
        iteration_limit = 2000,
        verbose = true,
        printing_frequency = 100,
        full_memory = false,
    )

    for (name, instance) in instances
        Xp = (instance.feature_matrix)
        y = instance.labels
        _, m = size(Xp)
        x_init = randn(m) / m
        for coeff in coefficients
            print("\n------------------------------------\n")
            Printf.@printf("Instance %s, coeff %12g\n", name, coeff)
            println("Solving with Convex.jl first")
            problem = compute_minimum_with_cvx(Xp, y, coeff)
            Printf.@printf("Obj=%12g\n", problem.optval,)

            objective = (w -> svm_objective(w, Xp, y, coeff) - problem.optval)
            gradient = (w -> svm_subgradient(w, Xp, y, coeff))
            poly_step_size = ((_, _, t) -> 1 / (coeff * t))

            println("\nAbout to solve a random SVM problem using subgradient method.")
            sol_subgradient = ProximalBundleMethod.solve(
                objective,
                gradient,
                params_subgradient,
                poly_step_size,
                x_init,
            )

            println("\nAbout to solve a random SVM problem using adaptive parallel method.")
            sol_bundle, iter_info_agents = ProximalBundleMethod.solve_adaptive(
                objective,
                gradient,
                params_bundle,
                step_sizes,
                x_init,
            )

            push!(
                errors,
                [
                    name,
                    coeff,
                    objective(sol_subgradient.solution),
                    objective(sol_bundle.solution),
                ],
            )

          # Uncomment this block of code if you want to save stats about each run.

          # csv_path_sol =
          #     "results/svm/results_subgradient_" * name * "_" * string(coeff) * "_svm.csv"
          # output_stepsize_plot =
          #     "results/svm/results_subgradient_" * name * "_" *
          #     string(coeff) *
          #     "_stepsize_svm.pdf"
          # output_objective_plot =
          #     "results/svm/results_subgradient_" * name * "_" *
          #     string(coeff) *
          #     "_objective_svm.pdf"

          # export_statistics(sol_subgradient, csv_path_sol)
          # plot_step_sizes(csv_path_sol, output_stepsize_plot)
          # plot_objective(csv_path_sol, output_objective_plot)

          # csv_path_sol =
          #     "results/svm/results_bundle_" * name * "_" * string(coeff) * "_svm.csv"
          # output_stepsize_plot =
          #     "results/svm/results_bundle_" * name * "_" *
          #     string(coeff) *
          #     "_stepsize_svm.pdf"
          # output_objective_plot =
          #     "results/svm/results_bundle_" * name * "_" *
          #     string(coeff) *
          #     "_objective_svm.pdf"

          # csv_path =
          #     "results/svm/results_" * name * "_" * string(coeff) * "_agents_svm.csv"
          # output_path =
          #     "results/svm/results_" * name * "_" * string(coeff) * "_agents_svm.pdf"

          # export_statistics(sol_bundle, csv_path_sol)
          # plot_step_sizes(csv_path_sol, output_stepsize_plot)
          # plot_objective(csv_path_sol, output_objective_plot)

          # export_losses_agents(iter_info_agents, step_sizes, csv_path)
          # plot_agent_losses(csv_path, output_path)

        end
    end
    print(errors)
    CSV.write("results/svm/results_svm.csv", errors)
end

main()
