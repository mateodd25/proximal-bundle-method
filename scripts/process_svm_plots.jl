import CSV
using DataFrames
include("plot_utils.jl")

function plot_accuracy_vs_coefficient(dataframe, output_path::String; lower_bound = 1e-7)
    general_setup()
    plot(
        dataframe[!, "coeff"],
        dataframe[!, "error_subg"],
        color = 2,
        line = (3, :solid),
        label = "Subgradient",
    )

    plot!(
        dataframe[!, "coeff"],
        dataframe[!, "error_bundle"],
        color = 1,
        line = (3, :solid),
        label = "Proximal bundle",
    )
    yaxis!("Objective gap")
    xaxis!("Regularizer coefficient")
    savefig(output_path)
end

function plot_log_accuracy_vs_coefficient(
    dataframe,
    output_path::String;
    lower_bound = 1e-7,
)
    general_setup()
    plot(
        dataframe[!, "coeff"],
        dataframe[!, "error_subg"],
        yaxis = (:log10, [lower_bound, :auto]),
        color = 2,
        line = (3, :solid),
        label = "Subgradient",
        legend = :bottomright,
    )

    plot!(
        dataframe[!, "coeff"],
        dataframe[!, "error_bundle"],
        color = 1,
        line = (3, :solid),
        label = "Proximal bundle",
    )
    yaxis!("Objective gap")
    xaxis!("Regularizer coefficient")
    savefig(output_path)
end


function main()
    results = DataFrame(CSV.File("results/svm/results_svm.csv"))
    instances = unique(results[!, "instance"])
    coefficients = unique(results[!, "coeff"])

    for name in instances
        plot_accuracy_vs_coefficient(
            results[(results.instance.==name).&(results.coeff.<10.0), :],
            "results/svm/svm_accuracy_vs_coefficient_" * string(name) * ".pdf",
        )
        plot_log_accuracy_vs_coefficient(
            results[(results.instance.==name).&(results.coeff.<10.0), :],
            "results/svm/svm_log_accuracy_vs_coefficient_" * string(name) * ".pdf",
        )
    end
end

main()
