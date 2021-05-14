using Plots
using CSV

function general_setup()
    gr()
    fntsm = Plots.font(pointsize = 12)
    fntlg = Plots.font(pointsize = 18)
    default(titlefont = fntlg, guidefont = fntlg, tickfont = fntsm, legendfont = fntsm)
end

function plot_agent_losses(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    general_setup()
    losses = DataFrame(CSV.File(csv_path))
    plot()
    for col in names(losses)
        plot!(
            losses[!, col],
            yaxis = (:log10, [lower_bound, upper_bound]),
            label = col,
            line = (2, :solid),
            legend = :bottomleft,
        )
    end
    xaxis!("Iteration count")
    yaxis!("Objective gap")
    savefig(plot_file)
end

function plot_step_sizes(csv_path, plot_file; lower_bound = 1.0, upper_bound = 1.0e15)
    general_setup()
    losses = DataFrame(CSV.File(csv_path))
    # lower_bound = minimum([s for s in losses[!, "step_size"] if s > 0.0]) / 2
    plot(
        losses[!, "step_size"],
        line = (3, :solid),
        yaxis = (:log10, [lower_bound, upper_bound]),
        label = false,
    )
    xaxis!("Iteration count")
    yaxis!("Ideal step size")
    savefig(plot_file)
end

function plot_objective(csv_path, plot_file; lower_bound = 1.0e-15, upper_bound = 1.0)
    general_setup()
    losses = DataFrame(CSV.File(csv_path))
    # lower_bound = minimum([s for s in losses[!, "objective"] if s > 0.0]) / 2
    plot(
        losses[!, "objective"],
        line = (3, :solid),
        yaxis = (:log10, [lower_bound, lower_bound]),
        color = 2,
        label = false,
    )
    xaxis!("Iteration")
    yaxis!("Objective gap")
    savefig(plot_file)
end
