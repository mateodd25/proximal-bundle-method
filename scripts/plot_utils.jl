using Plots
using CSV

function general_setup()
  gr()
  fntsm = Plots.font(pointsize=12)
  fntlg = Plots.font(pointsize=18)
  # default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
end

function plot_agent_losses(csv_path, plot_file)
  general_setup()
  losses = DataFrame(CSV.File(csv_path))
  plot()
  for col in names(losses)
    plot!(losses[!, col], yaxis=:log, label=col)
  end
  xaxis!("Iteration")
  yaxis!("Objective")
  savefig(plot_file)
end

function plot_step_sizes(csv_path, plot_file)
  general_setup()
  losses = DataFrame(CSV.File(csv_path))
  lower_bound = minimum([s for s in losses[!, "step_size"] if s > 0.0]) / 2
  plot(losses[!, "step_size"],  yaxis=(:log10, [lower_bound, :auto]), label=false)
  xaxis!("Iteration")
  yaxis!("Ideal step size")
  savefig(plot_file)
end

function plot_objective(csv_path, plot_file)
  general_setup()
  losses = DataFrame(CSV.File(csv_path))
  lower_bound = minimum([s for s in losses[!, "objective"] if s > 0.0]) / 2
  plot(losses[!, "objective"],  yaxis=(:log10, [lower_bound, :auto]), color=2, label=false)
  xaxis!("Iteration")
  yaxis!("Objective")
  savefig(plot_file)
end
