using CSV
using DataFrames

function export_losses_agents(iteration_info_agents, step_sizes, file_path)

  losses = [
    [iter_info.objective
     for iter_info in info_agent]
    for info_agent in iteration_info_agents]
  iter_max = iteration_info_agents[1][end].iteration
  # df = DataFrame([[0:iter_max]; losses], :auto)
  df = DataFrame(losses, :auto)
  step_sizes = map(string, step_sizes)
  rename!(df, step_sizes)
  # push!(df, [["iter"]; step_sizes])
  CSV.write(file_path, df)
end

function export_statistics(solver_output, file_path)
  df = DataFrame(iter = [iter_stats.iteration for iter_stats in solver_output.iteration_stats],
                 objective = [iter_stats.objective for iter_stats in solver_output.iteration_stats],
                 step_size = [iter_stats.step_size for iter_stats in solver_output.iteration_stats],
                 null = [iter_stats.is_null for iter_stats in solver_output.iteration_stats])
  CSV.write(file_path, df)
end
