using CSV
using DataFrames

function export_losses_dataframe(iteration_info_agents, file_path)
    losses = [
        [iter_info.objective
         for iter_info in info_agent]
        for info_agent in iteration_info_agents]
    print(size(losses))
    iter_max = iteration_info_agents[1][end].iteration
    step_sizes = [string(info_agent[1].step_size) for info_agent in iteration_info_agents]
    print(step_sizes)
    df = DataFrame([[0:iter_max]; losses], :auto)
    # push!(df, [["iter"]; step_sizes])
    CSV.write(file_path, df)
end
