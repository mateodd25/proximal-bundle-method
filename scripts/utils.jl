function compute_ideal_step_size(new_objective, new_point, min_value, solution)
  distance_to_solution = LinearAlgebra.norm(new_point - solution)
  if distance_to_solution > 0.0
    return (new_objective - min_value)/distance_to_solution^2
  end
  return 0.0
end

function compute_step_size(new_objective, min_value, diameter)
  return (new_objective - min_value)/diameter^2
end

function create_bundle_method_parameters(;
                                         eps_optimal=1e-6,
                                         iteration_limit=100,
                                         contraction_factor=0.5,
                                         printing_frequency=50,
                                         verbose=false,
                                         full_memory=false,
                                         )
  return ProximalBundleMethod.BundleMethodParams(
    eps_optimal, iteration_limit, contraction_factor, verbose, printing_frequency, full_memory)
end
