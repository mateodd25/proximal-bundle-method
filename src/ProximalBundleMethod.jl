module ProximalBundleMethod

import LinearAlgebra
import Printf
import Convex
import SCS
# import Gurobi

const dot = LinearAlgebra.dot
const norm = LinearAlgebra.norm
const minimize = Convex.minimize

include("solver.jl")

end # module
