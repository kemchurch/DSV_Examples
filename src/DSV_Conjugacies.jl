module DSV_Conjugacies

using PyCall, ForwardDiff, RadiiPolynomial

include("Import_Conjugacies.jl")
include("DSV_Functions.jl")
include("Numerics.jl")
include("Proofs.jl")
include("fp_Data.jl")

export fp_Rossler, fp_Kuramoto, fp_Lorenz, fp_MackeyGlass, data
export import_model, h_g_functions                          
export proofs, proof_sequential_1D, best_r★_proof_1D_outer 
export SELU, ReLU, sgn, abs_dsv, PWL3Sat, ⊘

end