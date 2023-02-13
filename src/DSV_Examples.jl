module DSV_Examples

using PyCall, RadiiPolynomial, JLD2, ForwardDiff, LinearAlgebra
using ForwardDiff: Chunk

include("Import_Conjugacies.jl")
include("DSV_Functions.jl")
include("Import_Interpolator.jl")
include("Numerics.jl")
include("Proofs.jl")
include("fp_Data.jl")
ssh_load = load("src\\streamfunction_data.jld2")
ssh_in = Float64.(ssh_load["val_inVar"])
ssh_out = Float64.(ssh_load["val_outVar"])

# Conjugacy export
export fp_Rossler, fp_Kuramoto, fp_Lorenz, fp_MackeyGlass, data
export import_conjugacy_model, h_g_functions  
export proofs, proofs_conjugacies, proof_conjugacy_sequential_1D, best_r★_proof_conjugacy_1D_outer 

# Streamfunction export
export FCC_model, import_ssh_interpolation_model
export ssh_in, ssh_out 
export proofs_ssh         

# DSV export
export SELU, ReLU, LeakyReLU, sgn, abs_dsv, PWL3Sat, ⊘

# Other
export Interval

end