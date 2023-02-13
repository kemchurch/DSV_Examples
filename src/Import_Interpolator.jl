struct FCC_model{T<:Real}
    dense_weight::Vector{LinearOperator{CartesianPower{ParameterSpace}, CartesianPower{ParameterSpace}, Matrix{T}}}
    dense_bias::Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}
    Batch_Gamma::Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}
    Batch_Beta::Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}
    Batch_MovMean::Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}
    Batch_MovVar::Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}
end

function BatchNormalization(x,Gamma,Beta,MovMean,MovVar;epsilon::Real=big(0.001))
    return (Gamma.*(x-MovMean))./sqrt.((MovVar .+ epsilon)) + Beta
end

function flatten(M::Array{T,2}) where T
    # Consistent with Keras, but we output a column vector because row vectors are unholy.
    return reshape(transpose(M),size(M)[1]*size(M)[2])[:,1]
end

function expand(M::Array{T}) where T
    # Inverse function of flatten provided flatten was fed a square matrix.
    m = Int(sqrt(length(M)))
    return Matrix(transpose(reshape(M,m,m)))
end

function interlace(y1,y2)
    # The first layers in the feedforward seems to interlace the two "sheets" of the tensor of type
    # (:,:,2). Not going to dig into the code to find out why. Just going to write a function to
    # do that at the level of the flattened matrices.
    if size(y1) !== size(y2)
        error("Inputs are not the same size.")
    end
    if length(y1) !== length(y2)
        error("Inputs of not the same length.")
    end
    return Vector(reshape(transpose([y1 y2]),length(y1)*2))
end

function import_ssh_interpolation_model(folder_string,T)
    N_Dense = 3
    N_BatchNormalization = 2
    tf = pyimport("tensorflow")
    model_pyobject = tf.keras.models.load_model(folder_string)
    dense_weight = Vector{LinearOperator{CartesianPower{ParameterSpace}, CartesianPower{ParameterSpace}, Matrix{T}}}(undef,N_Dense)
    dense_bias = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}(undef,N_Dense)
    Gamma = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}(undef,N_BatchNormalization)
    Beta = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}(undef,N_BatchNormalization)
    Moving_Mean = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}(undef,N_BatchNormalization)
    Moving_Variance = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{T}}}(undef,N_BatchNormalization)
    for n=1:N_Dense
        dense_weight[n] = LinearOperator(model_pyobject.layers[3*n].get_weights()[1]')
        dense_bias[n] = Sequence(model_pyobject.layers[3*n].get_weights()[2])
    end
    for m=1:N_BatchNormalization
        Gamma[m] = Sequence(model_pyobject.layers[3*(m-1)+4].get_weights()[1])
        Beta[m] = Sequence(model_pyobject.layers[3*(m-1)+4].get_weights()[2])
        Moving_Mean[m] = Sequence(model_pyobject.layers[3*(m-1)+4].get_weights()[3])
        Moving_Variance[m] = Sequence(model_pyobject.layers[3*(m-1)+4].get_weights()[4])
    end
    return FCC_model(dense_weight,dense_bias,Gamma,Beta,Moving_Mean,Moving_Variance)
end

function round_model(model)
    N_Dense = 3
    N_BatchNormalization = 2
    dense_weight = Vector{LinearOperator{CartesianPower{ParameterSpace}, CartesianPower{ParameterSpace}, Matrix{Interval{Float64}}}}(undef,N_Dense)
    dense_bias = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{Interval{Float64}}}}(undef,N_Dense)
    Gamma = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{Interval{Float64}}}}(undef,N_BatchNormalization)
    Beta = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{Interval{Float64}}}}(undef,N_BatchNormalization)
    Moving_Mean = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{Interval{Float64}}}}(undef,N_BatchNormalization)
    Moving_Variance = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{Interval{Float64}}}}(undef,N_BatchNormalization)
    for n=1:N_Dense
        dense_weight[n] = LinearOperator(RoundIntervalF64(coefficients(model.dense_weight[n])))
        dense_bias[n] = Sequence(RoundIntervalF64(coefficients(model.dense_bias[n])))
    end
    for m=1:N_BatchNormalization
        Gamma[m] = Sequence(RoundIntervalF64(coefficients(model.Batch_Gamma[m])))
        Beta[m] = Sequence(RoundIntervalF64(coefficients(model.Batch_Beta[m])))
        Moving_Mean[m] = Sequence(RoundIntervalF64(coefficients(model.Batch_MovMean[m])))
        Moving_Variance[m] = Sequence(RoundIntervalF64(coefficients(model.Batch_MovVar[m])))
    end
    return FCC_model(dense_weight,dense_bias,Gamma,Beta,Moving_Mean,Moving_Variance)
end

function midpoint_model(model)
    N_Dense = 3
    N_BatchNormalization = 2
    BaseType = eltype(model.dense_weight[1]).types[1]
    dense_weight = Vector{LinearOperator{CartesianPower{ParameterSpace}, CartesianPower{ParameterSpace}, Matrix{BaseType}}}(undef,N_Dense)
    dense_bias = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{BaseType}}}(undef,N_Dense)
    Gamma = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{BaseType}}}(undef,N_BatchNormalization)
    Beta = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{BaseType}}}(undef,N_BatchNormalization)
    Moving_Mean = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{BaseType}}}(undef,N_BatchNormalization)
    Moving_Variance = Vector{Sequence{CartesianPower{ParameterSpace}, Vector{BaseType}}}(undef,N_BatchNormalization)
    for n=1:N_Dense
        dense_weight[n] = LinearOperator(mid.(coefficients(model.dense_weight[n])))
        dense_bias[n] = Sequence(mid.(coefficients(model.dense_bias[n])))
    end
    for m=1:N_BatchNormalization
        Gamma[m] = Sequence(mid.(coefficients(model.Batch_Gamma[m])))
        Beta[m] = Sequence(mid.(coefficients(model.Batch_Beta[m])))
        Moving_Mean[m] = Sequence(mid.(coefficients(model.Batch_MovMean[m])))
        Moving_Variance[m] = Sequence(mid.(coefficients(model.Batch_MovVar[m])))
    end
    return FCC_model(dense_weight,dense_bias,Gamma,Beta,Moving_Mean,Moving_Variance)
end

function interpolate_ssh(y::Sequence,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    W = model.dense_weight
    B = model.dense_bias
    Gamma = model.Batch_Gamma
    Beta = model.Batch_Beta
    MMean = model.Batch_MovMean
    MVar = model.Batch_MovVar
    # Note: codes are specific to our example.
    # Layer 1
    y = W[1]*y + B[1]                 
    y = BatchNormalization(y,Gamma[1],Beta[1],MMean[1],MVar[1];epsilon=epsilon)
    y = LeakyReLU.(y;α=α)
    # Layer 2
    y = W[2]*y + B[2]
    y = BatchNormalization(y,Gamma[2],Beta[2],MMean[2],MVar[2];epsilon=epsilon)
    y = LeakyReLU.(y;α=α)
    # Layer 3
    y = W[3]*y + B[3]
    return y
end

function interpolate_ssh(y::Vector,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    return coefficients(interpolate_ssh(Sequence(y),model;epsilon=epsilon,α=α))
end