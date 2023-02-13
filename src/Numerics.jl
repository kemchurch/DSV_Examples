# Outer Poincaré map (n-dimensional conjugate map, n>1)
function G_DG_outer_conj(encoder,decoder,c_map,order)
    function G₁(x)
        return decoder(c_map(encoder(x)))
    end
    function G(x)
        y = copy(x)
        for n = 1 : order
            y = G₁(y)
        end
        return x - y
    end
    DG = x -> ForwardDiff.jacobian(G,x)
    return G,DG
end

# Outer Poincaré map (conjugate map dimension n=1)
function G_DG_outer_1D_conj(encoder,decoder,c_map,order)
    function G₁(x)
        return decoder(c_map.(encoder(x)))
    end
    function G(x)
        y = copy(x)
        for n = 1 : order
            y = G₁(y)
        end
        return x - y
    end
    DG = x -> ForwardDiff.jacobian(G,x)
    return G,DG
end

# Inner Poincaré map (n-dimensional conjugate map, n>1)
function G_DG_inner_conj(encoder,decoder,c_map,order)
    function G(x)
        y = encoder(copy(x))
        for n = 1 : order
            y = c_map(y)
        end
        return x - decoder(y)
    end
    DG = x -> ForwardDiff.jacobian(G,x)
    return G,DG
end

# Inner Poincaré map (conjugate map dimension n=1)
function G_DG_inner_1D_conj(encoder,decoder,c_map,order)
    function G(x)
        y = encoder(copy(x))
        for n = 1 : order
            y = c_map(y)
        end
        return x - decoder([y])
    end
    DG = x -> ForwardDiff.jacobian(G,x)
    return G,DG
end

# Sequential map set-up for the outer Poincaré map;
# 1-dimensional conjugate map.
function G_DG_outer_1D_sequential_conj(encoder,decoder,c_map,order)
    function step(x)
        return decoder([c_map(encoder([x]))])[1]
    end
    function G(x)
        y = Array{eltype(x)}(undef,order)
        y[1] = x[1] - step(x[order])
        for n=2:order
            y[n] = x[n] - step(x[n-1])
        end
        return y
    end
    DG = x -> ForwardDiff.jacobian(G,x)
    return G,DG
end

# Streamfunction stuff.
function map_F_streamfunction(ψ₀::Array,ψ₁::Array,x::Array,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    return ψ₁ - expand(interpolate_ssh(interlace(flatten(ψ₀),flatten(x)),model;epsilon,α))
end

function map_F_streamfunction(ψ₀::Vector,ψ₁::Vector,x::Vector,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    return ψ₁ - interpolate_ssh(interlace(ψ₀,x),model;epsilon,α)
end

function row_scale(A,b)
    # Scale the rows of A by rows of b. Much faster than doing B*diagm(b) when working with intervals.
    B = similar(A)
    for n=1:lastindex(b)
        B[n,:] = A[n,:]*b[n]
    end
    return B
end

function Dy_interpolate_ssh(y::Sequence,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    # Hard-coded derivative of the interpolation map, because calculating this with ForwardDiff is a bit too slow, especially with IntervalArithmetic.
    W = model.dense_weight
    B = model.dense_bias
    Gamma = model.Batch_Gamma
    Beta = model.Batch_Beta
    MMean = model.Batch_MovMean
    MVar = model.Batch_MovVar
    # Layer 1
    # -- Dense
    y = W[1]*y + B[1]
    D = W[1]
    ## -- BatchNormalization
    y = BatchNormalization(y,Gamma[1],Beta[1],MMean[1],MVar[1];epsilon=epsilon)
    d1 = coefficients(Gamma[1]./sqrt.((MVar[1] .+ epsilon)))
    D = row_scale(D,d1) 
    ## -- LeakyReLU
    y = LeakyReLU.(y;α=α)
    d1 = coefficients((y .< 0)*(α) + (y .> 0)*1)
    D = row_scale(D,d1)
    ## Layer 2
    # -- Dense
    y = W[2]*y + B[2]
    D = W[2]*D
    ## -- BatchNormalization
    y = BatchNormalization(y,Gamma[2],Beta[2],MMean[2],MVar[2];epsilon=epsilon)
    d1 = coefficients(Gamma[2]./sqrt.((MVar[2] .+ epsilon)))
    D = row_scale(D,d1)
    ## -- LeakyReLU
    y = LeakyReLU.(y;α=α)
    d1 = coefficients((y .< 0)*(α) + (y .> 0)*1)
    D = row_scale(D,d1)
    # Layer 3
    # -- Dense
    y = W[3]*y + B[3]
    D = W[3]*D
    return y,D
end

function Dy_interpolate_ssh(y::Vector,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    Dy = Dy_interpolate_ssh(Sequence(y),model;epsilon=epsilon,α=α)
    return coefficients(Dy[1]),coefficients(Dy[2])
end

function map_DF_streamfunction(ψ₀::Vector,x::Vector,model::FCC_model;epsilon::Real=big(0.001),α::Real=big(0.3))
    Dx = LinearOperator(ForwardDiff.jacobian(y->interlace(mid.(ψ₀),y), mid.(x)))    # interlace behaves like a permutation, so the result is a 0-1 matrix. 
    _,D_interpolate = Dy_interpolate_ssh(interlace(ψ₀,x),model;epsilon=epsilon,α=α)
    return -LinearOperator(D_interpolate)*Dx
end

function Newton(x,F,DF;verbose=true,tol=1E-30)
    defect = Inf
    iter = 0
    while defect>tol && iter<16
        Δx = DF(x)\F(x)
        x = x - Δx
        defect = norm(Δx)
        if verbose
            println("⋅ iter = $iter, |A⋅F(x)| = $defect")
        end
        iter += 1
    end
    return x
end

function RoundIntervalF64(x)
    # Round interval to Float64
    return interval.(Float64.(inf.(x),RoundDown),Float64.(sup.(x),RoundUp))
end