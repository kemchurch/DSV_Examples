# Outer Poincaré map (n-dimensional conjugate map, n>1)
function G_DG_outer(encoder,decoder,c_map,order)
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
function G_DG_outer_1D(encoder,decoder,c_map,order)
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
function G_DG_inner(encoder,decoder,c_map,order)
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
function G_DG_inner_1D(encoder,decoder,c_map,order)
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
function G_DG_outer_1D_sequential(encoder,decoder,c_map,order)
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

# Generic Newton iteration. Note that if eltype(x)!=BigFloat, this 
# will always iterate until iter=16 with stock tolerance tol=1E-30.
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