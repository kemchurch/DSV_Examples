# Data processing helpers; specific to fp_data.
function decode_point(decoder,fp_data)
    return decoder(fp_data[:,1])
end

function decode_points_1D(decoder,fp_data)
    vec = Array{eltype(fp_data)}(undef,size(fp_data,1))
    for n in axes(fp_data,1)
        vec[n] = decoder([fp_data[n]])[1]
    end
    return vec
end

# Primary proof function.
function proof(x,F,DF,r★)
    ix = interval.(big.(x))
    ball_ix = ix .+ r★*interval(-1,1)
    Fx = F(ix)
    DFx = DF(x)
    A = LinearOperator(interval.(inv(mid.(DFx))))
    Y = norm(A*Sequence(Fx),Inf)
    Z = opnorm(I - A*LinearOperator(DF(ball_ix)),Inf)
    r = nextfloat(sup(Y/(1-interval(sup(Z)))))
    if r>0 && sup(Y+r*(Z-1))<0
        return sup(Y),sup(Z),sup(r)
    else
        return sup(Y),sup(Z),-Inf
    end
end

# Batch proof function.
function proofs(encoder,decoder,conjugacy,fp_data,r★,conjugacy_dimension)
    decoded_points = Vector{Vector{BigFloat}}(undef,size(fp_data,1))
    corrected_points_inner = Vector{Vector{BigFloat}}(undef,size(fp_data,1))
    corrected_points_outer = Vector{Vector{BigFloat}}(undef,size(fp_data,1))
    r_inner = Vector{BigFloat}(undef,size(fp_data,1))
    r_outer = Vector{BigFloat}(undef,size(fp_data,1))
    normcorrection_inner = Vector{BigFloat}(undef,size(fp_data,1))
    normcorrection_outer = Vector{BigFloat}(undef,size(fp_data,1))
    for k in axes(fp_data,1)
        if conjugacy_dimension==1
            order = size(fp_data[k],2)
            p = decode_point(decoder,fp_data[k])
            G_in,DG_in = G_DG_inner_1D(encoder,decoder,conjugacy,order)
            G_out,DG_out = G_DG_outer_1D(encoder,decoder,conjugacy,order)
        elseif conjugacy_dimension>1
            order = size(fp_data[k],1)
            p = decode_point(decoder,fp_data[k][1])
            G_in,DG_in = G_DG_inner(encoder,decoder,conjugacy,order)
            G_out,DG_out = G_DG_outer(encoder,decoder,conjugacy,order)
        end
        decoded_points[k] = p
        printstyled("Newton iteration starting (inner Poincare map, data row $k).\n"; color=:blue)
        p_cor_in = Newton(big.(p),G_in,DG_in)
        if norm(G_in(p_cor_in),Inf)>1E-14
            println("Warning: row $k of input fp_data Newton iteration with inner conjugacy did not converge.")
        end
        corrected_points_inner[k] = p_cor_in
        normcorrection_inner[k] = norm(p-p_cor_in,Inf)
        printstyled("Newton iteration starting (outer Poincare map, data row $k).\n"; color=:blue)
        p_cor_out = Newton(big.(p),G_out,DG_out)
        if norm(G_out(p_cor_out),Inf)>1E-14
            println("Warning: row $k of input fp_data Newton iteration with outer conjugacy did not converge.")
        end
        printstyled("Starting proofs for row $k.\n";color=:yellow)
        corrected_points_outer[k] = p_cor_out
        normcorrection_outer[k] = norm(p-p_cor_out,Inf)
        _,_,r_inner[k] = proof(p_cor_in,G_in,DG_in,r★)
        _,_,r_outer[k] = proof(p_cor_out,G_out,DG_out,r★)
    end
    return decoded_points,corrected_points_inner,corrected_points_outer,normcorrection_inner,normcorrection_outer,r_inner,r_outer
end

# Special proof function for the sequential (higher-dimension), low-composition set-up.
function proof_sequential_1D(encoder,decoder,conjugacy,fp_data,r★)
    order = size(fp_data[1,:],1)
    p0 = BigFloat.(decode_points_1D(decoder,fp_data[1,:]))
    G,DG = G_DG_outer_1D_sequential(encoder,decoder,conjugacy,order)
    printstyled("Newton iteration starting.\n"; color=:blue)
    p_cor = Newton(p0,G,DG)
    printstyled("Starting proof.\n";color=:yellow)
    _,_,r = proof(p_cor,G,DG,r★)
    return p0,p_cor,norm(p0-p_cor,Inf),r
end

# Finds the biggest r★=10⁻ⁿ for n≥2 where we are successful for a 1D proof. Can be 
# slower due to the use of error handling (try/catch). Outer Poincare map only.
function best_r★_proof_1D_outer(encoder,decoder,conjugacy,fp_data)
    p0 = decode_point(decoder,fp_data[1,:])
    order = size(fp_data[1,:],2)
    G,DG = G_DG_outer_1D(encoder,decoder,conjugacy,order)
    printstyled("Newton iteration starting (outer Poincare map).\n"; color=:blue)
    p_cor = Newton(p0,G,DG)
    printstyled("Checking for best r★.\n"; color=:yellow)
    r = BigFloat(Inf)
    n=2
    while r==Inf
        try 
            r,_,_ = proof(p_cor,G,DG,10.0^(-n))
        catch
            n+=1
        end
    end
    return p0,p_cor,norm(p0-p_cor,Inf),r,n
end