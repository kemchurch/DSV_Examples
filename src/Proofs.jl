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
function proof(x,F,DF,r★;fast_domain_test=false)
    ix = interval.(x)
    ball_ix = ix .+ r★*interval(-1,1)
    if fast_domain_test
        test = try 
            F(ball_ix)
        catch
            ∅
        end
        if test == ∅
            return ∞,∞,-Inf
        else
            println("Domain check (for error handling): evaluation of F(x .+ [-r★,r★]) succeeded. Continuing.")
        end
    end
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

function proof_debug(x,F,DF,r★)
    ix = interval.(x)
    ball_ix = ix .+ r★*interval(-1,1)
    Fx = F(ix)
    DFx = DF(x)
    A = LinearOperator(interval.(inv(mid.(DFx))))
    Y = norm(A*Sequence(Fx),Inf)
    Z = opnorm(I - A*LinearOperator(DF(ball_ix)),Inf)
    r = nextfloat(sup(Y/(1-interval(sup(Z)))))
    if r>0 && sup(Y+r*(Z-1))<0
        return sup(Y),sup(Z),sup(r),A
    else
        return sup(Y),sup(Z),-Inf,A
    end
end

# Batch proof function.
function proofs_conjugacies(encoder,decoder,conjugacy,fp_data,r★,conjugacy_dimension)
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
            G_in,DG_in = G_DG_inner_1D_conj(encoder,decoder,conjugacy,order)
            G_out,DG_out = G_DG_outer_1D_conj(encoder,decoder,conjugacy,order)
        elseif conjugacy_dimension>1
            order = size(fp_data[k],1)
            p = decode_point(decoder,fp_data[k][1])
            G_in,DG_in = G_DG_inner_conj(encoder,decoder,conjugacy,order)
            G_out,DG_out = G_DG_outer_conj(encoder,decoder,conjugacy,order)
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
        _,_,r_inner[k] = proof(big.(p_cor_in),G_in,DG_in,r★)
        _,_,r_outer[k] = proof(big.(p_cor_out),G_out,DG_out,r★)
    end
    return decoded_points,corrected_points_inner,corrected_points_outer,normcorrection_inner,normcorrection_outer,r_inner,r_outer
end

# Special proof function for the sequential (higher-dimension), low-composition set-up.
function proof_conjugacy_sequential_1D(encoder,decoder,conjugacy,fp_data,r★)
    order = size(fp_data[1,:],1)
    p0 = BigFloat.(decode_points_1D(decoder,fp_data[1,:]))
    G,DG = G_DG_outer_1D_sequential_conj(encoder,decoder,conjugacy,order)
    printstyled("Newton iteration starting.\n"; color=:blue)
    p_cor = Newton(p0,G,DG)
    printstyled("Starting proof.\n";color=:yellow)
    _,_,r = proof(big.(p_cor),G,DG,r★)
    return p0,p_cor,norm(p0-p_cor,Inf),r
end

# Finds the biggest r★=10⁻ⁿ for n≥2 where we are successful for a 1D proof. Outer Poincare map only.
function best_r★_proof_conjugacy_1D_outer(encoder,decoder,conjugacy,fp_data)
    p0 = decode_point(decoder,fp_data)
    order = size(fp_data,2)
    G,DG = G_DG_outer_1D_conj(encoder,decoder,conjugacy,order)
    printstyled("Newton iteration starting (outer Poincare map).\n"; color=:blue)
    p_cor = Newton(p0,G,DG)
    printstyled("Checking for best r★.\n"; color=:yellow)
    r = -BigFloat(Inf)
    n=2
    while r==-Inf
        print("Trying r★ = 1E-",n,".")
        _,_,r = try 
            proof(big.(p_cor),G,DG,10.0^(-n))
        catch
            (Inf,Inf,-Inf)
        end
        if r==-Inf
            n+=1
        end
        print("\e[1G")
    end
    println("Best r★ found: 1E-",n,".")
    return p0,p_cor,norm(p0-p_cor,Inf),r,n
end

function proofs_ssh(ℳ_Interval_BigFloat)
    r = Vector{BigFloat}(undef,10)
    Y = Vector{BigFloat}(undef,10)
    Z = Vector{BigFloat}(undef,10)
    # Defining auxiliary models.
    ℳ_midpoint_BigFloat = midpoint_model(ℳ_Interval_BigFloat)   # Midpoint of Interval{BigFloat} model.
    ℳ_Interval_Float64 = round_model(ℳ_Interval_BigFloat)       # Float64 enclosure model.
    ℳ_midpoint_Float64 = midpoint_model(ℳ_Interval_Float64)     # Midpoint of Float64 model.
    for n = 1:10
        printstyled("Attempting proof $n. \n",color=:blue)
        ψ₀ = interval.(flatten(big.(ssh_in[n,:,:,1])));    ψ₀_64 = flatten(ssh_in[n,:,:,1])
        x = interval.(flatten(big.(ssh_in[n,:,:,2])));     x_64 = flatten(ssh_in[n,:,:,2])
        ψ₁ = interpolate_ssh(interlace(mid.(ψ₀),mid.(x)),ℳ_midpoint_BigFloat;epsilon=mid(@biginterval(0.001)),α=mid(@biginterval(0.3)))
        println("⋅⋅ Startup finished. Proceeding to bounds...")
        F = map_F_streamfunction(ψ₀,ψ₁,x,ℳ_Interval_BigFloat;epsilon=@biginterval(0.001),α=@biginterval(0.3))   # Compute F with bigfloat precision.
        DF_64 = map_DF_streamfunction(ψ₀_64,x_64,ℳ_midpoint_Float64;epsilon=0.001,α=0.3)     # Compute A and DF with 64-bit precision.
        A = interval.(inv(DF_64))       
        Y = opnorm(A,Inf)*norm(F,Inf)    # Majorization for speed; |F| is bigfloat-precision-small, so ||A|| won't hurt us.
        println("⋅⋅ Y = ",sup(Y))
        println("⋅⋅ Proceeding to Z bound.")
        m = 6
        r[n] = -Inf
        while r[n]==-Inf
            println("⋅⋅ Current r★ = 1E-",m)
            DF_ball = try 
                map_DF_streamfunction(RoundIntervalF64(ψ₀),RoundIntervalF64(x) .+ interval(-1,1)*(10.0^(-m)), ℳ_Interval_Float64;epsilon=@interval(0.001),α=@interval(0.3))
            catch 
                Inf*LinearOperator(ones(size(DF_64)))
            end
            if DF_ball[1,1]==Inf
                println("⋅⋅ DSV violation in calculation of DF(x+δ).")
                Z = interval(-Inf,Inf)
            else
                Z = opnorm(I - A*DF_ball)
            end
            println("⋅⋅ Z = ",sup(Z))
            if (sup(Z)<1) && (sup(Y/(1-Z))<10.0^(-m))
                r[n] = sup(Y/(1-Z))
            else
                r[n] = -Inf
            end
            if r[n]==-Inf
                println("⋅⋅ Proof failed at r★ = 1E-",m,". Trying with r★ = 1E-",m+1)
                m+=1
            end
        end
        println("⋅⋅ Proof $n complete. Radius = ",r[n])
    end
    return r
end