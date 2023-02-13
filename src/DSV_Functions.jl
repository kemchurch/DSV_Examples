function SELU(x::Real ;alpha::Real=1.67326324,scale::Real=1.05070098)
    if x<0
        return scale*alpha*(exp(x)-1)
    elseif x>0
        return scale*x
    else
        error("An argument to SELU violates DSV smoothness specification.")
    end
end

function ReLU(x::Real)
    if x<0
        return zero(typeof(x))
    elseif x>0
        return x
    else
        error("An argument to ReLU violates DSV smoothness specification.")
    end
end

function LeakyReLU(x::Real;α::Real=0.3)
    if x<0
        return α*x
    elseif x>0
        return one(typeof(α))*x
    else
        error("An argument to leaky ReLU violates DSV smoothness specification.")
    end
end

function sgn(x::Real)
    if x<0
        return -one(typeof(x))
    elseif x>0
        return one(typeof(x))
    else
        error("An argument to sgn violates DSV continuity specification.")
    end
end

function abs_dsv(x::Real)
    if x<0
        return -x
    elseif x>0
        return x
    else
        error("An argument to abs_dsv violates DSV smoothness specification.")
    end
end

function PWL3Sat(x,a,b,min,max)
    if a<b
        error("PWL3Sat(x,a,b,min,max) requires $a < $b.")
    elseif x<min
        return min*one(typeof(x))
    elseif a<x && x<b
        m = (max-min)/(b-a)
        return min + m*(x-a)
    elseif max<x
        return max*one(typeof(x))
    else
        error("An argument to x↦PWL3Sat(x,a,b,min,max) violates DSV smoothness specification.")
    end
end

function ⊘(a::Real,b::Real)
    if b<0 || 0<b
        return a/b
    else
        error("Second argument/denominator of ⊘  violates DSV smoothness specification.")
    end
end