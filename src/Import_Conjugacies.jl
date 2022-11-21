function import_model(folder_string,model)
    tf = pyimport("tensorflow")
    model_pyobject = tf.keras.models.load_model(folder_string)
    layers_enc = py"len"(model_pyobject.encoder.layers)
    layers_dec = py"len"(model_pyobject.decoder.layers)
    encoder_weights = Vector{Matrix{BigFloat}}(undef,layers_enc); encoder_biases = Vector{Vector{BigFloat}}(undef,layers_enc)
    decoder_weights = Vector{Matrix{BigFloat}}(undef,layers_dec); decoder_biases = Vector{Vector{BigFloat}}(undef,layers_dec)
    for n in 1:layers_enc
        encoder_weights[n] = model_pyobject.encoder.layers[n].get_weights()[1]'
        encoder_biases[n] = model_pyobject.encoder.layers[n].get_weights()[2]
    end
    for n in 1:layers_dec
        decoder_weights[n] = model_pyobject.decoder.layers[n].get_weights()[1]'
        decoder_biases[n] = model_pyobject.decoder.layers[n].get_weights()[2]
    end
    if model=="Rossler" || model=="Kuramoto1D" || model=="MackeyGlass"
        g_data = [model_pyobject.c1.numpy();model_pyobject.c2.numpy()]
    elseif model=="Lorenz1D"
        g_data = x -> [model_pyobject.c1.numpy();model_pyobject.c2.numpy()]
    elseif model=="Lorenz2D"
        g_data = [model_pyobject.c1.numpy();model_pyobject.c2.numpy();model_pyobject.d0.numpy(); model_pyobject.d1.numpy()]
    elseif model=="Kuramoto2D"
        g_data = [model_pyobject.c0.numpy();model_pyobject.c10.numpy();model_pyobject.c01.numpy();model_pyobject.c20.numpy();model_pyobject.c11.numpy();model_pyobject.c02.numpy();
        model_pyobject.d0.numpy();model_pyobject.d10.numpy();model_pyobject.d01.numpy();model_pyobject.d20.numpy();model_pyobject.d11.numpy();model_pyobject.d02.numpy()]
    end
    return encoder_weights,encoder_biases,decoder_weights,decoder_biases,g_data
end

function h_g_functions(e_weight,e_bias,d_weight,d_bias,g_data,model;alpha::T where T<:Real=BigFloat(1.67326324),scale::T where T<:Real=BigFloat(1.05070098))
    layers_enc = size(e_weight,1)
    layers_dec = size(d_weight,1)
    function e_func(x)
        y = Vector{Vector{eltype(x)}}(undef,layers_enc+1)
        y[1] = x
        for n = 1:layers_enc
            y[n+1] = SELU.(e_weight[n]*y[n] + e_bias[n];alpha,scale)
        end
        return y[layers_enc+1]
    end
    function d_func(x)
        y = Vector{Vector{eltype(x)}}(undef,layers_dec+1)
        y[1] = x
        for n = 1:layers_dec
            y[n+1] = SELU.(d_weight[n]*y[n] + d_bias[n];alpha,scale)
        end
        return y[layers_dec+1]
    end
    if model=="Rossler" || model=="Kuramoto1D" || model=="MackeyGlass"
        c_map = x -> x[1]*g_data[1] + g_data[2]*x[1]^2
    elseif model=="Lorenz2D"
        c_map = x -> [-sgn(x[1]) + x[1]*g_data[1] + g_data[2]*x[1]*abs_dsv(x[1]) ; g_data[3]*sgn(x[1]) + g_data[4]*x[2]]
    elseif model=="Kuramoto2D"
        c_map = x -> [g_data[1] + g_data[2]*x[1] + g_data[3]*x[2] + g_data[4]*x[1]^2 + g_data[5]*x[1]*x[2] + g_data[6]*x[2]^2 ;
        g_data[7] + g_data[8]*x[1] + g_data[9]*x[2] + g_data[10]*x[1]^2 + g_data[11]*x[1]*x[2] + g_data[12]*x[2]^2]
    end
    return e_func, d_func, c_map
end