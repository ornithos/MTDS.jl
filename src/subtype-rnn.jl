struct MTGRU_Single end
struct MTGRU_Double end

"""
    MTGRU_variational(enc_tform, init_enc, init_post, mt_enc, mt_post,
        x0decoder, mtbias_deocder, gen_model, d_y, mtbias_only)

A sequence-to-sequence model 𝒴→𝒴 which imbibes `T_steps` steps of the sequence,
and predicts `T_steps` (including the original `T_steps`), but unlike the
`Seq2SeqModel`, the `MTSeqModel_Pred` can *modulate* its prediction based on
hierarchical 'multi-task' variables. The `Pred` suffix refers to the function
of the encoders. I had previously made the `E3` suffix version (see docstring),
but this was optimized for learning a latent representation of the entire
sequence, which is effective for disentangled-like representation and likelihood
but less so for prediction, since optimizing the evidence of the **observed**
sequence appears to perform worse on future observations than an RNN trained
only to predict. For this model, the encoder learns a latent code that is
optimized for **predictive** performance, not **reconstructive** performance.
See the paper, and similar observations have previously been made -- see e.g.
Karl et al. 2016, Deep VB Filter.

As for the `Seq2SeqModel`, the posterior over the initial state of the generator
is learned variationally via the `init_enc` RNN with the final state passed to
the `init_post` layer which generates the mean and variance of the latent
posterior. Since the posterior is typically lower dimensional than the state of
the generator network, the latent variable is expanded via the `x0decoder`
network into the state of the `generator`. The MT variable is encoded with a
full-length sequence encoding via the `mt_enc` network, and the final state is
passed to the `mt_post` layer to generate the posterior mean and variance.
The MT variable enters the generator network either:

1. by predicting all parameters of the network via a `MTGRU_NoU` construction
(see docstring).
2. by forming a latent (constant) input at each time step, which functions as a
variable bias of the generator recurrent cell. This may be directly from the
latent variables, or via a decoder (`mtbias_decoder`).

Once these variables are sampled, the `generator` is iterated in an open-loop
fashion for `T_steps`, similar to the `Seq2SeqModel`. The penultimate slot `d`
is a record of the dimension of 𝒴, used currently just for `show`; and the
`mtbias_only` variable dictates whether either approach (1) or (2) is used as
above.

--------------------
This type can be called via
```julia
    m = MTSeqModel_Pred(...)
    m(y::Vector)
```
where `y` is a vector of slices of the sequence for the encoder. This means each
`y[i]` is an `AbstractMatrix` of size ``d \\times n_{\\text{batch}}`.

Options include `T_enc`, `T_steps` (already discussed), `reset` (which performs
a state reset before execution; true by default), and `stoch`, whether to
randomly sample the latent state, or use the state mean (default true).
"""
struct MTGRU_variational{U,A,B,V,W,S,N,M,T} <: MTDSModel
    enc_tform::U
    init_enc::A
    init_post::B
    mt_enc::A
    mt_post::B
    x0decoder::V
    mtbias_decoder::W
    generator::S
    dec_tform::N
    model_type::M
    logstd::T
    d::Int
    d_mt::Int
    mtbias_only::Bool
end


# Model utils
Flux.@treelike MTGRU_variational
model_type(m::MTGRU_variational{U,A,B,V,W,S,N,M,T}) where {U,A,B,V,W,S,N,M,T} = M

unpack(m::MTGRU_variational) = Flux.children(m)[1:9] # unpack struct; exclude `d`, `mtbias_only`.
unpack_inference(m::MTGRU_variational) = Flux.children(m)[1:5]
unpack_generative(m::MTGRU_variational) = (m.x0decoder, m.mtbias_decoder, m.generator, m.dec_tform)

load!(m::MTGRU_variational, fname::String) = load!(Flux.params(m), fname)

function Base.show(io::IO, l::MTGRU_variational)
    d_x0, d_mt = size(l.init_post.Dense1.W, 1), l.d_mt
    out_type = Dict(MTGRU_Single=>"Deterministic", 
                    MTGRU_Double=>"Probabilistic")[model_type(l)]
    mttype = l.mtbias_only ? "Bias-only" : "Full-MT Recurrent Cell (" * 
                    get_strtype_wo_params(l.generator) * ")"
    print(io, "MTGRU_variational(", mttype, ", d_x0=", d_x0, ", d_mt=", d_mt, ", ", out_type, ")")
end


################################################################################
##                                                                            ##
##                          Encoding the prior sequence                       ##
##                                                                            ##
################################################################################

function encode(m::MTGRU_variational, Y::AbstractArray{T}, U::AbstractArray{T}; T_enc=size(Y, 2), 
                T_x0=T_enc, stochastic=true) where T
    encode(m, vcat(Y, U); T_enc=T_enc, T_x0=T_x0, stochastic=stochastic)
end

encode(m::MTGRU_variational, Y::MaskedArray{T}, U::AbstractArray{T}; T_enc=size(Y, 2), T_x0=T_enc, 
       stochastic=true) where T = encode(m, vcat(stacked(Y), U); T_enc=T_enc, T_x0=T_x0,
                                         stochastic=stochastic)

function encode(m::MTGRU_variational, data_enc::AbstractArray; T_enc=size(data_enc,2), T_x0=T_enc,
                           stochastic=true, x0=nothing, z=nothing)
    enc_tform, init_enc, init_post, mt_enc, mt_post = unpack_inference(m)

    # RNN 1: Amortized inference for initial state, h0, of generator
    x0, μ_x0, σ_x0 = posterior_sample(init_enc, init_post, data_enc, T_enc, stochastic)

    # RNN 2: Amortized inference for z
    # Technically this should be conditioned on the sample x₀, nevertheless, the posterior
    # of x0 is usually very tight, and the `mt_enc` has access to the same information.
    z, μ_z, σ_z = posterior_sample(mt_enc, mt_post, data_enc, T_x0, stochastic)

    return (z, μ_z, σ_z), (x0, μ_x0, σ_x0)
end

# common superclass properties
is_amortized(m::MTGRU_variational) = true
is_amortized(m::MTGRU_variational{U,A,B,V,W,S,N,M,T}) where {A <: LookupTable, U,B,V,W,S,N,M,T} = false
has_x0_encoding(m::MTGRU_variational) = true

################################################################################
##                                                                            ##
##                                Forward Model                               ##
##                                                                            ##
################################################################################

function forward(m::MTGRU_variational, z, U; T_steps=size(U, 2))
    n_u, n_t, n_b = size(U)
    forward(m, zeros(eltype(U), size(m.x0decoder.W, 2), n_b), z, U; T_steps=T_steps)
end

function forward(m::MTGRU_variational, x0::AbstractVecOrMat{T}, z::AbstractVecOrMat{T},
        U::AbstractArray{T}; T_steps=size(U, 2)) where T <: AbstractFloat
    # note that T_steps is currently ignored (this is no great issue).
    @argcheck size(z, 2) == size(U, 3)

    # currently don't modulate the initial state.
    return _forward_given_latent(m, x0, z, U, T_steps)
end

    function _forward_given_latent(m::MTGRU_variational, x0::AbstractVector, z::AbstractVector,
        U::AbstractArray{T,2}, T_steps::Int) where T
        x0, z, U = ensure_matrix(x0), ensure_matrix(z), ensure_tensor(U)
        return dropdims(_forward_given_latent(m, x0, z, U, T_steps), dims=3)
    end

    function _forward_given_latent(m::MTGRU_variational, x0::AbstractMatrix, z::AbstractMatrix,
                                U::AbstractArray{T,3}, T_steps::Int) where T
        x0decoder, mtbias_decoder, gen_model, dec_tform = unpack_generative(m)
        
        if dec_tform.layers[2] isa MTDenseGenerator
            _mtdec = dec_tform.layers[2](z)
            decoder(x) = let l1=dec_tform.layers[1](x); _mtdec(l1) + dec_tform.layers[3](l1); end
        else
            decoder = dec_tform
        end

        if m.mtbias_only
            gen_model.state = x0decoder(x0)
            mtbias = mtbias_decoder(z)   # if linear, this is just the identity ∵ GRUCell.Wi
            yhat_ts = [unsqueeze(decoder(gen_model(mtbias)),2) for t in 1:T_steps]
        else
            # Get GRU models from samples from the MT GRU model
            posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
            # Set state (batch-wise) to x0 sample
            posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch
            # Run generative model
            
            yhat_ts = [unsqueeze(decoder(posterior_grus(U[:,tt,:])),2) for tt in 1:T_steps]
        end
        return hcat(yhat_ts...)
    end

function (m::MTGRU_variational)(x0::AbstractVecOrMat, z::AbstractVecOrMat, U::AbstractArray;
          T_steps=size(U,2))
    _forward_given_latent(m, x0, z, U, T_steps)
end

# function _posterior_sample(enc, dec, input::Vector, T_max, stochastic=true, input_cat=nothing)
#     Flux.reset!(enc)
#     if input_cat === nothing
#         for tt = 1:T_max; enc(input[tt]); end
#     else
#         d, nbatch = size(input[1])
#         if nbatch == 1 && nbatch != size(input_cat,2)
#             gmove = Flux.has_cuarrays() && input_cat isa Flux.CuArray ? gpu : identity
#             input_expander = gmove(ones(Float32, 1, size(input_cat, 2)))
#         else
#             input_expander = 1
#         end
#         for tt = 1:T_max; enc(vcat(input[tt] * input_expander, input_cat)); end
#     end
#     μ_, σ_ = dec(enc.state)
#     n, d = size(μ_)
#     smp = μ_ + randn_repar(σ_, n, d, stochastic)
#     return smp, μ_, σ_
# end

################################################################################
##                                                                            ##
##                             Model "Constructor"                            ##
##                                                                            ##
################################################################################

function create_mtgru(d_u, d_x, d_y, d_enc_state, d_enc_x0, d_mt; encoder=:GRU,
    out_heads=1, d_hidden=d_x, mtbias_only=false, d_hidden_mt=32, mt_is_linear=true, 
    fixb=false, fixb_version=nothing, spherical_var=false, lkp_init_std_logit=-2f0)

    @argcheck out_heads in 1:2
    @assert !(fixb && mtbias_only) "cannot do mtbias AND fix 'b'"
    @assert !fixb || something(fixb_version, 1) ∈ [1,2] "fixb_version must be in [1,2]."

    if fixb && fixb_version === nothing
        @warn "fixed bias chosen, but `fixb_version` not specified. Defaulting to v1."
        fixb_version = 1
    end

    # ENCODER TRANSFORM FROM OBS SPACE => ENC SPACE
    tform_enc = identity
    d_rnn_in = d_y+d_u
    model_type = out_heads == 1 ? MTGRU_Single() : MTGRU_Double()

    # 2× ENCODERS for inference of (a) initial state (init_enc); (b) sequence level MT var (mt_enc)
    @argcheck encoder in [:LSTM, :GRU, :Bidirectional, :LookupTable]

    if encoder == :LookupTable
        # args 2/3: initial posterior noise for mu/logstd
        init_enc = LookupTable(d_enc_x0, 0.01f0, lkp_init_std_logit)  
        mt_enc = LookupTable(d_mt, 0.01f0, lkp_init_std_logit)
    else
        rnn_constructor = Dict(:LSTM=>LSTM, :GRU=>GRU, :Bidirectional=>BRNNenc)[encoder]
        init_enc = rnn_constructor(d_rnn_in, d_enc_state)
        mt_enc = rnn_constructor(d_rnn_in, d_enc_state)
        (encoder == :Bidirectional) && (d_enc_state *= 2)
    end


    # POSTERIOR OVER LVM CORR. TO (a), (b). Ignored if using LookupTable
    init_post = MultiDense(Dense(d_enc_state, d_enc_x0, identity), Dense(d_enc_state, d_enc_x0, σ))
    init_post.Dense2.b.data .= -2   # initialize posteriors to be low variance
    mt_post = MultiDense(Dense(d_enc_state, d_mt, identity), Dense(d_enc_state, d_mt, σ))
    mt_post.Dense2.b.data .= -2

    # decode from LV (a) --> size of generative hidden state
    x0decoder = Dense(d_enc_x0, d_x, identity)

    # -------------- GENERATIVE MODEL -----------------
    ###################################################

    if !mtbias_only
        d_out = !fixb ? 3*d_x*(d_x+d_u+1) : (fixb_version == 1 ? 3*d_x*(d_x+d_u) : 
            3*d_x*(d_x+d_u) + 2*d_x)
        par_gen_net = mt_is_linear ? mlp(d_mt, d_out) : mlp(d_mt, d_hidden_mt, 
                                                                    d_out; activation=tanh)
        if !fixb
            gen_rnn = MTGRU(d_u, d_x, par_gen_net)
        else
            if fixb_version == 1
                MTGRU_b = Flux.param(randn(Float32, d_x*3))
                gen_rnn = MTGRU_fixb(d_u, d_x, par_gen_net, MTGRU_b)
            elseif fixb_version == 2
                MTGRU_b = Flux.param(randn(Float32, d_x))
                gen_rnn = MTGRU_fixb_v2(d_u, d_x, par_gen_net, MTGRU_b)
            else
                error("fixb_version: Unreachable error.")
            end
        end
        mtbias_decoder = identity
    else
        mtbias_decoder = mt_is_linear ? identity : Dense(d_mt, d_hidden_mt, swish)
        _d_in = mt_is_linear ? d_mt : d_hidden_mt
        gen_rnn = GRU(_d_in, d_x)
    end

    if out_heads == 1
        # decoder = Chain(Dense(d_x, d_hidden, relu), Dense(d_hidden, d_y, identity))
        decoder = Chain(Dense(d_x, d_hidden, relu), 
                        MTDenseGenerator(d_mt, d_hidden, d_y, identity),
                        Dense(d_hidden, d_y, identity))   # this is super hacky -- see forward pass
    elseif out_heads == 2
        @error "out_heads=2 reqs a little re-engineering of the forward pass due to the MTDense"*
        " layer. Really we need to make a generator for the whole Chain of the decoder, not simply"*
        " the MTDense part."
        decoder = Chain(Dense(d_x, d_hidden, relu), MultiDense(
            MTDenseGenerator(d_mt, d_hidden, d_y, identity), 
            MTDenseGenerator(d_mt, d_hidden, d_y, identity)))
    end

    emission_std = spherical_var ? Tracker.param([0.0f0]) : Tracker.param(zeros(Float32, d_y))

    m = MTGRU_variational(tform_enc, init_enc, init_post, mt_enc, mt_post,
        x0decoder, mtbias_decoder, gen_rnn, decoder, model_type, emission_std, 
        d_y, d_mt, mtbias_only)

    return m
end


"""
    create_model_opt_dict(...)
Takes same arguments as `create_model` but just wraps up arguments in a Dict for
saving out as JSON/YAML/BSON etc.
"""
function create_model_opt_dict(d_x, d_x0, d_y, d_enc_state, d_mt, d_chaos=0; encoder=:GRU,
    cnn=false, out_heads=1, d_hidden=d_x, mtbias_only=false, d_hidden_mt=32,
    mt_is_linear=true, decoder_fudge_layer=false, model_purpose=:llh, fixb=false,
    fixb_version=nothing)

    Dict("d_x"=>d_x, "d_x0"=>d_x0, "d_y"=>d_y, "d_enc_state"=>d_enc_state, "d_mt"=>d_mt, "d_chaos"=>d_chaos,
    "encoder"=>encoder, "cnn"=>cnn, "out_heads"=>out_heads, "d_hidden"=>d_hidden, "mtbias_only"=>mtbias_only,
    "d_hidden_mt"=>d_hidden_mt, "mt_is_linear"=>mt_is_linear, "decoder_fudge_layer"=>decoder_fudge_layer,
    "model_purpose"=>model_purpose, "fixb"=>fixb, "fixb_version"=>fixb_version)
end


"""
    load_model_from_def(ymlfile)
We have YAML files saved with metadata associated with the model parameters
in the /data folder. These YAML files point to the relevant parameters too, and
so suffices to construct the model and load the parameters. That is the purpose
of this function.
"""
function load_model_from_def(ymlfile::String)
    model_details = YAML.load_file(ymlfile)
    filename = model_details["filename"]
    constructor = model_details["constructor"]
    @argcheck constructor == "mtmodel.create_model"
    D = model_details["model_def"]
    D["encoder"] = Symbol(D["encoder"])
    D["model_purpose"] = Symbol(D["model_purpose"])
    m = create_model(D["d_x"], D["d_x0"], D["d_y"], D["d_enc_state"], D["d_mt"], D["d_chaos"];
        encoder=D["encoder"], cnn=D["cnn"], out_heads=D["out_heads"], d_hidden=D["d_hidden"],
        mtbias_only=D["mtbias_only"], d_hidden_mt=D["d_hidden_mt"], mt_is_linear=D["mt_is_linear"],
        decoder_fudge_layer=D["decoder_fudge_layer"], model_purpose=D["model_purpose"], fixb=D["fixb"],
        fixb_version=D["fixb_version"])
    modelutils.load!(m, filename)
    return m
end

# # Example saving model info
# YAML.write_file("saved_models/mtgru_video_fixbgrp_pred_450.yml", Dict(
#     "filename"=>"data/mtgru_video_fixbgrp_pred_450.bson",
#     "description"=>"MT GRU model with posterior mean GROUPED BY SEQ IDENTITY fixed (v2) bias. 450x2 epochs. T=80, Tenc=20.",
#     "constructor"=>"mtmodel.create_model",
#     "model_def"=>mtmodel.create_model_opt_dict(64, 6, 4, 40, 2; encoder=:GRU, cnn=true, d_hidden=128,
#     model_purpose=:pred, mtbias_only=false, decoder_fudge_layer=true, fixb=true, fixb_version=2)
# ))


####################################################################################################
#                                                                                                  #
#     Overloading some objective subroutines in order to accommodate both z and x0 latent vars     #
#                                                                                                  #
####################################################################################################

function elbo_w_kl(m::MTGRU_variational, y::AbstractArray, u::AbstractArray, 
                   data_enc::AbstractArray; kl_coeff=1.0f0, stochastic=true, logstd_prior=nothing, 
                   T_enc=size(u, 2))
    nbatch = size(y,3)
    zstuff, x0stuff = encode(m, data_enc; stochastic=stochastic, T_enc=T_enc)
    zsmp, zμ, zσ = zstuff
    x0smp, x0μ, x0σ = x0stuff
    ŷ = m(x0smp, zsmp, u)
    nllh = _gauss_nllh_batchavg(y, ŷ, m.logstd)
    kl = stochastic * kl_coeff * (-0.5f0 * sum(1 .+ 2*log.(zσ) - zμ.*zμ - zσ.*zσ)) / nbatch
    kl += stochastic * kl_coeff * (-0.5f0 * sum(1 .+ 2*log.(x0σ) - x0μ.*x0μ - x0σ.*x0σ)) / nbatch
    # Prior regularization of emission stdev.
    if !(logstd_prior === nothing)
        nllh += sum(x->x^2, (m.logstd .- logstd_prior[1])./logstd_prior[2])/2
    end
    nllh + kl, kl
end

"""
    forward_multiple_z(m::MTDSModel, x0, z, u::AbstractMatrix; maxbatch=200)
Performs multiple forward passes, partitioned according to the `maxbatch` argument, with a single
`x0` and `u`, but different `z` values. Returns a matrix of size d_y × T × n_b.
"""
function forward_multiple_z(m::MTGRU_variational, x0, z, u::AbstractMatrix; maxbatch=200)
    M = size(z, 2)
    ŷs = map(Iterators.partition(1:M, maxbatch)) do cbatch
        u_rep = repeat(u, outer=(1, 1, length(cbatch)));
        x0_rep = repeat(x0, outer=(1, length(cbatch))); 
        m(x0_rep, z[:, cbatch], u_rep)
    end
    cat(ŷs..., dims=3)   # problematic if length(ŷs) is too large, but there's no nice `reduce` version in Julia yet.
end