module mtrnn

using Flux, ArgCheck, StatsBase, YAML
using Flux: gate
using Flux.Tracker: istracked

using ..modelutils
import ..modelutils: load!, randn_repar, mlp, get_strtype_wo_params
export load!

const D_IM = 32


################################################################################
##                                                                            ##
##              Define a basic MTGRU model (with no inputs)                   ##
##                                                                            ##
################################################################################

"""
    batch_matvec(A::Tensor, X::Matrix)

for A ∈ ℝ^{n × m × d}, X ∈ ℝ^{n × d}. Performs ``n`` matrix-vec multiplications

    A[i,:,:] * X[i,:]

for ``i \\in 1,\\ldots,n``. This is performed via expansion of `X` and can be
efficiently evaluated using BLAS operations, which (I have assumed!) will often
be faster than a loop, especially in relation to AD and GPUs.
"""
batch_matvec(A::AbstractArray{T,3}, X::AbstractArray{T,2}) where T =
    dropdims(sum(A .* unsqueeze(X, 1), dims=2), dims=2)

Flux.gate(x::AbstractArray{T,3}, h, n) where T = x[gate(h,n),:,:]
Flux.gate(x::AbstractArray{T,4}, h, n) where T = x[gate(h,n),:,:,:]


mutable struct MTGRU_NoU{F}
    N::Int
    G::F
end

"""
    MTGRU_NoU(N, G)
Multi-task GRU (Gated Recurrent Unit) model. Produces a GRU layer
for each value of `z` ∈ Rᵈ which depends on the map `G` inside
the MTGRU. N = out dimension, G: Rᵈ->R^{nLSTM}. z can be batched
as `z` ∈ R^{d × nbatch}.

In order to simplify the implementation, there is no input depen-
dent evolution (hence no `U`s, and hence no `Wi`). It is
straight-forward to extend to this case, should it be reqd.
"""
function (m::MTGRU_NoU)(z, h=nothing)
    N, nB = m.N, size(z, 2)
    λ = m.G(z)
    Wh = reshape(λ[1:N*N*3,:], N*3, N, nB)
    b = λ[N*N*3+1:(N+1)*N*3, :]
    open_forget_gate = zero(b)   # no backprop, and hence not tracked, even if `b` is.
    open_forget_gate[gate(N, 2), :] .= 1
    b += open_forget_gate
    h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

    Flux.Recur(BatchedGRUCell_NoU(Wh, b, h))
end

Flux.@treelike MTGRU_NoU
Base.show(io::IO, l::MTGRU_NoU) =
  print(io, "Multitask-GRU(", l.N, ", ", typeof(l.G), ", no inputs)")


"""
  MTGRU_NoU_fixb(N, G, b)
Multi-task GRU (Gated Recurrent Unit) model. Produces a GRU layer
for each value of `z` ∈ Rᵈ which depends on the map `G` inside
the MTGRU. N = out dimension, G: Rᵈ->R^{nLSTM}. z can be batched
as `z` ∈ R^{d × nbatch}. This version of the `MTGRU_NoU` does not
adapt the offset vector `b` in the GRU, but learns a fixed value
for all `z`.

In order to simplify the implementation, there is no input depen-
dent evolution (hence no `U`s, and hence no `Wi`). It is
straight-forward to extend to this case, should it be reqd.
"""
mutable struct MTGRU_NoU_fixb{F,V}
  N::Int
  G::F
  b::V
end

"""
  MTGRU_NoU_fixb_v2(N, G, b)
As per `MTGRU_NoU_fixb`, except now only the *state* bias is fixed across all
tasks. The gate biases depend on `z` just like the `MTGRU_NoU`.
"""
mutable struct MTGRU_NoU_fixb_v2{F,V}
  N::Int
  G::F
  b::V
end

function (m::MTGRU_NoU_fixb)(z, h=nothing)
  N, nB = m.N, size(z, 2)
  λ = m.G(z)
  Wh = reshape(λ, N*3, N, nB)
  b = m.b
  open_forget_gate = zero(b)   # no backprop, and hence not tracked, even if `b` is.
  open_forget_gate[gate(N, 2), :] .= 1
  b += open_forget_gate
  h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

  Flux.Recur(BatchedGRUCell_NoU(Wh, b, h))
end

function (m::MTGRU_NoU_fixb_v2)(z, h=nothing)
    N, nB = m.N, size(z, 2)
    λ = m.G(z)
    Wh = reshape(λ[1:N*N*3,:], N*3, N, nB)
    gmove = Flux.has_cuarrays() && Tracker.data(z) isa Flux.CuArray ? gpu : identity
    expander = gmove(ones(eltype(Tracker.data(λ)), 1, nB))
    b = vcat(λ[N*N*3+1:N*N*3+N*2, :], m.b * expander)
    open_forget_gate = zero(b)
    open_forget_gate[gate(N, 2), :] .= 1
    b += open_forget_gate
    h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

    Flux.Recur(BatchedGRUCell_NoU(Wh, b, h))
end

Flux.@treelike MTGRU_NoU_fixb
Base.show(io::IO, l::MTGRU_NoU_fixb) =
print(io, "Multitask-GRU-fixb(", l.N, ", ", typeof(l.G), ", bias vector: ", typeof(l.b), "; no inputs)")

Flux.@treelike MTGRU_NoU_fixb_v2
Base.show(io::IO, l::MTGRU_NoU_fixb_v2) =
print(io, "Multitask-GRU-fixb_v2(", l.N, ", ", typeof(l.G), ", bias vector (state only): ", typeof(l.b), "; no inputs)")


"""
    BatchedGRUCell_NoU(Wh, b, h)
Multi-task GRU Cell (which takes no input).
`Wh`, `b`, `h` are the (concatenated) transformation matrices,
offsets, and initial hidden state respectively.

Each time the cell is called (no arguments) it will perform a one-
step evolution of the hidden state and return the current value.
The cell is implemented in batch-mode, and the final dimension of
each quantity is the batch index.
"""
mutable struct BatchedGRUCell_NoU{A,W}
    Wh::A
    b::W
    h::W
end


function (m::BatchedGRUCell_NoU)(h, x=nothing)
  b, o = m.b, size(h, 1)
  gh = batch_matvec(m.Wh, h)
  r = σ.(gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

Flux.@treelike BatchedGRUCell_NoU
Base.show(io::IO, l::BatchedGRUCell_NoU) =
  print(io, "Batched GRU Cell(dim=", size(l.Wh,2), ", batch=", size(l.Wh,3), ", no inputs)")
Flux.hidden(m::BatchedGRUCell_NoU) = m.h


################################################################################
##                                                                            ##
##           Define *THE* multi-task model used in our experiments            ##
##                                                                            ##
################################################################################


struct MTSeq_CNN end
struct MTSeq_Single end
struct MTSeq_Double end

"""
    MTSeqModel_Pred(enc_tform, init_enc, init_post, mt_enc, mt_post,
        x0decoder, mtbias_deocder, gen_model, d_y, mtbias_only)

A sequence-to-sequence model 𝒴→𝒴 which imbibes `T_enc` steps of the sequence,
and predicts `T_steps` (including the original `T_enc`), but unlike the
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
struct MTSeqModel_Pred{U,A,B,V,W,S,N,M}
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
    d::Int64
    mtbias_only::Bool
end

"""
    MTSeqModel_E3(enc_tform, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post,
        x0decoder, mtbias_deocder, gen_model, d_y, mtbias_only)

A sequence-to-sequence model 𝒴→𝒴 which imbibes `T_enc` steps of the sequence,
and predicts `T_steps` (including the original `T_enc`), but unlike the
`Seq2SeqModel`, the `MTSeqModel_E3` can *modulate* its prediction based on
hierarchical 'multi-task' variables. (The `E3` suffix refers to the number of
encoders: previously I had `E1`, `E2` and `E3` types, but this seems redundant.)

As for the `Seq2SeqModel`, the posterior over the initial state of the generator
is learned variationally via the `init_enc` RNN with the final state passed to
the `init_post` layer which generates the mean and variance of the latent
posterior. Since the posterior is typically lower dimensional than the state of
the generator network, the latent variable is expanded via the `x0decoder`
network into the state of the `generator`. The MT variable is encoded with a
full-length sequence encoding via the `mt_enc` network, and the final state is
passed to the `mt_post` layer to generate the posterior mean and variance. A
similar set of operations is performed for the more local 'chaos variable'
using the `chaos_enc` and `chaos_post` networks. The 'chaos variable' encodes
sequence variation due to inherent unpredictability of a system, perhaps due to
chaos. This allows different long term predictions from the same state. The MT
and chaos variables enter the generator network either:

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
    m = MTSeqModel_E3(...)
    m(y::Vector, yfull::Vector)
```
where `y` is a vector of slices of the sequence for the encoder, and `yfull` is
the entire sequence (inc the future) on which to infer the latent MT variable
`z`. For both `y` and `yfull`, `y[i]` is an `AbstractMatrix` of size
``d \\times n_{\\text{batch}}`.

Options include `T_enc`, `T_steps` (already discussed), `reset` (which performs
a state reset before execution; true by default), and `stoch`, whether to
randomly sample the latent state, or use the state mean (default true).
"""
struct MTSeqModel_E3{U,A,B,V,W,S,N,M}
    enc_tform::U
    init_enc::A
    init_post::B
    mt_enc::A
    mt_post::B
    chaos_enc::A
    chaos_post::B
    x0decoder::V
    mtbias_decoder::W
    generator::S
    dec_tform::N
    model_type::M
    d::Int64
    mtbias_only::Bool
end

# Define union type for convenience
MTSeqModels{U,A,B,V,W,S,N,M} = Union{MTSeqModel_Pred{U,A,B,V,W,S,N,M}, MTSeqModel_E3{U,A,B,V,W,S,N,M}}

# Model utils
Flux.@treelike MTSeqModel_Pred
Flux.@treelike MTSeqModel_E3
model_type(m::MTSeqModels{U,A,B,V,W,S,N,M}) where {U,A,B,V,W,S,N,M} = M

unpack(m::MTSeqModel_Pred) = Flux.children(m)[1:9] # unpack struct; exclude `d`, `mtbias_only`.
unpack(m::MTSeqModel_E3) = Flux.children(m)[1:11]
unpack_inference(m::MTSeqModel_Pred) = Flux.children(m)[1:5]
unpack_inference(m::MTSeqModel_E3) = Flux.children(m)[1:7]
unpack_generative(m::MTSeqModels) = (m.x0decoder, m.mtbias_decoder, m.generator, m.dec_tform)

modelutils.load!(m::MTSeqModels, fname::String) = load!(Flux.params(m), fname)

function Base.show(io::IO, l::MTSeqModel_Pred)
    d_x0, d_mt = size(l.init_post.Dense1.W, 1), size(l.mt_post.Dense1.W, 1)
    out_type = Dict(MTSeq_CNN=>"CNN", MTSeq_Single=>"Deterministic", MTSeq_Double=>"Probabilistic")[model_type(l)]
    mttype = l.mtbias_only ? "Bias-only" : "Full-MT Recurrent Cell (" * get_strtype_wo_params(l.generator) * ")"
    print(io, "MTSeqModel_Pred(", mttype, ", d_x0=", d_x0, ", d_mt=", d_mt, ", ", out_type, ") -- for prediction")
end

function Base.show(io::IO, l::MTSeqModel_E3)
    d_x0, d_mt, d_c = size(l.init_post.Dense1.W, 1), size(l.mt_post.Dense1.W, 1), size(l.chaos_post.Dense1.W, 1)
    out_type = Dict(MTSeq_CNN=>"CNN", MTSeq_Single=>"Deterministic", MTSeq_Double=>"Probabilistic")[model_type(l)]
    mttype = l.mtbias_only ? "Bias-only" : "Full-MT Recurrent Cell (" * get_strtype_wo_params(l.generator) * ")"
    print(io, "MTSeqModel_E3(", mttype, ", d_x0=", d_x0, ", d_mt=", d_mt, ", d_chaos=", d_c, ", ", out_type, ") -- for llh")
end


################################################################################
##                                                                            ##
##                                Forward Model                               ##
##                                                                            ##
################################################################################

function (m::MTSeqModel_Pred)(y::AbstractVector; T_steps=70, T_enc=10, stoch=true)
    x0, z = posterior_samples(m, y; T_enc=T_enc, stoch=stoch)[1]
    return m(x0, z; T_steps=T_steps)
end

function (m::MTSeqModel_E3)(y::AbstractVector, yfull::AbstractVector; T_steps=70, T_enc=10, stoch=true)
    x0, z, c = posterior_samples(m, y, yfull; T_enc=T_enc, stoch=stoch)[1]
    return m(x0, z, c; T_steps=T_steps)
end

# ------------------- Forward Model (inference section) ----------------------

function posterior_samples(m::MTSeqModel_Pred, y::AbstractVector, yfull::AbstractVector=[];
    T_enc=10, stoch=true, x0=nothing, z=nothing, c=nothing)

    enc_tform, init_enc, init_post, mt_enc, mt_post = unpack_inference(m)
    (x0 === nothing || z === nothing) && (enc_yslices = [enc_tform(yy) for yy in y])

    # RNN 1: Amortized inference for initial state, h0, of generator
    if x0 === nothing
        x0, μ_x0, σ_x0 = _posterior_sample(init_enc, init_post, enc_yslices, T_enc, stoch)
    else
        μ_x0, σ_x0 = nothing, nothing
    end

    # RNN 2: Amortized inference for z
    # Technically this should be conditioned on the sample x₀, nevertheless, the posterior
    # of x0 is usually very tight, and the `mt_enc` has access to the same information.
    if z === nothing
        z, μ_z, σ_z = _posterior_sample(mt_enc, mt_post, enc_yslices, T_enc, stoch)
    else
        μ_z, σ_z = nothing, nothing
    end

    return (x0,z), (μ_x0, μ_z), (σ_x0, σ_z)
end


function posterior_samples(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
    T_enc=10, stoch=true, x0=nothing, z=nothing, c=nothing)

    enc_tform, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post = unpack_inference(m)

    # RNN 1: Amortized inference for initial state, h0, of generator
    if x0 === nothing
        enc_yslices = [enc_tform(yy) for yy in y]
        x0, μ_x0, σ_x0 = _posterior_sample(init_enc, init_post, enc_yslices, T_enc, stoch)
    else
        μ_x0, σ_x0 = nothing, nothing
    end

    # RNN 2: Amortized inference for z
    if z === nothing
        enc_fullseqs = [enc_tform(yy) for yy in yfull]
        z, μ_z, σ_z = _posterior_sample(mt_enc, mt_post, enc_fullseqs, length(yfull), stoch)
    else
        μ_z, σ_z = nothing, nothing
    end

    # RNN 3: Encode the chaos
    # Want a lowdim rep. of departure from expected trajected.
    if c === nothing
        !(x0 === nothing) && (enc_yslices = [enc_tform(yy) for yy in y])
        if !(z === nothing) && size(x0, 2) == 1 && size(z,2) > 1
            gmove = Flux.has_cuarrays() && Tracker.data(z) isa Flux.CuArray ? gpu : identity
            x0z = vcat(x0 * gmove(ones(Float32, 1, size(z,2))), z)
        else
            x0z = vcat(x0, z)
        end
        c, μ_c, σ_c = _posterior_sample(chaos_enc, chaos_post, enc_yslices, length(y),
            stoch, x0z)  # note conditioning on (x0, z) here.
    else
        μ_c, σ_c = nothing, nothing
    end
    return (x0,z,c), (μ_x0, μ_z, μ_c), (σ_x0, σ_z, σ_c)
end


# ------------------- Forward Model (generative section) ----------------------
# Note in Julia 1.1.1 (where this is developed), it is not possible to add
# methods to abstract types; hence (m::MTSeqModels)(x0, z, ...) is not possible
# and so we have the following "private"(!) function which is dispatched to by
# both types ∈ MTSeqModels. This is fixed in future versions of julia (I think)
# by https://github.com/JuliaLang/julia/pull/31916.
function _forward_given_latent(m::MTSeqModels, x0::AbstractVecOrMat, z::AbstractVecOrMat; T_steps=70)
    x0decoder, mtbias_decoder, gen_model, dec_tform = unpack_generative(m)

    if m.mtbias_only
        gen_model.state = x0decoder(x0)
        mtbias = mtbias_decoder(z)   # if linear, this is just the identity ∵ GRUCell.Wi
        return [dec_tform(gen_model(mtbias)) for t in 1:T_steps]
    else
        # Get GRU models from samples from the MT GRU model
        posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
        # Set state (batch-wise) to x0 sample
        posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch
        # Run generative model
        return [dec_tform(posterior_grus()) for t in 1:T_steps]
    end
end


(m::MTSeqModel_Pred)(x0::AbstractVecOrMat, z::AbstractVecOrMat; T_steps=70) =
    _forward_given_latent(m, x0, z; T_steps=T_steps)

# Allow signature for `E3` model to call with separate z, c vars..
(m::MTSeqModel_E3)(x0::AbstractVecOrMat, z::AbstractVecOrMat,
    c::AbstractVecOrMat; T_steps=70) = _forward_given_latent(m, x0, vcat(z, c); T_steps=T_steps)


# -------------------  Utils for inference section ----------------------------

function _posterior_sample(enc, dec, input, T_max, stochastic=true, input_cat=nothing)
    Flux.reset!(enc)
    if input_cat === nothing
        for tt = 1:T_max; enc(input[tt,:,:]); end
    else
        for tt = 1:T_max; enc(vcat(input[tt,:,:], input_cat)); end
    end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

function _posterior_sample(enc, dec, input::Vector, T_max, stochastic=true, input_cat=nothing)
    Flux.reset!(enc)
    if input_cat === nothing
        for tt = 1:T_max; enc(input[tt]); end
    else
        d, nbatch = size(input[1])
        if nbatch == 1 && nbatch != size(input_cat,2)
            gmove = Flux.has_cuarrays() && input_cat isa Flux.CuArray ? gpu : identity
            input_expander = gmove(ones(Float32, 1, size(input_cat, 2)))
        else
            input_expander = 1
        end
        for tt = 1:T_max; enc(vcat(input[tt] * input_expander, input_cat)); end
    end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

################################################################################
##                                                                            ##
##                             Model "Constructor"                            ##
##                                                                            ##
################################################################################

function create_model(d_x, d_x0, d_y, d_enc_state, d_mt, d_chaos=0; encoder=:GRU,
    cnn=false, out_heads=1, d_hidden=d_x, mtbias_only=false, d_hidden_mt=32,
    mt_is_linear=true, decoder_fudge_layer=false, model_purpose=:llh, fixb=false,
    fixb_version=nothing)

    @assert !(out_heads > 1 && cnn) "cannot have multiple output heads and CNN."
    @argcheck out_heads in 1:2
    @argcheck model_purpose in [:llh, :pred]
    @assert !(fixb && mtbias_only) "cannot do mtbias AND fix 'b'"
    @assert !fixb || something(fixb_version, 1) ∈ [1,2] "fixb_version must be in [1,2]."

    is_pred_model = model_purpose == :pred
    if fixb && fixb_version === nothing
        @warn "fixed bias chosen, but `fixb_version` not specified. Defaulting to v1."
        fixb_version = 1
    end

    # ENCODER TRANSFORM FROM OBS SPACE => ENC SPACE
    if cnn
        N_FILT=32
        d_conv_result = 4*4*N_FILT
        tform_enc = Chain(
            Conv((3, 3), 2=>N_FILT, relu, stride=2, pad=1),        # out: 16 × 16 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 8 × 8 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 4 × 4 × 32 × nbatch
            x->reshape(x, d_conv_result, :),
            Dense(d_conv_result, d_hidden, swish)
        )
        d_rnn_in = d_hidden
        model_type = MTSeq_CNN()
    else
        tform_enc = identity
        d_rnn_in = d_y
        model_type = out_heads == 1 ? MTSeq_Single() : MTSeq_Double()
    end

    # 2/3× ENCODERS for inference of
    # (a) initial state (init_enc)
    # (b) sequence level MT variable (mt_enc)
    # (c) local variability due to sensitivity/chaos (mt_chaos)
    @argcheck encoder in [:LSTM, :GRU, :Bidirectional]
    rnn_constructor = Dict(:LSTM=>LSTM, :GRU=>GRU, :Bidirectional=>BRNNenc)[encoder]
    init_enc = rnn_constructor(d_rnn_in, d_enc_state)             # (a)
    mt_enc = rnn_constructor(d_rnn_in, d_enc_state)               # (b)
    !(is_pred_model) && (chaos_enc = rnn_constructor(d_rnn_in+d_mt+d_x0, d_enc_state))  # (c)
    (encoder == :Bidirectional) && (d_enc_state *= 2)


    # POSTERIOR OVER LVM CORR. TO (a), (b), (c)
    init_post = MultiDense(Dense(d_enc_state, d_x0, identity), Dense(d_enc_state, d_x0, σ))
    init_post.Dense2.b.data .= -2   # initialize posteriors to be low variance
    mt_post = MultiDense(Dense(d_enc_state, d_mt, identity), Dense(d_enc_state, d_mt, σ))
    mt_post.Dense2.b.data .= -2
    if !is_pred_model
        chaos_post = MultiDense(Dense(d_enc_state, d_chaos, identity), Dense(d_enc_state, d_chaos, σ))
        chaos_post.Dense2.b.data .= -2
    end

    # decode from LV (a) --> size of generative hidden state
    x0decoder = Dense(d_x0, d_x, identity)

    # -------------- GENERATIVE MODEL -----------------
    ###################################################

    if !mtbias_only
        d_out = !fixb ? 3*d_x*(d_x+1) : (fixb_version == 1 ? 3*d_x^2 : 3*d_x^2 + 2*d_x)
        par_gen_net = mt_is_linear ? mlp(d_mt+d_chaos, d_out) : mlp(d_mt+d_chaos, d_hidden_mt, d_out; activation=tanh)
        if !fixb
            gen_rnn = MTGRU_NoU(d_x, par_gen_net)
        else
            if fixb_version == 1
                MTGRU_b = Flux.param(randn(Float32, d_x*3))
                gen_rnn = MTGRU_NoU_fixb(d_x, par_gen_net, MTGRU_b)
            elseif fixb_version == 2
                MTGRU_b = Flux.param(randn(Float32, d_x))
                gen_rnn = MTGRU_NoU_fixb_v2(d_x, par_gen_net, MTGRU_b)
            else
                error("fixb_version: Unreachable error.")
            end
        end
        mtbias_decoder = identity
    else
        mtbias_decoder = mt_is_linear ? identity : Dense(d_mt+d_chaos, d_hidden_mt, swish)
        _d_in = mt_is_linear ? d_mt+d_chaos : d_hidden_mt
        gen_rnn = GRU(_d_in, d_x)
    end

    if decoder_fudge_layer
        # accidental additional layer in decoder for full MT-model.
        # (This is really low capacity. 64->64 unit relu; perhaps as likely to reduce as to improve performance.)
        fudge_layer = Dense(d_x, d_x, relu)
    else
        fudge_layer = identity
    end

    if out_heads == 1 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), Dense(d_hidden, d_y, identity))
    elseif out_heads == 2 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), MultiDense(Dense(d_hidden, d_y, identity), Dense(d_hidden, d_y, identity)))
    else    # cnn
        decoder = Chain(
            fudge_layer,
            Dense(d_x, d_conv_result, identity),
            x->reshape(x, 4, 4, N_FILT, :),                               # out: 4 × 4 × 32 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2),         # out: 9 × 9 × 16 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2, pad=1),  # out: 17 × 17 × 8 × nbatch
            ConvTranspose((3,3), N_FILT=>2, identity, stride=2, pad=1),   # out: 33 × 33 × 1 × nbatch
            x->x[1:D_IM, 1:D_IM, :, :]   # out: 32 × 32 × 2 × nbatch
        )
    end

    if is_pred_model
        m = MTSeqModel_Pred(tform_enc, init_enc, init_post, mt_enc, mt_post,
            x0decoder, mtbias_decoder, gen_rnn, decoder, model_type, d_y, mtbias_only)
    else
        m = MTSeqModel_E3(tform_enc, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post,
            x0decoder, mtbias_decoder, gen_rnn, decoder, model_type, d_y, mtbias_only)
    end

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

################################################################################
##                                                                            ##
##                        Objective(s) / Loss Function(s)                     ##
##                                                                            ##
################################################################################

function online_inference_BCE(m::MTSeqModels, x0::AbstractVecOrMat, z::AbstractVecOrMat,
    y::AbstractVector; T_steps=length(y))

    x0decoder, mtbias_decoder, gen_model, dec_tform = unpack_generative(m)

    # Generative model
    # --------------------------------------------------
    if m.mtbias_only
        gen_model.state = x0decoder(x0)
        mtbias = mtbias_decoder(z)   # if linear, this is just the identity ∵ GRUCell.Wi

        nllh = map(1:T_steps) do t
            _nllh_bernoulli(dec_tform(gen_model(mtbias)), y[t])  # y broadcasts over the batch implicitly
        end
    else
        posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
        posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch

        nllh = map(1:T_steps) do t
            _nllh_bernoulli(dec_tform(posterior_grus()), y[t])  # y broadcasts over the batch implicitly
        end
    end
    return reduce(vcat, nllh)
end

# Allow signature for `E3` model to call with separate z, c vars..
online_inference_BCE(m::MTSeqModel_E3, x0::AbstractVecOrMat, z::AbstractVecOrMat, c::AbstractVecOrMat,
    y::AbstractVector; T_steps=length(y)) = online_inference_BCE(m, x0, vcat(z,c), y; T_steps=T_steps)



function online_inference_single_BCE(m::MTSeqModels, h0::AbstractVecOrMat, z::AbstractVecOrMat,
    y::AbstractArray)

    mtbias_decoder, gen_model, dec_tform = unpack(m)[9:11]

    # Generative model
    # --------------------------------------------------
    if m.mtbias_only
        gen_model.state = h0
        mtbias = mtbias_decoder(z)
        h_new = gen_model(mtbias)
        nllh = _nllh_bernoulli(dec_tform(h_new), y)
    else
        posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
        posterior_grus.state = h0  # 2×linear d_x0 × nbatch → d_x × nbatch
        h_new = posterior_grus()
        nllh = _nllh_bernoulli(dec_tform(h_new), y)
    end
    return vec(nllh), h_new
end

# Allow signature for `E3` model to call with separate z, c vars..
online_inference_single_BCE(m::MTSeqModel_E3, h0::AbstractVecOrMat, z::AbstractVecOrMat,
    c::AbstractVecOrMat, y::AbstractArray) = online_inference_single_BCE(m, h0, vcat(z,c), y)


function _nllh_bernoulli(ŷ::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractArray{M,N} where {M,N}
    @assert N > 1 "Avoiding infinite recursion. " *
    "Current implementation does not permit nllh of bernoulli vector of vectors. Try unsqueezing."
    return [_nllh_bernoulli(ŷŷ, yy) for (ŷŷ, yy) in zip(ŷ, y)]
end

_nllh_bernoulli(ŷ::AbstractArray, y::AbstractArray) =
    let n=size(ŷ)[end]; res=Flux.logitbinarycrossentropy.(ŷ, y); sum(reshape(res, :, n), dims=1); end

_nllh_gaussian(ŷ_lσ::AbstractVector, y::AbstractVector) =
    [((ŷŷ, ll) = ŷl; δ=yy-ŷŷ; sum(δ.*δ./(exp.(2*ll)))/2+sum(ll)/2) for (ŷl, yy) in zip(ŷ_lσ, y)]

_nllh_gaussian_constvar(ŷ::AbstractVector, y::AbstractVector, logstd::Number) =
    [(let n=size(ŷ)[end]; δ=yy-ŷŷ; res=sum(δ.*δ./(exp.(2*logstd)))/2+sum(logstd)/2;
    sum(reshape(res, :, n), dims=1); end) for (ŷŷ, yy) in zip(ŷ, y)]

function nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector=[];
    T_steps=70, T_enc=10, stoch=true) where {U,A,B,V,W,S,N,M <: MTSeq_CNN}
    x0_z_c = posterior_samples(m, y, yfull; T_enc=T_enc, stoch=stoch)[1]
    online_inference_BCE(m::MTSeqModels, x0_z_c..., y; T_steps=T_steps)
end

nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) where {U,A,B,V,W,S,N,M <: MTSeq_Single} =
        vcat(_nllh_gaussian_constvar(m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch), y, logstd)...)

nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true) where {U,A,B,V,W,S,N,M <: MTSeq_Double} =
        vcat(_nllh_gaussian(m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch), y)...)

nllh(::Type{MTSeq_CNN}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_bernoulli(ŷ, y)
nllh(::Type{MTSeq_Single}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian_constvar(ŷ, y, logstd)
nllh(::Type{MTSeq_Double}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian(ŷ, y)


StatsBase.loglikelihood(m::MTSeqModels, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) = -nllh(m, y, yfull, T_steps=T_steps,
        T_enc=T_enc, stoch=stoch, logstd=logstd)


function _gauss_kl_to_N01(μ::AbstractVecOrMat{T}, σ::AbstractVecOrMat{T}) where T
    T(0.5) * sum(1 .+ 2*log.(σ.*σ) - μ.*μ - σ.*σ)  # x.^2 -> x.*x due to old version of CuArrays / GPU (culiteral pow issue)
end
_gauss_kl_to_N01_deterministic(μ::AbstractVecOrMat{T}) where T  = T(0.5) * sum(1 .- μ.*μ)

function _kl_penalty_stoch(β::Vector{T}, μs::Vector{T}, σs::Vector{T}) where T <: Float32
    @argcheck length(β) == length(μs) == length(σs)
    kl = zero(T)
    for j in 1:length(β)
        kl += β[j] * _gauss_kl_to_N01(μs[j], σs[j])
    end
    return kl
end

function _kl_penalty_deterministic(β::Vector{T}, μs::Vector{T}) where T <: Float32
    @argcheck length(β) == length(μs)
    kl = zero(T)
    for j in 1:length(β)
        kl += β[j] * _gauss_kl_to_N01_deterministic(μs[j])
    end
    return kl
end


function elbo(m::MTSeqModels, y::AbstractVector, yfull::AbstractVector=[]; T_steps=70,
    T_enc=10, stoch=true, kl_coeff=1f0, βkl=ones(Float32, 3))

    smps, μs, σs = posterior_samples(m, y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)
    model_out = m(x0, vcat(z, c); T_steps=T_steps)
    recon = -nllh(M, model_out, y)

    # ⇒ Initially no KL (Hard EM), and then an annealing sched. cf. Bowman et al. etc?
    kl = stoch ? _kl_penalty_stoch(β, μs, σs) : _kl_penalty_deterministic(β, μs)
    kl = kl * kl_coeff

    return - (recon + kl)
end




end
