"""
    MTLDS_variational(mt_enc, mt_post, hphi, emission, d, flag_mt_emission,
    flag_mt_x0)

A deterministic MT-LDS model (subtype of `MTDSModel`) following the equations:

`` x_t = A_z x_{t-1} + B_z u_{t} ``

`` y_t = C_z x_y + d_z + s .* ϵ_t``

for ``ϵ_t \\sim 𝑵(0, I)`` and ``x_0 = 0``. Note the following restrictions vs a
full LDS:

1. Latent system is deterministic.
2. The inputs cannot affect the emission process (i.e. no ``D_z u_t`` term).
3. The latent state ``x_0`` is constant and equal to zero.
4. The emission covariance is diagonal.

The limitations exist because this was all I needed for my experiments, but it
is not difficult to extend the model to relax each of these, especially (2-4).

## Struct description
The model is variational in ``z``, and contains an encoder (`mt_enc`) which is
an RNN which encodes the sequences `Y` and `U` (which should be of size ``d × T ×
n_b``, where ``d`` is the size of each vector, ``T`` is the sequence length, and
``n_b`` is the batch size) into a single vector. One can instead perform standard
(non-amortized VI by using a LookupTable here. The `mt_post` network is a
network which returns a tuple of (mean, (diagonal) sqrt variance) defining a
variational Gaussian posterior. The method `encode` will return two tuples, the
first of which is a sample from this posterior (using reparameterization),
elements 2 and 3 are the variational posterior mean and stdev components. (The
second tuple is empty -- this is reserved for ``x_0`` if it has a separate
encoder.) Note that these variational networks need not be used, e.g. if one is
using an alternative MCO approach.

The function `hphi` is a network ``h_ϕ: \\text{dom}(z) → \\text{dom}(θ)`` which
takes a ``k × n_b`` matrix ``z`` and returns the LDS parameter vector θ. The
`emission` is either a pre-specified network which is optimized without any MT
modulation, or an Int which specifies the emission dimension, for which a MT
affine emission is constructed (``C_z + d_z``). Finally ``flag_mt_emission`` is
a boolean which is true in the latter case (MT affine emission) and false
otherwise, and `flag_mt_x0` is a placeholder for whether the ``x_0`` is
modulated via ``z``.

## Methods
As per all `MTDSModel` objects, the following interface is available:

- `encode(m::MTLDS_variational, Y::AbstractArray{T}, U::AbstractArray{T})`,
  which encodes the sequences `Y` and `U` into a sample and variational
  posterior parameters of ``z``.
- `forward(m::MTLDS_variational, z::AbstractVecOrMat{T}, U::AbstractArray{T})`,
  the forward model, which constructs the LDS for the latent variable `z` and
  calculates the output mean.
- `reconstruct(m::MTDSModel, Y, U)` which performs both `encode` and `forward`
  operations, effectively reconstructing `Y` through the bottleneck of the
  inference of ``z``.

### Constructor
See `create_mtlds` for a more helpful constructor.

"""
struct MTLDS_variational{U,V,W,S,T} <: MTDSModel
    mt_enc::U
    mt_post::V
    hphi::W
    emission::S
    logstd::T
    d::Int
    flag_mt_emission::Bool
    flag_mt_x0::Bool
end

Flux.@treelike MTLDS_variational
# common superclass properties
is_amortized(m::MTLDS_variational) = true
is_amortized(m::MTLDS_variational{U,V,W,S,T}) where {U <: LookupTable, V,W,S,T} = false
has_x0_encoding(m::MTLDS_variational) = false

function Base.show(io::IO, l::MTLDS_variational)
    d_x0, d_mt = l.d, size(l.mt_post.Dense1.W, 1)
    d_y = get_final_layer_dim(l.emission)
    if (d_y === nothing) || l.mt_enc isa LookupTable
        d_uy = d_y === nothing ? "" : format(", in=?, out={:d}", d_y)
    else
        d_u = size(l.mt_enc.cell.Wi, 2) - something(d_y, 0)
        d_uy = format(", in={:d}, out={:d}", d_u, d_y)
    end
    emission_type = l.flag_mt_emission ? "" : ", *non-MT emission function*"
    h0_type = l.flag_mt_x0 ? "'MT'" : "'non-MT'"
    enc_type = l.mt_enc isa LookupTable ? "'Standard VI'" : "'Amortized VI'"
    print(io, "MTLDS_variational(state=", d_x0, d_uy, ", d_mt=", d_mt, ", ", emission_type,
    "x0_type=", h0_type,  ", enc_type=", enc_type,")")
end

is_amortized(m::MTLDS_variational{U,V,W,S,T}) where {U <: LookupTable, V, W, S, T} = false


forward(m::MTLDS_variational, z, U; T_steps=size(U, 2)) = forward(m, nothing, z, U; T_steps=T_steps)
function forward(m::MTLDS_variational, x0, z::AbstractVecOrMat{T}, U::AbstractArray{T}; T_steps=size(Y,2)) where T <: AbstractFloat
    # note that T_steps is currently ignored (this is no great issue).
    !(x0 === nothing) && @warn "MTLDS_variational currently doesn't accept x0 -- fixed at 0. Ignoring..."
    @argcheck size(z,2) == size(U,3)
    θ = m.hphi(z)
    # currently don't modulate the initial state.
    return _mtlds_model_forward_parvector(m, θ, U)
end

    function _mtlds_model_forward_parvector(m::MTLDS_variational, θ::AbstractVector{T},
            U::AbstractMatrix{T}, h0::AbstractVector{T}=zeros(T, m.d)) where T <: AbstractFloat
        n_u, tT = size(U)
        _A, _B = A(θ, m.d), B(θ, m.d, n_u)
        lds = Flux.Recur(LDSCell_simple_u(_A, _B, h0))
        lds_state = [lds(U[:,t]) for t in 1:tT]
        X = hcat(lds_state...)
        if m.flag_mt_emission
            _C, d = C(θ, m.d, n_u, size(Y, 1)), dmt(θ, m.d, n_u, size(Y, 1))
            return _C * X .+ d
        end
        return m.emission(X)
    end

    function _mtlds_model_forward_parvector(m::MTLDS_variational, θ::AbstractMatrix{T},
            U::AbstractArray{T}, h0::AbstractVecOrMat{T}=zeros(T, m.d)) where T <: AbstractFloat
        n_u, tT, n_b = size(U)
        h0 = ensure_matrix(h0)
        _Bs = B(θ, m.d, n_u)
        _As = cat([A(θ[:,b], m.d) for b in 1:n_b]..., dims=3)
        lds = Flux.Recur(LDSCell_batch_u(_As, _Bs, h0))

        if m.flag_mt_emission
            _C, d = C(θ, m.d, n_u, m.emission), dmt(θ, m.d, n_u, m.emission)
            lds_state = [unsqueeze(lds(U[:,t,:]), 2) for t in 1:tT]
            X = cat(lds_state..., dims=2)
            return batch_matmul(_C, X) .+ unsqueeze(d, 2)
        else
            Ys = [unsqueeze(m.emission(lds(U[:,t,:])), 2) for t in 1:tT]
            return cat(Ys..., dims=2)
        end
    end


encode(m::MTLDS_variational, Y::AbstractArray{T}, U::AbstractArray{T}; T_enc=size(Y, 2), 
    stochastic=true) where T = encode(m, vcat(Y, U); T_enc=T_enc, stochastic=stochastic)

encode(m::MTLDS_variational, Y::MaskedArray{T}, U::AbstractArray{T}; T_enc=size(Y, 2), 
    stochastic=true) where T = 
    encode(m, vcat(stacked(Y), U); T_enc=T_enc, stochastic=stochastic)


function encode(m::MTLDS_variational, data_enc::AbstractArray; T_enc=size(data_enc, 2), stochastic=true)
    m.flag_mt_x0 && error("Unable to modulate x0 currently. Change encoder code, and `_mtlds_model_forward`.")
    smpz, μ_z, σ_z = posterior_sample(m.mt_enc, m.mt_post, data_enc, T_enc, stochastic)
    return (smpz, μ_z, σ_z), (nothing, nothing, nothing)   # second tuple is for if x0 has separate encoder.
end

# Cannot define function on an abstract type, so we have to def these inside each subtype.
(m::MTLDS_variational)(z::AbstractArray{T,2}, U::AbstractArray{T,3}) where T = forward(m, z, U)
(m::MTLDS_variational)(Y::AbstractArray{T,3}, U::AbstractArray{T,3}) where T = reconstruct(m, Y, U)


"""
    create_mtlds(d_x, d_in, d_y, d_enc_state, d_mt; encoder=:LSTM,
    emission=:linear, d_hidden=64)

Create `MTLDS_variational` with the dimensions `d_x` of the latent system,
`d_in` of the input vectors, `d_y` of the output vectors, `d_enc_state` of
the RNN encoder, `d_mt` of the latent hierarchical variable ``z``.

### Optional arguments:

- `encoder` is the encoding RNN for the variational posterior. Can be in {`:RNN`
, `:GRU`, `:LSTM`, `:LookupTable`}, with the latter providing standard VI.
- `emission` specifies the system emission equation, which by
  default is `:linear`, which constructs a MT affine output. Otherwise, any
  callable function that maps a state vector (matrix) → outputs can be given, but
  it will not use any MT adaptation.
- `d_hidden`: number of hidden units in the one-hidden-layer MLP used for ``h_ϕ``.
- `spherical_var`: whether the emission covariance is spherical (and hence has a
  single parameter) (default: true), or is an axis-aligned ellipsoid with `n_y`
  parameters.
- `hphi`: default `:MLP`, can be `:Linear`. This determines whether the prior over
  θ is linear or nonlinear. If == `:MLP`, `d_hidden` specifies layers, o.w. it's ignored.
"""
function create_mtlds(d_x, d_in, d_y, d_enc_state, d_mt; encoder=:LSTM, emission=:linear,
    d_hidden=64, spherical_var=false, hphi=:MLP)

    if encoder == :LSTM
        init_enc = LSTM(d_y+d_in, d_enc_state)
    elseif encoder == :GRU
        init_enc = GRU(d_y+d_in, d_enc_state)
    elseif encoder == :RNN
        init_enc = RNN(d_y+d_in, d_enc_state)
    elseif encoder == :LookupTable
        init_enc = LookupTable(d_mt, 0.01f0, -1f0)  # 2/3: initial posterior noise for mu/logstd
    else
        error("encoder must be specified as :LSTM, :GRU, :RNN, :LookupTable")
    end

    enc_post = MultiDense(Dense(d_enc_state, d_mt, identity), Dense(d_enc_state, d_mt, σ))
    enc_post.Dense2.b.data .= -2   # initialize posteriors to be low variance

    n_dec = d_x*(d_x+1)/2 + d_x*d_in*2
    if emission == :linear
        emission = d_y  ### C(θ), d(θ)
        n_dec += d_y * d_x + d_y
        flag_mt_emission = true
    else
        (supertype(typeof(emission)) === Any) || error("unexpected emission object: expecting fn or :linear")
        flag_mt_emission = false
    end

    if hphi == :MLP
        hphi = mlp(d_mt, d_hidden..., Int(n_dec))
    elseif hphi == :Linear
        hphi = Dense(d_mt, Int(n_dec))
    else
        error("hphi must be specified as :MLP, :Linear")
    end

    emission_std = spherical_var ? Tracker.param([0.0f0]) : Tracker.param(zeros(Float32, d_y))
    MTLDS_variational(init_enc, enc_post, hphi, emission, emission_std, d_x, flag_mt_emission, false)
end
