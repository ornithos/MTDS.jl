"""
    MTPD_variational(mt_enc, mt_post, hphi, emission_coeffs, logstd, d, flag_mt_x0)

A deterministic MT Pharmacodynamic model (subtype of `MTDSModel`) following the equations:

`` [β₁, β₂, β₃, θ₁, ..., θd, α] = h_ϕ(z)``

`` x_t = diag(β₁) x_{t-1} + β₂ u_{t} ``

`` y_t = g_θ(x_t + β₃) + α + s .* ϵ_t``

for ``ϵ_t \\sim 𝑵(0, I)`` and ``x_0 = 0``. This is a discrete time version of the usual 
pharmacodynamic model using an effect site per dimension, but with a more general sigmoidal emission
function than simply the Hill Equation (and also more numerically stable to boot). Notable
characteristics:

1. Latent system is deterministic.
2. The latent state ``x_0`` is constant and equal to zero.
3. The emission covariance is diagonal.

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
takes a ``k × n_b`` matrix ``z`` and returns the LDS parameter vector θ. The emission
function is a flexible sigmoidal function, a kind of 1-hidden layer MLP with fixed
parameters (``emission\\_coeffs``) for the first layer. ``d`` is both the dimension of ``x``
and ``y``, since there is one latent dimension per output dimension.
``emission\\_coeffs`` is an L × 2 matrix (columns give a, b respectively, see paper),
and finally  `flag_mt_x0` is a placeholder for whether the ``x_0`` is modulated via 
``z`` (not yet implemented).

## Methods
As per all `MTDSModel` objects, the following interface is available:

- `encode(m::MTPD_variational, Y::AbstractArray{T}, U::AbstractArray{T})`,
  which encodes the sequences `Y` and `U` into a sample and variational
  posterior parameters of ``z``.
- `forward(m::MTPD_variational, z::AbstractVecOrMat{T}, U::AbstractArray{T})`,
  the forward model, which constructs the LDS for the latent variable `z` and
  calculates the output mean.
- `reconstruct(m::MTDSModel, Y, U)` which performs both `encode` and `forward`
  operations, effectively reconstructing `Y` through the bottleneck of the
  inference of ``z``.

### Constructor
See `create_mtpd` for a more helpful constructor.

"""
struct MTPD_variational{U,V,W,S,T} <: MTDSModel
    mt_enc::U
    mt_post::V
    hphi::W
    emission_coeffs::S
    logstd::T
    d::Int
    flag_mt_x0::Bool
end

Flux.@treelike MTPD_variational

function Base.show(io::IO, l::MTPD_variational)
    d_x0, d_mt = l.d, size(l.mt_post.Dense1.W, 1)
    d_uy = format("effect_sites={:d}", l.d)
    h0_type = l.flag_mt_x0 ? "'MT'" : "'non-MT'"
    enc_type = l.mt_enc isa LookupTable ? "'Standard VI'" : "'Amortized VI'"
    print(io, "MTPD_variational(", d_uy, ", d_mt=", d_mt, 
    ", x0_type=", h0_type,  ", enc_type=", enc_type,")")
end

is_amortized(m::MTPD_variational{U,V,W,S,T}) where {U <: LookupTable,V,W,S,T} = false
getL(m::MTPD_variational) = size(m.emission_coeffs, 1)

function f_theta_unpack(θ::AbstractVector, m) 
    dx = dy = m.d
    L = getL(m)
    βs = σ.(θ[1:dx]), softplus.(θ[(dx+1):2dx]), θ[(2dx+1):3dx]
    ηs = [-softplus.(θ[(3dx+(i-1)*L+1):(3dx+i*L)]) for i in 1:dy]
    α = θ[(3dx+L*dy+1):(3dx+L*dy+dy)]
    return βs, ηs, α
end
function f_theta_unpack(θ::AbstractMatrix, m) 
    dx = dy = m.d
    L = getL(m)
    βs = σ.(θ[1:dx, :]), softplus.(θ[(dx+1):2dx, :]), θ[(2dx+1):3dx, :]
    ηs = [-softplus.(θ[(3dx+(i-1)*L+1):(3dx+i*L), :]) for i in 1:dy]
    α = θ[(3dx+L*dy+1):(3dx+L*dy+dy), :]
    return βs, ηs, α
end

forward(m::MTPD_variational, z, U; T_steps=size(U, 2)) = forward(m, nothing, z, U; T_steps=T_steps)
function forward(m::MTPD_variational, x0, z::AbstractVecOrMat{T}, U::AbstractArray{T}; T_steps=size(U, 2)) where T <: AbstractFloat
    # note that T_steps is currently ignored (this is no great issue).
    !(x0 === nothing) && @warn "MTPD_variational currently doesn't accept x0 -- fixed at 0. Ignoring..."
    @argcheck size(z, 2) == size(U, 3)
    θ = m.hphi(z)
    # currently don't modulate the initial state.
    return _mtpd_model_forward_parvector(m, θ, U)
end

    function _mtpd_model_forward_parvector(m::MTPD_variational, θ::AbstractVector{T},
            U::AbstractMatrix{T}, h0::AbstractVector{T}=zeros(T, m.d)) where T <: AbstractFloat
        n_u, tT = size(U)
        βs, ηs, α = f_theta_unpack(θ, m)

        lds = Flux.Recur(LDSCell_diag_u(βs[1], reshape(βs[2], m.d, n_u), h0))
        lds_state = [lds(U[:,t]) for t in 1:tT]
        X = hcat(lds_state...) .+ βs[3]
        a, b = m.emission_coeffs[:,1], m.emission_coeffs[:,2]
        # h = kron(X, a) - repeat(b, m.n_y)
        yhat = map(1:m.d) do j
            unsqueeze(ηs[j],1) * σ.(a .* X[j:j,:] .- b) .+ α[j]
        end
        return vcat(yhat...)
    end

    function _mtpd_model_forward_parvector(m::MTPD_variational, θ::AbstractMatrix{T},
            U::AbstractArray{T,3}, h0::AbstractVecOrMat{T}=zeros(T, m.d)) where T <: AbstractFloat
        n_u, tT, n_b = size(U)
        h0 = ensure_matrix(h0)
        βs, ηs, α = f_theta_unpack(θ, m)
        
        lds = Flux.Recur(LDSCell_diag_batch_u(βs[1], 
                                              reshape(βs[2], m.d, n_u, n_b), 
                                              h0))
        lds_state = [unsqueeze(lds(U[:,t,:]),2) for t in 1:tT]
        X = hcat(lds_state...) .+ unsqueeze(βs[3], 2)
        a, b = m.emission_coeffs[:,1], m.emission_coeffs[:,2]

        yhat = map(1:m.d) do j
            batch_matmul(unsqueeze(ηs[j],1), σ.(a .* X[j:j,:,:] .- b)) .+ unsqueeze(α[j:j,:], 2)
        end
        return vcat(yhat...)
    end


encode(m::MTPD_variational, Y::AbstractArray{T}, U::AbstractArray{T}; T_steps=size(Y, 2), 
    stochastic=true) where T = encode(m, vcat(U, Y); T_steps=T_steps, stochastic=stochastic)

encode(m::MTPD_variational, Y::MaskedArray{T}, U::AbstractArray{T}; T_steps=size(Y, 2), 
    stochastic=true) where T = 
    encode(m, vcat(stacked(Y), U); T_steps=T_steps, stochastic=stochastic)

function encode(m::MTPD_variational, data_enc::AbstractArray; T_steps=size(data_enc, 2), stochastic=true)
    m.flag_mt_x0 && error("Unable to modulate x0 currently. Change encoder code, and `_mtlds_model_forward`.")
    smpz, μ_z, σ_z = posterior_sample(m.mt_enc, m.mt_post, data_enc, size(data_enc, 2), stochastic)
    return (smpz, μ_z, σ_z), (nothing, nothing, nothing)   # second tuple is for if x0 has separate encoder.
end


#  Cannot define function on an abstract type, so we have to def these inside each subtype.
(m::MTPD_variational)(z::AbstractArray{T,2}, U::AbstractArray{T,3}) where T = forward(m, z, U)
(m::MTPD_variational)(Y::AbstractArray{T,3}, U::AbstractArray{T,3}) where T = reconstruct(m, Y, U)


"""
    create_mtpd(d, d_in, d_enc_state, d_mt; encoder=:LSTM,
    d_hidden=64, hphi=:MLP, emission_pars=:default)

Create `MTPD_variational` with ``d`` effect sites, and hence the dimension of both
the latent system and observations is ``d``. ``d_in`` specifies the dimension of 
the input vectors, `d_enc_state` the latent dim of the RNN encoder, `d_mt` of the 
latent hierarchical variable ``z``.

### Optional arguments:

- `encoder` is the encoding RNN for the variational posterior. Can be in {`:RNN`
, `:GRU`, `:LSTM`, `:LookupTable`}, with the latter providing standard VI.
- `d_hidden`: number of hidden units in the one-hidden-layer MLP used for ``h_ϕ``.
- `spherical_var`: whether the emission covariance is spherical (and hence has a
  single parameter) (default: false), or is an axis-aligned ellipsoid with `d_y`
  parameters.
- `hphi`: default `:MLP`, can be `:Linear`. This determines whether the prior over
  θ is linear or nonlinear. If == `:MLP`, `d_hidden` specifies layers, o.w. it's ignored.
"""
function create_mtpd(d, d_in, d_enc_state, d_mt; encoder=:LSTM, d_hidden=64, 
    spherical_var=false, hphi=:MLP, emission_coeffs=:default, indep_biases::Bool=false)

    # Encoding z from the U, Y sequences
    if encoder == :LSTM
        init_enc = LSTM(d + d_in, d_enc_state)
    elseif encoder == :GRU
        init_enc = GRU(d + d_in, d_enc_state)
    elseif encoder == :RNN
        init_enc = RNN(d + d_in, d_enc_state)
    elseif encoder == :LookupTable
        init_enc = LookupTable(d_mt, 0.01f0, -2.0f0)  # 2/3: initial posterior noise for mu/logstd
    else
        error("encoder must be specified as :LSTM, :GRU, :RNN, :LookupTable")
    end

    # Latent z *encoding* --> posterior E(z|Y,U) and V(z|Y,U)
    enc_post = MultiDense(Dense(d_enc_state, d_mt, identity), Dense(d_enc_state, d_mt, σ))
    enc_post.Dense2.b.data .= -2   # initialize posteriors to be low variance

    # Emission function first layer
    nl_ab =  [0.0   0.008  0.0259    0.1248  0.6012   0.1257  0.0261  0.008;
              0.0  -500.0  -150.0  -18.1187  0.4374  18.8683   150.0  500.0]' |> f32 |> Matrix
    emission_coeffs = (emission_coeffs == :default) ? nl_ab : emission_coeffs
    @assert emission_coeffs isa AbstractMatrix
    @assert size(emission_coeffs, 2) == 2 "Expecting an L×2 matrix for emission_coeffs or :default."
    L = size(emission_coeffs, 1)   # number of basis functions for emission fn.

    # Multi-task network z --> θ
    n_dec = indep_biases ? 3*d + d*L : 3*d + d*L + d
    d_mt_h = indep_biases ? d_mt - d : d_mt
    @assert d_mt_h >= 0 "Independent biases selected, but d_mt is too small to permit this."
    # -- Major mapping z --> θ
    if hphi == :MLP
        hphi = mlp(d_mt_h, d_hidden..., Int(n_dec))
    elseif hphi == :Linear
        hphi = Dense(d_mt_h, Int(n_dec))
    else
        error("hphi must be specified as :MLP, :Linear")
    end
    # -- z --> biases if independent
    if indep_biases
        k_not_bias, k_bias = 1:d_mt_h, (d_mt_h+1):d_mt
        hphi = PartialSplit(k_not_bias, k_bias, 1, hphi, identity)
    end

    # Emission noise standard deviation: either same per channel (spherical) or not (diagonal).
    emission_std = spherical_var ? Tracker.param([0.0f0]) : Tracker.param(zeros(Float32, d))
    
    MTPD_variational(init_enc, enc_post, hphi, emission_coeffs, emission_std, d, false)
end


