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

# get_final_layer_dim is purely used for pretty printing (Base.show)
get_final_layer_dim(l) = nothing
get_final_layer_dim(l::Int) = l
get_final_layer_dim(l::Chain) = get_final_layer_dim(l.layers[end])
get_final_layer_dim(l::Dense) = length(l.b)
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


function encode(m::MTLDS_variational, Y::AbstractArray{T}, U::AbstractArray{T}; T_steps=size(Y, 2), stochastic=true) where T
    m.flag_mt_x0 && error("Unable to modulate x0 currently. Change encoder code, and `_mtlds_model_forward`.")
    smpz, μ_z, σ_z = posterior_sample(m.mt_enc, m.mt_post, vcat(U, Y), size(U, 2), stochastic)
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
"""
function create_mtlds(d_x, d_in, d_y, d_enc_state, d_mt; encoder=:LSTM, emission=:linear, d_hidden=64,
    spherical_var=false)

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
    hphi = mlp(d_mt, d_hidden, Int(n_dec))

    emission_std = spherical_var ? Tracker.param([0.0f0]) : Tracker.param(zeros(Float32, d_y))
    MTLDS_variational(init_enc, enc_post, hphi, emission, emission_std, d_x, flag_mt_emission, false)
end



################################################################################
##                                                                            ##
##                        Objective(s) / Loss Function(s)                     ##
##                                                                            ##
################################################################################

function check_inputs_outputs(m::MTLDS_variational, Ys::AbstractArray{T1}, Us::AbstractArray{T2}) where {T1, T2}
    @assert T1 === T2 format("Y is type {:s}, but U is type {:s}", string(T1), string(T2))
    no_inputs, has_inputs = false, true        # for readability of return value
    @argcheck ndims(Ys) == 3
    @argcheck ndims(Us) == 3
    @argcheck size(Ys, 2) == size(Us, 2)
    # (size(Us, 3) == 1) && all(Us .≈ 0) && return no_inputs # idiom for no inputs
    (size(Us, 3) == 1) && return no_inputs # idiom for no inputs (or same inputs ∀ seqs i)
    @argcheck size(Ys, 3) == size(Us, 3)
    return has_inputs
end


nllh_of_recon_batchavg(m::MTLDS_variational, y, mu, logstd) = _gauss_nllh_batchavg(y, mu, logstd)

function _gauss_nllh_batchavg(y::AbstractArray{T,3}, mu::AbstractArray{T,3}, logstd::AbstractVector) where T
    n_y, tT, n_b = size(y)
    Δ = (y - mu) ./ exp.(logstd)
    (sum(Δ.*Δ) + tT*n_b*sum(logstd)) / (2*n_b)   # ignoring -0.5tT*n_y*log(2π)
end

# d \times   T
function _gauss_nllh_individual_bybatch(y::AbstractArray{T,3}, mu::AbstractArray{T,2}, logstd::AbstractVector; dims::Int=0) where T
    n_y, tT, n_b = size(y)
    Δ = (y .- mu) ./ exp.(logstd)
    (sum(Δ.*Δ; dims=(1,2))[:] .+ tT*sum(logstd)) / 2   # ignoring -0.5tT*n_y*log(2π)
end

function nllh(m::MTLDS_variational, y, u; kl_coeff=1.0f0, stochastic=true,
        logstd_prior=nothing)
    zsmp, zμ, zσ = encode(m, y, u; stochastic=stochastic)[1]
    ŷ = m(zsmp, u)
    nllh = _gauss_nllh_batchavg(y, ŷ, m.logstd)

    # Prior regularization of emission stdev.
    if !(logstd_prior === nothing)
        loss += sum(x->x^2, (m.logstd - logstd_prior[1])./logstd_prior[2])/2
    end
    loss
end


function elbo_w_kl(m::MTLDS_variational, y, u; kl_coeff=1.0f0, stochastic=true,
        logstd_prior=nothing)
    nbatch = size(y,3)
    zsmp, zμ, zσ = encode(m, y, u; stochastic=stochastic)[1]
    ŷ = m(zsmp, u)
    nllh = _gauss_nllh_batchavg(y, ŷ, m.logstd)
    kl = stochastic * kl_coeff * (-0.5f0 * sum(1 .+ 2*log.(zσ) - zμ.*zμ - zσ.*zσ)) / nbatch

    # Prior regularization of emission stdev.
    if !(logstd_prior === nothing)
        nllh += sum(x->x^2, (m.logstd .- logstd_prior[1])./logstd_prior[2])/2
    end
    nllh + kl, kl
end

"""
    elbo(m::MTLDS_variational, y, u; kl_coeff=1.0f0, stochastic=true, logstd_prior=nothing)
Calculate ELBO ``=\\, <\\log p(y, z | u)>_{q(z|y, u)} - \\,\\text{KL}(q(z|y, u)\\,||\\,p(z))``
for MTLDS with emissions `y` and inputs `u`.

### Optional arguments:
- `kl_coeff`: essentially the β of the β-VAE, useful for annealing or disentangling.
- `stochastic`: whether to sample from posterior or use mean (boolean).
- `logstd_prior`: prior distirbution (Gaussian) regularizing the emission log std. Specify tuple (prior_mean, prior_std)

It is sometimes useful to monitor the two terms (reconstruction and KL) separately.
In this case, the function `elbo_w_kl` (same arguments) returns a tuple of (elbo, kl).
"""
elbo(m::MTLDS_variational, y, u; kl_coeff=1.0f0, stochastic=true, logstd_prior=nothing) =
    elbo_w_kl(m, y, u; kl_coeff=kl_coeff, stochastic=stochastic, logstd_prior=logstd_prior)[1]


"""
    aggregate_importance_smp(m::MTLDS_variational, y::Tensor, u::Tensor; tT=size(u,2), M=200,
        maxbatch=200, suppresswarn=false)

Take `M` samples from the prior and use as an aggregate importance sample for *all* `y`,
where y is a ``n_y × tT × batch`` matrix (i.e. for each ``y`` in the batch). This uses the
`forward_multiple_z` function on a `MTLDS_variational` object with possible inputs `u`. This
cannot be efficiently used where each sequence has a different input `u`, and the function
will yield a warning in this case that it is defaulting to non-aggregated sampling.

Returns the sample matrix `Z` (``d_z × M``), the batch-sample weight matrix, `W`, and the
total (summed) log probability.
"""
function aggregate_importance_smp(m::MTLDS_variational, y::AbstractArray{T,3}, u::AbstractArray{T,3}=zeros(T, size(y)[1,2]..., 1);
        tT=size(u,2), M=200, maxbatch=200, suppresswarn=false) where T
    # !(sum(abs, diff(u, dims=3)) ≈ 0)
    !suppresswarn && size(u,3) > 1 && @warn "different batches of u can result in slow execution. Defaulting to non-aggregated importance sampling. Works best with zero inputs."

    # draw Quasi-Monte Carlo samples
    d_z = size(m.mt_post.Dense1.W, 1)
    n_y, _, N = size(y)
    Z = f32(util.sobol_gaussian(M, d_z)')

    # if each y has a different input, we cannot amortize the draws, so dispatch to individual fn:
    if size(u,3) > 1
        @argcheck size(u,3) == size(y, 3)
        return _aggregate_importance_smp_noamortize(m, y, u, Z; tT=tT, M=M, maxbatch=maxbatch)
    end

    yhats = forward_multiple_z(m, Z, dropdims(u, dims=3); maxbatch=maxbatch)
    W, logp_y = _aggregate_importance_smp(yhats, y, m.logstd, tT)
    return Z, W, logp_y
end

"""
    _aggregate_importance_smp(y_mc::Tensor, y_query::Tensor, logσ, tT::Int)

Internal aggregate importance sample function which creates the batch-sample importance
weight matrix, `W` between the `y_mc` samples and `y_query` queries; and the estimated
marginal log probability of each.
"""
function _aggregate_importance_smp(y_mc::AbstractArray{T,3}, y_query::AbstractArray{T,3}, logσ, tT) where T
    s = exp.(logσ)
    n_y = length(logσ)
    M = size(y_mc, 3)
    y_query, y_mc = util.collapsedims1_2(y_query ./ s), util.collapsedims1_2(y_mc ./ s)

    logW = - util.sq_diff_matrix(y_mc', y_query') / 2   # [log p(y_i | z_j) i = 1..N, j = 1..M]
    logW .-= tT * (sum(logσ)*2 + n_y*log(2π)) / 2   # normalizing constant

    W, lse = util.softmax_lse!(logW; dims=1)
    logp_y = lse .- log(M)  # W, log(1/M sum_j exp(log_j))
    return W, logp_y
end

function _aggregate_importance_smp_noamortize(m::MTLDS_variational, y::AbstractArray{T,3}, u::AbstractArray{T,3}, Z::AbstractArray{T,2};
        tT=-1, M=-1) where T
    n_b = size(y,3)
    W, logp_ys = Matrix{Float32}(undef, M, n_b), Vector{Float32}(undef, n_b)
    for i in 1:n_b
        w, logp_y = _importance_smp_individual(m, y[:,:,i], u[:,:,i], Z; tT=tT, M=M)
        W[:,i] = w
        logp_ys[i] = logp_y
    end
    return Z, W, sum(logp_ys)
end

function _importance_smp_individual(m::MTLDS_variational, y::AbstractArray{T,2},
        u::AbstractArray{T,2}, z::AbstractArray{T,2}; tT=80, M=200, maxbatch=200) where T
    yhats = forward_multiple_z(m, z, u; maxbatch=maxbatch)    # n_y × tT × n_b

    llh_by_z = - _gauss_nllh_individual_bybatch(yhats, y, m.logstd)
    W, lse = util.softmax_lse!(unsqueeze(llh_by_z, 2); dims=1)
    logp_y = sum(lse) - log(M)  # W, log(1/M sum_j exp(log_j))
    return vec(W), logp_y
end
