module mtlinear

using LinearAlgebra
using Flux, ArgCheck, StatsBase, YAML
using Flux: Tracker
using Flux.Tracker: istracked, @grad

using ..modelutils
import ..modelutils: load!, randn_repar
export load!




"""
    batch_matvec(A::Tensor, X::Matrix)

for A ∈ ℝ^{n × m × d}, X ∈ ℝ^{n × d}. Performs ``n`` matrix-vec multiplications

    A[i,:,:] * X[i,:]

for ``i \\in 1,\\ldots,n``. This is performed via expansion of `X` and can be
efficiently evaluated using BLAS operations, which (I have assumed!) will often
be faster than a loop, especially in relation to AD and GPUs. Cf. `multitask.jl`
"""
batch_matvec(A::AbstractArray{T,3}, X::AbstractArray{T,2}) where T =
    dropdims(sum(A .* unsqueeze(X, 1), dims=2), dims=2)

"""
    batch_matmul(A::MatrixOrArray, X::Array)

for A ∈ ℝ^{n × m × d}, X ∈ ℝ^{m × r × d}. Performs ``n`` matrix multiplications

    A[:,:,i] * X[:,:,i]

for ``i \\in 1,\\ldots,n``. This is performed via expansion of `X` and can be
efficiently evaluated using BLAS operations, which (I have assumed!) will often
be faster than a loop in relation to AD and GPUs.

We also permit A ∈ ℝ^{n × m}, X ∈ ℝ^{m × r × d} => ℝ^{m × r × d}.
"""
batch_matmul(A::AbstractArray{T,3}, X::AbstractArray{T,3}) where T =
    dropdims(sum(unsqueeze(A,3) .* unsqueeze(X, 1), dims=2), dims=2)

batch_matmul(A::AbstractArray{T,2}, X::AbstractArray{T,3}) where T =
    dropdims(sum(unsqueeze(unsqueeze(A,3),4) .* unsqueeze(X, 1), dims=2), dims=2)

ensure_matrix(x::AbstractMatrix) = x
ensure_matrix(x::AbstractVector) = unsqueeze(x, 2)

eye(d) = Matrix(I, d, d)

################################################################################
##                                                                            ##
##                         Special Matrix Constructors                        ##
##          *** THIS IS TAKEN DIRECTLY FROM MY JULIA UTILS PKG,               ##
##          BUT COPIED HERE FOR COMPLETENESS / TO SATISFY ANONYMITY ***       ##
##                                                                            ##
################################################################################

#= have a CuArray version in CUDA file. Cannot make more generally available
   because CuArrays.jl will not compile (e.g. on my machine) if no GPU. See
   @requires in [REDACTED].jl and CUDA.jl files in this project. =#

"""
    make_lt(x::AbstractVector{T}, d::Int)

Make lower triangular matrix (``d \\times d``) from vector `x`.
"""
function make_lt(x::AbstractVector{T}, d::Int)::Array{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d+1)/2))
    make_lt!(zeros(T, d, d), x, d)
end

function make_lt!(out::AbstractMatrix{T}, x::AbstractArray{T,1}, d::Int)::Array{T,2} where T <: Real
    x_i = 1
    for j=1:d, i=j:d
        out[i,j] = x[x_i]
        x_i += 1
    end
    return out
end

# Gradients
function unmake_lt(M::AbstractMatrix{T}, d)::Array{T,1} where T <: Real
    return M[tril!(trues(d,d))]
end

make_lt(x::TrackedArray, d::Int) = Tracker.track(make_lt, x, d)

@grad function make_lt(x, d::Int)
    return make_lt(Tracker.data(x), d), Δ -> (unmake_lt(Δ, d), nothing)
end


"""
    make_lt_strict(x::AbstractVector{T}, d::Int)

Make *strictly* lower triangular matrix (``d \\times d``) from vector `x`.
"""
function make_lt_strict(x::AbstractVector{T}, d::Int)::Array{T,2} where T <: Real
    @argcheck (length(x) == Int(d*(d-1)/2))
    return make_lt_strict!(zeros(T, d, d), x, d)
end

function make_lt_strict!(out::AbstractMatrix{T}, x::AbstractVector, d::Int)::Array{T,2} where T <: Real
    x_i = 1
    for j=1:d-1, i=j+1:d
        out[i,j] = x[x_i]
        x_i += 1
    end
    return out
end

make_lt_strict(x::TrackedArray, d::Int) = Tracker.track(make_lt_strict, x, d)

@grad function make_lt_strict(x, d::Int)
    return make_lt_strict(Tracker.data(x), d), Δ -> (unmake_lt_strict(Δ, d), nothing)
end


function unmake_lt_strict(M::AbstractMatrix{T}, d::Int)::Array{T,1} where T <: Real
    return M[tril!(trues(d,d), -1)]
end

"""
    make_skew(x::AbstractArray{T,1}, d::Int)

Make ``d \\times d`` skew-symmetric matrix from vector `x`.
"""
function make_skew(x::AbstractArray{T,1}, d::Int)::AbstractArray{T,2} where T <: Real
    S = make_lt_strict(x, d)
    return S - S'
end


function cayley_orthog(x::AbstractArray{T,1}, d)::AbstractArray{T,2} where T <: Real
    S = make_skew(x, d)
    I = eye(d)
    return (I - S) / (I + S)  # (I - S)^{-1}(I + S). Always nonsingular.
end


function inverse_cayley_orthog(Q::AbstractMatrix{T}, d::Int=size(Q,1)) where T <: Real
    unmake_lt_strict((I - Q ) / (I + Q), d)
end


"""
    diag0(x::Array{T,1})
Make diagonal matrix from vector `x`. This may have been superseded by `Diag`,
but is here for legacy reasons since Flux's version used to result in Tracked{Tracked}.
"""
function diag0(x::Array{T,1})::Array{T,2} where T <: Real
    d = length(x)
    M = zeros(T, d,d)
    M[diagind(M)] = x
    return M
end

diag0(x::TrackedArray) = Tracker.track(diag0, x)

@grad function diag0(x)
    return diag0(Tracker.data(x)), Δ -> (Δ[diagind(Δ)],)
end

"""
    LDSCell_simple_u{U,V,W}
Simple LDS struct containing three fields: A, B (for transition and input
matrices) and h, the initial state.
"""
mutable struct LDSCell_simple_u{U,V,W}
    A::U
    B::V
    h::W
end

# Operation
function (m::LDSCell_simple_u)(h, x)
    h = m.A * h + m.B * x
    return h, h
end

Flux.hidden(m::LDSCell_simple_u) = m.h
Flux.@treelike LDSCell_simple_u


"""
    LDSCell_batch_u{U,V,W}
Simple LDS struct containing three fields: A, B (for transition and input
matrices) and h, the initial state. Permits 3D arrays for A, B and performs
batch updates on the inputs.
"""
mutable struct LDSCell_batch_u{U,V,W}
    A::U
    B::V
    h::W
end

# Operation
function (m::LDSCell_batch_u)(h, x)
    h = batch_matvec(m.A, h) + batch_matvec(m.B, x)
    return h, h
end

Flux.hidden(m::LDSCell_batch_u) = m.h
Flux.@treelike LDSCell_batch_u

################################################################################
##                                                                            ##
##                         Model specific utilities                           ##
##                                                                            ##
################################################################################

"""
    feat_extract(z)

A fixed-form feature extractor converting a vector ``\\mathbf{z}`` into the
vector ``[\\mathbf{z}, \\sin(\\mathbf{z}), \\cos(\\mathbf{z}), \\|\\mathbf{z}\\|]``.
"""
feat_extract(z) = reduce(vcat, [z ,sin.(z), cos.(z), sqrt.(sum(x->x^2, z, dims=1))])
feat_extract(z::TrackedArray) = Tracker.track(feat_extract, z)


function feat_extract_deriv(Δ, f, d)
    out = Δ[1:d,:]
    for dd in 1:d
        out[dd,:] .+= (f[d*2+dd,:] .* Δ[d+dd,:])   # cos z_dd * ∇_{d+dd}
        out[dd,:] .+= -(f[d*1+dd,:] .* Δ[2*d+dd,:])  # - sin z_dd * ∇_{2d+dd}
        out[dd,:] .+= (f[dd,:] .* Δ[3*d+1,:]) ./ (f[3*d+1,:] .+ 1e-16) # (z_dd /||z||) * ∇_{3d+1}
    end
    return out
end

@grad function feat_extract(z::AbstractVector)
    zd = Tracker.data(z)
    d = size(z, 1)
    f = feat_extract(zd)
    return f, Δ->let g=feat_extract_deriv(Δ, f, d); (vec(g), ); end
end

@grad function feat_extract(z::AbstractMatrix)
    zd = Tracker.data(z)
    d = size(z, 1)
    f = feat_extract(zd)
    return f, Δ->let g=feat_extract_deriv(Δ, f, d); (g, ); end
end

"""
    A(ψ, d)
Construct transition matrix `A` using reduced Cayley form from parameter vector
ψ (extracting relevant elements).
"""
function A(ψ, d)
    n_skew = Int(d*(d-1)/2)
    x_S, x_V = ψ[1:d], ψ[d+1:d+n_skew]
    V = cayley_orthog(x_V/10, d)
    S = diag0(σ.(x_S))
    return S * V
end

"""
    B(ψ, d)
Construct transition matrix `B` using sparse construction from parameter vector
ψ (extracting relevant elements).
"""
function B(ψ::AbstractVector, d, n_u)
    n_skew = Int(d*(d-1)/2)
    i_init = d+n_skew
    ψ_b1 = reshape(ψ[(i_init+1):(i_init+n_u*d)], d, n_u)
    ψ_b2 = reshape(ψ[(i_init + n_u*d +1):(i_init + 2*n_u*d)], d, n_u)
    return σ.(ψ_b1) .* tanh.(ψ_b2)
end

function B(ψ::AbstractMatrix, d, n_u)
    n_skew = Int(d*(d-1)/2)
    n_b = size(ψ, 2)
    i_init = d+n_skew
    ψ_b1 = reshape(ψ[(i_init+1):(i_init+n_u*d), :], d, n_u, n_b)
    ψ_b2 = reshape(ψ[(i_init + n_u*d +1):(i_init + 2*n_u*d), :], d, n_u, n_b)
    return σ.(ψ_b1) .* tanh.(ψ_b2)
end

"""
    C(ψ, d)
Construct emission matrix `C` with no special constructor from parameter vector
ψ (extracting relevant elements).
"""
function C(ψ::AbstractVector, d, n_u, n_out)
    n_skew = Int(d*(d-1)/2)
    i_init = d + n_skew + 2*n_u*d
    ψ_c= ψ[(i_init+1):(i_init+n_out*d)]
    return reshape(ψ_c, n_out, d)
end

function C(ψ::AbstractMatrix, d, n_u, n_out)
    n_skew = Int(d*(d-1)/2)
    n_b = size(ψ, 2)
    i_init = d + n_skew + 2*n_u*d
    ψ_c= ψ[(i_init+1):(i_init+n_out*d), :]
    return reshape(ψ_c, n_out, d, n_b)
end

"""
    dmt(ψ, d)
Construct emission bias `d` with no special constructor from parameter vector
ψ (extracting relevant elements).
"""
function dmt(ψ::AbstractVector, d, n_u, n_out)
    n_skew = Int(d*(d-1)/2)
    i_init = d + n_skew + 2*n_u*d +n_out*d
    return ψ[(i_init+1):(i_init+n_out)]
end

function dmt(ψ::AbstractMatrix, d, n_u, n_out)
    n_skew = Int(d*(d-1)/2)
    i_init = d + n_skew + 2*n_u*d +n_out*d
    return ψ[(i_init+1):(i_init+n_out), :]
end

################################################################################
##                                                                            ##
##                              Define MT-LDS                                 ##
##                                                                            ##
################################################################################

struct MTLDS_variational{U,V,W,S}
    mt_enc::U
    mt_post::V
    hphi::W
    emission::S
    d::Int
    flag_mt_emission::Bool
    flag_mt_h0::Bool
end

Flux.@treelike MTLDS_variational

function _mtlds_model_forward(m::MTLDS_variational, z::AbstractVecOrMat{T}, U::AbstractArray{T}) where T <: AbstractFloat
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


function _posterior_sample(enc, dec, input, T_max, stochastic=true)
    Flux.reset!(enc)
    for tt = 1:T_max; enc(input[:,tt,:]); end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

function _mtlds_enc(m::MTLDS_variational, Y::AbstractArray{T}, U::AbstractArray{T}; stochastic=true) where T
    m.flag_mt_h0 && error("Unable to modulate h0 currently. Change encoder code, and `_mtlds_model_forward`.")
    _posterior_sample(m.mt_enc, m.mt_post, vcat(U, Y), size(U, 2), stochastic)
end

function _mtlds_model_recon(m::MTLDS_variational, Y::AbstractArray{T}, U::AbstractArray{T}; stochastic=true) where T
    @argcheck size(Y,2) == size(U,2)
    smp, μ, σ = _mtlds_enc(m, Y, U; stochastic=stochastic)
    Yhat = _mtlds_model_forward(m, smp, U)
    return Yhat
end


(m::MTLDS_variational)(z::AbstractArray{T,2}, U::AbstractArray{T,3}) where T = _mtlds_model_forward(m, z, U)
(m::MTLDS_variational)(Y::AbstractArray{T,3}, U::AbstractArray{T,3}) where T = _mtlds_model_recon(m, Y, U)


"""
    create_model(d_x, d_x0, d_y, d_enc_state; encoder=:LSTM,
    cnn=false, out_heads=1, d_hidden=d_x)

Create `Seq2SeqModel` with the number of hidden units in the recurrent
generative model as `d_x`, the size of the latent variable for the initial state
as `d_x0` (typically ``d_x0 \\ll d_x``), `d_y` the dimension of the observations
and `d_enc_state` the number of hidden units in the recurrent encoder. This
encoder can be specified as `:LSTM`, `:GRU` or `:Bidirectional`. If the
observations are video data and one wishes to use a CNN to encode and decode,
specify `cnn=true`. By default the `Seq2SeqModel` looks to make a point estimate
of the future series, but specifying `out_heads=2` will result in uncertainty
estimation too. Finally `d_hidden` specifies the number of hidden units in non
CNN decoders.
"""
function create_model(d_x, d_in, d_y, d_enc_state, d_mt; encoder=:LSTM, emission=:linear, d_hidden=64)

    if encoder == :LSTM
        init_enc = LSTM(d_y+d_in, d_enc_state)
    elseif encoder == :GRU
        init_enc = GRU(d_y+d_in, d_enc_state)
    elseif encoder == :RNN
        init_enc = RNN(d_y+d_in, d_enc_state)
    else
        error("encoder must be specified as :LSTM, :GRU, :RNN")
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
        lag_mt_emission = false
    end
    hphi = mlp(d_mt, d_hidden, Int(n_dec))

    MTLDS_variational(init_enc, enc_post, hphi, emission, d_x, flag_mt_emission, false)
end

end

