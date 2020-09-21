

"""
    batch_matvec(A::Tensor, X::Matrix)

for A ∈ ℝ^{m × d × n}, X ∈ ℝ^{d × n}. Performs ``n`` matrix-vec multiplications

    A[:,:,i] * X[:,i]

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

ensure_tensor(x::AbstractMatrix) = unsqueeze(x, 3)
ensure_tensor(x::AbstractVector) = reshape(x, length(x), 1, 1)
ensure_tensor(x::AbstractArray{T,3}) where T = x

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
    LDSCell_diag_u{U,V,W}
Simple LDS struct containing three fields: a (transition coeffs), B (input
matrix) and h, the initial state.
"""
mutable struct LDSCell_diag_u{U,V,W}
    a::U
    B::V
    h::W
end

# Operation
function (m::LDSCell_diag_u)(h, x)
    h = m.a .* h + m.B * x
    return h, h
end

Flux.hidden(m::LDSCell_diag_u) = m.h
Flux.@treelike LDSCell_diag_u


"""
    LDSCell_diag_batch_u{U,V,W}
Simple LDS struct containing three fields: a (transition coeffs), B (input
matrix) and h, the initial state. Permits 3D arrays for a, B and performs
batch updates on the inputs.
"""
mutable struct LDSCell_diag_batch_u{U,V,W}
    a::U
    B::V
    h::W
end

# Operation
function (m::LDSCell_diag_batch_u)(h, x)
    h = m.a .* h + batch_matvec(m.B, x)
    return h, h
end

Flux.hidden(m::LDSCell_diag_batch_u) = m.h
Flux.@treelike LDSCell_diag_batch_u



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
