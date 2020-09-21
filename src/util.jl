module util
using Base.Threads: @threads
using ..Formatting, ..Distributions, ..StatsBase
using Sobol
using Random: GLOBAL_RNG, MersenneTwister

export unsqueeze


################################################################################
##                                                                            ##
##                                Generic                                     ##
##                                                                            ##
################################################################################


e_k(T, n, k) = begin; out = zeros(T, n); out[k] = 1; out; end
e_k(n, k) = e_k(Float32, n, k)
e1(T, n) = e_k(T, n, 1)
e1(n) = e_k(Float32, n, 1)

unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...));
vflatten(x) = reduce(vcat, x)
collapsedims1_2(X::AbstractArray) = reshape(X, (prod(size(X)[1:2]), size(X)[3]))
"""
    get_strtype_wo_params(x)

Useful for highly-parameterized models. The print (or typeof) for these structs
result in the overall type, plus a load of parameters, or details in
parentheses. This function just returns the first part: the overall type,
without all the other junk. This uses the Base.show methods, but captures the
output via IOBuffer.
"""
function get_strtype_wo_params(x)
    _io = IOBuffer()
    print(_io, x)    # (does not print to stdout)
    tstr = String(take!(_io))
    close(_io)
    ixparen = findfirst("(", tstr)
    ixparen === nothing && return tstr
    return tstr[1:first(ixparen)-1]
end

function sq_diff_matrix(X, Y)
    #=
    Constructs $(x_i - y_j)^T (x_i - y_j)$ matrix where
    X = Array(n_x, d)
    Y = Array(n_y, d)
    return: Array(n_x, n_y)
    =#
    normsq_x = sum(a -> a^2, X, dims=2)
    normsq_y = sum(a -> a^2, Y, dims=2)
    @assert size(normsq_x, 2) == 1 && size(normsq_y, 2) == 1
    out = normsq_x .+ normsq_y'    # outer
    @assert size(out) == (size(normsq_x, 1), size(normsq_y, 1))
    out .-= 2*X * Y'
    return out
end

function _softmax_lse!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where T<:AbstractFloat
    #=
    Simultaneous softmax and logsumexp. Useful for calculating an estimate of logp(x) out of
    lots of MC samples, as well as calculating the approximate posterior.
    We essentially get the Logsumexp for free here.
    (adapted from Mike I's code in NNlib.)
    =#
    lse = Array{T, 1}(undef, size(out, 2))

    @threads for j = 1:size(xs, 2)
        @inbounds begin
            # out[end, :] .= maximum(xs, 1)
            m = xs[1, j]
            for i = 2:size(xs, 1)
                m = max(m, xs[i, j])
            end
            # out .= exp(xs .- out[end, :])
            for i = 1:size(out, 1)
                out[i, j] = exp(xs[i, j] - m)
            end
            # out ./= sum(out, 1)
            s = zero(eltype(out))
            for i = 1:size(out, 1)
                s += out[i, j]
            end
            for i = 1:size(out, 1)
                out[i, j] /= s
            end
            lse[j] = log(s) + m
        end
    end
    return out, lse
end

function softmax_lse!(out, xs; dims=1)
    if dims == 1
        _softmax_lse!(out, xs)
    elseif dims == 2
        xsT, outT = Matrix(xs'), out'  # Matrix = faster to avoid strides for larger matrix
        s, l = _softmax_lse!(outT, xsT)
        s', l
    else
        error("Unsupported dims argument (expecting (1,2)): $dims")
    end
end

softmax_lse!(xs; dims=1) = softmax_lse!(xs, xs; dims=dims)
softmax_lse(xs; dims=1) = softmax_lse!(similar(xs), xs; dims=dims)


################################################################################
##                                                                            ##
##                             Random samples                                 ##
##    These are all copied from my AxUtil package to avoid further imports    ##
##                                                                            ##
################################################################################



sobol_gaussian(n, d) = sobol_gaussian(GLOBAL_RNG, n, d)
"""
    sobol_gaussian([rng::MersenneTwister,] n, d)

Randomized Quasi Monte Carlo samples from a `d` dimensional unit Gaussian. This
function returns `n` such samples in a ``n × d`` matrix. The QMC sequence used
here is the Sobol sequence.
"""
function sobol_gaussian(rng::MersenneTwister, n, d)
    (n == 0) && return zeros(d,0)'
    s = SobolSeq(d)
    p = reduce(hcat, [next!(s) for i = 1:n])
    ϵ = rand(rng, d)
    prand = [(p[j,:] .+ ϵ[j]) .% 1.0 for j in 1:d]
    p = reduce(vcat, [quantile.(Normal(), prand[j])' for j in 1:d])'
    return p
end


randomised_sobol(n, d) = randomised_sobol(GLOBAL_RNG, n, d)
"""
    randomised_sobol([rng::MersenneTwister,] n, d)
Generate `n` samples from `d`-dimensional Sobol sequence, and applying a random
affine mapping to the collection on the [0,1]^d torus.
"""
function randomised_sobol(rng::MersenneTwister, n, d)
    s = SobolSeq(d)
    p = reduce(hcat, [next!(s) for i = 1:n])
    ϵ = rand(rng, d)
    prand = [(p[j,:] .+ ϵ[j]) .% 1.0 for j in 1:d]
    return hcat(prand...)
end


# randomised Sobol within rectangle
uniform_rand_sobol(n, lims...) = uniform_rand_sobol(GLOBAL_RNG, n, lims...)
"""
    uniform_rand_sobol([rng::MersenneTwister,] n, lims)

Randomised sobol sequence (see `randomised_sobol`), mapped to be a quasi random
sample of a uniform distribution on a hyper-cuboid with limits specified by
`lims`. For example:

    uniform_rand_sobol(N, [x_l, x_u], [y_l, y_u])

generates a QMC sample on the rectangle ``[x_l, x_u] × [y_l, y_u]```, with
the first coordinate drawn at random in ``[x_l, x_u]`` and so on.
"""
function uniform_rand_sobol(rng::MersenneTwister, n, lims...)
    d = length(lims)
    rsob = randomised_sobol(rng, n, d)
    for (i, interval) in enumerate(lims)
        @assert (length(interval) == 2) format("interval {:d} does not have length 2", i)
        rsob[:,i] .*= diff(interval)
        rsob[:,i] .+= interval[1]
    end
    return rsob
end



function categorical_sampler(p::AbstractVector, n::Int)
    """
        categorical_sampler(p::AbstractVector, n::Int)

    Categorical sampling using a linear scan strategy, which is surprisingly
    efficient for small `n`. Unlike `rand(Categorical(p), n)`, this function
    accepts Float32 as well as Float64, and is robust in case of `sum(p) != 1`.
    """
    m = length(p)
    x = zeros(Int32, n)

    function linearsearch(p::AbstractArray, m::Int64, rn)
        cs = 0.0
        for ii in 1:m
            @inbounds cs += p[ii]
            if cs > rn
                return ii
            end
        end
        return m
    end

    sump = sum(p)
    for i in 1:n
        rn = rand()*sump
        @inbounds x[i] = linearsearch(p, m, rn)
    end
    return x
end




################################################################################
##                                                                            ##
##                    Preprocessing: StandardScaler                           ##
##                                                                            ##
################################################################################
# This was originally from my Mocap project, but works ~ like Sklearn's version

"""
    fit(MyStandardScaler, X, dims)
Fit a standardisation to a matrix `X` s.t. that the rows/columns have mean zero and standard deviation 1.
This operation calculates the mean and standard deviation and outputs a MyStandardScaler object `s` which
allows this same standardisation to be fit to any matrix using `transform(s, Y)` for some matrix `Y`. Note
that the input matrix `X` is *not* transformed by this operation. Instead use the above `transform` syntax
on `X`.
Note a couple of addendums:
1. Any columns/rows with constant values will result in a standard deviation of 1.0, not 0. This is to avoid NaN errors from the transformation (and it is a natural choice).
2. If only a subset of the rows/columns should be standardised, an additional argument of the indices may be given as:
    `fit(MyStandardScaler, X, operate_on, dims)`
    The subset to operate upon will be maintained through all `transform` and `invert` operations.
"""
mutable struct MyStandardScaler{T}
    μ::AbstractArray{T}
    σ::AbstractArray{T}
    dims
end

Base.copy(s::MyStandardScaler) = MyStandardScaler(copy(s.μ), copy(s.σ), copy(s.operate_on), copy(s.dims))

StatsBase.fit(::Type{MyStandardScaler}, X::AbstractArray, dims=1) = 
    MyStandardScaler(mean(MaskedArray(X), dims=dims), 
    std(MaskedArray(X), dims=dims), dims) |> post_fit

function post_fit(s::MyStandardScaler)
    bad_ixs = (s.σ .== 0)
    s.σ[bad_ixs] .= 1
    s
end

function scale_transform(s::MyStandardScaler, X::AbstractArray, dims=s.dims)
    (dims != s.dims) && @warn "dim specified in transform is different to specification during fit."
    mu, sigma = s.μ, s.σ
    (X .- mu) ./ sigma
end

function invert(s::MyStandardScaler, X::AbstractArray, dims=s.dims)
    (dims != s.dims) && @warn "dim specified in inversion is different to specification during fit."
    mu, sigma = s.μ, s.σ
    (X .* sigma) .+ mu
end



################################################################################
##                                                                            ##
##               Handling missing (NaN) values: MaskedArray                   ##
##                                                                            ##
################################################################################

"""
    MaskedArray(x::AbstractArray)

I've been dissatisfied for some time about how AD engines etc tend to handle missing values (historically, they basically couldn't, and required a hack. The hack is not terribly hard, but it requires additional overhead for coding and data structures etc. This was an attempt to show that this could be easily handled using a new type of array. But... turns out that a lot more work is needed and I need to understand more about Julia internals to make this work properly (especially Broadcasting), and would need to spend quite some time integrating with Flux's Tracker which is now ~ defunct anyway. So I gave up halfway! Nevertheless, I found the resulting class moderately useful.

The idea is that missing values (represented by NaNs¹) can be handled by a (`.mask`) matrix which determines where the missing values go. The array is then represented with a (`.data`) matrix **only** with numbers in ℝ -- where the missing values might be replaced with any value, but often zero. To recover the original matrix with missing values, the `as_nan` function creates a copy of the `.data` and places the `NaN`s back in their correct place from the mask. 

This allows a couple of useful properties for writing AD-enabled code:

1. The missing value mask is carried around *with* the data, rather than requiring separate book-keeping.
1. The AD code can just operate on the `.data`, but then the mask is used to zero out all the gradients which should be ignored.
1. We can natively extract the dimensions of the missing entries e.g. to use as an input mask for a network using `.mask`.
1. Useful versions of `sum`, `mean`, `std` and `var` can be (and have been) defined which ignores the `NaN`s as usually required rather than needing to use some alternative library like NaNMath.

**Broadcasting Issues**:

THIS DOES NOT ALLOW BROADCASTING, and getting to grips with how broadcasting works is necessary to sort this. Simply subtyping from AbstractArray prob doesn't work. The major problem right now is that by default broadcasting iterates over the entire object *one element at a time*. This results in using single indices (scalar values) from the object, which the `.getindex` method implicitly doesn't support :(. To get this to work I'll need to catch when none of the indices are slices and force the result back into an array prior to --> MaskedArray. But even this won't help in the end, as it will result in an Array of 1x1 MaskedArrays. There's no inspiration I can get from TrackedArrays or DualNumbers etc. Utimately I'll need to rtfm here: https://docs.julialang.org/en/latest/manual/interfaces/#man-interfaces-broadcasting-1

... BUT even then I can't natively plug these arrays into Flux, since I'll need to write lots of custom methods to interact with TrackedArrays, and this could easily go wrong. It's almost certainly best just to use MaskedArrays as a useful container and apply the data and mask explicitly during the Flux forward pass... 😂


**Footnote:**

¹I know Julia introduced `missing` with very good support and apparently performance some time ago, but this idea predates this being well supported. I don't know how well it is supported in AD these days. This whole data structure might be entirely unnecessary these days.
"""
struct MaskedArray{T, N} <: AbstractArray{T, N}
    data::AbstractArray{T, N}
    mask::AbstractArray{Bool, N}
end
MaskedArray(T::DataType, x::AbstractArray, b::AbstractArray) = MaskedArray(T.(x), T.(b))

function make_masked_array(x::AbstractArray)
    mask = isnan.(x)
    x = copy(x)
    x[mask] .= 0
    return MaskedArray(x, .!mask)
end
MaskedArray(T::DataType, x::AbstractArray) = make_masked_array(T.(x))
MaskedArray(x::AbstractArray) = make_masked_array(x)

data(x::MaskedArray) = x.data
mask(x::MaskedArray) = x.mask

# A key problem is here => this works fine provided >=1 of the indices are  
Base.getindex(x::MaskedArray, I...) = MaskedArray(x.data[I...], x.mask[I...])
Base.size(x::MaskedArray) = size(data(x))
Base.size(x::MaskedArray, j::Int) = size(data(x), j)
Base.length(x::MaskedArray) = length(data(x))
Base.show(io::IO, ::MIME"text/plain", x::MaskedArray) = (println("Masked array with data:"); 
    flush(stdout); display(as_nan(x)))
Base.display(io::IO, x::MaskedArray) = (println("Masked array with data:"); 
    flush(stdout);  display(as_nan(x)))

function _as_missing(x::MaskedArray{T,N}) where {T,N}
    y::AbstractArray{Union{Missing, T}} = copy(x.data)
    y[.!x.mask] .= missing
    y
end

# (Possibly) useful ops re applying the mask
as_nan(x::MaskedArray{T,N}) where {T,N} = replace(_as_missing(x), missing=>T(NaN))
rm_nan(x::MaskedArray{T,N}) where {T,N} = collect(skipmissing(_as_missing(x)))
zero_maskvals(x::MaskedArray, y::AbstractArray) = y .* mask(x)
zero_maskvals(x::MaskedArray) = zero_maskvals(x, data(x))
apply_mask(x::MaskedArray, y::AbstractArray) = MaskedArray(zero_maskvals(x, y), mask(x))


stacked(x::MaskedArray) = vcat(data(x), mask(x))
stacked(x::MaskedArray, I...) = vcat(data(x)[I...], mask(x)[I...])
hstacked(x::MaskedArray) = hcat(data(x), mask(x))
hstacked(x::MaskedArray, I...) = hcat(data(x)[I...], mask(x)[I...])
const vstacked = stacked

# # For vcat in encode method
# Base.vcat(x::MaskedArray, y::MaskedArray) = MaskedArray(vcat(data(x), data(y)), 
#     vcat(mask(x), mask(y)))

# For comparison and use as Dict keys
Base.hash(x::MaskedArray, h::UInt) = hash((data(x), mask(x)), h)
Base.:(==)(x::MaskedArray, y::MaskedArray) = hash(x) == hash(y)

# Standard operators
Base.Array(x::MaskedArray) = as_nan(x)
Base.:+(x::MaskedArray, y::AbstractArray) = MaskedArray(data(x) + y, mask(x))
Base.:+(x::AbstractArray, y::MaskedArray) = MaskedArray(x + data(y), mask(y))
Base.:-(x::MaskedArray, y::AbstractArray) = MaskedArray(data(x) - y, mask(x))
Base.:-(x::AbstractArray, y::MaskedArray) = MaskedArray(x - data(y), mask(y))
Base.:*(x::MaskedArray, y::AbstractArray) = MaskedArray(data(x) * y, mask(x))
Base.:*(x::AbstractArray, y::MaskedArray) = MaskedArray(x * data(y), mask(y))
Base.:/(x::MaskedArray, y::AbstractArray) = MaskedArray(data(x) / y, mask(x))
Base.:/(x::AbstractArray, y::MaskedArray) = MaskedArray(x / data(y), mask(y))
Base.sum(x::MaskedArray) = sum(zero_maskvals(x))
Base.sum(x::MaskedArray; dims) = sum(zero_maskvals(x), dims=dims)
StatsBase.mean(x::MaskedArray; dims) = sum(x, dims=dims) ./ sum(mask(x), dims=dims)
function StatsBase.var(x::MaskedArray; dims)
    mu = mean(x, dims=dims)
    return sum(z->z^2, (data(x) .- mu) .* mask(x), dims=dims) ./ (sum(mask(x), dims=dims) .- 1)
end
StatsBase.std(x::MaskedArray; dims) = sqrt.(var(x; dims=dims))

Base.selectdim(A::MaskedArray, d::Integer, i) = MaskedArray(Array(selectdim(data(A), d, i)),
    Array(selectdim(mask(A), d, i)))
end
