module util
using Base.Threads: @threads
using ..Formatting, ..Distributions
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

end
