module util
using Base.Threads: @threads

e_k(T, n, k) = begin; out = zeros(T, n); out[k] = 1; out; end
e_k(n, k) = e_k(Float32, n, k)
e1(T, n) = e_k(T, n, 1)
e1(n) = e_k(Float32, n, 1)

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

end

