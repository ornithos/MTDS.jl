################################################################################
##                                                                            ##
##                        Objective(s) / Loss Function(s)                     ##
##                                                                            ##
################################################################################

function check_inputs_outputs(m::MTDSModel, Ys::AbstractArray{T1}, Us::AbstractArray{T2}) where {T1, T2}
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


# ======== Gaussian (negative) log likelihood calcs =============================

nllh_of_recon_batchavg(m::MTDSModel, y, mu, logstd) = _gauss_nllh_batchavg(y, mu, logstd)

function _gauss_nllh_batchavg(y::AbstractArray{T,3}, mu::AbstractArray{T,3}, logstd::AbstractArray) where T
    n_y, tT, n_b = size(y)
    Δ = (y - mu) ./ exp.(logstd)
    (sum(Δ.*Δ) + tT*n_b*sum(logstd)) / (2*n_b)   # ignoring -0.5tT*n_y*log(2π)
end

# d \times   T
function _gauss_nllh_individual_bybatch(y::AbstractArray{T,3}, mu::AbstractArray{T,2}, logstd::AbstractArray; dims::Int=0) where T
    n_y, tT, n_b = size(y)
    Δ = (y .- mu) ./ exp.(logstd)
    (sum(Δ.*Δ; dims=(1,2))[:] .+ tT*mean(logstd)) / 2  # ignoring -0.5tT*n_y*log(2π)
end

# MaskedArray variants (mask out NaNs with zero for all gradient calcs, and change denom)
function _gauss_nllh_batchavg(y::MaskedArray{T,3}, mu::AbstractArray{T,3}, logstd::AbstractArray) where T
    n_y, tT, n_b = size(y)
    Δ = (y.data - mu) ./ exp.(logstd)
    Δ = Δ .* y.mask
    avg_nonzero = sum(y.mask) / n_y   # ≈ tT * n_b, but counting only observed vals.
    (sum(Δ.*Δ) + avg_nonzero*sum(logstd)) / (2*n_b)   # ignoring -0.5tT*n_y*log(2π)
end

function _gauss_nllh_individual_bybatch(y::MaskedArray{T,3}, mu::AbstractArray{T,N}, logstd::AbstractArray; dims::Int=0) where {T, N}
    n_y, tT, n_b = size(y)
    Δ = (y.data .- mu) ./ exp.(logstd)
    Δ = Δ .* y.mask
    num_nonzero = sum(y.mask, dims=(1,2))[:] / n_y
    # length(logstd) is either 1 or n_y depending on constructor
    (sum(Δ.*Δ; dims=(1,2))[:] .+ num_nonzero*mean(logstd)) / 2  # ignoring -0.5tT*n_y*log(2π)
end



# ======== Objective functions: NLLH, ELBO ==================================

function nllh(m::MTDSModel, y, u; kl_coeff=1.0f0, stochastic=true,
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


function elbo_w_kl(m::MTDSModel, y::AbstractArray, u::AbstractArray, data_enc::AbstractArray;
    kl_coeff=1.0f0, stochastic=true, logstd_prior=nothing)
    nbatch = size(y,3)
    zsmp, zμ, zσ = encode(m, data_enc; stochastic=stochastic)[1]
    ŷ = m(zsmp, u)
    nllh = _gauss_nllh_batchavg(y, ŷ, m.logstd)
    kl = stochastic * kl_coeff * (-0.5f0 * sum(1 .+ 2*log.(zσ) - zμ.*zμ - zσ.*zσ)) / nbatch

    # Prior regularization of emission stdev.
    if !(logstd_prior === nothing)
        nllh += sum(x->x^2, (m.logstd .- logstd_prior[1])./logstd_prior[2])/2
    end
    nllh + kl, kl
end

elbo_w_kl(m::MTDSModel, y::AbstractArray, u::AbstractArray; kl_coeff=1.0f0, stochastic=true,
        logstd_prior=nothing) = (elbo_w_kl(m, y, u, vcat(y, u); kl_coeff=kl_coeff,
        stochastic=stochastic, logstd_prior=logstd_prior))

elbo_w_kl(m::MTDSModel, y::MaskedArray, u::AbstractArray; kl_coeff=1.0f0, stochastic=true,
        logstd_prior=nothing) = (elbo_w_kl(m, y, u, vcat(stacked(y), u); kl_coeff=kl_coeff, 
        stochastic=stochastic, logstd_prior=logstd_prior))

elbo_w_kl(m::MTDSModel, y::AbstractArray, u::MaskedArray; kl_coeff=1.0f0, stochastic=true,
    logstd_prior=nothing) = error("ELBO not implemented yet for when `u` has missing values." *
    " No current models accept missing inputs -- try passing through `stacked(u)` as inputs.")

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
elbo(m::MTDSModel, y, u; kl_coeff=1.0f0, stochastic=true, logstd_prior=nothing) =
    elbo_w_kl(m, y, u; kl_coeff=kl_coeff, stochastic=stochastic, logstd_prior=logstd_prior)[1]


# ======== Objective function: Monte Carlo Objective =============================

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
function aggregate_importance_smp(m::MTDSModel, y::AbstractArray{T,3}, u::AbstractArray{T,3}=zeros(T, size(y)[1,2]..., 1);
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

function _aggregate_importance_smp_noamortize(m::MTDSModel, y::AbstractArray{T,3}, u::AbstractArray{T,3}, Z::AbstractArray{T,2};
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

function _importance_smp_individual(m::MTDSModel, y::AbstractArray{T,2},
        u::AbstractArray{T,2}, z::AbstractArray{T,2}; tT=80, M=200, maxbatch=200) where T
    yhats = forward_multiple_z(m, z, u; maxbatch=maxbatch)    # n_y × tT × n_b

    llh_by_z = - _gauss_nllh_individual_bybatch(yhats, y, m.logstd)
    W, lse = util.softmax_lse!(unsqueeze(llh_by_z, 2); dims=1)
    logp_y = sum(lse) - log(M)  # W, log(1/M sum_j exp(log_j))
    return vec(W), logp_y
end



################################################################################
##                                                                            ##
##                              Training Loops                                ##
##                                                                            ##
################################################################################



Base.@kwdef mutable struct training_params_elbo
    nbatch::Int = 40
    niter::Int = 20
    kl_anneal::Float32 = 1f-4
    kl_pct::Float32 = 0f-4
    hard_em_epochs::Int = 5
    kl_mult::Float32 = 1.0f0
    opt::Any = ADAM(3e-3)
    lr_decay::Float64 = 0.99
end

Base.@kwdef mutable struct training_params_mco
    nbatch::Int = 40
    niter::Int = 20
    samples::Int = 500
    m_resamp::Int = 1
    opt::Any = ADAM(3e-3)
    lr_decay::Float64 = 0.99
end



function train_mco!(m, Ys, Us, nepochs; opt_pars=training_params_mco(), tT=size(Ys,2),
        ps=Flux.params(m), verbose=true, logstd_prior=nothing, maxbatch=200)
    has_inputs = check_inputs_outputs(m, Ys, Us)

    # untracked (no-AD) version of model
    m_u = mapleaves(Tracker.data, m)

    # useful shorthand
    M = opt_pars.samples
    M_rsmp = opt_pars.m_resamp
    N = size(Ys, 3)
    Nb = opt_pars.nbatch
    mco_warn = true

    history = ones(nepochs) * NaN
    for ee in 1:nepochs
        epoch_loss, epoch_logp = 0, 0
        start_time = time()

        for i in 1:opt_pars.niter
            ixs = rand(1:N, Nb)
            y = selectdim(Ys, ndims(Ys), ixs) # batch = final dim.
            u_is = has_inputs ? selectdim(Us, ndims(Us), ixs) : Us
            u_fwd = has_inputs ? u_is : repeat(u_is, outer=(1,1,Nb*M_rsmp))

            # Get posterior samples via Importance Sampling
            # suppresswarn = if Us vary per batch (i.e. has inputs), warn to say this is bad the 1st time.
            Z, W, logp = aggregate_importance_smp(m_u, y, u_is; tT=tT, M=M, suppresswarn=!mco_warn, maxbatch=maxbatch)
            smps = [util.categorical_sampler(W[:, i], M_rsmp) for i ∈ 1:Nb]  |> util.vflatten
            zsmp = Z[:, smps]

            # Get reconstruction error for these samples
            ŷ = m(zsmp, u_fwd)
            y_rep = repeat(y, inner=(1,1,M_rsmp))
            loss = nllh_of_recon_batchavg(m, y_rep, ŷ, m.logstd)

            # Regularization of stdev (if specified -- can be useful for stabilization nr maximum)
            if !(logstd_prior === nothing)
                loss += sum(x->x^2, (m.logstd .- logstd_prior[1])./logstd_prior[2])/2
            end

            epoch_logp += sum(logp) / Nb
            epoch_loss += Tracker.data(loss)

            # Take gradient
            Tracker.back!(loss)
            for (p_i, p) in enumerate(ps)
                Tracker.update!(opt_pars.opt, p, Tracker.grad(p))
            end
            mco_warn = false
        end
        opt_pars.opt.eta *= opt_pars.lr_decay

        # Logging
        history[ee] = epoch_logp / opt_pars.niter
        etime = time() - start_time
        verbose && (printfmtln("Epoch {:04d} Loss: {:.2f}, logp: {:.2f} ({:02.1f}s)",
            ee, epoch_loss / opt_pars.niter, epoch_logp / opt_pars.niter, etime); flush(stdout))
    end
    history
end


function train_elbo!(m, Ys, Us, nepochs; opt_pars=training_params_elbo(), tT=size(Ys,2),
        T_enc=tT, ps=Flux.params(m), verbose=true, logstd_prior=nothing)
    has_inputs = check_inputs_outputs(m, Ys, Us)

    # useful shorthand
    β_kl = opt_pars.kl_mult
    N = size(Ys, 3)
    Nb = opt_pars.nbatch

    history = ones(nepochs) * NaN
    for ee in 1:nepochs
        epoch_loss, epoch_kl = 0, 0
        start_time = time()
        is_hard_em = (opt_pars.hard_em_epochs > 0)

        for i in 1:opt_pars.niter
            ixs = rand(1:N, Nb)
            y = selectdim(Ys, ndims(Ys), ixs) # batch = final dim.
            u = has_inputs ? selectdim(Us, ndims(Us), ixs) : repeat(Us, outer=(1,1,Nb))

            # Calculate ELBO
            loss, kl = elbo_w_kl(m, y, u;
                kl_coeff = β_kl * opt_pars.kl_pct,
                stochastic = !is_hard_em,
                logstd_prior = logstd_prior)
            
            # loss += 0.01*sum(abs, m.hphi.W)   # regularization of MT network

            # add any new VI posteriors (never before seen sequences) into Params
            is_amortized(m) || (ps = Flux.params(ps, Flux.params(m.mt_enc)));

            # update KL annealing
            is_hard_em && (opt_pars.kl_pct = min(opt_pars.kl_pct + opt_pars.kl_anneal, 1.0f0))

            epoch_loss += loss.data
            epoch_kl += kl.data

            # Take gradient
            Tracker.back!(loss)
            for (p_i, p) in enumerate(ps)
                Tracker.update!(opt_pars.opt, p, Tracker.grad(p))
            end
        end
        opt_pars.opt.eta *= opt_pars.lr_decay
        opt_pars.hard_em_epochs = max(opt_pars.hard_em_epochs -1, 0)

        # Logging
        history[ee] = epoch_loss
        etime = time() - start_time
        logstd_penalty = opt_pars.niter*tT*sum(Tracker.data(m.logstd)) / 2
        verbose && (printfmtln("Epoch {:04d} Loss: {:.2f}, kl: {:.2f}, " *
            "p_logstd: {:.2f} ({:02.1f}s)",
            ee, epoch_loss  / opt_pars.niter, epoch_kl  / opt_pars.niter, logstd_penalty  / opt_pars.niter, etime); flush(stdout))
    end
    history
end
