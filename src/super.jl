abstract type MTDSModel end

# Common interface

forward(m::MTDSModel, x0, z, U; T_steps=size(U, 2)) =
    error("Not Implemented Yet: {:s}. Please implement a forward proc (given z).", string(typof(m)))
encode(m::MTDSModel, Y, U; T_steps=size(Y, 2), stochastic=true) =
    error("Not Implemented Yet: {:s}. Please implement an encode proc.", string(typof(m)))

function reconstruct(m::MTDSModel, Y, U; T_steps=size(Y, 2), enc_steps=T_steps, stochastic=true)
    @argcheck size(Y,2) == size(U,2)
    (smpz, μ_z, σ_z), (smpx0, μ_x0, σ_x0) = encode(m, Y, U; T_steps=enc_steps, stochastic=stochastic)
    forward(m, smpx0, smpz, U; T_steps=T_steps)
end

forward_multiple_z(m::MTDSModel, z, u::AbstractMatrix) = (u_rep = repeat(u, outer=(1,1,size(z, 2))); m(z, u_rep))
forward_multiple_z(m::MTDSModel, z, u::AbstractVector) = (u_rep = repeat(reshape(u,1,length(u),1), outer=(1,1,size(z, 2))); m(z, u_rep))

do_importance_smp(m::MTDSModel, y, u, tT=80, M=200) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))

nllh(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
nllh_of_recon_batchavg(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
elbo(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
elbo_w_kl(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))


check_inputs_outputs(m::MTDSModel, Ys, Us) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))





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
        ps=Flux.params(m), verbose=true, logstd_prior=nothing)
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
            Z, W, logp = aggregate_importance_smp(m_u, y, u_is; tT=tT, M=M, suppresswarn=!mco_warn)
            smps = [util.categorical_sampler(W[:, i], M_rsmp) for i ∈ 1:N]  |> util.vflatten
            zsmp = Z[:, smps]

            # Get reconstruction error for these samples
            ŷ = m(zsmp, u_fwd)
            y_rep = repeat(y, inner=(1,1,M_rsmp))
            loss = nllh_of_recon_batchavg(m, y_rep, ŷ, m.logstd)

            # Regularization of stdev (if specified -- can be useful for stabilization nr maximum)
            if !(logstd_prior === nothing)
                loss += sum(x->x^2, (m.logstd .- logstd_prior[1])./logstd_prior[2])/2
            end

            epoch_logp += logp / Nb
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
        history[ee] = epoch_logp
        etime = time() - start_time
        verbose && (printfmtln("Epoch {:04d} Loss: {:.2f}, logp: {:.2f} ({:02.1f}s)",
            ee, epoch_loss, epoch_logp, etime); flush(stdout))
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

        for i in 1:opt_pars.niter
            ixs = rand(1:N, Nb)
            y = selectdim(Ys, ndims(Ys), ixs) # batch = final dim.
            u = has_inputs ? selectdim(Us, ndims(Us), ixs) : repeat(Us, outer=(1,1,Nb))

            # Calculate ELBO
            loss, kl = elbo_w_kl(m, y, u;
                kl_coeff = β_kl * opt_pars.kl_pct,
                stochastic = (opt_pars.hard_em_epochs <= 0),
                logstd_prior = logstd_prior)

            # update KL annealing
            opt_pars.kl_pct = min(opt_pars.kl_pct + opt_pars.kl_anneal, 1.0f0)

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
        verbose && (printfmtln("Epoch {:04d} Loss: {:.2f}, kl: {:.2f}, p_logstd: {:.2f} ({:02.1f}s)",
            ee, epoch_loss, epoch_kl, logstd_penalty, etime); flush(stdout))
    end
    history
end
