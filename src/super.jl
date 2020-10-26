abstract type MTDSModel end

# Common interface
is_amortized(m::MTDSModel) = error(format("It's not known whether {:s} uses amortized inference. "*
 "Please implement `is_amortized(m)` for this type.", get_strtype_wo_params(m)))
has_x0_encoding(m::MTDSModel) = error(format("It's not known whether {:s} uses an x0 (recurrent "*
 "state) encoder. Please implement `has_x0_encoding(m)` for this type.", get_strtype_wo_params(m)))

forward(m::MTDSModel, x0, z, U; T_steps=size(U, 2)) =
    error(format("Not Implemented Yet: {:s}. Please implement a forward proc (given z).", string(typeof(m))))
encode(m::MTDSModel, Y, U; kl_coeff=1f0, stochastic=true, logstd_prior=nothing, T_enc=size(U, 2)) =
    error(format("Not Implemented Yet: {:s}. Please implement an encode proc.", string(typeof(m))))

function reconstruct(m::MTDSModel, Y, U; T_steps=size(Y, 2), T_enc=T_steps, stochastic=true)
    @argcheck size(Y,2) == size(U,2)
    (smpz, μ_z, σ_z), (smpx0, μ_x0, σ_x0) = encode(m, Y, U; T_enc=T_enc, stochastic=stochastic)
    forward(m, smpx0, smpz, U; T_steps=T_steps)
end

function forward_multiple_z(m::MTDSModel, z, u::AbstractMatrix; maxbatch=200)
    M = size(z, 2)
    ŷs = map(Iterators.partition(1:M, maxbatch)) do cbatch
        u_rep = repeat(u, outer=(1, 1, length(cbatch)));
        m(z[:, cbatch], u_rep)
    end
    cat(ŷs..., dims=3)   # problematic if length(ŷs) is too large, but there's no nice `reduce` version in Julia yet.
end
forward_multiple_z(m::MTDSModel, z, u::AbstractVector; maxbatch=200) = forward_multiple_z(m, z, unsqueeze(u,1); maxbatch=maxbatch)

do_importance_smp(m::MTDSModel, y, u, tT=80, M=200) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))

nllh(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
nllh_of_recon_batchavg(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
elbo(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
elbo_w_kl(m::MTDSModel, args...; kwargs...) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))


check_inputs_outputs(m::MTDSModel, Ys, Us) = error(format("Not implemented yet for model type: {:s}", get_strtype_wo_params(m)))
