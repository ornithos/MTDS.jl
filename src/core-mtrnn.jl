
const D_IM = 32


################################################################################
##                                                                            ##
##              Define a basic MTGRU model (with no inputs)                   ##
##                                                                            ##
################################################################################


Flux.gate(x::AbstractArray{T,3}, h, n) where T = x[gate(h,n),:,:]
Flux.gate(x::AbstractArray{T,4}, h, n) where T = x[gate(h,n),:,:,:]


mutable struct MTGRU{F}
    d_input::Int
    N::Int
    G::F
end

"""
    MTGRU(N, G)
Multi-task GRU (Gated Recurrent Unit) model. Produces a GRU layer
for each value of `z` ∈ Rᵈ which depends on the map `G` inside
the MTGRU. d_input = in dimension, N = out dimension, G: Rᵈ->R^{nLSTM}. z can be batched
as `z` ∈ R^{d × nbatch}.
"""
function (m::MTGRU)(z, h=nothing)
    d, d_x, nB = m.N, m.d_input, size(z, 2)
    λ = m.G(z)
    Wh = reshape(λ[1:3*d*(d+d_x),:], d*3, d+d_x, nB)
    b = λ[3*d*(d+d_x)+1:3*d*(d+d_x)+3*d, :]
    open_forget_gate = zero(b)   # no backprop, and hence not tracked, even if `b` is.
    open_forget_gate[gate(d, 2), :] .= 1
    b += open_forget_gate
    h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

    Flux.Recur(BatchedGRUCell(Wh, b, h))
end

Flux.@treelike MTGRU
Base.show(io::IO, l::MTGRU) =
  print(io, "Multitask-GRU(state=", l.N, ", in=", l.d_input, ", ", typeof(l.G), ")")


"""
  MTGRU_fixb(N, G, b)
Multi-task GRU (Gated Recurrent Unit) model. Produces a GRU layer
for each value of `z` ∈ Rᵈ which depends on the map `G` inside
the MTGRU. N = out dimension, G: Rᵈ->R^{nLSTM}. z can be batched
as `z` ∈ R^{d × nbatch}. This version of the `MTGRU` does not
adapt the offset vector `b` in the GRU, but learns a fixed value
for all `z`.
"""
mutable struct MTGRU_fixb{F,V}
  d_input::Int
  N::Int
  G::F
  b::V
end

"""
  MTGRU_fixb_v2(N, G, b)
As per `MTGRU_fixb`, except now only the *state* bias is fixed across all
tasks. The gate biases depend on `z` just like the `MTGRU`.
"""
mutable struct MTGRU_fixb_v2{F,V}
  d_input::Int
  N::Int
  G::F
  b::V
end

function (m::MTGRU_fixb)(z, h=nothing)
  d, d_input, nB = m.N, m.d_input, size(z, 2)
  λ = m.G(z)
  Wh = reshape(λ, d*3, d+d_input, nB)
  b = m.b
  open_forget_gate = zero(b)   # no backprop, and hence not tracked, even if `b` is.
  open_forget_gate[gate(d, 2), :] .= 1
  b += open_forget_gate
  h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

  Flux.Recur(BatchedGRUCell(Wh, b, h))
end

function (m::MTGRU_fixb_v2)(z, h=nothing)
    d, d_input, nB = m.N, m.d_input, size(z, 2)
    λ = m.G(z)
    Wh = reshape(λ[1:d*d*3,:], d*3, d+d_input, nB)
    gmove = Flux.has_cuarrays() && Tracker.data(z) isa Flux.CuArray ? gpu : identity
    expander = gmove(ones(eltype(Tracker.data(λ)), 1, nB))
    b = vcat(λ[d*d*3+1:d*d*3+d*2, :], m.b * expander)
    open_forget_gate = zero(b)
    open_forget_gate[gate(d, 2), :] .= 1
    b += open_forget_gate
    h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

    Flux.Recur(BatchedGRUCell(Wh, b, h))
end

Flux.@treelike MTGRU_fixb
Base.show(io::IO, l::MTGRU_fixb) =
print(io, "Multitask-GRU-fixb(", l.N, ", ", typeof(l.G), ", bias vector: ", typeof(l.b), ")")

Flux.@treelike MTGRU_fixb_v2
Base.show(io::IO, l::MTGRU_fixb_v2) =
print(io, "Multitask-GRU-fixb_v2(", l.N, ", ", typeof(l.G), ", bias vector (state only): ", typeof(l.b), ")")


"""
    BatchedGRUCell(Wh, b, h)
Multi-task GRU Cell (which takes no input).
`Wh`, `b`, `h` are the (concatenated) transformation matrices,
offsets, and initial hidden state respectively.

Each time the cell is called (no arguments) it will perform a one-
step evolution of the hidden state and return the current value.
The cell is implemented in batch-mode, and the final dimension of
each quantity is the batch index.
"""
mutable struct BatchedGRUCell{A,W}
    Wh::A
    b::W
    h::W
end


(m::BatchedGRUCell)(h, x) = _batchgrucell(m, h, x)

function _batchgrucell(m::BatchedGRUCell, h::AbstractArray, u::AbstractArray)
  b, o = m.b, size(m.Wh, 1) ÷ 3
  gh = batch_matvec(m.Wh, vcat(h, u))
  r = σ.(gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(r .* gate(gh, o, 3) .+ gate(b, o, 3)) 
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

Flux.@treelike BatchedGRUCell
Base.show(io::IO, l::BatchedGRUCell) =
  print(io, "Batched GRU Cell(dim=", size(l.Wh,2), ", batch=", size(l.Wh,3), ")")
Flux.hidden(m::BatchedGRUCell) = m.h



################################################################################
##                                                                            ##
##                        Objective(s) / Loss Function(s)                     ##
##            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            ##
##*~*~*~*~*~*~These are yet to be properly integrated with the pkg*~*~*~*~*~*~##
##                                                                            ##
################################################################################

# function online_inference_BCE(m::MTSeqModels, x0::AbstractVecOrMat, z::AbstractVecOrMat,
#     y::AbstractVector; T_steps=length(y))
#
#     x0decoder, mtbias_decoder, gen_model, dec_tform = unpack_generative(m)
#
#     # Generative model
#     # --------------------------------------------------
#     if m.mtbias_only
#         gen_model.state = x0decoder(x0)
#         mtbias = mtbias_decoder(z)   # if linear, this is just the identity ∵ GRUCell.Wi
#
#         nllh = map(1:T_steps) do t
#             _nllh_bernoulli(dec_tform(gen_model(mtbias)), y[t])  # y broadcasts over the batch implicitly
#         end
#     else
#         posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
#         posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch
#
#         nllh = map(1:T_steps) do t
#             _nllh_bernoulli(dec_tform(posterior_grus()), y[t])  # y broadcasts over the batch implicitly
#         end
#     end
#     return reduce(vcat, nllh)
# end
#
# # Allow signature for `E3` model to call with separate z, c vars..
# online_inference_BCE(m::MTSeqModel_E3, x0::AbstractVecOrMat, z::AbstractVecOrMat, c::AbstractVecOrMat,
#     y::AbstractVector; T_steps=length(y)) = online_inference_BCE(m, x0, vcat(z,c), y; T_steps=T_steps)
#
#
#
# function online_inference_single_BCE(m::MTSeqModels, h0::AbstractVecOrMat, z::AbstractVecOrMat,
#     y::AbstractArray)
#
#     mtbias_decoder, gen_model, dec_tform = unpack(m)[9:11]
#
#     # Generative model
#     # --------------------------------------------------
#     if m.mtbias_only
#         gen_model.state = h0
#         mtbias = mtbias_decoder(z)
#         h_new = gen_model(mtbias)
#         nllh = _nllh_bernoulli(dec_tform(h_new), y)
#     else
#         posterior_grus = gen_model(z) # output: BatchedGRU (def. above)
#         posterior_grus.state = h0  # 2×linear d_x0 × nbatch → d_x × nbatch
#         h_new = posterior_grus()
#         nllh = _nllh_bernoulli(dec_tform(h_new), y)
#     end
#     return vec(nllh), h_new
# end
#
# # Allow signature for `E3` model to call with separate z, c vars..
# online_inference_single_BCE(m::MTSeqModel_E3, h0::AbstractVecOrMat, z::AbstractVecOrMat,
#     c::AbstractVecOrMat, y::AbstractArray) = online_inference_single_BCE(m, h0, vcat(z,c), y)
#
#
# function _nllh_bernoulli(ŷ::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractArray{M,N} where {M,N}
#     @assert N > 1 "Avoiding infinite recursion. " *
#     "Current implementation does not permit nllh of bernoulli vector of vectors. Try unsqueezing."
#     return [_nllh_bernoulli(ŷŷ, yy) for (ŷŷ, yy) in zip(ŷ, y)]
# end
#
# _nllh_bernoulli(ŷ::AbstractArray, y::AbstractArray) =
#     let n=size(ŷ)[end]; res=Flux.logitbinarycrossentropy.(ŷ, y); sum(reshape(res, :, n), dims=1); end
#
# _nllh_gaussian(ŷ_lσ::AbstractVector, y::AbstractVector) =
#     [((ŷŷ, ll) = ŷl; δ=yy-ŷŷ; sum(δ.*δ./(exp.(2*ll)))/2+sum(ll)/2) for (ŷl, yy) in zip(ŷ_lσ, y)]
#
# _nllh_gaussian_constvar(ŷ::AbstractVector, y::AbstractVector, logstd::Number) =
#     [(let n=size(ŷ)[end]; δ=yy-ŷŷ; res=sum(δ.*δ./(exp.(2*logstd)))/2+sum(logstd)/2;
#     sum(reshape(res, :, n), dims=1); end) for (ŷŷ, yy) in zip(ŷ, y)]
#
# function nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector=[];
#     T_steps=70, T_enc=10, stoch=true) where {U,A,B,V,W,S,N,M <: MTSeq_CNN}
#     x0_z_c = posterior_samples(m, y, yfull; T_enc=T_enc, stoch=stoch)[1]
#     online_inference_BCE(m::MTSeqModels, x0_z_c..., y; T_steps=T_steps)
# end
#
# nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
#     T_steps=70, T_enc=10, stoch=true, logstd=-2.5) where {U,A,B,V,W,S,N,M <: MTSeq_Single} =
#         vcat(_nllh_gaussian_constvar(m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch), y, logstd)...)
#
# nllh(m::MTSeqModels{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
#     T_steps=70, T_enc=10, stoch=true) where {U,A,B,V,W,S,N,M <: MTSeq_Double} =
#         vcat(_nllh_gaussian(m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch), y)...)
#
# nllh(::Type{MTSeq_CNN}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_bernoulli(ŷ, y)
# nllh(::Type{MTSeq_Single}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian_constvar(ŷ, y, logstd)
# nllh(::Type{MTSeq_Double}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian(ŷ, y)
#
#
# StatsBase.loglikelihood(m::MTSeqModels, y::AbstractVector, yfull::AbstractVector;
#     T_steps=70, T_enc=10, stoch=true, logstd=-2.5) = -nllh(m, y, yfull, T_steps=T_steps,
#         T_enc=T_enc, stoch=stoch, logstd=logstd)
#
#
# function _gauss_kl_to_N01(μ::AbstractVecOrMat{T}, σ::AbstractVecOrMat{T}) where T
#     T(0.5) * sum(1 .+ 2*log.(σ.*σ) - μ.*μ - σ.*σ)  # x.^2 -> x.*x due to old version of CuArrays / GPU (culiteral pow issue)
# end
# _gauss_kl_to_N01_deterministic(μ::AbstractVecOrMat{T}) where T  = T(0.5) * sum(1 .- μ.*μ)
#
# function _kl_penalty_stoch(β::Vector{T}, μs::Vector{T}, σs::Vector{T}) where T <: Float32
#     @argcheck length(β) == length(μs) == length(σs)
#     kl = zero(T)
#     for j in 1:length(β)
#         kl += β[j] * _gauss_kl_to_N01(μs[j], σs[j])
#     end
#     return kl
# end
#
# function _kl_penalty_deterministic(β::Vector{T}, μs::Vector{T}) where T <: Float32
#     @argcheck length(β) == length(μs)
#     kl = zero(T)
#     for j in 1:length(β)
#         kl += β[j] * _gauss_kl_to_N01_deterministic(μs[j])
#     end
#     return kl
# end
#
#
# function elbo(m::MTSeqModels, y::AbstractVector, yfull::AbstractVector=[]; T_steps=70,
#     T_enc=10, stoch=true, kl_coeff=1f0, βkl=ones(Float32, 3))
#
#     smps, μs, σs = posterior_samples(m, y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)
#     model_out = m(x0, vcat(z, c); T_steps=T_steps)
#     recon = -nllh(M, model_out, y)
#
#     # ⇒ Initially KL (Hard EM), and then an annealing sched. cf. Bowman et al. etc?
#     kl = stoch ? _kl_penalty_stoch(β, μs, σs) : _kl_penalty_deterministic(β, μs)
#     kl = kl * kl_coeff
#
#     return - (recon + kl)
# end
