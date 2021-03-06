module modelutils

using ..Flux   # must be Flux@0.90
using ..BSON
using ..ArgCheck, ..Dates

export save, load!


if Flux.has_cuarrays()
    using CuArrays
    # note I'm using cu(zeros(...)), cu(randn(...)), CuArrays.randn(...) in particular occasionally
    # (apparently non-deterministically) kept bugging out on the GPU side :/.
    randn_repar(σ::CuArray, n, d, stochastic=true) = !stochastic ? cu(zeros(Float32, n, d)) : σ .* cu(randn(Float32, n,d))
    randn_repar(σ::TrackedArray{Float32, N, CuArray{Float32,N}}, n, d, stochastic=true) where N =
        !stochastic ? cu(zeros(Float32, n, d)) : σ .* cu(randn(Float32, n,d))
end

# Add a third channel to a 2-channel image Tensor
chan3cat(x::AbstractArray{T,4}) where T = cat(x, zero(x)[:,:,1:1,:], dims=3)
chan3cat(x::AbstractArray{T,3}) where T = cat(x, zero(x)[:,:,1:1], dims=3)


# get_final_layer_dim from an arbitrary Chain of layers.
# This was created for pretty printing (Base.show) where n_y was unavailable in MT-LDS.
get_final_layer_dim(l) = nothing
get_final_layer_dim(l::Int) = l
get_final_layer_dim(l::Chain) = get_final_layer_dim(l.layers[end])
get_final_layer_dim(l::Dense) = length(l.b)

get_first_layer_dim(l) = nothing
get_first_layer_dim(l::Int) = l
get_first_layer_dim(l::Chain) = get_first_layer_dim(l.layers[1])
get_first_layer_dim(l::Dense) = size(l.W, 2)

"""
    LookupTable(d, mu_sd, logstd_sd, lkp)

A lookup table (Wrapped Dict) for use instead of an encoder for use in standard variational
inference. The lookup table maps training data (arrays) to mean/logstd Flux.params for
a Gaussian variational posterior. Where a queried array does not have an entry, LookupTable
creates one using standard Gaussian variates with `mu_sd` standard deviation for the mean
and `logstd_sd` standard deviation for the logstd.
"""
struct LookupTable
    d::Int
    mu_sd::Float32
    logstd_sd::Float32
    lkp::Dict
end
LookupTable(d, mu_sd=0.001f0, logstd_sd=-1f0) = LookupTable(d, mu_sd, logstd_sd, Dict())
function Flux.mapleaves(f, s::LookupTable)
    for (k,v) in s
        s[k] = f(v)
    end
end

(lkp::LookupTable)(x::Flux.TrackedArray) = error("Unable to accept tracked arrays in LookupTable.")
function (lkp::LookupTable)(x::AbstractVecOrMat{T}) where T
    # seems inefficient to *always* allocate the default value, but `get` doesn't support
    # lazy evaluation (so it seems), and below is *MUCH* faster than using try/catch which
    # adds O(100μs) to what is otherwise a O(1μs) operation.
    def_value = (Flux.param(randn(T, lkp.d, 1)*lkp.mu_sd), Flux.param(ones(T, lkp.d, 1)*lkp.logstd_sd))
    mu_sd = get!(lkp.lkp, vec(x), def_value)
    return (mu_sd[1], σ.(mu_sd[2]))   # pass 'logstd' (!) through sigmoid
end

function (lkp::LookupTable)(x::AbstractArray{T, 3}) where T
    mu_sds = [lkp(x[:,:,i]) for i in 1:size(x, 3)]
    return hcat([mu for (mu, sd) in mu_sds]...), hcat([sd for (mu, sd) in mu_sds]...)
end

# play nicely with Flux's params.
Flux.children(m::Iterators.Flatten) = m
Flux.children(m::Flux.Params) = m        # params is currently not idempotent: this fixes it. 
                                         # Not sure if unintended consequences though?
Flux.children(lkp::LookupTable) = () #Iterators.Flatten(lkp.lkp.vals[findall(lkp.lkp.slots .== 1)])
Flux.params(lkp::LookupTable) = Flux.params(Flux.children(lkp))


"""
    PartialSplit(ixs1, ixs2, split_on_d, fn1, fn2)

Splits an input in two on dimension ``split_on_d`` according to the two UnitRange vectors
``ix1`` and ``ix2``, and passes the result into functions ``fn1`` and ``fn2`` respectively.
"""
struct PartialSplit{V,W}
    ixs1::UnitRange
    ixs2::UnitRange
    split_on_d::Int
    fn1::V
    fn2::W
end

function (ps::PartialSplit)(x::AbstractArray)
    x1 = selectdim(x, ps.split_on_d, ps.ixs1)
    x2 = selectdim(x, ps.split_on_d, ps.ixs2)
    fx1 = ps.fn1(x1)
    fx2 = ps.fn2(x2)
    return vcat(fx1, fx2)
end

Flux.@treelike PartialSplit    # new Flux calls this '@functor'. May need to replace.

function Base.show(io::IO, l::PartialSplit)
    l1 = l.fn1 == identity ? "" : format("ixs: {:s} => {:s}", string(l.ixs1), 
        string(typeof(l.fn1)))
    l2 = l.fn2 == identity ? "" : format("ixs: {:s} => {:s}", string(l.ixs2), 
        string(typeof(l.fn2)))
    out = length(l1) == 0 ? (length(l2) == 0 ? "" : l2) : (length(l2) == 0 ? l1 : l1 * "\n" * l2)
    print(io, "PartialSplit(\n" * out * "\n)")
end

"""
    MultiDense(Dense(in_1, out_1), Dense(in_2, out_2))

2× Dense layer for convenience for variational (μ/σ) like application. Allows
one to keep both layers together, and can be called like:

    m(x)  # = (m.Dense1(x), m.Dense2(x))

for an instance `m`; the result is a Tuple.
"""
struct MultiDense{F1, F2, S, T}
    Dense1::Dense{F1,S,T}
    Dense2::Dense{F2,S,T}
end
Flux.@treelike MultiDense    # new Flux calls this '@functor'. May need to replace.
(a::MultiDense)(x::AbstractArray) = (a.Dense1(x), a.Dense2(x))

function Base.show(io::IO, l::MultiDense)
  fs = [d.σ == identity ? "" : ", " * string(d.σ) for d in [l.Dense1, l.Dense2]]
  print(io, "MultiDense((", size(l.Dense1.W, 2), ", ", size(l.Dense1.W, 1), fs[1], ")")
  print(io, ", (", size(l.Dense2.W, 2), ", ", size(l.Dense2.W, 1), fs[2], "))")
end

"""
    mlp(units::Tuple; activation::Function, final::Function)

A convenience wrapper for defining a fully connected feedforward (MLP)
component. `units` is a tuple which specifies the number of units in each layer,
with the first and last components taken to be the inputs / outputs of the MLP;
all intermediate units are hidden layers.

Example:
    mlp((20,128,128,40)) === Chain(Dense(20,128,relu), Dense(128,128,relu), Dense(128, 40))

The activation functions can be customized via `activation=...` (for hidden layers)
and `final=...` for the final layer.
"""
function mlp(units::Tuple; activation::Function=relu, final::Function=identity)
    last = length(units)-1
    layers = [Dense(i,j, l==last ? final : activation) for (l,(i,j)) in
            enumerate(zip(units[1:end-1], units[2:end]))]
    return last == 1 ? layers[1] : Chain(layers...)
end
mlp(x...; activation::Function=relu, final::Function=identity) = mlp(x, activation=activation, final=final)

"""
    BRNNenc(d_in, d_state)

Bidirectional RNN using an LSTM in each direction with `d_in` dimensional inputs
and `d_state` dimension for _each_. This is an encoder variant which only uses
the final state (of each LSTM), concatenated. Therefore, its usage is as
follows for `xs`, a Vector where each element corresponds to the input at time
``t``:

    m(xs)::Vector

for an instance `m`, where the output vector is the final state concatenation.
"""
struct BRNNenc{V}
    forward::V
    backward::V
end

BRNNenc(in::Integer, out::Integer) = BRNNenc(LSTM(in, out), LSTM(in, out))

function (m::BRNNenc)(xs::AbstractVector)
    m.forward.(xs)
    m.backward.(reverse(xs))
    return vcat(m.forward.state[1], m.backward.state[1])
end

Flux.@treelike BRNNenc

function Base.show(io::IO, l::BRNNenc)
  print(io, "Bidirectional RNN. (Forward / Backward):")
  show(l.forward); print(", "); show(l.backward)
end


################################################################################
##  Generate samples from posterior (using encoder) via reparameterization    ##
################################################################################

randn_repar(σ::AbstractArray, n, d, stochastic=true) = !stochastic ? zeros(Float32, n, d) : σ .* randn(Float32, n,d)

function posterior_sample(enc, dec, input, T_max, stochastic=true)
    Flux.reset!(enc)
    for tt = 1:T_max; enc(input[:,tt,:]); end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

function posterior_sample(enc::LookupTable, dec, input, T_max, stochastic=true)
    μ_, σ_ = enc(input)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

################################################################################
##  Save and Load utils for parameters of arbitrary models.                   ##
##                                                                            ##
##  These utils can be used for *any* models as they don't save the model def ##
##  itself, but uses the fact that the parameter 'vector' can be extracted    ##
##  easily using Flux's treelike structure and similarly loaded easily.       ##
################################################################################

"""
    save(fname::String, ps::Flux.Params, opt::Flux.ADAM; timestamp::Bool=false, force::Bool=false)
    save(fname::String, ps::Flux.Params; timestamp::Bool=false, force::Bool=false)

Save a parameter vector extracted via `Flux.params(models...)` to `fname`. in
BSON format. Since the state of the optimizer is critical for continuing
optimization (should it be required), the optimizer can optionally be given too.
At the moment this is limited to Adam, but this is easily extended.

Two options:
   (i) `timestamp = true` will insert a timestamp just before the extension.
   (ii) `force=true` will overwrite an existing file of the same name. The
        default behaviour is to throw an error instead.
"""
function save(fname::String, ps::Flux.Params, opt::Flux.ADAM; timestamp::Bool=false, force::Bool=false)
    fname = _get_and_check_fname(fname, timestamp, force)
    BSON.bson(fname, ps=[cpu(Tracker.data(p)) for p in ps], etabeta=(opt.eta, opt.beta),
                opt_state=[mapleaves(cpu, opt.state[p]) for p in ps])
end

function save(fname::String, ps::Flux.Params; timestamp::Bool=false, force::Bool=false)
    fname = _get_and_check_fname(fname, timestamp, force)
    BSON.bson(fname, ps=[cpu(Tracker.data(p)) for p in ps])
end

function _get_and_check_fname(fname::String, timestamp::Bool, force::Bool)
    fn = splitext(fname)
    @argcheck fn[2] in [".bson", ""]
    timestamp && (fname = fn[1] * Dates.format(Dates.now(), "yyyymmddTHH:MM:SS") * fn[2])
    !force && (@assert !isfile(fname) "fname already exists. To overwrite, use `force=true`.")
    return fname
end


"""
    load!(ps::Flux.Params, fname::String)
    load!(ps::Flux.Params, opt::Flux.ADAM, fname::String)

The inverse operation to save; loads a parameter vector from a BSON file created
per the `save` function. This operation will overwrite the parameter values in
the `ps` parameter 'vector' and optionally also the optimizer state (and η, β)
if specified as an argument.

The function takes an _existing_ parameter vector in order to link it to some
pre-specified models; this allows the file format to be agnostic to the original
model.
"""
function load!(ps::Flux.Params, fname::String)
    tf = Flux.has_cuarrays() && Tracker.data(first(ps)) isa CuArray ? gpu : identity
    _load_pars!(ps, BSON.load(fname)[:ps], tf)
end

function load!(ps::Flux.Params, opt::Flux.ADAM, fname::String)
    tf = Flux.has_cuarrays() && Tracker.data(first(ps)) isa CuArray ? cu : identity
    f=BSON.load(fname)
    _load_pars!(ps, f[:ps], tf)
    opt.eta, opt.beta = f[:etabeta]
    for (p, p_saved) in zip(ps, f[:opt_state])
        opt.state[p] = tf(p_saved)
    end
end


function _load_pars!(ps_to::Flux.Params, ps_from::Vector, tf::Function)
    try
        for (p, p_saved) in zip(ps_to, ps_from)
            p.data .= tf(p_saved)
        end
    catch e
        # Debugging dimension mismatch:
        if e isa DimensionMismatch
            @warn "Specified file has parameters of different dimensions:"; flush(stderr)
            for (p, p_saved) in zip(ps_to, ps_from)
                txt = ("(cur/new): ", size(p.data), " <-- ", size(p_saved) , "\n")
                if size(p.data) == size(p_saved)
                    print(txt...)
                else
                    printstyled(IOContext(stdout, :color => true), txt...; bold=true)
                end
            end
        end
        rethrow(e)
    end
end


end
