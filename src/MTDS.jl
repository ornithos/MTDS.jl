module MTDS

using LinearAlgebra, StatsBase, Distributions
using Flux   # must be Flux@0.90
using BSON, YAML, ArgCheck, Dates, Formatting

export create_mtlds, MTLDS_variational

include("util.jl")
include("modelutils.jl")
using .util
using .modelutils


module model

    using ..LinearAlgebra
    using ..Flux, ..ArgCheck, ..StatsBase, ..YAML
    using ..Flux: Tracker
    using ..Flux.Tracker: istracked, @grad
    using ..Distributions, Formatting

    import ..save, ..load!
    import ..modelutils: randn_repar, posterior_sample, MultiDense, mlp
    using ..util
    import ..util: unsqueeze, get_strtype_wo_params

    export load!, create_mtlds, MTLDS_variational,
           forward, encode, reconstruct, elbo, elbo_w_kl

    # Supertype
    include("super.jl")

    # Subtypes
    include("core-mtlds.jl")
    include("subtype-lds.jl")
    # include("core-mtrnn.jl")
    # include("mtrnn-dblpend.jl")   # => refactor later
end

using .model


end # module
