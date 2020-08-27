module MTDS

using LinearAlgebra, StatsBase, Distributions
using Flux   # must be Flux@0.90
using BSON, YAML, ArgCheck, Dates, Formatting

export create_mtlds, MTLDS_variational, create_mtpd, MTPD_variational

include("util.jl")
include("modelutils.jl")
using .util
using .modelutils


module model

    using ..LinearAlgebra
    using ..Flux, ..ArgCheck, ..StatsBase, ..YAML
    using ..Flux: Tracker, gate
    using ..Flux.Tracker: istracked, @grad
    using ..Distributions, Formatting

    import ..save, ..load!
    import ..modelutils: randn_repar, posterior_sample, LookupTable, PartialSplit, MultiDense, mlp, 
        get_final_layer_dim, get_first_layer_dim
    using ..util
    import ..util: unsqueeze, get_strtype_wo_params, MaskedArray, stacked

    export load!, forward, encode, reconstruct, elbo, elbo_w_kl,
           MTLDS_variational, create_mtlds, 
           MTPD_variational, create_mtpd
           
    # Supertype
    include("super.jl")

    # Subtypes
    include("core-mtlds.jl")
    include("subtype-lds.jl")
    include("subtype-mtpd.jl")
    include("objective.jl")
    # include("core-mtrnn.jl")
    # include("mtrnn-dblpend.jl")   # => refactor later
end

using .model


end # module
