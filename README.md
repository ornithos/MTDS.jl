# MTDS.jl: Multi Task Dynamical Systems implementation

This is a consolidated implementation of my multi-task dynamical system (MTDS) models developed during my phd. The original implementations are spread-out over a variety of repos and projects, but these are not easily extricated, nor can they easily include all features discussed in the thesis.

*Placeholder for link to thesis*

### Getting Started
This repo is a work-in-progress, and has not reached completion yet. If you're looking to start playing around with these models, I suggest beginning with the [example MT-LDS notebook](./notebooks/mtlds-example.ipynb) which provides a deep dive into some examples of Multi-Task Linear Dynamical Systems on synthetic data. This roughly follows chapter 4 in the thesis, although the experimental setup and benchmark models will not be available here. This is in a (messy) private repo at the moment; if you're interested, please raise an issue :).


### Rough organisation

* `super.jl`. This contains the MTDS supertype with the interface defined mostly by a bunch of placeholders.
* `subtype-lds.jl` defines the MT-LDS model type including the  `struct`, constructor, encoding networks, forward pass and encoding methods. The internals can be found in `core-mtlds.jl`, which includes the Cayley parameterisation of the transition matrix.
* `subtype-rnn.jl` defines the MT-GRU model type with the same methods defined as the MT-LDS. Note the addition of an encoder for the initial `x0` dynamic state too. (I'm afraid the basic MT-RNN is not yet implemented, although this will be simpler than the MT-GRU, naturally). The MT-GRU cell is defined in the file `core-mtrnn.jl`, along with a few variants, and a MT-Dense network too, required in case one wishes to modulate the emission network.
* `subtype-mtpd.jl` defines the MT pharmacodynamic (MT-PD) model with the same methods defined as above. 
* `objective.jl` defines a wrapper for performing optimization either via the ELBO or via MCO methods. This includes a variety of loss functions which are called by the wrapper, specialised for different arguments.

While all types have the suffix `_variational`, they need not be learned variationally (i.e. one can use a MCO approach and ignore the encoder). Nevertheless, it is neater to include a variational encoder in all cases, and simply ignore it when not needed.


### Future development
While I fully intend to continue development of this package, I concede even at this stage that it may fall through the cracks: I'm already busy with other things. Therefore, if something isn't clear, or appears to be missing, please raise an issue. This will provide me an incentive to continue to refactor/organise everything.
