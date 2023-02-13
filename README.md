# DSV_Examples.jl
These codes may be used to complete the computer-assisted proofs from https://www.researchgate.net/publication/365613446_Computer-assisted_proofs_with_deep_neural_networks.

## Installation
This package requires python with TensorFlow to be installed, in addition to Julia v1.6 or higher. It makes use of [PyCall](https://github.com/JuliaPy/PyCall.jl) to run TensorFlow commands in order to import the models from two sources:

- [Deep-Conjuacies](https://github.com/jbramburger/Deep-Conjugacies), of which data is included included with permission at this repository in the saved_models folder. 

- A model trained with the feedforward architecture described at [doi.org/10.1029/2019MS001965](doi.org/10.1029/2019MS001965) with data from the Zenodo repository cited therein.

Usage of PyCall is platform-specific; consult the relevant documentation to ensure PyCall is working correctly. Additionally, a TensorFlow installation is required.

To install, clone this repository, and activate/instantiate the package as follows:
```julia
import Pkg
Pkg.activate("path/to/package") # edit the path accordingly
Pkg.instantiate()
```

## Usage
To complete the proofs described in Section 3 of the paper, first import the package.

```julia
using DSV_Examples
```

### Batch proofs: Conjugacies

To complete one of the "batch" proofs from Section 3.2--3.5, the following sequence of commands should be completed.

First, import the model. This will output a bunch of tensorflow information (possibly also warnings) to the REPL; this is normal.
```julia
model_name = "Name" # set this this to one of "Rossler", "Lorenz2D", "Kuramoto1D" or "MackeyGlass"
model_path = ".../saved_models/Name" # edit the path accordingly, make sure the model matches with model_name
eW,eB,dW,dB,gcoeffs = import_conjugacy_model(model_path,model_name)
```

Next, generate the encoder, decoder and conjugate mapping, and import the periodic points / dimension of the conjugate mapping.
```julia
enc,dec,g = h_g_functions(eW,eB,dW,dB,gcoeffs,model_name)
in_points,dim = data(model_name)
```

Finally, run the batch proofs.
```julia
_,_,_,norm_correction_inner,norm_correction_outer,r_inner,r_outer = proofs_conjugacies(enc,dec,g,in_points,r★,dim)  # set r★ as in the paper.
```

### Sequential proof of Section 3.1.5 and r★ optimization
First, load the relevant data.
```julia
model_name = "Rossler"
model_path = "path/to/model/folder" # edit the path to the Rossler c=11 model folder.
eW,eB,dW,dB,gcoeffs = import_conjugacy_model(model_path,model_name)
enc,dec,g = h_g_functions(eW,eB,dW,dB,gcoeffs,model_name)
in_points,dim = data(model_name)
```

Next, to complete the proof for the period 6 point using the six-dimensional formulation (with minimal composition), run 
````julia
_,_,norm_correction,r = proof_conjugacy_sequential_1D(enc,dec,g,in_points[6],1E-8)
````

To optimize r★=10⁻ⁿ over n≥2 for the outer Poincare map, run 
```julia
_,_,norm_correction,r,n = best_r★_proof_conjugacy_1D_outer(enc,dec,g,in_points[6])
```

### Sea surface height interpolation examples
```julia
model_path = ".../saved_models/surface_sea_height_1536"; #edit the path accordingly
model = import_ssh_interpolation_model(model_path,Interval{BigFloat});
r = proofs_ssh(model)
