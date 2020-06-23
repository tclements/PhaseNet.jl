# PhaseNet
PhaseNet.jl is a Julia implementation of the Generalized Phase Detection (GPD) framework for seismic phase detection with deep learning using Pytorch. GPD uses deep convolutional networks to learn generalized representations of millions of P-wave, S-wave, and noise seismograms that can be used for phase detection and picking. The framework is described in

```
Ross, Z. E., Meier, M.-A., Hauksson, E., and T. H. Heaton (2018). Generalized Seismic Phase Detection with Deep Learning, Bull. Seismol. Soc. Am., doi: 10.1785/0120180080 [arXiv:1805.01075]
```

# Installation

You can install the latest version of PhaseNet using the Julia package manager (Press `]` to enter `pkg`). 
From the Julia command prompt:

```julia
julia>]
(@v1.4) pkg> add https://github.com/tclements/PhaseNet.jl
```

Or, equivalently, via the `Pkg` API:

```julia
julia> using Pkg; Pkg.add(PackageSpec(url="https://github.com/tclements/PhaseNet.jl", rev="master"))
```

This will install the latest version of PyTorch in your `.julia` directory and the P-wave and S-wave model weights in the PhaseNet package directory. We recommend using the latest version of PhaseNet by updating with the Julia package manager:

```julia 
(@v1.4) pkg> update PhaseNet
```

# Usage 
Here is an example of how the package can be used to detect P-waves from a set of seismic data. We will use the test data from the 2016 Mw 5.2 Anza, California sequence. For loading/downloading seismic data in Julia, we recommend using [SeisIO](https://github.com/jpjones76/SeisIO.jl).  First import the `PhaseNet` package: 

```julia
julia> using PhaseNet, SeisIO 
```

PhaseNet requires three-component seismic data. Here, we'll use test data set from the 2016 Mw 5.2 Anza, California sequence and apply minimal processing with [SeisIO](https://github.com/jpjones76/SeisIO.jl)

```julia
julia> S = load_test_data() # load HHE, HHN and HHZ component data 
julia> detrend!(S) # remove mean and linear trend 
julia> sync!(S) # synchonize channel starttimes 
```

Now convert the data to a PyTorch `tensor` object 

```julia 
julia> window_samples = 1600
julia> X = seisdata2torch(S,window_samples)
```

Here `window_samples` controls the number of samples per detection window (`1600 samples / 100 Hz = 16s` per window). 

Next load the model for p-wave detection 

```julia
julia> model_P = load_model("P") 
PyObject UNet(
  (relu): ReLU()
  (maxpool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv11): Conv1d(3, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv12): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv21): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv22): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv31): Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv32): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv41): Conv1d(256, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv42): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv51): Conv1d(512, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv52): Conv1d(1024, 1024, kernel_size=(3,), stride=(1,), padding=(1,))
  (uconv6): ConvTranspose1d(1024, 512, kernel_size=(2,), stride=(2,))
  (conv61): Conv1d(1024, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv62): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (uconv7): ConvTranspose1d(512, 256, kernel_size=(2,), stride=(2,))
  (conv71): Conv1d(512, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv72): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (uconv8): ConvTranspose1d(256, 128, kernel_size=(2,), stride=(2,))
  (conv81): Conv1d(256, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv82): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (uconv9): ConvTranspose1d(128, 64, kernel_size=(2,), stride=(2,))
  (conv91): Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv92): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (conv93): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
  (sigmoid): Sigmoid()
)
```

and run the detector with a specified `batch_size` 

```julia
julia> batch_size = 256 
julia> probs_P = detect(X,model_P,batch_size)
```

this will return the probability, `probs_P` that each each sample is a p-wave. Detections will run on the GPU if an NVIDA GPU is installed. Finally, use the `get_picks` function to detect earthquakes based on certain detection thresholds

```julia
julia> on_trigger = 0.5 
julia> off_trigger = 0.25
julia> P_picks, tt = get_picks(S,probs_P,on_trigger,off_trigger)
```

P-wave starttimes are returned as an array of `Dates.DateTime` structures

```julia
julia> P_picks
59-element Array{Dates.DateTime,1}:
 2016-06-10T01:10:55.658
 2016-06-10T02:01:35.628
 2016-06-10T03:10:07.958
 2016-06-10T03:32:47.928
 2016-06-10T03:54:39.998
 2016-06-10T04:15:59.958
 2016-06-10T04:23:11.928
 ⋮
 2016-06-10T21:57:35.798
 2016-06-10T22:32:47.988
 2016-06-10T23:43:00.558
 2016-06-10T23:49:19.958
 2016-06-10T23:56:15.648
 2016-06-10T23:57:35.958
 ```
 
 S-wave phase-detection can be achieved similarly by loading the s-wave model with `model_S = load_model("S")`. 

