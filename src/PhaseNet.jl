module PhaseNet

# Julia translation from:
# Automatic picking of seismic waves using Generalized Phase Detection
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#
# Original Author: Zachary E. Ross (2018)
# Contact: zross@gps.caltech.edu
# Website: http://www.seismolab.caltech.edu/ross_z.html

using Dates
using PyCall, SeisIO

# load torch modules
export torch,nn, UNet

const torch = PyNULL()
const nn = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
    copy!(nn, pyimport("torch.nn"))
    include(joinpath(@__DIR__, "UNet.jl"))
end

# include files
include("detection.jl")

end
