using Pycall, Conda, Dates

#############################
# Install Python PyTorch
#############################

const cuda_cur_version = "10.2"

if "pytorch" ∈ Conda._installed_packages()
    try
        pyimport("torch")
        println("Hello!")
    catch ee
        typeof(ee) <: PyCall.PyError || rethrow(ee)
        error("""
Python PyTorch not installed
Please either:
 - Rebuild PyCall to use Conda, by running in the julia REPL:
    - `ENV["PYTHON"]=""; Pkg.build("PyCall"); Pkg.build("PhaseNet")`
 - Or install the python binding yourself, eg by running conda
    - `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
    - then rebuilding PhaseNet.jl via `Pkg.build("PhaseNet")` in the julia REPL
    - make sure you run the right pip, for the instance of python that PyCall is looking at.
"""
    )
    end
else
    Conda.add(["pytorch","torchvision","cudatoolkit=$cuda_cur_version"],channel="pytorch")
end

############################
# Install P + S models
############################
# base = dirname(@__FILE__)
# download_dir = joinpath(base, "downloads")
# download_dir = joinpath(base, "downloads")
# mkpath(download_dir)
# pmodel = joinpath(download_dir,"model_P.pt")
# smodel = joinpath(download_dir,"model_S.pt")
#
# if !isfile(pmodel)
#     println("Downloading P-wave model... ",now())
#     url = "https://github.com/tclements/PhaseNet.jl/releases/download/Models/model_P.pt"
#     download(url,pmodel)
#     println("P-wave model complete... ",now())
# end
#
# if !isfile(smodel)
#     println("Downloading S-wave model... ",now())
#     url = "https://github.com/tclements/PhaseNet.jl/releases/download/Models/model_S.pt"
#     download(url,smodel)
#     println("S-wave model complete... ",now())
# end
