using Dates
############################
# Install P + S models
############################
base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")
download_dir = joinpath(base, "downloads")
mkpath(download_dir)
pmodel = joinpath(download_dir,"model_P.pt")
smodel = joinpath(download_dir,"model_S.pt")

if !isfile(pmodel)
    println("Downloading P-wave model... ",now())
    url = "https://github.com/tclements/PhaseNet.jl/releases/download/Models/model_P.pt"
    download(url,pmodel)
    println("P-wave model complete... ",now())
end
if !isfile(smodel)
    println("Downloading S-wave model... ",now())
    url = "https://github.com/tclements/PhaseNet.jl/releases/download/Models/model_S.pt"
    download(url,smodel)
    println("S-wave model complete... ",now())
end
