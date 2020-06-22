using Dates
############################
# Install P + S models + sample data
############################
base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")
download_dir = joinpath(base, "downloads")
mkpath(download_dir)
pmodel = joinpath(download_dir,"model_P.pt")
smodel = joinpath(download_dir,"model_S.pt")
edata = joinpath(download_dir,"2016.162.00.00.00.008.AZ.TRO.HHE.SAC")
ndata = joinpath(download_dir,"2016.162.00.00.00.008.AZ.TRO.HHN.SAC")
zdata = joinpath(download_dir,"2016.162.00.00.00.008.AZ.TRO.HHZ.SAC")

function get_release_data(file::String)
    toget = basename(file)
    url = "https://github.com/tclements/PhaseNet.jl/releases/download/Models/"
    url = joinpath(url,toget)
    println("Downloading $toget... ",now())
    download(url,file)
    println("$toget complete... ",now())
    return nothing
end

for dat in [pmodel,smodel,edata,ndata,zdata]
    if !isfile(dat)
        get_release_data(dat)
    end
end
