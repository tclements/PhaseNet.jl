export load_model, get_device, seisdata2torch, slide, detect, trigger_onset
export get_picks, load_test_data

"""
  load_model(phase;device=nothing)

Load generalized-phase-detection model onto device.

# Arguments
- `phase::String`: Seismic phase to load - "P" or "S".

# Keywords
- `device::Union{PyObject,Nothing}`: Pytorch device - specify "gpu" or "cpu".
"""
function load_model(phase::String;device::Union{PyObject,Nothing}=nothing)
    # check phase input
    if uppercase(phase) ∉ ["P","S"]
        error("Only available models are 'P' and 'S'.")
    end

    # get correct model
    model_dir = joinpath(dirname(@__FILE__),"..","deps","downloads")
    modelP = joinpath(model_dir,"model_P.pt")
    modelS = joinpath(model_dir,"model_S.pt")
    model_file = uppercase(phase) == "P" ? modelP : modelS

    # load model
    if isa(device,Nothing)
        device = get_device()
    end
    model = UNet(3,1).to(device)
    checkpoint = torch.load(model_file, map_location=device)

    # load state dictionary
    state_dict = Dict()
    for key in keys(checkpoint["model_state_dict"])
        if occursin("tracked",key)
            continue
        end

        state_dict[key] = checkpoint["model_state_dict"][key]
    end
    model.load_state_dict(state_dict)
    model.eval()
    return model
end

"""
  get_device()

Returns current PyTorch device.
"""
get_device() = torch.device(torch.cuda.is_available() ? "cuda" : "cpu")

"""
  seisdata2torch(S)

Convert 3 component `SeisData` struct `S` into Torch tensor.

# Arguments
- `S::SeisData`: 3-component `SeisData` for detection.
- `window_samples::Int`: Number of samples per detection window.
"""
function seisdata2torch(S::SeisData,window_samples::Int)
    # check only 3 channels in SeisChannel
    @assert S.n == 3
    # create sliding windows
    X = cat([slide(S[ii].x,window_samples) for ii = 1:S.n]...,dims=3)
    X ./= maximum(abs.(X),dims=(1,3))
    return torch.from_numpy(X).float().permute(1,2,0)
end

"""
  slide(A,window_samples)

Split vector `A` into non-overlapping windows of length `window_samples`.

# Arguments
- `S::SeisData`: 3-component `SeisData` for detection.
- `window_samples::Int`: Number of samples per detection window.
"""
function slide(A::AbstractVector, window_samples::Int)
    N = size(A,1)
    if N % window_samples == 0
        return Array(reshape(A,window_samples,N ÷ window_samples))
    else
        return Array(reshape(A[1 : N - N % window_samples], window_samples, N ÷ window_samples))
    end
end

"""
  detect(X,model,batch_size;device=nothing)

Run generalized-phase-detection on tensor `X` in batches of size `batch_size`.

# Arguments
- `X::PyObject`: Torch tensor.
- `model::PyObject`: Torch `UNet` model.
- `batch_size::Int`: Number of windows per detection. Specifying too large a
    batch size will cause the GPU to run out of VRAM.

# Keywords
- `device::Union{PyObject,Nothing}`: Pytorch device - specify "gpu" or "cpu".
"""
function detect(X,model,batch_size::Int;device::Union{PyObject,Nothing}=nothing)
    # use default device
    if isa(device,Nothing)
        device = get_device()
    end

    # run predictions in batches
    Y_pred = zeros(Float32, X.size(2),X.size(0))
    for ii in 1:batch_size:size(Y_pred,2)
        i_start = ii
        i_stop = min(size(Y_pred,2),ii+batch_size-1)
        X_test = get(X,i_start-1:i_stop-1)
        X_test = X_test.to(device)
        @pywith torch.no_grad() begin
            out = model(X_test)
            Y_pred[:,i_start:i_stop] .= out.cpu().permute(2,0,1).numpy()[:,:,1]
        end
        GC.gc(false)
    end
    return Y_pred
end

"""
  trigger_onset(charfct, thresh1, thresh2; max_len=1e99, max_len_delete=false)

Calculate trigger on and off times.

Given `thresh1` and `thresh2` calculate trigger on and off times from
characteristic function.

# Arguments
- `charfct::AbstractArray`: Characteristic function of e.g. STA/LTA trigger.
- `thresh1::Real`: Value above which trigger (of characteristic function)
    is activated (higher threshold).
- `thresh2::Real`: Value below which trigger (of characteristic function)
    is deactivated (lower threshold).

# Keywords
- `max_len::Int`: Maximum length of triggered event in samples. A new
    event will be triggered as soon as the signal reaches again above thresh1.
- `max_len_delete::Bool`: Do not write events longer than max_len into report file.
"""
function trigger_onset(charfct::AbstractArray, thresh1::Real, thresh2::Real;
    max_len::Int=10^10, max_len_delete::Bool=false,
)
    ind1 = findall(charfct .> thresh1)
    if length(ind1) == 0
        return []
    end
    ind2 = findall(charfct .> thresh2)

    on = [ind1[1]]
    off = [-1]

    # determine the indices where charfct falls below off-threshold
    ind2_ = Array{Bool}(undef,length(ind2))
    ind2_[1:end-1] .= diff(ind2) .> 1
    # last occurence is missed by diff, add it manually
    ind2_[end] = true
    append!(off,ind2[ind2_])
    append!(on,ind1[findall(diff(ind1) .> 1) .+ 1])
    # include last pick if trigger is on or drop it
    if max_len_delete
        # drop it
        append!(off,max_len)
        append!(on,on[end])
    else
        # include it
        append!(off,ind2[end])
    end

    pick = []
    while on[end] > off[1]
        while on[1] <= off[1]
            deleteat!(on,1)
        end
        while off[1] < on[1]
            deleteat!(off,1)
        end
        if off[1] - on[1] > max_len
            if max_len_delete
                deleteat!(on,1)
                continue
            end
            prepend!(off,on[1] + max_len)
        end
        push!(pick,[on[1],off[1]])
    end
    return permutedims(hcat(pick...))
end

"""
  get_picks(S,probs,thresh1,thresh2;min_trig_dur=0.)

Get picks from detection probabilities `probs` from SeisData `S`.

# Arguments
- `S::SeisData`: 3-component `SeisData` for detection.
- `probs::AbstractArray`: Array of probabilities of phase detection per window.
- `thresh1::Real`: Probability (0-1) above which trigger is activated (higher threshold).
- `thresh2::Real`: Probability (0-1) below which trigger is deactivated (lower threshold).

# Keywords
`min_trig_dur::Real`: Minimum duration (in seconds) of trigger.

"""
function get_picks(
    S::SeisData,
    probs::AbstractArray,
    thresh1::Real,
    thresh2::Real;
    min_trig_dur::Real=0.,
)
    @assert 0. < thresh2 < thresh1 <= 1.
    tt = collect(0:length(probs)-1) ./ S.fs[1]
    picks = Array{DateTime}(undef,0)
    trigs = trigger_onset(vcat(probs...),thresh1,thresh2)
    for ii in 1:size(trigs,1)
        if trigs[ii,1] == trigs[ii,2]
            continue
        end

        # check trigger duration
        trig_dur = tt[trigs[ii,2]] - tt[trigs[ii,1]]
        if trig_dur < min_trig_dur
            continue
        end

        pick = argmax(probs[trigs[ii,1]:trigs[ii,2]]) + trigs[ii,1]

        tstamp = u2d(S.t[1][1,2] * 1e-6+ tt[pick])
        push!(picks,tstamp)
    end
    return picks, tt
end

"""
  load_test_data()

Load 3-component seismic data from the 2016 Mw 5.2 Anza, California sequence.
"""
function load_test_data()
    sacfiles = joinpath(dirname(@__FILE__),"..","deps","downloads","*.SAC")
    return read_data("sac",sacfiles)
end
