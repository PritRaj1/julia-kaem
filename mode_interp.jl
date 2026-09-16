using ConfParser, Random, JLD2, ComponentArrays, Lux, Reactant, Statistics, LinearAlgebra
using CairoMakie, LaTeXStrings, Colors

ENV["DEVICE"] = "gpu"
CairoMakie.activate!(type = "png")

dataset = length(ARGS) >= 1 ? ARGS[1] : "CELEBA"
mode = length(ARGS) >= 2 ? ARGS[2] : "vanilla"

dataset_configs = Dict(
    "CELEBA" => (config = "config/celeba_config.ini", resize = (64, 64)),
    "SVHN" => (config = "config/svhn_config.ini", resize = (32, 32)),
    "CIFAR10" => (config = "config/cifar_config.ini", resize = (32, 32)),
)

haskey(dataset_configs, dataset) || error("Unknown dataset: $dataset. Use one of: $(keys(dataset_configs))")
ds = dataset_configs[dataset]

conf = ConfParse(ds.config)
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

prior_type = parse(Bool, retrieve(conf, "MixtureModel", "use_mixture_prior")) ? "mixture" : "univariate"
file_loc = (
    mode == "thermo" ? "logs/Thermodynamic/$(dataset)/ULA/$(prior_type)/" :
        "logs/Vanilla/$(dataset)/ULA/$(prior_type)/"
)
save_dir = "figures/interpolations/$(dataset)/$(mode)/$(prior_type)/"
mkpath(save_dir)

num_interp_steps = 8
num_pairs = 6
num_prior_samples = 500
num_density_dims = 3

include("src/utils.jl")
using .Utils

include("src/pipeline/trainer.jl")
using .trainer
using .trainer: seed_rand
using .trainer.KAEM_model: load_params
saved_data = load(file_loc * "saved_model.jld2")
rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset; img_resize = ds.resize, file_loc = file_loc, save_model = false)

t.ps = load_params(saved_data) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

model = t.model
ps = t.ps
st_kan = t.st_kan
st_lux = Lux.testmode(t.st_lux)
q_size = model.prior.q_size
batch_size = model.batch_size

# Prior samples
num_batches = num_prior_samples ÷ batch_size
z_all = zeros(Float32, q_size, 1, num_batches * batch_size)
for i in 1:num_batches
    st_rng = seed_rand(model; rng = rng)
    z, _ = Reactant.@jit model.sample_prior(ps, st_kan, st_lux, st_rng)
    idx = ((i - 1) * batch_size + 1):(i * batch_size)
    z_all[:, :, idx] = Array(z)
end
println("Sampled $(size(z_all, 3)) latent vectors")

# Compile decoder
function decode(ps_gen, st_kan_gen, st_lux_gen, z)
    x̂, _ = model.lkhood.generator(ps_gen, st_kan_gen, st_lux_gen, z)
    return model.lkhood.output_activation(x̂)
end

z_dummy = pu(zeros(Float32, q_size, 1, batch_size))
decode_compiled = Reactant.@compile decode(ps.gen, st_kan.gen, st_lux.gen, z_dummy)
println("Decoder compiled.")

# Gaussian KDE per dim
function silverman_bandwidth(samples::AbstractVector)
    n = length(samples)
    σ = std(samples)
    iqr = quantile(samples, 0.75f0) - quantile(samples, 0.25f0)
    a = min(σ, iqr / 1.34f0)
    a = a > 0 ? a : (σ > 0 ? σ : 1.0f0)
    return Float32(0.9 * a * n^(-0.2))
end

function kde_on_grid(samples::AbstractVector{Float32}, grid::AbstractVector{Float32})
    h = silverman_bandwidth(samples)
    invh = 1.0f0 / h
    norm = invh / (sqrt(2.0f0 * π) * length(samples))
    density = similar(grid)
    @inbounds for i in eachindex(grid)
        z = grid[i]
        acc = 0.0f0
        for s in samples
            d = (z - s) * invh
            acc += exp(-0.5f0 * d * d)
        end
        density[i] = acc * norm
    end
    return density
end

# Local maxima above prominence floor
function kde_peaks(
        grid::AbstractVector{Float32},
        density::AbstractVector{Float32};
        rel_prominence = 0.2f0,
        min_sep = 0.1f0,
    )
    n = length(grid)
    dmax = maximum(density)

    candidates = Tuple{Float32, Float32}[]

    for i in 2:(n - 1)
        if density[i] > density[i - 1] &&
                density[i] > density[i + 1] &&
                density[i] > rel_prominence * dmax
            push!(candidates, (grid[i], density[i]))
        end
    end

    isempty(candidates) && return [grid[argmax(density)]]

    sort!(candidates, by = x -> x[2], rev = true)
    peaks = Float32[]

    for (loc, _) in candidates
        if all(abs(loc - p) > min_sep for p in peaks)
            push!(peaks, loc)
        end
    end

    sort!(peaks)
    return peaks
end

struct DimDistribution
    sorted::Vector{Float32}
    grid::Vector{Float32}
    density::Vector{Float32}
    modes::Vector{Float32}
end

function ecdf_value(d::DimDistribution, z::Float32)
    n = length(d.sorted)
    z <= d.sorted[1] && return 0.0f0
    z >= d.sorted[end] && return 1.0f0
    idx = searchsortedlast(d.sorted, z)
    z_lo = d.sorted[idx]
    z_hi = d.sorted[idx + 1]
    span = max(z_hi - z_lo, eps(Float32))
    return Float32(idx - 1 + (z - z_lo) / span) / Float32(n - 1)
end

function ecdf_inverse(d::DimDistribution, u::Float32)
    n = length(d.sorted)
    u <= 0.0f0 && return d.sorted[1]
    u >= 1.0f0 && return d.sorted[end]
    pos = u * Float32(n - 1) + 1.0f0
    lo = clamp(floor(Int, pos), 1, n - 1)
    frac = pos - Float32(lo)
    return (1.0f0 - frac) * d.sorted[lo] + frac * d.sorted[lo + 1]
end

println("Building per-dimension empirical CDFs and locating prior modes...")
dim_dists = Vector{DimDistribution}(undef, q_size)
for q in 1:q_size
    samples = vec(z_all[q, 1, :])
    sorted = sort(samples)
    pad = 0.05f0 * (sorted[end] - sorted[1] + eps(Float32))
    grid = collect(range(sorted[1] - pad, sorted[end] + pad; length = 256))
    density = kde_on_grid(samples, grid)
    modes = kde_peaks(grid, density)
    dim_dists[q] = DimDistribution(sorted, grid, density, modes)
end
mode_counts = [length(d.modes) for d in dim_dists]
multimodal_dims = findall(>(1), mode_counts)
println("Identified $(length(multimodal_dims)) multimodal dimensions out of $q_size.")

num_flip_dims = 10

function decode_many(zs::AbstractMatrix{Float32})
    n = size(zs, 2)
    out = nothing
    for start in 1:batch_size:n
        stop = min(start + batch_size - 1, n)
        z_pad = zeros(Float32, q_size, 1, batch_size)
        z_pad[:, 1, 1:(stop - start + 1)] .= zs[:, start:stop]
        x = Array(decode_compiled(ps.gen, st_kan.gen, st_lux.gen, pu(z_pad)))
        chunk = x[:, :, :, 1:(stop - start + 1)]
        out = out === nothing ? chunk : cat(out, chunk; dims = 4)
    end
    return out
end

# Rank multimodal dims by L2 change in decoded image
function decoder_sensitivities(anchor, dim_dists, mode_counts, q_size)
    multimodal = findall(>=(2), mode_counts)
    n = length(multimodal)
    n == 0 && return Int[], Float32[]

    z_test = zeros(Float32, q_size, n + 1)
    z_test[:, 1] .= anchor
    for (i, q) in enumerate(multimodal)
        modes_q = dim_dists[q].modes
        cur = argmin(abs.(anchor[q] .- modes_q))
        nxt = cur == 1 ? 2 : 1
        z_test[:, i + 1] .= anchor
        z_test[q, i + 1] = modes_q[nxt]
    end

    imgs = decode_many(z_test)
    base = imgs[:, :, :, 1]
    sens = Float32[
        sqrt(sum((imgs[:, :, :, i + 1] .- base) .^ 2)) for i in 1:n
    ]
    return multimodal, sens
end

function sample_mode_pair(rng_pair, dim_dists, mode_counts, q_size, z_all, n_flip)
    n_samples = size(z_all, 3)
    anchor = vec(z_all[:, 1, rand(rng_pair, 1:n_samples)])
    z_a = copy(anchor)
    z_b = copy(anchor)

    multimodal, sens = decoder_sensitivities(anchor, dim_dists, mode_counts, q_size)
    isempty(multimodal) && return z_a, z_b

    k = min(n_flip, length(multimodal))
    flip_dims = multimodal[sortperm(sens; rev = true)[1:k]]

    for q in flip_dims
        modes_q = dim_dists[q].modes
        # cur = argmin(abs.(anchor[q] .- modes_q))
        # others = [i for i in 1:length(modes_q) if i != cur]
        # nxt = others[rand(rng_pair, 1:length(others))]
        cur = argmin(abs.(anchor[q] .- modes_q))
        nxt = argmax(abs.(modes_q .- modes_q[cur]))
        z_a[q] = modes_q[cur]
        z_b[q] = modes_q[nxt]
    end
    return z_a, z_b
end

num_cols = num_interp_steps + 2
ts = collect(range(0.0f0, 1.0f0; length = num_cols))

for pair_idx in 1:num_pairs
    rng_pair = Random.MersenneTwister(pair_idx * 31 + 7)
    z_a, z_b = sample_mode_pair(rng_pair, dim_dists, mode_counts, q_size, z_all, num_flip_dims)

    u_a = Float32[ecdf_value(dim_dists[q], z_a[q]) for q in 1:q_size]
    u_b = Float32[ecdf_value(dim_dists[q], z_b[q]) for q in 1:q_size]

    z_a_col = reshape(z_a, q_size, 1)
    z_batch = repeat(z_a_col, 1, 1, batch_size)
    for (ci, t_val) in enumerate(ts)
        u_t = (1.0f0 - t_val) .* u_a .+ t_val .* u_b
        z_t = Float32[ecdf_inverse(dim_dists[q], u_t[q]) for q in 1:q_size]
        z_batch[:, :, ci] .= reshape(z_t, q_size, 1)
    end

    x_decoded = Array(decode_compiled(ps.gen, st_kan.gen, st_lux.gen, pu(z_batch)))

    all_rgb = Matrix{RGB{Float32}}[]
    for ci in 1:num_cols
        raw = clamp.(x_decoded[:, :, :, ci], 0.0f0, 1.0f0)
        rgb = RGB.(raw[:, :, 1], raw[:, :, 2], raw[:, :, 3])
        push!(all_rgb, rot180(rgb))
    end

    # Dims the path actually traverses
    delta_u = abs.(u_a .- u_b)
    scores = delta_u .* Float32.(mode_counts)
    top_dims = sortperm(scores; rev = true)[1:num_density_dims]

    fig = Figure(size = (700, 550), backgroundcolor = :white)

    for col in 1:num_cols
        ax = CairoMakie.Axis(fig[2, col], aspect = DataAspect())
        hidedecorations!(ax)
        hidespines!(ax)
        image!(ax, all_rgb[col])
    end

    Label(fig[1, 1], L"z_A", fontsize = 18, halign = :center, valign = :bottom)
    Label(fig[1, div(num_cols, 2):(div(num_cols, 2) + 1)], L"\longrightarrow", fontsize = 18, halign = :center, valign = :bottom)
    Label(fig[1, num_cols], L"z_B", fontsize = 18, halign = :center, valign = :bottom)

    dim_colors = [:royalblue, :firebrick, :forestgreen]
    for (di, dim) in enumerate(top_dims)
        is_last = di == num_density_dims
        ax = CairoMakie.Axis(
            fig[2 + di, 1:num_cols],
            ylabel = L"p_{f,%$dim}(z_{%$dim})",
            xlabel = is_last ? L"z_q" : "",
            xgridvisible = false,
            ygridvisible = false,
        )
        !is_last && hidexdecorations!(ax; ticks = false)
        hidespines!(ax, :t, :r)

        d = dim_dists[dim]
        lines!(
            ax, d.grid, d.density;
            color = dim_colors[di], linewidth = 1.5
        )
        band!(
            ax, d.grid, zeros(Float32, length(d.grid)), d.density;
            color = (dim_colors[di], 0.15)
        )
        vlines!(ax, [z_a[dim]]; color = :black, linewidth = 1.5, linestyle = :solid)
        vlines!(ax, [z_b[dim]]; color = :black, linewidth = 1.5, linestyle = :dash)
    end

    Legend(
        fig[2 + num_density_dims + 1, 1:num_cols],
        [
            LineElement(color = :black, linewidth = 1.5, linestyle = :solid),
            LineElement(color = :black, linewidth = 1.5, linestyle = :dash),
        ],
        [L"z_A", L"z_B"],
        orientation = :horizontal,
        framevisible = false,
        labelsize = 18,
    )

    for col in 1:num_cols
        colsize!(fig.layout, col, Relative(1 / num_cols))
    end
    rowgap!(fig.layout, 4)
    rowgap!(fig.layout, 2, 12)

    save(save_dir * "mode_cdf_$(pair_idx).png", fig, px_per_unit = 5)
    println("Saved mode_cdf_$(pair_idx).png")
end

println("Results in $save_dir")
