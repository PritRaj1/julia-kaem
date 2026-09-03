module ModelGridUpdating

export GridUpdater

using Accessors, Lux, ComponentArrays, ConfParser, Reactant, Statistics

using ..Utils
using ..KAEM_model
using ..KAEM_model.UnivariateFunctions
using ..KAEM_model.EBM_Model
using ..KAEM_model.Quadrature: get_gausslegendre

include("kan/grid_updating.jl")
include("posterior_sampling/langevin.jl")
using .GridUpdating
using .LangevinSampling

function rbf_scale(fcn, st_layer, new_grid)
    fcn.spline_string == "RBF" || return st_layer.scale
    scale = (maximum(new_grid) - minimum(new_grid)) / (size(new_grid, 2) - 1) |> Lux.f32
    return scale .+ zero(st_layer.scale)
end

function replace_leaf(ps, old_leaf, new_leaf)
    data = getdata(ps)
    axes = getaxes(ps)

    # old_leaf is a view into the flat data array
    idx = parentindices(old_leaf)[1]
    start_idx = first(idx)
    end_idx = last(idx)
    n = length(data)

    # Build new data via concat
    new_data = if start_idx == 1 && end_idx == n
        vec(new_leaf)
    elseif start_idx == 1
        vcat(vec(new_leaf), data[(end_idx + 1):n])
    elseif end_idx == n
        vcat(data[1:(start_idx - 1)], vec(new_leaf))
    else
        vcat(data[1:(start_idx - 1)], vec(new_leaf), data[(end_idx + 1):n])
    end

    return ComponentArray(new_data, axes)
end

struct GridUpdater
    model
    update_frequency
    update_decay
    update_prior_grid
    update_llhood_grid
    nogrid_prior
    sampler
end

function GridUpdater(model, conf::ConfParse)
    grid_update_frequency =
        parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))
    grid_update_decay =
        parse(Float32, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    update_prior_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_prior_grid"))
    update_llhood_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_llhood_grid"))

    prior_func = retrieve(conf, "EbmModel", "spline_function")
    gen_func = retrieve(conf, "GeneratorModel", "spline_function")
    prior_grid_trainable = parse(Bool, retrieve(conf, "EbmModel", "grid_trainable"))
    gen_grid_trainable = parse(Bool, retrieve(conf, "GeneratorModel", "grid_trainable"))
    nogrid_prior = prior_grid_trainable || prior_func == "FFT" || prior_func == "Cheby" || prior_func == "Wavelet"
    nogrid_gen = gen_grid_trainable || gen_func == "FFT" || gen_func == "Cheby" || gen_func == "Wavelet"

    N = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
    η = parse(Float32, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))
    ula = initialize_ULA_sampler(model; η = η, N = N, prior_sampling_bool = true)

    return GridUpdater(
        model,
        grid_update_frequency,
        grid_update_decay,
        update_prior_grid,
        update_llhood_grid && !model.lkhood.CNN && !model.lkhood.SEQ && !nogrid_gen,
        nogrid_prior,
        ula
    )
end

function (gu::GridUpdater)(
        ps,
        st_kan,
        st_lux,
        st_rng,
    )
    """Update KAN grids using prior samples."""

    model = gu.model
    x = first(model(ps, st_kan, st_lux, st_rng))
    if gu.update_prior_grid
        ula_bool = model.prior.bool_config.ula || model.sampler_type != "importance" || model.N_t > 1
        z = first(gu.sampler(ps, st_kan, st_lux, x, st_rng))

        # Must update domain for inverse transform sampling
        if (ula_bool && gu.nogrid_prior)
            red_dim = model.prior.bool_config.mixture_model ? (2, 3) : (1, 3)
            lo_bound = dropdims(minimum(z; dims = red_dim); dims = red_dim)
            hi_bound = dropdims(maximum(z; dims = red_dim); dims = red_dim)
            @reset st_kan.ebm.a.min = lo_bound .* 1.0f0
            @reset st_kan.ebm.a.max = hi_bound .* 1.0f0
        end

        if !gu.nogrid_prior
            prior_copy = model.prior
            Q, P, B = prior_copy.q_size, prior_copy.p_size, model.batch_size

            if !prior_copy.bool_config.ula && !prior_copy.bool_config.mixture_model
                for i in 1:prior_copy.depth
                    @reset prior_copy.fcns_qp[i].basis_function.S = Q * B
                end
                @reset prior_copy.s_size = Q * B
                z = reshape(z, P, Q * B)
            else
                z = dropdims(z; dims = 2)
            end

            mid_size = prior_copy.bool_config.mixture_model ? P : Q
            outer_dim = prior_copy.bool_config.mixture_model ? Q * B : Q * P * B

            for i in 1:prior_copy.depth
                if prior_copy.bool_config.layernorm && i != 1
                    z, st_ebm = Lux.apply(
                        prior_copy.layernorms[i - 1],
                        z,
                        ps.ebm.layernorm[symbol_map[i]],
                        st_lux.ebm[symbol_map[i]],
                    )
                    @reset st_lux.ebm[symbol_map[i]] = st_ebm
                end

                new_grid, new_coef = update_fcn_grid(
                    prior_copy.fcns_qp[i],
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                    z,
                )
                ps = replace_leaf(ps, ps.ebm.fcn[symbol_map[i]].coef, vec(new_coef))
                @reset st_kan.ebm[symbol_map[i]].grid = new_grid

                @reset st_kan.ebm[symbol_map[i]].scale = rbf_scale(
                    prior_copy.fcns_qp[i],
                    st_kan.ebm[symbol_map[i]],
                    new_grid
                )

                z = Lux.apply(
                    prior_copy.fcns_qp[i],
                    z,
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                )
                z =
                    (i == 1 && !prior_copy.bool_config.ula) ? reshape(z, mid_size, outer_dim) :
                    dropdims(sum(z, dims = 1); dims = 1)
            end
        end
        new_nodes, new_weights = get_gausslegendre(
            model.prior,
            st_kan.ebm,
            st_kan.quad.init_nodes,
            st_kan.quad.init_weights
        )
        @reset st_kan.quad.nodes = new_nodes
        @reset st_kan.quad.weights = new_weights
    end

    # Only update if KAN-type generator requires
    if gu.update_llhood_grid
        z = dropdims(sum(first(gu.sampler(ps, st_kan, st_lux, x, st_rng)); dims = 2); dims = 2)

        for i in 1:model.lkhood.generator.depth
            if model.lkhood.generator.bool_config.layernorm
                z, st_gen = Lux.apply(
                    model.lkhood.generator.layernorms[i],
                    z,
                    ps.gen.layernorm[symbol_map[i]],
                    st_lux.gen[symbol_map[i]],
                )
                @reset st_lux.gen[symbol_map[i]] = st_gen
            end

            if !(
                    model.lkhood.generator.Φ_fcns[i].spline_string == "FFT" ||
                        model.lkhood.generator.Φ_fcns[i].spline_string == "Cheby"
                )
                new_grid, new_coef = update_fcn_grid(
                    model.lkhood.generator.Φ_fcns[i],
                    ps.gen.fcn[symbol_map[i]],
                    st_kan.gen[symbol_map[i]],
                    z,
                )
                ps = replace_leaf(ps, ps.gen.fcn[symbol_map[i]].coef, vec(new_coef))
                @reset st_kan.gen[symbol_map[i]].grid = new_grid

                @reset st_kan.gen[symbol_map[i]].scale = rbf_scale(
                    model.lkhood.generator.Φ_fcns[i],
                    st_kan.gen[symbol_map[i]],
                    new_grid
                )
            end

            z = Lux.apply(
                model.lkhood.generator.Φ_fcns[i],
                z,
                ps.gen.fcn[symbol_map[i]],
                st_kan.gen[symbol_map[i]],
            )
            z = dropdims(sum(z, dims = 1); dims = 1)
        end
    end

    return ps, st_kan, st_lux
end

end
