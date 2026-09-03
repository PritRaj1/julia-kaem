module Quadrature

export GaussLegendreQuadrature, get_gausslegendre

using LinearAlgebra, Random, ComponentArrays, Accessors

using ..Utils

struct GaussLegendreQuadrature <: AbstractQuadrature end

function qfirst_exp_kernel(f, π0)
    return exp.(f) .* PermutedDimsArray(view(π0, :, :, :), (1, 3, 2))
end

function pfirst_exp_kernel(f, π0)
    return exp.(f) .* PermutedDimsArray(view(π0, :, :, :), (3, 1, 2))
end

function apply_mask(exp_fg, component_mask)
    return dropdims(
        sum(
            PermutedDimsArray(
                view(exp_fg, :, :, :, :), (1, 2, 4, 3)
            ) .* component_mask; dims = 2
        ); dims = 2
    )
end

function weight_kernel(trapz, weights)
    return PermutedDimsArray(
        view(weights, :, :, :), (3, 1, 2)
    ) .* trapz
end

function gauss_kernel(
        trapz,
        weights
    )
    return PermutedDimsArray(
        view(weights, :, :, :), (1, 3, 2)
    ) .* trapz
end

function get_gausslegendre(
        ebm,
        st_kan,
        init_nodes,
        init_weights,
    )
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    a, b = ebm.bool_config.no_grid ?
        (st_kan[:a].min, st_kan[:a].max) :
        (dropdims(minimum(st_kan[:a].grid; dims = 2), dims = 2), dropdims(maximum(st_kan[:a].grid; dims = 2), dims = 2))

    nodes = ((a .+ b) ./ 2 .+ (b .- a) ./ 2) * init_nodes
    weights = ((b .- a) ./ 2) * init_weights
    return nodes, weights
end

function mix_return(nodes, π_nodes, weights, component_mask)
    exp_fg = qfirst_exp_kernel(nodes, π_nodes)
    trapz = apply_mask(exp_fg, component_mask)
    trapz = gauss_kernel(trapz, weights)
    return trapz
end

function mix_return(nodes, π_nodes, weights)
    exp_fg = qfirst_exp_kernel(nodes, π_nodes)
    trapz = gauss_kernel(exp_fg, weights)
    return trapz
end

function univar_return(nodes, π_nodes, weights)
    exp_fg = pfirst_exp_kernel(nodes, π_nodes)
    exp_fg = weight_kernel(exp_fg, weights)
    return exp_fg
end

_quad_return(::UnivariateMode, nodes, π_nodes, weights, _component_mask) =
    univar_return(nodes, π_nodes, weights)
_quad_return(::MixtureMode, nodes, π_nodes, weights, component_mask) =
    mix_return(nodes, π_nodes, weights, component_mask)
_quad_return(::MixtureMode, nodes, π_nodes, weights) =
    mix_return(nodes, π_nodes, weights)

function (gq::GaussLegendreQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad,
        component_mask;
        mode::AbstractSamplingMode = UnivariateMode(),
    )
    """Gauss-Legendre quadrature for numerical integration."""
    nodes, weights = st_quad.nodes, st_quad.weights
    I, O = first(ebm.fcns_qp).in_dim, first(ebm.fcns_qp).out_dim
    Q, P, S = ebm.q_size, ebm.p_size, ebm.N_quad

    transpose_bool = ebm.prior_type ∈ ("learnable_gaussian", "kl_gaussian")
    nodes_in = transpose_bool ? nodes' : nodes
    π_nodes = ebm.π_pdf(nodes_in, ps.dist.π_μ, ps.dist.π_σ)
    π_nodes = transpose_bool ? π_nodes' : π_nodes

    for i in 1:ebm.depth
        @reset ebm.fcns_qp[i].basis_function.S = S
    end

    @reset ebm.s_size = S

    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    result = _quad_return(mode, nodes, π_nodes, weights, component_mask)
    return result, st_quad.nodes, st_lyrnorm_new
end

function (gq::GaussLegendreQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        mode::AbstractSamplingMode = UnivariateMode(),
    )
    """Gauss-Legendre quadrature for numerical integration without mix masking."""
    nodes, weights = st_quad.nodes, st_quad.weights
    I, O = first(ebm.fcns_qp).in_dim, first(ebm.fcns_qp).out_dim
    Q, P, S = ebm.q_size, ebm.p_size, ebm.N_quad

    transpose_bool = ebm.prior_type ∈ ("learnable_gaussian", "kl_gaussian")
    nodes_in = transpose_bool ? nodes' : nodes
    π_nodes = ebm.π_pdf(nodes_in, ps.dist.π_μ, ps.dist.π_σ)
    π_nodes = transpose_bool ? π_nodes' : π_nodes

    for i in 1:ebm.depth
        @reset ebm.fcns_qp[i].basis_function.S = S
    end

    @reset ebm.s_size = S

    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    result = _quad_return(mode, nodes, π_nodes, weights)
    return result, st_quad.nodes, st_lyrnorm_new
end

end
