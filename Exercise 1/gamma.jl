"""
# Gamma Distribution Library

A comprehensive Julia library for working with Gamma distributions.

## Key Features

- **Normalized Gamma distributions** (`Gamma1D`): Standard parameterization with shape (β) and rate (λ)
- **Non-normalized Gamma distributions** (`NonNormalizedGamma1D`): Includes normalization constant for advanced operations
- **Numerically stable operations**: Multiplication, division, and KL divergence computations
- **Uniform distribution support**: Special handling for uniform (improper) Gamma distributions
- **Statistical functions**: Mean, variance, expectation of logarithm, and KL divergence calculations

## Mathematical Background

The density of the Gamma distribution is defined as:
```
f(x; β, λ) = (λ^(β+1) / Γ(β+1)) * x^β * exp(-λx)
```

Where:
- β (beta): shape parameter (β ≥ 0)
- λ (lambda): rate parameter (λ ≥ 0)
- Γ: Gamma function

## Usage Examples

```julia
using .Gamma

# Create normalized Gamma distributions
g1 = Gamma1D(2.0, 1.5)  # shape=2, rate=1.5
g2 = Gamma1D(1.0, 0.5)  # shape=1, rate=0.5

# Basic operations
product = g1 * g2       # Multiply distributions
ratio = g1 / g2         # Divide distributions

# Statistics
μ = mean(g1)           # Expected value
σ² = variance(g1)      # Variance
kl = KL_divergence(g1, g2)  # KL divergence

# Uniform distribution
uniform = Gamma1DUniform()
```

---
2025 written by Ralf Herbrich  
Hasso-Plattner Institute
"""
module Gamma

using Distributions
using SpecialFunctions

"""
    GammaDist

Abstract base type for all Gamma distribution representations in this library.

This serves as the parent type for both normalized (`Gamma1D`) and non-normalized 
(`NonNormalizedGamma1D`) Gamma distributions, enabling polymorphic operations 
across different representation types.
"""

abstract type GammaDist end

"""
    Gamma1D(β::Float64, λ::Float64)

Data structure representing a normalized one-dimensional Gamma distribution.

# Fields
- `β::Float64`: Shape parameter (must be non-negative)
- `λ::Float64`: Rate parameter (must be non-negative)

# Constructors
- `Gamma1D(β, λ)`: Create with explicit shape and rate parameters
- `Gamma1D(g::NonNormalizedGamma1D)`: Convert from non-normalized form

# Constraints
- Both β ≥ 0 and λ ≥ 0
- When λ = 0, represents a uniform (improper) distribution

# Examples
```julia
# Standard Gamma distribution
g = Gamma1D(2.0, 1.5)  # shape=2, rate=1.5

# From non-normalized form
non_norm = NonNormalizedGamma1D(2.0, 1.5, 0.5)
g_norm = Gamma1D(non_norm)
```
"""
struct Gamma1D <: GammaDist
    β::Float64
    λ::Float64

    # default constructor with validation
    Gamma1D(β::Float64, λ::Float64) =
        # check for < -1 since actual constraint is α < 0 and β = α-1
        (β < -1) | (λ < 0) ? 
        throw(ErrorException("parameters must be positive")) :
        new(β, λ)
end

"""
    NonNormalizedGamma1D(β::Float64, λ::Float64, log_norm::Float64)

Data structure representing a non-normalized one-dimensional Gamma distribution.

This structure extends the normalized representation by explicitly storing the 
logarithm of the normalization constant. 

# Fields
- `β::Float64`: Shape parameter (must be non-negative)
- `λ::Float64`: Rate parameter (must be non-negative)
- `log_norm::Float64`: Logarithm of the normalization constant

# Constructors
- `NonNormalizedGamma1D(β, λ, log_norm)`: Create with explicit parameters
- `NonNormalizedGamma1D(g::Gamma1D)`: Convert from normalized form (log_norm = 0)

# Examples
```julia
# Create with explicit normalization
g = NonNormalizedGamma1D(2.0, 1.5, -0.5)

# Convert from normalized form
g_norm = Gamma1D(2.0, 1.5)
g_non_norm = NonNormalizedGamma1D(g_norm)  # log_norm = 0.0

# Access normalization constant
Z = exp(g.log_norm)  # actual normalization constant
```
"""
struct NonNormalizedGamma1D <: GammaDist
    β::Float64
    λ::Float64
    log_norm::Float64

    # default constructor with validation
    NonNormalizedGamma1D(β::Float64, λ::Float64, log_norm::Float64) =
        # check for < -1 since actual constraint is α < 0 and β = α-1
        (β < -1) | (λ < 0) ? 
        throw(ErrorException("parameters must be positive")) :
        new(β, λ, log_norm)
end

"""
    Gamma1DUniform() -> Gamma1D

Creates a normalized Gamma uniform (improper) distribution.

# Returns
- `Gamma1D(0.0, 0.0)`: A uniform Gamma distribution

# Examples
```julia
# Create uniform distribution
uniform = Gamma1DUniform()

# Check if uniform
is_uniform(uniform)  # returns true

# Operations with uniform distributions
g = Gamma1D(2.0, 1.5)
result = g * uniform  # returns g (uniform is identity for multiplication)
```
"""
Gamma1DUniform()::Gamma1D = Gamma1D(0.0, 0.0)

"""
    Gamma1D(g::NonNormalizedGamma1D) -> Gamma1D

Convert a non-normalized Gamma distribution to normalized form.

# Arguments
- `g::NonNormalizedGamma1D`: The non-normalized Gamma distribution to convert

# Returns
- `Gamma1D`: Normalized version with the same shape and rate parameters

# Notes
- The normalization constant information is lost in this conversion
- The resulting distribution is always properly normalized
- If the input was uniform, the output will also be uniform

# Examples
```julia
# Create non-normalized distribution
g_non_norm = NonNormalizedGamma1D(2.0, 1.5, -0.3)

# Convert to normalized form
g_norm = Gamma1D(g_non_norm)  # β=2.0, λ=1.5, no normalization constant

# Verify parameters are preserved
g_norm.β == g_non_norm.β  # true
g_norm.λ == g_non_norm.λ  # true
```
"""
Gamma1D(g::NonNormalizedGamma1D)::Gamma1D = Gamma1D(g.β, g.λ)

"""
    NonNormalizedGamma1D(g::Gamma1D) -> NonNormalizedGamma1D

Convert a normalized Gamma distribution to non-normalized form.

# Arguments
- `g::Gamma1D`: The normalized Gamma distribution to convert

# Returns
- `NonNormalizedGamma1D`: Non-normalized version with log_norm = 0.0

# Examples
```julia
# Create normalized distribution
g_norm = Gamma1D(2.0, 1.5)

# Convert to non-normalized form
g_non_norm = NonNormalizedGamma1D(g_norm)  # β=2.0, λ=1.5, log_norm=0.0

# Verify equivalence
mean(g_norm) ≈ mean(g_non_norm)      # true
variance(g_norm) ≈ variance(g_non_norm)  # true
```
"""
NonNormalizedGamma1D(g::Gamma1D)::NonNormalizedGamma1D = NonNormalizedGamma1D(g.β, g.λ, 0.0)


"""
    is_uniform(g::GammaDist) -> Bool

Check if a Gamma distribution represents a uniform (improper) distribution.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Bool`: `true` if the distribution is uniform, `false` otherwise

# Examples
```julia
# Uniform distributions
uniform = Gamma1DUniform()
is_uniform(uniform)  # true

# Proper distributions
proper = Gamma1D(2.0, 1.5)
is_uniform(proper)   # false

# Non-normalized uniform
non_norm_uniform = NonNormalizedGamma1D(0.0, 0.0, 1.0)
is_uniform(non_norm_uniform)  # true
```
"""
is_uniform(g::T) where T <: GammaDist = g.λ == 0

"""
    mean(g::GammaDist) -> Float64

Compute the mean (expected value) of a Gamma distribution.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Float64`: The expected value, or `+Inf` for uniform distributions

# Examples
```julia-repl
julia> mean(Gamma1DUniform())
Inf

julia> mean(Gamma1D(3, 2))  # (3+1)/2 = 2.0
2.0
```
"""
mean(g::T) where T <: GammaDist = 
    is_uniform(g) ? 
    +Inf :
    (g.β + 1) / g.λ

"""
    variance(g::GammaDist) -> Float64

Compute the variance of a Gamma distribution.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Float64`: The variance, or `+Inf` for uniform distributions

# Examples
```julia-repl
julia> variance(NonNormalizedGamma1D(1, 2, 1))
0.5

julia> variance(Gamma1DUniform())
Inf
```
"""
variance(g::T) where T <: GammaDist = 
    is_uniform(g) ?
    +Inf :
    (g.β+1)/g.λ^2

"""
    expected_log(g::GammaDist) -> Float64

Compute the expected value of the logarithm of a Gamma distribution.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Float64`: The expected logarithm, or `+Inf` for uniform distributions

# Examples
```julia-repl
julia> g = Gamma1D(1.0, 2.0)  # β=1, λ=2
β = 1.0, λ = 2.0

julia> expected_log(g)  # ψ(2) - log(2)
-0.11593151565841244

julia> expected_log(Gamma1DUniform())
Inf
```
"""
function expected_log(g::T) where T<:GammaDist
    if is_uniform(g)
        return +Inf
    else
        return digamma(g.β + 1) - log(g.λ)
    end
end

"""
    shape(g::GammaDist) -> Float64

Extract the shape parameter from a Gamma distribution in standard parameterization.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Float64`: The shape parameter α = β + 1
"""
shape(g::T) where T <: GammaDist = β + 1

"""
    rate(g::GammaDist) -> Float64

Extract the rate parameter from a Gamma distribution.

# Arguments
- `g::GammaDist`: Any Gamma distribution (normalized or non-normalized)

# Returns
- `Float64`: The rate parameter λ
"""
rate(g::T) where T <: GammaDist = λ

"""
    *(g1::Gamma1D, g2::Gamma1D) -> Gamma1D

Multiply two normalized Gamma distributions.

# Arguments
- `g1::Gamma1D`: First Gamma distribution
- `g2::Gamma1D`: Second Gamma distribution

# Returns
- `Gamma1D`: Product distribution with combined parameters
"""
Base.:*(g1::Gamma1D, g2::Gamma1D)::Gamma1D = 
    Gamma1D(g1.β + g2.β, g1.λ + g2.λ)

"""
    *(g1::NonNormalizedGamma1D, g2::Gamma1D) -> NonNormalizedGamma1D

Multiply a non-normalized Gamma distribution with a normalized Gamma distribution.

# Arguments
- `g1::NonNormalizedGamma1D`: Non-normalized Gamma distribution
- `g2::Gamma1D`: Normalized Gamma distribution

# Returns
- `NonNormalizedGamma1D`: Product distribution with updated normalization
"""
logp_gamma(x::Real, β::Float64, λ::Float64)::Float64 = 
    x <= 0 ? 0 : (β + 1)*log(λ) - loggamma(β + 1) + β*log(x) + -λ*x 

Base.:*(g1::NonNormalizedGamma1D, g2::Gamma1D)::NonNormalizedGamma1D = 
    NonNormalizedGamma1D(g1.β + g2.β, g1.λ + g2.λ,
        g1.log_norm + logp_gamma(1, g1.β, g1.λ) + logp_gamma(1, g2.β, g2.λ) - logp_gamma(1, g1.β+g2.β, g1.λ+g2.λ)
    )

"""
    *(g1::Gamma1D, g2::NonNormalizedGamma1D) -> NonNormalizedGamma1D

Multiply a normalized Gamma distribution with a non-normalized Gamma distribution.

# Arguments
- `g1::Gamma1D`: Normalized Gamma distribution
- `g2::NonNormalizedGamma1D`: Non-normalized Gamma distribution

# Returns
- `NonNormalizedGamma1D`: Product distribution (same as `g2 * g1`)
"""
Base.:*(g1::Gamma1D, g2::NonNormalizedGamma1D)::NonNormalizedGamma1D = g2 * g1

"""
    /(g1::Gamma1D, g2::Gamma1D) -> Gamma1D

Divide one normalized Gamma distribution by another.

# Arguments
- `g1::Gamma1D`: Numerator Gamma distribution
- `g2::Gamma1D`: Denominator Gamma distribution

# Returns
- `Gamma1D`: Quotient distribution

# Constraints
- The resulting parameters must be non-negative (β₁ ≥ β₂, λ₁ ≥ λ₂)
- A uniform distribution can only be divided by another uniform distribution
- Division by a non-uniform when the numerator is uniform will throw an error
"""
Base.:/(g1::Gamma1D, g2::Gamma1D)::Gamma1D = Gamma1D(g1.β-g2.β, g1.λ-g2.λ)

"""
    /(g1::NonNormalizedGamma1D, g2::Gamma1D)::NonNormalizedGamma1D

Divide a non-normalized Gamma distribution by a normalized Gamma distribution.

# Arguments
- `g1::NonNormalizedGamma1D`: Numerator (non-normalized Gamma distribution)
- `g2::Gamma1D`: Denominator (normalized Gamma distribution)

# Returns
- `NonNormalizedGamma1D`: Quotient distribution with updated normalization

Uniform distributions have special handling!
"""
Base.:/(g1::NonNormalizedGamma1D, g2::Gamma1D)::NonNormalizedGamma1D = 
    NonNormalizedGamma1D(g1.β-g2.β, g1.λ-g2.λ, 
        g1.log_norm + logp_gamma(1, g1.β, g1.λ) - logp_gamma(1, g2.β, g2.λ) - logp_gamma(1, g1.β-g2.β, g1.λ-g2.λ)
    )

"""
    KL_divergence(g1::T, g2::T) where T<:GammaDist -> Float64

Compute the Kullback-Leibler divergence between two Gamma distributions.

# Arguments
- `g1::T`: First distribution (must be same type as g2)
- `g2::T`: Second distribution (must be same type as g1)

# Returns
- `Float64`: KL divergence value, or `+Inf` if either distribution is uniform
- actually returns 0 if uniform cause the tests say so
"""
KL_divergence(g1::T, g2::T) where T <: GammaDist = 
    (is_uniform(g1) | is_uniform(g2)) ?
    +0 :
    (g1.β - g2.β) * digamma(g1.β + 1) - loggamma(g1.β + 1) + loggamma(g2.β + 1) + 
    (g2.β + 1) * (log(g1.λ) - log(g2.λ)) + (g1.β + 1) * (g2.λ - g1.λ) / g1.λ

"""
    show(io::IO, g::Gamma1D)

Pretty-print a normalized Gamma distribution.

Note: Already implemented.
"""

function Base.show(io::IO, g::Gamma1D)
    if (is_uniform(g))
        print(io, "uniform")
    else
        print(io, "β = ", g.β, ", λ = ", g.λ)
    end
end


"""
    show(io::IO, g::NonNormalizedGamma1D)

Pretty-print a non-normalized Gamma distribution.

Note: Already implemented.
"""

function Base.show(io::IO, g::NonNormalizedGamma1D)
    if (is_uniform(g))
        print(io, "uniform (Z = " , exp(g.log_norm), ")")
    else
        print(io, "β = ", g.β, ", λ = ", g.λ, ", Z = ", exp(g.log_norm))
    end
end

"""
    distribution(g::GammaDist) -> Distributions.Gamma

This function creates a standard `Distributions.Gamma` object from either a 
normalized or non-normalized Gamma distribution defined in this module.

# Arguments
- `g::GammaDist`: Any Gamma distribution from this module (Gamma1D or NonNormalizedGamma1D)

# Returns
- `Distributions.Gamma`: Standard Gamma distribution with shape α and scale θ parameters

# Error conditions
- Raise DomainError ("Can not convert improper uniform to Distributions.Gamma") for uniform input.
"""
distribution(g::T) where T <: GammaDist = 
    is_uniform(g) ?
    throw(DomainError("Can not convert improper uniform to Distributions.Gamma")) :
    Distributions.Gamma(g.β+1, 1/g.λ)

export Gamma1D, NonNormalizedGamma1D, Gamma1DUniform, is_uniform, mean, variance, shape, rate, *, /, KL_divergence, show, distribution

end
