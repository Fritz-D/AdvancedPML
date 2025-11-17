"""
# Gaussian Distribution Library

A comprehensive Julia library for working with Gaussian (Normal) distributions in both normalized and non-normalized forms using precision parameterization. This library provides numerically stable operations for multiplication, division, and statistical computations on Gaussian distributions.

## Key Features

- **Precision parameterization**: Uses τ (precision-weighted mean) and ρ (precision) for numerical stability
- **Normalized Gaussian distributions** (`Gaussian1D`): Standard representation with implicit normalization
- **Non-normalized Gaussian distributions** (`NonNormalizedGaussian1D`): Includes explicit normalization constant
- **Numerically stable operations**: Multiplication, division, and KL divergence computations
- **Uniform distribution support**: Special handling for uniform (improper) Gaussian distribution
- **Multiple constructors**: Create from mean/variance or precision parameters

## Mathematical Background

The pdf of the Gaussian distribution is given by (in mean/variance parametrization):
```
f(x; μ, σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```

This library uses precision parameterization:
- τ (tau): precision-weighted mean = μ/σ² = μ * ρ
- ρ (rho): precision = 1/σ²

Where:
- μ: mean parameter
- σ²: variance parameter

## Usage Examples

```julia
using .Gaussian


julia> Gaussian1D(2.0, 0.5) # Create from precision parameters
N(μ = 4.0, σ² = 2.0) resp. G(τ = 2.0, ρ = 0.5)

julia> Gaussian1DFromMeanVariance(4, 2) # Create from mean and variance
N(μ = 4.0, σ² = 2.0) resp. G(τ = 2.0, ρ = 0.5)

julia> # μ=4.0, σ²=2.0 → τ=2.0, ρ=0.5

julia> Gaussian1DUniform()  # Create (improper) Uniform
Uniform()

julia> NonNormalizedGaussian1D(2.0, 0.5, 3.0)  # Create non-normalized Gaussian
Non-normalized: N(μ = 4.0, σ² = 2.0, Z = 20.085536923187668) resp. G(τ = 2.0, ρ = 0.5, Z)

julia> NonNormalizedGaussian1D(0.0, 0.0, 3.0)  # Create non-normalized Uniform
Non-normalized: Uniform(Z = 20.085536923187668)

julia> Gaussian1D(2.0, 0.5) * Gaussian1D(1.4, 3.5) # product
N(μ = 0.85, σ² = 0.25) resp. G(τ = 3.4, ρ = 4.0)

julia> Gaussian1D(7, 8) / Gaussian1D(4, 3) # ratio
N(μ = 0.6, σ² = 0.2) resp. G(τ = 3.0, ρ = 5.0)

julia> mean(Gaussian1D(2, .5))
4.0

julia> variance(Gaussian1D(2, .5))
2.0

julia> KL_divergence(Gaussian1D(2, .5), Gaussian1D(3, .6))
0.3088392216030226
```

---
2025 written by Ralf Herbrich  
Hasso-Plattner Institute
"""
module Gaussian

using Distributions

"""
    GaussianDist

Abstract base type for all Gaussian distribution representations in this library.

This serves as the parent type for both normalized (`Gaussian1D`) and non-normalized (`NonNormalizedGaussian1D`) Gaussian distributions, enabling polymorphic operations across different representation types.

# Note:
This is already implemented as an example. No need to modify anything.
"""

abstract type GaussianDist end

"""
    Gaussian1D(τ::Float64, ρ::Float64)

Data structure representing a normalized one-dimensional Gaussian distribution using precision parameterization.
This structure stores the precision-weighted mean (τ) and precision (ρ) parameters. 

# Fields
- `τ::Float64`: Precision-weighted mean (ℝ) as Float64
- `ρ::Float64`: Precision (ℝ, must be non-negative) as Float64

# Constructors
- `Gaussian1D(τ, ρ)`: Create with explicit precision parameters
- `Gaussian1D()`: Standard normal distribution (μ=0, σ²=1)
- `Gaussian1D(g::NonNormalizedGaussian1D)`: Convert from non-normalized form
- `Gaussian1DFromMeanVariance(μ, σ²)`: Create from mean and variance

# Constraints
- ρ ≥ 0 (precision must be non-negative)
- When ρ = 0, represents a uniform (improper) distribution

# Examples
```julia
# Standard normal distribution
standard = Gaussian1D()  # μ=0, σ²=1

# Custom distribution
g = Gaussian1D(4.0, 2.0)  # τ=4.0, ρ=2.0 → μ=2.0, σ²=0.5

# Uniform distribution (special case)
uniform = Gaussian1D(0.0, 0.0)  # ρ=0 → uniform

# From mean and variance
g_mv = Gaussian1DFromMeanVariance(2.0, 0.5)  # μ=2.0, σ²=0.5
```

# Note
This is already implemented as an example. No need to change anything here.
"""
struct Gaussian1D <: GaussianDist
    τ::Float64
    ρ::Float64

    # default constructor with validation
    Gaussian1D(τ, ρ) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ)...)
end

"""
    NonNormalizedGaussian1D(τ::Float64, ρ::Float64, log_norm::Float64)

Data structure representing a non-normalized one-dimensional Gaussian distribution.

This structure extends the normalized representation by explicitly storing the logarithm of the normalization constant.
This is essential for operations that modify the normalization, such as message passing in probabilistic inference, variational methods, or Bayesian updates where the normalization constant carries important information about evidence or marginal likelihood.

# Fields
- `τ::Float64`: as in `Gaussian1D` 
- `ρ::Float64`: as in `Gaussian1D`
- `log_norm::Float64`: Logarithm of the normalization constant (ℝ) as Float64

# Constructors
- `NonNormalizedGaussian1D(τ, ρ, log_norm)`: Create with explicit parameters
- `NonNormalizedGaussian1D(g::Gaussian1D)`: Convert from normalized form (log_norm = 0)

# Usage Examples
```julia
# Create with explicit normalization
g = NonNormalizedGaussian1D(4.0, 2.0, -0.5)  # τ=4.0, ρ=2.0, log_norm=-0.5

# Convert from normalized form
g_norm = Gaussian1D(4.0, 2.0)
g_non_norm = NonNormalizedGaussian1D(g_norm)  # log_norm = 0.0

# Access normalization constant
Z = exp(g.log_norm)  # actual normalization constant

# Compute mean and variance (same as normalized version)
μ = mean(g)  # τ/ρ = 4.0/2.0 = 2.0
σ² = variance(g)  # 1/ρ = 1/2.0 = 0.5
```
"""
struct NonNormalizedGaussian1D <: GaussianDist
    τ::Float64
    ρ::Float64
    log_norm::Float64

    # default constructor with validation
    NonNormalizedGaussian1D(τ::Float64, ρ::Float64, log_norm::Float64) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ, log_norm)...)
end

"""
    Gaussian1D()

This convenience constructor creates a standard normal distribution with mean 0 
and variance 1, using the precision parameterization τ = 0 and ρ = 1, by not supplying any arguments to `Gaussian1D`.

# Returns
- `Gaussian1D(0.0, 1.0)`: Standard normal distribution

# Note
This is already implemented as an example. No need to change anything.
"""
Gaussian1D() = Gaussian1D(0.0, 1.0)

"""
    Gaussian1DFromMeanVariance(μ::Real, σ²::Real) -> Gaussian1D

Create a normalized Gaussian distribution from mean and variance parameters.

# Arguments
- `μ::Float64`: Mean of the distribution (ℝ) as Float64
- `σ²::Float64`: Variance of the distribution (ℝ, must be positive) as Float64

# Returns
- `Gaussian1D`: Distribution with precision parameters τ = μ/σ² and ρ = 1/σ²

# Error Conditions
```julia
# This will throw an error
Gaussian1DFromMeanVariance(1.0, 0.0)  # Error: zero variance not allowed
```
It must raise the error "Dirac delta cannot be initialized with GaussianFromMeanVariance".
Errors can be raised using `error()`.

# Examples
```julia
g1 = Gaussian1DFromMeanVariance(2.0, 0.5)  # μ=2.0, σ²=0.5

# Verify the conversion
mean(g1)      # 2.0
variance(g1)  # 0.5

# Internal precision parameters
g1.τ          # 4.0 = 2.0/0.5
g1.ρ          # 2.0 = 1/0.5
```
"""
Gaussian1DFromMeanVariance(μ::Real, σ²::Real)::Gaussian1D = 
    (σ² <= 0) ? error("variance of a Gaussian must be positive") : Gaussian1D(μ/σ², 1/σ²)

"""
    Gaussian1DUniform() -> Gaussian1D

This convenience constructor creates a Gaussian distribution with precision ρ = 0, which represents a uniform distribution over the entire real line. 
This is an improper distribution (doesn't integrate to 1) but is useful in Bayesian inference as a non-informative prior.

# Returns
- `Gaussian1D(0.0, 0.0)`: Uniform Gaussian distribution
"""
Gaussian1DUniform()::Gaussian1D = Gaussian1D(0.0, 0.0)

"""
    Gaussian1D(g::NonNormalizedGaussian1D) -> Gaussian1D

Convert a non-normalized Gaussian distribution to normalized form.

# Arguments
- `g::NonNormalizedGaussian1D`: The non-normalized Gaussian distribution to convert

# Returns
- `Gaussian1D`: Normalized version with the same τ and ρ parameters

# Notes
- The normalization constant information is lost in this conversion
- The resulting distribution is always properly normalized
- Mean and variance are preserved exactly
- If the input was uniform, the output will also be uniform

#  Usage Examples
```julia
# Create non-normalized distribution
g_non_norm = NonNormalizedGaussian1D(4.0, 2.0, -0.3)

# Convert to normalized form
g_norm = Gaussian1D(g_non_norm)  # τ=4.0, ρ=2.0, no normalization constant

# Verify parameters and statistics are preserved
g_norm.τ == g_non_norm.τ           # true
g_norm.ρ == g_non_norm.ρ           # true
mean(g_norm) ≈ mean(g_non_norm)    # true
variance(g_norm) ≈ variance(g_non_norm)  # true
```
"""
Gaussian1D(g::NonNormalizedGaussian1D)::Gaussian1D = Gaussian1D(g.τ, g.ρ)

"""
    NonNormalizedGaussian1D(g::Gaussian1D) -> NonNormalizedGaussian1D

This constructor creates a non-normalized representation of a normalized 
Gaussian distribution by setting the log normalization constant to zero.

# Arguments
- `g::Gaussian1D`: The normalized Gaussian distribution to convert

# Returns
- `NonNormalizedGaussian1D`: Non-normalized version with log_norm = 0.0

# Notes
- The precision parameters (τ, ρ) are preserved exactly
- The log normalization constant is initialized to 0.0
- This represents the same probability distribution as the input
- Mean and variance remain unchanged

# Examples
```julia
# Create normalized distribution
g_norm = Gaussian1DFromMeanVariance(2.0, 0.5)

# Convert to non-normalized form
g_non_norm = NonNormalizedGaussian1D(g_norm)  # τ=4.0, ρ=2.0, log_norm=0.0

# Verify equivalence
mean(g_norm) ≈ mean(g_non_norm)          # true
variance(g_norm) ≈ variance(g_non_norm)  # true
exp(g_non_norm.log_norm)                 # 1.0 (properly normalized)
```
"""
NonNormalizedGaussian1D(g::Gaussian1D)::NonNormalizedGaussian1D = NonNormalizedGaussian1D(g.τ, g.ρ, 0.0)

"""
    mean(g::GaussianDist) -> Float64

Compute the expected value (mean) of a Gaussian distribution.

# Arguments
- `g::GaussianDist`: Any Gaussian distribution (normalized or non-normalized). Here, the parent type `GaussianDist` comes in handy.

# Returns
- `Float64`: The expected value, or `0.0` for uniform distributions

# Note
This is already implemented as an example. No need to change anything.
"""
function mean(g::T)::Float64 where T<:GaussianDist
    if g.ρ == 0.0   # Check if uniform
        return 0.0
    else
        return g.τ / g.ρ
    end
end

"""
    variance(g::GaussianDist) -> Float64

Compute the variance of a Gaussian distribution.

# Arguments
- `g::GaussianDist`: Any Gaussian distribution (normalized or non-normalized)

# Returns
- `Float64`: The variance, or `+Inf` for uniform distributions
"""
function variance(g::T)::Float64 where T<:GaussianDist
    if g.ρ == 0.0   # Check if uniform
        return +Inf
    else
        return 1 / g.ρ
    end
end

"""
    *(g1::Gaussian1D, g2::Gaussian1D) -> Gaussian1D

This operation multiplies two Gaussian pdfs.
It overwrites the base multiplication operator *.

# Arguments
- `g1::Gaussian1D`: First Gaussian distribution
- `g2::Gaussian1D`: Second Gaussian distribution

# Returns
- `Gaussian1D`: Product density with combined parameters

# Special Cases
- Multiplying by a uniform distribution returns the non-uniform distribution
- Multiplying two uniform distributions returns a uniform distribution

# Note
This is already implemented as an example. No need to change anything.
"""
Base.:*(g1::Gaussian1D, g2::Gaussian1D)::Gaussian1D = 
    Gaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ)

"""
    *(g1::NonNormalizedGaussian1D, g2::Gaussian1D) -> NonNormalizedGaussian1D

Multiply a non-normalized Gaussian distribution with a normalized Gaussian distribution (from the left).
It overwrites the base multiplication operator *.

# Arguments
- `g1::NonNormalizedGaussian1D`: Non-normalized Gaussian distribution
- `g2::Gaussian1D`: Normalized Gaussian distribution

# Returns
- `NonNormalizedGaussian1D`: Product distribution with updated normalization


# Examples
```julia-repl
julia> NonNormalizedGaussian1D(1,1,2) * Gaussian1D(2,3)
Non-normalized: N(μ = 0.75, σ² = 0.25, Z = 2.448691410992965) resp. G(τ = 3.0, ρ = 4.0, Z)

julia> NonNormalizedGaussian1D(1,1,2) * Gaussian1DUniform()
Non-normalized: N(μ = 1.0, σ² = 1.0, Z = 7.38905609893065) resp. G(τ = 1.0, ρ = 1.0, Z)

julia> NonNormalizedGaussian1D(Gaussian1DUniform()) * Gaussian1D(1,1)
Non-normalized: N(μ = 1.0, σ² = 1.0, Z = 1.0) resp. G(τ = 1.0, ρ = 1.0, Z)
```
"""
function Base.:*(g1::NonNormalizedGaussian1D, g2::Gaussian1D)::NonNormalizedGaussian1D
    v1, v2, m1, m2 = variance(g1), variance(g2), mean(g1), mean(g2)
    addlognorm = (is_uniform(g1) || is_uniform(g2)) ? 0 : 
        - 0.5 * (log(2*pi*(v1+v2)) + (m1-m2)^2 / (v1+v2))
    NonNormalizedGaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ, g1.log_norm + addlognorm)
end

"""
    *(g1::Gaussian1D, g2::NonNormalizedGaussian1D) -> NonNormalizedGaussian1D

Multiply a non-normalized Gaussian distribution with a normalized Gaussian distribution (now from the right).
This can be easily implemented using the left multiplication defined above.
"""
Base.:*(g1::Gaussian1D, g2::NonNormalizedGaussian1D)::NonNormalizedGaussian1D = 
    g2 * g1

"""
    /(g1::Gaussian1D, g2::Gaussian1D) -> Gaussian1D

Divide one normalized Gaussian distribution by another.

# Arguments
- `g1::Gaussian1D`: Numerator Gaussian distribution
- `g2::Gaussian1D`: Denominator Gaussian distribution

# Returns
- `Gaussian1D`: Quotient distribution

# Constraints
- The resulting precision must be non-negative (ρ₁ ≥ ρ₂)
- A uniform distribution can only be divided by another uniform distribution

# Error constraints
A division by a non-uniform when the numerator is uniform must throw the error "A uniform cannot be divided by anything else than a Gaussian uniform".
"""
function Base.:/(g1::Gaussian1D, g2::Gaussian1D)::Gaussian1D
    if (g1.ρ == 0.0) & (g2.ρ != 0.0) return error("Uniform numerator divided by non-uniform") end
    if g1.ρ < g2.ρ return error("Resulting precision is negative (ρ₁ < ρ₂)") end
    return Gaussian1D(g1.τ-g2.τ, g1.ρ-g2.ρ)
end

"""
    /(g1::NonNormalizedGaussian1D, g2::Gaussian1D) -> NonNormalizedGaussian1D

Divide a non-normalized Gaussian distribution by a normalized Gaussian distribution. This operation computes the quotient while carefully tracking the normalization constant. 

# Arguments
- `g1::NonNormalizedGaussian1D`: Numerator (non-normalized Gaussian distribution)
- `g2::Gaussian1D`: Denominator (normalized Gaussian distribution)

# Returns
- `NonNormalizedGaussian1D`: Quotient distribution with updated normalization

# Examples
```julia-repl
julia> NonNormalizedGaussian1D(2,5,1) / Gaussian1D(1,1)
Non-normalized: N(μ = 0.25, σ² = 0.25, Z = 9.540160496601143) resp. G(τ = 1.0, ρ = 4.0, Z)

julia> NonNormalizedGaussian1D(1,1,1) / Gaussian1DUniform()
Non-normalized: N(μ = 1.0, σ² = 1.0, Z = 2.718281828459045) resp. G(τ = 1.0, ρ = 1.0, Z)

julia> g = NonNormalizedGaussian1D(1,2,3)
Non-normalized: N(μ = 0.5, σ² = 0.5, Z = 20.085536923187668) resp. G(τ = 1.0, ρ = 2.0, Z)

julia> h = Gaussian1D(1,1)
N(μ = 1.0, σ² = 1.0) resp. G(τ = 1.0, ρ = 1.0)

julia> # Verify round trip: (g * h) / h = g
julia> (g / h) * h
Non-normalized: N(μ = 0.5, σ² = 0.5, Z = 20.085536923187675) resp. G(τ = 1.0, ρ = 2.0, Z)

julia> g
Non-normalized: N(μ = 0.5, σ² = 0.5, Z = 20.085536923187668) resp. G(τ = 1.0, ρ = 2.0, Z)
```
"""
function Base.:/(g1::NonNormalizedGaussian1D, g2::Gaussian1D)::NonNormalizedGaussian1D 
    v1, v2, m1, m2 = variance(g1), variance(g2), mean(g1), mean(g2)
    v_diff, m_diff = v2-v1, m1-m2
    addlognorm = (is_uniform(g1) || is_uniform(g2)) ? 0 : 
        log(v2) + 0.5 * (log(2 * pi / v_diff) + m_diff^2 / v_diff)
    return NonNormalizedGaussian1D(g1.τ - g2.τ, g1.ρ - g2.ρ, g1.log_norm + addlognorm)
end

"""
    KL_divergence(g1::T, g2::T) where T<:GaussianDist -> Float64

Compute the Kullback-Leibler divergence between two Gaussian distributions.

# Arguments
- `g1::T`: First distribution
- `g2::T`: Second distribution (must be same type as g1)

# Returns
- `Float64`: KL divergence value, or `+Inf` if either distribution is uniform

# Examples
```julia-repl
julia> g = NonNormalizedGaussian1D(1,2,3)
Non-normalized: N(μ = 0.5, σ² = 0.5, Z = 20.085536923187668) resp. G(τ = 1.0, ρ = 2.0, Z)

julia> h1 = Gaussian1D(1,1)
N(μ = 1.0, σ² = 1.0) resp. G(τ = 1.0, ρ = 1.0)

julia> h2 = Gaussian1D(2,3)
N(μ = 0.6666666666666666, σ² = 0.3333333333333333) resp. G(τ = 2.0, ρ = 3.0)

julia> u = Gaussian1DUniform()
Uniform()

julia> KL_divergence(g, g)
0.0

julia> KL_divergence(g, h1)
ERROR: MethodError: no method matching KL_divergence(::NonNormalizedGaussian1D, ::Gaussian1D)

julia> KL_divergence(h1, h2)
0.617360522332612

julia> KL_divergence(h2, h1)
0.2715283665562771

julia> KL_divergence(h2, u)
Inf

julia> KL_divergence(u, u)
0.0
```
"""

function KL_divergence(g1::GaussianDist, g2::GaussianDist)::Float64 
    if is_uniform(g1) && is_uniform(g2) return 0.0 end
    if is_uniform(g1) || is_uniform(g2) return Inf end
    v1, v2, m1, m2 = variance(g1), variance(g2), mean(g1), mean(g2)
    log(sqrt(v2/v1)) + (v1+(m1-m2)^2)/2/v2 - 0.5
end

"""
    is_uniform(g::GaussianDist) -> Bool

Check if a Gaussian distribution represents a uniform (improper) distribution.

A Gaussian distribution is considered uniform when its precision ρ equals zero.
In this case, the distribution is improper (doesn't integrate to 1) and represents
complete uncertainty over the real line. This is commonly used as a non-informative
prior in Bayesian inference.

# Arguments
- `g::GaussianDist`: Any Gaussian distribution (normalized or non-normalized)

# Returns
- `Bool`: `true` if the distribution is uniform (ρ = 0), `false` otherwise

# Examples
```julia
# Uniform distributions
uniform = Gaussian1DUniform()
is_uniform(uniform)  # true

# Proper distributions
proper = Gaussian1DFromMeanVariance(2.0, 1.0)
is_uniform(proper)   # false

# Non-normalized uniform
non_norm_uniform = NonNormalizedGaussian1D(0.0, 0.0, 1.0)
is_uniform(non_norm_uniform)  # true

# High precision (very non-uniform)
precise = Gaussian1DFromMeanVariance(0.0, 0.01)  # very small variance
is_uniform(precise)  # false
```
"""
is_uniform(g::GaussianDist) = (g.ρ == 0.0)

"""
    show(io::IO, g::Gaussian1D)

Pretty-print a normalized Gaussian distribution.

# Note
This function is already implemented. Nothing to change here.
"""
function Base.show(io::IO, g::Gaussian1D)
    if (is_uniform(g))
        print(io, "Uniform()")
    else
        print(io, "N(μ = ", mean(g), ", σ² = ", variance(g), ") resp. G(τ = ", g.τ, ", ρ = ", g.ρ, ")")
    end
end


"""
    show(io::IO, g::NonNormalizedGaussian1D)

Pretty-print a non-normalized Gaussian distribution.

# Note
This function is already implemented. Nothing to change here.
"""
function Base.show(io::IO, g::NonNormalizedGaussian1D)
    if (is_uniform(g))
        print(io, "Non-normalized: Uniform(Z = " , exp(g.log_norm), ")")
    else
        print(io, "Non-normalized: N(μ = ", mean(g), ", σ² = ", variance(g), ", Z = ", exp(g.log_norm), ") resp. G(τ = ", g.τ, ", ρ = ", g.ρ, ", Z)")
    end
end
 
"""
    distribution(g::GaussianDist) -> Distributions.Normal

This function creates a `Distributions.Normal` object from either a jnormalized or non-normalized Gaussian distribution defined in this module.
The conversion enables interoperability with the broader Julia ecosystem and access to standard distribution functions like sampling, PDF evaluation, CDF computation, and quantile functions.

# Arguments
- `g::GaussianDist`: Any Gaussian distribution from this module (Gaussian1D or NonNormalizedGaussian1D)

# Returns
- `Distributions.Normal`: Standard Normal distribution with mean μ and standard deviation σ

Note that Distributions.jl uses (μ, σ) while this module uses precision
parameterization (τ, ρ).

# Error Condition
Uniform distributions: Throws `DomainError` since uniform distributions over ℝ cannot be represented as proper Normal distributions (they have infinite variance)

# Examples
```julia-repl
julia> import Pkg; Pkg.add("Distributions")

julia> using Distributions

julia> g = Gaussian1DFromMeanVariance(2, 3)
N(μ = 2.0, σ² = 3.0) resp. G(τ = 0.6666666666666666, ρ = 0.3333333333333333)

julia> d = distribution(g)
Normal{Float64}(μ=2.0, σ=1.7320508075688772)

julia> mean(g)
2.0

julia> Distributions.mean(d)
2.0

julia> variance(g)
3.0

julia> Distributions.var(d)
2.9999999999999996

julia> rand(d)
4.027138196278347

julia> pdf(d, 0)
0.11825507390945918

julia> cdf(d, 2)
0.5

julia> quantile(d, .25)
0.8317494834759462

julia> distribution(NonNormalizedGaussian1D(1,2,3))
Normal{Float64}(μ=0.5, σ=0.7071067811865476)
```

# Error Examples
```julia-repl
julia> distribution(Gaussian1DUniform())
ERROR: DomainError with Uniform distribution cannot be converted to a Distributions.Normal object:
```
"""
distribution(g::GaussianDist) = 
    is_uniform(g) ? 
    throw(DomainError(g, "Uniform distribution cannot be converted to a Distributions.Normal object")) : 
    Distributions.Normal(mean(g), sqrt(variance(g))) 

export Gaussian1D, NonNormalizedGaussian1D, Gaussian1DFromMeanVariance, Gaussian1DUniform, is_uniform
export mean, variance, *, /, KL_divergence, show, distribution

end;
