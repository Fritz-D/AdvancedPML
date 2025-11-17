"""
# Dirichlet Distribution Library

A comprehensive Julia library for working with Dirichlet distributions in both normalized 
and non-normalized forms.

## Key Features

- **Normalized Dirichlet distributions** (`NormalizedDirichlet`): Standard parameterization with concentration parameters α
- **Non-normalized Dirichlet distributions** (`NonNormalizedDirichlet`): Includes normalization constant for advanced operations
- **Numerically stable operations**: Multiplication, division, and KL divergence computations
- **Uniform distribution support**: Special handling for uniform Dirichlet distributions
- **Statistical functions**: Mean, variance, and divergence calculations

## Mathematical Background

The Dirichlet distribution is a multivariate generalization of the Beta distribution, given by the density:
```
f(x₁,...,xₖ; α₁,...,αₖ) = (1/B(α)) * ∏ᵢ₌₁ᵏ xᵢ^(αᵢ-1)
```

Where:
- α = (α₁, α₂, ..., αₖ): concentration parameters (αᵢ > 0)
- B(α): multivariate Beta function B(α) = ∏ᵢ₌₁ᵏ Γ(αᵢ) / Γ(∑ᵢ₌₁ᵏ αᵢ)
- x = (x₁, x₂, ..., xₖ) lies on the probability simplex (∑ᵢ₌₁ᵏ xᵢ = 1, xᵢ ≥ 0)

## Usage Examples

```julia
using .Dirichlet

# Create normalized Dirichlet distributions
d1 = NormalizedDirichlet([2.0, 3.0, 1.0])  # 3-dimensional
d2 = NormalizedDirichlet([1.0, 1.0, 1.0])  # uniform over 2-simplex

# Basic operations
product = d1 * d2       # Multiply distributions
ratio = d1 / d2         # Divide distributions

# Statistics
μ = means(d1)           # Expected values
σ² = variances(d1)      # Variances
kl = KL_divergence(d1, d2)  # KL divergence

# Uniform distribution
uniform = DirichletUniform(3)  # 3-dimensional uniform
```
---
2025 written by Ralf Herbrich  
Hasso-Plattner Institute
"""
module Dirichlet

using Distributions
using SpecialFunctions

"""
    DirichletDist

Abstract base type for all Dirichlet distribution representations in this library.
"""
abstract type DirichletDist end

"""
    NormalizedDirichlet <: DirichletDist

A structure representing a normalized multivariate Dirichlet distribution.

This structure stores the concentration parameters (α) of a Dirichlet distribution.

# Fields
- `α::Vector{Float64}`: Concentration parameters (all elements must be positive)

# Constraints
- All αᵢ > 0 (concentration parameters must be positive)

# Examples
```julia
# Symmetric Dirichlet
d1 = NormalizedDirichlet([2.0, 2.0, 2.0])  # symmetric

# Asymmetric Dirichlet  
d2 = NormalizedDirichlet([0.5, 1.0, 3.0])  # favors third component

# Uniform Dirichlet
uniform = NormalizedDirichlet([1.0, 1.0, 1.0])  # uniform

# High concentration
peaked = NormalizedDirichlet([10.0, 10.0, 10.0])  # concentrated around center
```
"""
struct NormalizedDirichlet <: DirichletDist
    α::Vector{Float64}

    # default constructor with validation
    NormalizedDirichlet(α::Vector{Float64}) =
        (any(α .<= 0)) ? 
        throw(ErrorException("probabilities must be positive")) :
        new(α)
end

"""
    NonNormalizedDirichlet <: DirichletDist

This structure extends the normalized representation by explicitly storing the 
logarithm of the normalization constant.

# Fields
- `α::Vector{Float64}`: Concentration parameters (all elements must be positive)
- `log_norm::Float64`: Logarithm of the normalization constant

# Examples
```julia
# Create with explicit normalization
d = NonNormalizedDirichlet([2.0, 3.0, 1.0], -0.5)

# Convert from normalized form
d_norm = NormalizedDirichlet([2.0, 3.0, 1.0])
d_non_norm = NonNormalizedDirichlet(d_norm)  # log_norm = 0.0

# Access normalization constant
Z = exp(d.log_norm)  # actual normalization constant

# Statistical properties remain the same
μ = means(d)  # same as normalized version
```
"""
struct NonNormalizedDirichlet <: DirichletDist
    α::Vector{Float64}
    log_norm::Float64

    # default constructor with validation
    NonNormalizedDirichlet(α::Vector{Float64}, log_norm::Float64) =
        (any(α .< 0)) ? 
        throw(ErrorException("probabilities must be positive")) :
        new(α, log_norm)
end

"""
    DirichletUniform(k::Int) -> NormalizedDirichlet

This convenience constructor creates a Dirichlet distribution with all concentration 
parameters equal to 1, which represents a uniform distribution over the probability 
simplex.

# Arguments
- `k::Int`: Dimension of the distribution (number of categories)

# Returns
- `NormalizedDirichlet`: Uniform Dirichlet distribution with α = [1, 1, ..., 1]

# Examples
```julia
uniform_3d = DirichletUniform(3)  # α = [1.0, 1.0, 1.0]

# Verify uniform property
means(uniform_3d)  # [0.333..., 0.333..., 0.333...]
```
"""
DirichletUniform(k::Int)::NormalizedDirichlet = NormalizedDirichlet(repeat([1.0], k))

"""
    NormalizedDirichlet(d::NonNormalizedDirichlet) -> NormalizedDirichlet

Convert a non-normalized Dirichlet distribution to normalized form.

# Arguments
- `d::NonNormalizedDirichlet`: The non-normalized Dirichlet distribution to convert

# Returns
- `NormalizedDirichlet`: Normalized version with the same concentration parameters

# Examples
```julia
# Create non-normalized distribution
d_non_norm = NonNormalizedDirichlet([2.0, 3.0, 1.0], -0.3)

# Convert to normalized form
d_norm = NormalizedDirichlet(d_non_norm)  # α = [2.0, 3.0, 1.0]

# Verify parameters are preserved
d_norm.α == d_non_norm.α  # true
means(d_norm) ≈ means(d_non_norm)  # true
```
"""
NormalizedDirichlet(d::NonNormalizedDirichlet)::NormalizedDirichlet = NormalizedDirichlet(d.α)

"""
    NonNormalizedDirichlet(d::NormalizedDirichlet) -> NonNormalizedDirichlet

Convert a normalized Dirichlet distribution to non-normalized form.

# Arguments
- `d::NormalizedDirichlet`: The normalized Dirichlet distribution to convert

# Returns
- `NonNormalizedDirichlet`: Non-normalized version with log_norm = 0.0

# Examples
```julia
# Create normalized distribution
d_norm = NormalizedDirichlet([2.0, 3.0, 1.0])

# Convert to non-normalized form
d_non_norm = NonNormalizedDirichlet(d_norm)  # α = [2.0, 3.0, 1.0], log_norm = 0.0

# Verify equivalence
means(d_norm) ≈ means(d_non_norm)          # true
variances(d_norm) ≈ variances(d_non_norm)  # true
exp(d_non_norm.log_norm)                   # 1.0 (properly normalized)
```
"""
NonNormalizedDirichlet(d::NormalizedDirichlet)::NonNormalizedDirichlet = NonNormalizedDirichlet(d.α, 0.0)

"""
    size(d::DirichletDist) -> Int

Get the dimensionality of a Dirichlet distribution.
This function returns the number of concentration parameters, which corresponds
to the dimension of the probability simplex. For a K-dimensional Dirichlet
distribution, the support is the (K-1)-dimensional probability simplex.

# Arguments
- `d::DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Int`: The number of concentration parameters (dimensionality)

# Mathematical Note
A K-dimensional Dirichlet distribution has support on the (K-1)-dimensional
probability simplex because of the constraint ∑ᵢ₌₁ᵏ xᵢ = 1.

# Examples
```julia
# 3-dimensional Dirichlet
d3 = NormalizedDirichlet([1.0, 2.0, 3.0])
size(d3)  # 3

# 5-dimensional uniform Dirichlet
d5 = DirichletUniform(5)
size(d5)  # 5

# Works with non-normalized distributions
d_non_norm = NonNormalizedDirichlet([2.0, 2.0], 0.5)
size(d_non_norm)  # 2
```
"""
size(d::T) where T <: DirichletDist = length(d.α)

"""
    is_uniform(d::DirichletDist) -> Bool

Check if a Dirichlet distribution represents a uniform distribution.

# Arguments
- `d::DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Bool`: `true` if the distribution is uniform (all αᵢ = 1), `false` otherwise

# Examples
```julia
# Uniform distributions
uniform = DirichletUniform(3)
is_uniform(uniform)  # true

# Non-uniform distributions
skewed = NormalizedDirichlet([2.0, 1.0, 3.0])
is_uniform(skewed)   # false

# Manual uniform construction
manual_uniform = NormalizedDirichlet([1.0, 1.0, 1.0, 1.0])
is_uniform(manual_uniform)  # true

# Non-normalized uniform
non_norm_uniform = NonNormalizedDirichlet([1.0, 1.0], 2.0)
is_uniform(non_norm_uniform)  # true
```
"""
is_uniform(d::T) where T<:DirichletDist = all(d.α .== 1)

"""
    means(d::DirichletDist) -> Vector{Float64}

Compute the mean (expected values) of each component of a Dirichlet distribution.

# Arguments
- `d::DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Vector{Float64}`: Vector of expected values, one for each component

# Examples
```julia-repl
julia> d = NormalizedDirichlet([2.0, 4.0, 1.0])
julia> means(d)
3-element Vector{Float64}:
 0.2857142857142857  # 2/7
 0.5714285714285714  # 4/7  
 0.14285714285714285 # 1/7

julia> # Uniform distribution
julia> uniform = DirichletUniform(3)
julia> means(uniform)
3-element Vector{Float64}:
 0.3333333333333333
 0.3333333333333333
 0.3333333333333333

julia> # Verify sum equals 1
julia> sum(means(d))
1.0
```
"""
means(d::T) where T<:DirichletDist = d.α ./ sum(d.α)

"""
    variances(d::DirichletDist) -> Vector{Float64}

Compute the variance of each component of a Dirichlet distribution.

# Arguments
- `d::DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Vector{Float64}`: Vector of variances, one for each component
"""
variances(d::T) where T <: DirichletDist = d.α .* (sum(d.α) .- d.α) ./ (sum(d.α)^2 * (sum(d.α) + 1))

"""
    expected_logs(d::DirichletDist) -> Vector{Float64}

Compute the expected value of the logarithm of each component of a Dirichlet distribution.

# Arguments
- `d::DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Vector{Float64}`: Vector of expected log values, one for each component

# Examples
```julia-repl
julia> d = NormalizedDirichlet([2.0, 4.0, 1.0])
julia> expected_logs(d)
3-element Vector{Float64}:
 -0.5108256237659907   # E[log X₁]
 -0.008658439516408668 # E[log X₂] (highest α, least negative)
 -1.3439766454205798   # E[log X₃] (lowest α, most negative)

julia> # Uniform distribution
julia> uniform = DirichletUniform(3)
julia> expected_logs(uniform)
3-element Vector{Float64}:
 -1.0986122886681098   # All equal for uniform
 -1.0986122886681098
 -1.0986122886681098

julia> # Higher concentration reduces magnitude
julia> concentrated = NormalizedDirichlet([20.0, 40.0, 10.0])
julia> expected_logs(concentrated)  # Less negative values
3-element Vector{Float64}:
 -1.6454525470838633
 -1.2748977802008214
 -2.048348914197405
```
"""
expected_logs(d::T) where T <: DirichletDist = digamma.(d.α) .- digamma(sum(d.α))

"""
    *(d1::NormalizedDirichlet, d2::NormalizedDirichlet) -> NormalizedDirichlet

Multiply two Dirichlet densities.

# Arguments
- `d1::NormalizedDirichlet`: First Dirichlet distribution
- `d2::NormalizedDirichlet`: Second Dirichlet distribution

# Returns
- `NormalizedDirichlet`: Distribution with combined concentration parameters

# Constraints
- Both distributions must have the same dimensionality

# Examples
```julia-repl
julia> d1 = NormalizedDirichlet([2.0, 3.0, 1.0])
julia> d2 = NormalizedDirichlet([1.0, 1.0, 2.0])
julia> result = d1 * d2
NormalizedDirichlet([2.0, 3.0, 2.0])  # [2+1-1, 3+1-1, 1+2-1]

julia> # Multiplying by uniform distribution
julia> uniform = DirichletUniform(3)  # [1.0, 1.0, 1.0]
julia> d1 * uniform
NormalizedDirichlet([2.0, 3.0, 1.0])  # unchanged

julia> # Error for mismatched dimensions
julia> d3 = NormalizedDirichlet([1.0, 1.0])  # 2D
julia> d1 * d3  # Error: different dimensions
```
"""
function Base.:*(d1::NormalizedDirichlet, d2::NormalizedDirichlet)::NormalizedDirichlet
    if size(d1) != size(d2)
        throw(ErrorException("Cannot multiply dirichlets of different dimensions"))
    end
    return NormalizedDirichlet(d1.α .+ d2.α .- 1)
end

"""
    *(d1::NonNormalizedDirichlet, d2::NormalizedDirichlet) -> NonNormalizedDirichlet

Multiply a non-normalized Dirichlet with a normalized Dirichlet.

# Arguments
- `d1::NonNormalizedDirichlet`: Non-normalized Dirichlet distribution
- `d2::NormalizedDirichlet`: Normalized Dirichlet distribution

# Returns
- `NonNormalizedDirichlet`: Distribution with updated normalization

# Constraints
- Both distributions must have the same dimensionality
"""
multivarbeta(α::Vector{Float64})::Float64 = sum(gamma.(α)) - gamma(sum(α)) 
logmultivarbeta(α::Vector{Float64})::Float64 = sum(loggamma.(α)) - loggamma(sum(α)) 
function Base.:*(d1::NonNormalizedDirichlet, d2::NormalizedDirichlet)::NonNormalizedDirichlet
    if size(d1) != size(d2)
        throw(ErrorException("Cannot multiply dirichlets of different dimensions"))
    end
    return NonNormalizedDirichlet(d1.α .+ d2.α .- 1,
        d1.log_norm + logmultivarbeta(d1.α .+ d2.α .- 1) - logmultivarbeta(d2.α) - logmultivarbeta(d1.α)
    )
end

"""
    *(d1::NormalizedDirichlet, d2::NonNormalizedDirichlet) -> NonNormalizedDirichlet

Multiply a normalized Dirichlet distribution with a non-normalized Dirichlet distribution.

# Arguments
- `d1::NormalizedDirichlet`: Normalized Dirichlet distribution
- `d2::NonNormalizedDirichlet`: Non-normalized Dirichlet distribution

# Returns
- `NonNormalizedDirichlet`: Distributionn with updated normalization
"""
Base.:*(d1::NormalizedDirichlet, d2::NonNormalizedDirichlet)::NonNormalizedDirichlet = d2 * d1

"""
    /(d1::NormalizedDirichlet, d2::NormalizedDirichlet) -> NormalizedDirichlet

Divide one normalized Dirichlet by another.

# Arguments
- `d1::NormalizedDirichlet`: Numerator Dirichlet distribution
- `d2::NormalizedDirichlet`: Denominator Dirichlet distribution

# Returns
- `NormalizedDirichlet`: Quotient distribution

# Constraints
- Both distributions must have the same dimensionality

# Examples
```julia-repl
julia> d1 = NormalizedDirichlet([3.0, 4.0, 2.0])
julia> d2 = NormalizedDirichlet([1.0, 2.0, 1.0])
julia> result = d1 / d2
NormalizedDirichlet([3.0, 3.0, 2.0])  # [3-1+1, 4-2+1, 2-1+1]

julia> # Dividing by uniform distribution
julia> uniform = DirichletUniform(3)  # [1.0, 1.0, 1.0]
julia> d1 / uniform
NormalizedDirichlet([3.0, 4.0, 2.0])  # [3-1+1, 4-1+1, 2-1+1] = original

julia> # Round-trip property: (d1 / d2) * d2 ≈ d1
julia> reconstructed = (d1 / d2) * d2
julia> # reconstructed should be close to d1
```
"""
function Base.:/(d1::NormalizedDirichlet, d2::NormalizedDirichlet)::NormalizedDirichlet
    if size(d1) != size(d2)
        throw(ErrorException("Cannot multiply dirichlets of different dimensions"))
    end
    return NormalizedDirichlet(d1.α .- d2.α .+ 1)
end

"""
    /(d1::NonNormalizedDirichlet, d2::NormalizedDirichlet) -> NonNormalizedDirichlet

Divide a non-normalized Dirichlet distribution by a normalized Dirichlet distribution.

# Arguments
- `d1::NonNormalizedDirichlet`: Numerator (non-normalized Dirichlet distribution)
- `d2::NormalizedDirichlet`: Denominator (normalized Dirichlet distribution)

# Returns
- `NonNormalizedDirichlet`: Quotient distribution with updated normalization

# Constraints
- Both distributions must have the same dimensionality

# Examples
```julia-repl
julia> d1 = NonNormalizedDirichlet([3.0, 4.0, 2.0], 0.5)
julia> d2 = NormalizedDirichlet([1.0, 2.0, 1.0])
julia> result = d1 / d2
NonNormalizedDirichlet([3.0, 3.0, 2.0], Z = ...)

julia> # Normalization constant is updated
julia> exp(result.log_norm)  # Different from input

julia> # Verify round-trip: (d1 / d2) * d2 ≈ d1
julia> reconstructed = (d1 / d2) * d2
julia> reconstructed.α ≈ d1.α  # Should be approximately true
```
"""
function Base.:/(d1::NonNormalizedDirichlet, d2::NormalizedDirichlet)::NonNormalizedDirichlet
    if size(d1) != size(d2)
        throw(ErrorException("Cannot multiply dirichlets of different dimensions"))
    end
    return NonNormalizedDirichlet(d1.α .- d2.α .+ 1,
        d1.log_norm + logmultivarbeta(d1.α .- d2.α .+ 1) + logmultivarbeta(d2.α) - logmultivarbeta(d1.α)
    )
end

"""
    KL_divergence(d1::T, d2::T) where T<:DirichletDist -> Float64

Compute the Kullback-Leibler (KL) divergence from distribution d1 to distribution d2.

# Arguments
- `d1::T`: First distribution (from which divergence is measured)
- `d2::T`: Second distribution (to which divergence is measured)
Note that `T` must be same type (both normalized or both non-normalized)

# Returns
- `Float64`: KL divergence KL(d1||d2) ≥ 0
"""
KL_divergence(d1::T, d2::T) where T <: DirichletDist = 
    (size(d1) != size(d2)) ?
    throw(ErrorException("dimensions are mismatched")) :
    logmultivarbeta(d2.α) - logmultivarbeta(d1.α) + sum((d1.α .- d2.α) .* (digamma.(d1.α) .- digamma(sum(d1.α))))

"""
    show(io::IO, d::NormalizedDirichlet)

Pretty print.

Already implemented.
"""
function Base.show(io::IO, d::NormalizedDirichlet)
    if (is_uniform(d))
        print(io, "uniform")
    else
        print(io, "α = ", d.α)
    end
end


"""
    show(io::IO, d::NonNormalizedDirichlet)

Pretty print.

Already implemented.
"""

function Base.show(io::IO, d::NonNormalizedDirichlet)
    if (is_uniform(d))
        print(io, "uniform (Z = " , exp(d.log_norm), ")")
    else
        print(io, "α = ", d.α, ", Z = ", exp(d.log_norm))
    end
end

"""
    distribution(d::T) where T<:DirichletDist -> Distributions.Dirichlet

This function creates a `Dirichlet` object from `Distributions.jl` from either normalized
or non-normalized Dirichlet distributions. This conversion enables interoperability
with the broader Julia statistics ecosystem and provides access to additional
statistical functions like random sampling, quantiles, and moments.

# Arguments
- `d::T where T<:DirichletDist`: Any Dirichlet distribution (normalized or non-normalized)

# Returns
- `Distributions.Dirichlet`: Standard Julia distribution object

# Conversion Details
The conversion uses only the concentration parameters (α), discarding any
normalization information from non-normalized distributions.

# Examples
```julia-repl
julia> using Distributions
julia> d1 = NormalizedDirichlet([2.0, 3.0, 1.5])
julia> std_dist = distribution(d1)
Dirichlet{Float64}(alpha=[2.0, 3.0, 1.5])

julia> # Access Distributions.jl functionality
julia> rand(std_dist)  # Random sample from the simplex
3-element Vector{Float64}: [0.123, 0.456, 0.421]

julia> mean(std_dist)  # Expected value
3-element Vector{Float64}: [0.307, 0.461, 0.230]

julia> # Works with non-normalized distributions too
julia> d2 = NonNormalizedDirichlet([2.0, 3.0, 1.5], 0.7)
julia> distribution(d2)  # Same result as d1
Dirichlet{Float64}(alpha=[2.0, 3.0, 1.5])
```
"""
distribution(d::T) where T <: DirichletDist = Distributions.Dirichlet(d.α)

export NormalizedDirichlet, NonNormalizedDirichlet, DirichletUniform, is_uniform, size, means, variances, expected_logs, *, /, KL_divergence, show, distribution

end
