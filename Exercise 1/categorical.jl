"""
# Categorical Distribution Library

A comprehensive Julia library for working with categorical distributions using log-probability parameterization. 

## Key Features

- **Log-probability parameterization**: Numerically stable representation using η = log(π)
- **Normalized categorical distributions** (`NormalizedCategorical`): Standard probability distributions
- **Non-normalized categorical distributions** (`NonNormalizedCategorical`): Includes normalization constant for factor operations
- **Numerically stable operations**: Multiplication, division, and KL divergence computations in log-space
- **Uniform distribution support**: Special handling for uniform (maximum entropy) categorical distributions
- **Statistical functions**: Mean, variance, and information-theoretic measures

## Mathematical Background

### Standard Parameterization
The categorical distribution models discrete outcomes with K categories.
In standard form, it uses probability vector π = [π₁, π₂, ..., πₖ] where ∑ᵢ πᵢ = 1.

### Log-Probability Parameterization  
This library uses log-probability parameterization for numerical stability:
```
η = log(π) = [log(π₁), log(π₂), ..., log(πₖ)]
π = exp(η) / ∑ⱼ exp(ηⱼ)  (softmax normalization)
```

### Probability Mass Function
For a categorical distribution with log-probabilities η:
```
P(X = k) = exp(ηₖ) / ∑ⱼ exp(ηⱼ)
```

## Advantages of Log-Parameterization
- **Avoids underflow**: Small probabilities represented exactly in log-space
- **Stable operations**: Addition/subtraction in log-space corresponds to multiplication/division
- **Natural operations**: Factor graph operations map directly to log-space arithmetic
- **Reduced normalization**: Many operations can defer expensive softmax computation

## Usage Examples

```julia
using .Categorical

# Create distributions using log-probabilities
η = [0.0, -0.5, -1.0]  # log([1.0, 0.607, 0.368])
g1 = NormalizedCategorical(η)

# Create uniform distribution
uniform = CategoricalUniform(3)

# Create from probabilities (converted to log-space)
g2 = CategoricalFromProbabilities([0.5, 0.3, 0.2])

# Basic operations (performed in log-space)
product = g1 * g2       # Element-wise addition of log-probabilities
quotient = g1 / g2      # Element-wise subtraction of log-probabilities

# Statistics (converted from log-space)
π = means(g1)          # Probability vector
σ² = variances(g1)     # Variance vector
kl = KL_divergence(g1, g2)  # KL divergence

# Interoperability
std_dist = distribution(g1)  # Convert to Distributions.jl
```
---
© 2025 by Ralf Herbrich  
Hasso-Plattner Institute
"""

module Categorical

using Distributions

"""
    CategoricalDist

Abstract type for categorical distribution representations.
"""

abstract type CategoricalDist end

"""
    NormalizedCategorical <: CategoricalDist

Represents a categorical distribution where the probability vector π satisfies
the normalization constraint ∑ᵢ πᵢ = 1. 

# Fields
- `η::Vector{Float64}`: Log-probabilities [log(π₁), log(π₂), ..., log(πₖ)]
"""

struct NormalizedCategorical <: CategoricalDist
    η::Vector{Float64}

    # default constructor with validation
    NormalizedCategorical(η::Vector{Float64}) =
        (any(exp.(η) .< 0)) ? 
        error("probabilities must be non-negative") :
        new(η)
end

"""
    NonNormalizedCategorical <: CategoricalDist

Represents a categorical distribution where the log-probability vector η may not
correspond to a normalized distribution, along with an explicit log normalization 
constant. 

# Fields
- `η::Vector{Float64}`: Unnormalized log-probabilities
- `log_norm::Float64`: Logarithm of the normalization constant

# Constructor Usage
```julia
NonNormalizedCategorical([0.0, -0.5, -1.0], 0.2)  # Valid: log-probabilities
```

# Examples
```julia-repl
julia> # Factor graph message (unnormalized log-probabilities)
julia> message = NonNormalizedCategorical([1.0, 0.5, 0.0], -0.5)
π = [0.422, 0.256, 0.155], Z = 0.606...

julia> # Access components
julia> message.η
3-element Vector{Float64}: [1.0, 0.5, 0.0]

julia> message.log_norm
-0.5

julia> # Normalization constant
julia> exp(message.log_norm)
0.6065306597126334

julia> # Convert to normalized form when needed
julia> normalized = NormalizedCategorical(message)
```
"""

struct NonNormalizedCategorical <: CategoricalDist
    η::Vector{Float64}
    log_norm::Float64

    # default constructor with validation
    NonNormalizedCategorical(η::Vector{Float64}, log_norm::Float64) =
        (any(exp.(η) .< 0)) ? 
        error("probabilities must be non-negative") :
        new(η, log_norm)
end

"""
    CategoricalUniform(k::Int) -> NormalizedCategorical

Create a uniform categorical distribution over k categories.

# Arguments
- `k::Int`: Number of categories (must be positive)

# Returns
- `NormalizedCategorical`: Uniform distribution with π = [1/k, 1/k, ..., 1/k]

# Error Conditions
- Throws ArgumentError if k ≤ 0 or not an integer
"""

CategoricalUniform(k::Int) = 
    (k <= 0 | !isinteger(k)) ? 
    throw(ArgumentError("k must be a positive integer")) : 
    NormalizedCategorical(log.(repeat([1/k], k)))

"""
    CategoricalFromProbabilities(π::Vector{Float64}) -> NormalizedCategorical

This convenience constructor takes a standard probability vector π (where ∑πᵢ = 1)
and converts it to the internal log-probability representation η = log(π).

# Arguments
- `π::Vector{Float64}`: Probability vector where ∑πᵢ = 1 and πᵢ ≥ 0

# Returns
- `NormalizedCategorical`: Distribution with η = log(π)

# Examples
```julia-repl
julia> # Create from standard probabilities
julia> probs = [0.5, 0.3, 0.2]
julia> dist = CategoricalFromProbabilities(probs)

julia> # Extract probabilities (should match input)
julia> means(dist) ≈ probs
true

julia> # Compare with direct log-probability construction
julia> log_probs = log.(probs)
julia> direct = NormalizedCategorical(log_probs)
julia> means(direct) ≈ means(dist)
true

julia> # Handles small probabilities safely
julia> tiny_probs = [0.99, 0.009, 0.001]
julia> safe_dist = CategoricalFromProbabilities(tiny_probs)
julia> means(safe_dist) ≈ tiny_probs
true
```

# Error conditions
Throw ArgumentError if one of the following is invalidated:
- π must be non-negative
- π must sum to one (within tolerance)
"""

CategoricalFromProbabilities(π::Vector{Float64}) = 
    (any(π .< 0) | !(sum(π) ≈ 1)) ?
    throw(ArgumentError("Probabilities must be non-negative and sum to one")) :
    NormalizedCategorical(log.(π))

"""
    NormalizedCategorical(d::NonNormalizedCategorical) -> NormalizedCategorical

Convert a non-normalized categorical distribution to normalized form.

# Arguments
- `d::NonNormalizedCategorical`: Input unnormalized distribution

# Returns
- `NormalizedCategorical`: Properly normalized categorical distribution

# Examples
```julia-repl
julia> # Convert unnormalized factor result
julia> unnorm = NonNormalizedCategorical([0.5, 0.3, 0.2], -1.0)
non-normalized

julia> norm = NormalizedCategorical(unnorm)
categorical

julia> norm.π
3-element Vector{Float64}: [0.5, 0.3, 0.2]

julia> # Verify normalization
julia> sum(norm.π)
1.0
```
"""

NormalizedCategorical(d::NonNormalizedCategorical) = 
    NormalizedCategorical(d.η)

"""
    NonNormalizedCategorical(d::NormalizedCategorical) -> NonNormalizedCategorical

Convert a normalized categorical distribution to non-normalized form.

# Arguments
- `d::NormalizedCategorical`: Input normalized distribution

# Returns
- `NonNormalizedCategorical`: Equivalent unnormalized representation

# Examples
```julia-repl
julia> # Convert normalized distribution
julia> norm = NormalizedCategorical([0.4, 0.3, 0.3])
categorical

julia> unnorm = NonNormalizedCategorical(norm)
non-normalized

julia> means(unnorm)
3-element Vector{Float64}: [0.4, 0.3, 0.3]

julia> unnorm.log_norm
0.0

julia> # Round-trip conversion preserves probabilities
julia> norm2 = NormalizedCategorical(unnorm)
julia> norm2.η ≈ norm.η
true
```
"""

NonNormalizedCategorical(d::NormalizedCategorical) = 
    NonNormalizedCategorical(d.η, 0.0)

"""
    size(d::CategoricalDist) -> Int

Return the number of categories in a categorical distribution.

# Arguments
- `d::CategoricalDist`: Any categorical distribution (normalized or unnormalized)

# Returns
- `Int`: Number of categories in the distribution
"""

function size(d::T) where T<:CategoricalDist
    return length(d.η)
end

"""
    is_uniform(d::CategoricalDist) -> Bool

Check if a categorical distribution is uniform across all categories.

# Arguments
- `d::CategoricalDist`: Any categorical distribution to test

# Returns
- `Bool`: true if uniform (all probabilities equal), false otherwise

# Special Cases
- Single category distributions are always uniform
- Empty distributions return true by convention
- Works with both normalized and unnormalized distributions
"""
function is_uniform(d::T) where T <:CategoricalDist
    return sum(d.η) == 0 ? true : all(d.η .≈ sum(d.η)/size(d))
end

"""
    means(d::CategoricalDist) -> Vector{Float64}

Return the mean (expected value) vector of a categorical distribution.

# Arguments
- `d::CategoricalDist`: Any categorical distribution

# Returns
- `Vector{Float64}`: Probability vector [π₁, π₂, ..., πₖ] where πᵢ = exp(ηᵢ)/∑exp(ηⱼ)
"""
function means(d::T) where T <: CategoricalDist
    return exp.(d.η) ./ sum(exp.(d.η))
end

"""
    variances(d::CategoricalDist) -> Vector{Float64}

Return the variance vector of a categorical distribution.

# Arguments
- `d::CategoricalDist`: Any categorical distribution

# Returns
- `Vector{Float64}`: Variance vector [π₁(1-π₁), π₂(1-π₂), ..., πₖ(1-πₖ)]
"""
function variances(d::T) where T <: CategoricalDist
    return exp.(d.η) .* (1 .- exp.(d.η))
end

"""
    *(d1::NormalizedCategorical, d2::NormalizedCategorical) -> NormalizedCategorical

Multiply two normalized categorical distributions element-wise in log-space.
Performs element-wise addition of log-probabilities and maintains proper
normalization through the softmax operation.

# Arguments
- `d1::NormalizedCategorical`: First categorical distribution
- `d2::NormalizedCategorical`: Second categorical distribution (must have same size)

# Returns
- `NormalizedCategorical`: Normalized product distribution


# Examples
```julia-repl
julia> # Combine two log-probability assessments
julia> prior = NormalizedCategorical([0.0, -0.5, -1.0])    # log([1.0, 0.607, 0.368])
julia> likelihood = NormalizedCategorical([-0.5, -1.6, -1.6])  # log([0.6, 0.2, 0.2])
julia> posterior = prior * likelihood
julia> means(posterior)  # Convert to probabilities
3-element Vector{Float64}: [0.545..., 0.136..., 0.136...]

julia> # Verify: this equals element-wise product normalized
julia> p1, p2 = means(prior), means(likelihood)
julia> expected = (p1 .* p2) ./ sum(p1 .* p2)
log((p1 .* p2) ./ sum(p1 .* p2))
(d1.η .+ d2.η) .- log(sum(exp.(d1.η) .* exp.(d2.η)))
julia> means(posterior) ≈ expected
true

julia> # Uniform prior (all zeros) doesn't change likelihood
julia> uniform = CategoricalUniform(3)  # η = [0.0, 0.0, 0.0]
julia> result = uniform * likelihood
julia> means(result) ≈ means(likelihood)
true
```

# Error Conditions
- Throws ArgumentError("Cannot multiply categorical distributions with different dimensions") if distributions have different dimensions
- Handles -Inf log-probabilities (zero probabilities) naturally
"""
function Base.:*(d1::NormalizedCategorical, d2::NormalizedCategorical)::NormalizedCategorical
    if (size(d1) != size(d2))
        throw(ArgumentError("Cannot multiply categorical distributions with different dimensions"))
    end
    return NormalizedCategorical((d1.η .+ d2.η) .- log(sum(exp.(d1.η .+ d2.η))))
end

"""
    *(d1::NonNormalizedCategorical, d2::NormalizedCategorical) -> NonNormalizedCategorical

Multiply a non-normalized and normalized categorical distribution in log-space.
Performs element-wise addition of log-probabilities while properly tracking the 
normalization constant. 

# Arguments
- `d1::NonNormalizedCategorical`: Non-normalized categorical distribution
- `d2::NormalizedCategorical`: Normalized categorical distribution (same size required)

# Returns
- `NonNormalizedCategorical`: Product with updated normalization constant

# Examples
```julia-repl
julia> # Factor graph message update in log-space
julia> message = NonNormalizedCategorical([1.0, 0.5, 0.0], -0.3)  # log-probabilities
julia> prior = NormalizedCategorical([-0.916, -0.916, -1.609])     # log([0.4, 0.4, 0.2])
julia> updated = message * prior
π = [0.615..., 0.307..., 0.076...], Z = 0.462...

julia> # Log normalization properly tracked
julia> updated.log_norm
-1.073...

julia> # Convert to normalized form when needed
julia> final = NormalizedCategorical(updated)
```

# Error Conditions
- Throws ArgumentError("Cannot multiply categorical distributions with different dimensions") if distributions have different dimensions
"""
function Base.:*(d1::NonNormalizedCategorical, d2::NormalizedCategorical)::NonNormalizedCategorical
    if size(d1) != size(d2)
        throw(ArgumentError("Cannot multiply categorical distributions with different dimensions"))
    end
    return NonNormalizedCategorical((d1.η .+ d2.η), 
        d1.log_norm + log(sum(exp.(d1.η .+ d2.η))) - log(sum(exp.(d1.η))) - log(sum(exp.(d2.η)))
    )
end

"""
    *(d1::NormalizedCategorical, d2::NonNormalizedCategorical) -> NonNormalizedCategorical

Multiply a normalized and non-normalized categorical distribution.

# Arguments
- `d1::NormalizedCategorical`: Normalized categorical distribution
- `d2::NonNormalizedCategorical`: Non-normalized categorical distribution

# Returns
- `NonNormalizedCategorical`: Same result as d2 * d1
"""
Base.:*(d1::NormalizedCategorical, d2::NonNormalizedCategorical) = d2 * d1

"""
    /(d1::NormalizedCategorical, d2::NormalizedCategorical) -> NormalizedCategorical

Divide one normalized categorical distribution by another element-wise in log-space.
Performs element-wise subtraction of log-probabilities with automatic normalization.

# Arguments
- `d1::NormalizedCategorical`: Numerator distribution
- `d2::NormalizedCategorical`: Denominator distribution (must have same size)

# Returns
- `NormalizedCategorical`: Normalized quotient distribution

# Error Conditions
- Throws ArgumentError("Cannot divide categorical distributions with different dimensions") if distributions have different dimensions
- Handles extreme log-probability differences gracefully through softmax
"""
function Base.:/(d1::NormalizedCategorical, d2::NormalizedCategorical)::NormalizedCategorical
    if size(d1) != size(d2)
        throw(ArgumentError("Cannot divide categorical distributions with different dimensions"))
    end
    return NormalizedCategorical(d1.η .- d2.η .- log(sum(exp.(d1.η .- d2.η))))
end

"""
    /(d1::NonNormalizedCategorical, d2::NormalizedCategorical) -> NonNormalizedCategorical

Divide a non-normalized categorical distribution by a normalized one.

# Arguments
- `d1::NonNormalizedCategorical`: Non-normalized numerator distribution
- `d2::NormalizedCategorical`: Normalized denominator distribution (same size required)

# Returns
- `NonNormalizedCategorical`: Quotient with updated normalization constant

# Error Conditions
- Throws ArgumentError("Cannot divide categorical distributions with different dimensions") if distributions have different dimensions
"""
function Base.:/(d1::NonNormalizedCategorical, d2::NormalizedCategorical)::NonNormalizedCategorical
    if size(d1) != size(d2)
        throw(ArgumentError("Cannot divide categorical distributions with different dimensions"))
    end
    return NonNormalizedCategorical(d1.η .- d2.η, 
        d1.log_norm + log(sum(exp.(d1.η .- d2.η))) + log(sum(exp.(d2.η))) - log(sum(exp.(d1.η)))
    )
end

"""
    KL_divergence(d1::CategoricalDist, d2::CategoricalDist) -> Float64

Compute the Kullback-Leibler divergence from distribution d2 to d1.

# Arguments
- `d1::CategoricalDist`: Target distribution (can be normalized or unnormalized)
- `d2::CategoricalDist`: Reference distribution (same type and size as d1)

# Returns
- `Float64`: KL divergence D(d1 || d2) in nats (natural logarithm units)

# Examples
```julia-repl
julia> # KL divergence between uniform and skewed distributions (log-space)
julia> uniform = CategoricalUniform(3)  # η = [0.0, 0.0, 0.0]
julia> skewed = NormalizedCategorical(log.([0.1, 0.8, 0.1]))  # Convert to log-space
julia> KL_divergence(skewed, uniform)
0.822...  # High divergence due to concentration

julia> KL_divergence(uniform, skewed)
0.311...  # Different value (asymmetric)

julia> # Self-divergence is always zero
julia> dist = NormalizedCategorical(log.([0.3, 0.4, 0.3]))
julia> KL_divergence(dist, dist)
0.0

julia> # Measure convergence in iterative algorithms
julia> prior = CategoricalUniform(2)  # η = [0.0, 0.0]
julia> posterior = NormalizedCategorical(log.([0.8, 0.2]))  # η = [-0.223, -1.609]
julia> information_gain = KL_divergence(posterior, prior)
0.500...  # Information gained through observation

julia> # Direct log-probability specification
julia> log_dist1 = NormalizedCategorical([0.0, -1.0, -2.0])  # log([1.0, 0.368, 0.135])
julia> log_dist2 = NormalizedCategorical([0.0, 0.0, 0.0])    # uniform
julia> KL_divergence(log_dist1, log_dist2)
0.313...  # KL divergence computed in log-space
```

# Special Cases
- If d2 has zero probability where d1 has positive probability: result is +∞
- Self-divergence: D(d || d) = 0
- Uniform reference: D(d || uniform) = log(k) - H(d), where H is entropy

# Error conditions
- Throw ArgumentError("Cannot compute KL divergence for distributions with different dimensions")
"""
function KL_divergence(d1::T, d2::T) where T <: CategoricalDist
    if size(d1) != size(d2)
        throw(ArgumentError("Cannot compute KL divergence for distributions with different dimensions"))
    end
    return sum(exp.(d1.η) .* (d1.η-d2.η))
end

"""
    show(io::IO, d::NormalizedCategorical)

Custom display method for normalized categorical distributions.

# Arguments
- `io::IO`: Output stream for display
- `d::NormalizedCategorical`: Distribution to display

Already implemented.
"""
function Base.show(io::IO, d::NormalizedCategorical)
    if (is_uniform(d))
        print(io, "uniform")
    else
        print(io, "π = ", means(d))
    end
end

"""
    show(io::IO, d::NonNormalizedCategorical)

Custom display method for non-normalized categorical distributions.

# Arguments
- `io::IO`: Output stream for display
- `d::NonNormalizedCategorical`: Distribution to display

Alredy implemented.
"""
function Base.show(io::IO, d::NonNormalizedCategorical)
    if (is_uniform(d))
        print(io, "uniform (Z = " , exp(d.log_norm), ")")
    else
        print(io, "π = ", means(d), ", Z = ", exp(d.log_norm))
    end
end

"""
    distribution(d::CategoricalDist) -> Distributions.Categorical

Creates a Distributions.jl Categorical object from our custom categorical distribution types.

# Arguments
- `d::CategoricalDist`: Any categorical distribution (normalized or unnormalized)

# Returns
- `Distributions.Categorical`: Standard library categorical distribution

# Examples
```julia-repl
julia> # Convert custom distribution to standard library (created from log-probabilities)
julia> custom = NormalizedCategorical(log.([0.3, 0.5, 0.2]))  # Convert to log-space
julia> standard = distribution(custom)
Categorical{Float64, Vector{Float64}}(support=Base.OneTo(3), p=[0.3, 0.5, 0.2])

julia> # Use with standard library functions
julia> using Random, Distributions
julia> samples = rand(standard, 1000)
julia> mean(samples)  # Should be around 1.9 (weighted average of 1,2,3)

julia> # Access standard statistical functions
julia> entropy(standard)
1.485...

julia> pdf(standard, 2)  # Probability of category 2
0.5

julia> # Works with unnormalized distributions too (log-space)
julia> unnorm = NonNormalizedCategorical(log.([0.6, 0.4]), 0.0)
julia> std_unnorm = distribution(unnorm)

julia> # Direct log-probability specification
julia> log_dist = NormalizedCategorical([0.0, -0.693, -1.099])  # log([1.0, 0.5, 0.33])
julia> std_log = distribution(log_dist)
```
# See Also
- `Distributions.jl` documentation for full API
"""
distribution(g::T) where T <: CategoricalDist = Distributions.Categorical(means(g))

export NormalizedCategorical, NonNormalizedCategorical, CategoricalUniform, CategoricalFromProbabilities
export is_uniform, size, means, variances, *, /, KL_divergence, show, distribution

end
