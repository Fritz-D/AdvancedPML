using LinearAlgebra

function swaprow!(M::Matrix,i::Integer,j::Integer)
    for c in axes(M, 2)
        temp = M[i, c]
        M[i, c] = M[j, c]
        M[j, c] = temp
    end
end

function inv_naive(M::Matrix{T}) where T <: Real
    # naive inverse calculation on square matrices with row reduction
    if size(M, 1) != size(M, 2) throw(ArgumentError("Matrix M must be square")) end
    A = hcat(Float64.(M), I)
    m, n = size(A)
    for j = 1:m
        for i = j+1:m
            factor = -A[i, j] / A[j, j]
            A[i,:] = A[i, :] + A[j, :] * factor
        end
    end
    for j in range(m, 1, step=-1)
        for i = range(j-1, 1, step=-1)
            factor = -A[i, j] / A[j, j]
            A[i,:] = A[i, :] + A[j, :] * factor
        end
    end
    for i in 1:m
        A[i, :] = A[i, :] / A[i,i]
    end
    return A[:, (m+1):n]
end

function cholesky_crout(A::Matrix)
    # Cholesky decomposition
    #
    # 2025 by Ralf Herbrich
    # Hasso-Plattner Institute
    # check that the matrix is square
    if (size(A)[1] != size(A)[2])
        error("matrix must be square")
    end
    n = size(A)[1]

    # create a zero matrix
    L = zeros(n, n)

    # run the Cholesky decomposition
    for j = 1:n
        sum = 0
        for k = 1:(j-1)
            sum += L[j, k] * L[j, k]
        end
        L[j, j] = sqrt(A[j, j] - sum)

        for i = (j+1):n
            sum = 0
            for k = 1:(j-1)
                sum += L[i, k] * L[j, k]
            end
            L[i, j] = (A[i, j] - sum) / L[j, j]
        end
    end

    return (L)
end

function main()
    A = [[8,5,2] [5,5,0] [2,0,8]]
    b = [1,2,3]

    # (a) naive inverse to obtain x
    x1 = inv_naive(A) * b
    println("inv(A) * b = ", round.(x1; digits=2))

    # (b) cholesky decomp to obtain x courtesy of Ralf Herbrich
    L = cholesky_crout(A)
    y = L \ b
    x2 = transpose(L) \ y
    println("L^T\\(L\\b) = ", round.(x2; digits=2))

    # (c) apply and check correctness
    println("x1 ≈ x2: ", x1 ≈ x2)
    println("A * x1 ≈ b: ", A * x1 ≈ b)
    println("A * x2 ≈ b: ", A * x2 ≈ b)

    # (c) super bad naive timing
    t = time()
    for _ in 1:100000
        inv_naive(A)
    end
    time_invnaive = time()-t
    t = time()
    for _ in 1:100000
        cholesky_crout(A)
    end
    time_cholesky = time()-t
    println("Cholesky was $(time_invnaive/time_cholesky)x faster")
end

main()