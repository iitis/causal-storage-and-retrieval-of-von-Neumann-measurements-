using LinearAlgebra, MKL
using Convex, SCS
using Combinatorics
using QuantumInformation
using SparseArrays

# SDP program for calculating the average fidelity function for causal learning scheme of qubit von Neumann measurements

function projector_onto_commutator_subspace(N=1, atol=1e-3)
    """
        Function creating the projector onto commutator subspace.
        Usage: Take matrix X. Define |Y>> := projector_onto_commutator_subspace(d,N) |X>>.
        Then, [Y,U^{⊗N} ⊗ U*] = 0 for all U.
    """

    projector = zeros(2^(2N+2), 2^(2N+2))
    idx = CartesianIndices(Tuple(fill(2, N+1)))

    for perm in permutations(1:N+1)
        x = zeros(fill(2, 2N+2)...)
        for k in idx
            k0 = CartesianIndex(Tuple(k)[perm])
            x[k0, k] = 1
        end
        x = vec(x)
        projector += x * x'
    end

    F = eigen(projector)
    Λ = Diagonal([x > atol ? 1 : 0 for x in F.values])
    P = F.vectors * Λ * F.vectors'
    asym_perm = Array(1:(2N+2))
    asym_perm[N+1], asym_perm[2N+2] = 2N + 2, N + 1
    sparse(permutesystems(P, fill(2, 2N + 2), asym_perm))
end

function Qpartialtrace(Q, tuple, N)
    """
    Function defining partial trace of Q over qubit spaces X_j such that tuple[j] = 0.
    """
    M = N
    for j=N:-1:1
        if tuple[j] == 0
            Q = partialtrace(Q, j, fill(2, M))
            M -= 1
        end
    end
    return Q
end

function get_network_causality_properties(Rs, N)
    """ Function defining causal learning network such that its storage is described by a process matrix. """
    S = [sum(Rs[i, j] for j=1:2) for i ∈ CartesianIndices(Tuple(fill(2, N)))]
    S = reshape(S, fill(2, N)...)
    constraints = [S[i] == partialtrace(S[i], N+1, fill(2, N+1)) ⊗ I(2)/2 for i=1:2^N]
    S = [partialtrace(X, N+1, fill(2, N+1))/2 for X ∈ S]
    S = reshape(S, fill(2, N)...)
    trW = sum(tr(X) for X in S)
    constraints += [real(trW) == 2^N]
    # Here S should be N-partite block-diagonal process matrix

    # Changing the storage S as follow:
    # S[tn, ..., t1] -> na Sn x ... x S1
    IDX = CartesianIndices(Tuple(fill(0:1, N)))[2:2^N]
    for i in IDX
        Si = [X for X in S]
        Si = [Qpartialtrace(X, i, N) for X in Si]
        Si = reshape(Si, fill(2, N)...)
        Hi = sum((-1)^(dot(Tuple(i), Tuple(j))) * Si[j] for j in CartesianIndices(Tuple(fill(2, N))))
        constraints += [Hi == zeros(2^sum(Tuple(i)), 2^sum(Tuple(i)))]
    end

    constraints
end

function calc_causal_fidelity(N, P::SparseMatrixCSC{Float64, Int64}, epsx = 1e-08)
    """Funtion maximizing the average fidelity function of causal learning scheme"""
    Rs = [ComplexVariable(2^(N+1), 2^(N+1)) for _=1:2^(N+1)]
    Rs = reshape(Rs, fill(2, N+1)...)

    constraints = [R in :SDP for R in Rs]
    constraints += [P * vec(R) == vec(R) for R in Rs]
    constraints += get_network_causality_properties(Rs, N)

    linear = LinearIndices(Rs)
    t = 1/2 * real(
        sum(
            Rs[CartesianIndex(reverse(Tuple(a)))][linear[a], linear[a]] for a in CartesianIndices(Rs)
        )
        )
    problem = maximize(t, constraints)
    print("start \n")
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => epsx, "eps_rel" => epsx))

    return (string(problem.status), problem.optval)
end

function results(N, epsx=1e-05)
    """"Function printing obtained results"""
    P = projector_onto_commutator_subspace(N, epsx)
    ans = calc_causal_fidelity(N, P, epsx)
    print("C, N $(N), $(ans[2]) \n")
end

results(1)
results(2)
results(3)
results(4)
results(5)
