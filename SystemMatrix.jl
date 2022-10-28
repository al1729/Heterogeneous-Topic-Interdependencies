#=
SystemMatrix.jl
---
Contains functions used for generating system matrices that describe opinion dynamics on networks of agents with heterogeneous belief preferences.
=#

using DifferentialEquations
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Distributions
using Random

function identityMatrix(n)
    #=
    n: size of matrix
    returns: an n x n identity matrix.
    =#
    m = zeros(n,n)
    for i in 1:n
        m[i,i] = 1
    end
    return m
end

function CMatrix(n, p)
    #=
    n: number of topics
    p: probability of repeated eigenvalues after the first eigenvalue (0 <= p <= 1)

    returns: a randomly generated n x n logic matrix to represent a single agent's belief system.
    =#
    satisfies_conditions = false
    C = zeros(n,n)

    # Repeatedly generate matrices until Assumptions are satisfied.
    while !(satisfies_conditions)
        satisfies_conditions = true
        # Randomly generate eigenvalues of our matrix.
        eigs = zeros(n)
        eigs[1] = 1
        lastVal = 1

        for i in 2:n
            if rand() < p && lastVal < 1
                eigs[i] = lastVal
            else
                newEig = -1 + rand()*(lastVal + 1)
                lastVal = newEig
                eigs[i] = newEig
            end
        end

        # Generates the Jordan Canonical Form of c_i given the eigenvalues.
        jcf = zeros(n, n)
        jcf[1,1]=1
        for i in 2:n # 1 assumed semisimple
            if eigs[i] == eigs[i-1]
                jcf[i-1, i] = 1
            end
            jcf[i,i] = eigs[i]
                end  
        S = rand(n,n)
        C = S * jcf * inv(S)
        
        # Bounds the eigenvector product according to Assumption 1.
        eig = eigvecs(C) 
        eigT = eigvecs(transpose(C))
        eig1 = real(eig[:,n][:,1])
        eig2 = real(eigT[:,n][:,1])
        norm = sqrt(abs(transpose(eig2) * eig1))
        for i in 1:n
            C[i,i] -= 1
        end

        C = C ./ norm
        for i in 1:n
            C[i,i] += 1
        end        
        for i in 1:n
            if C[i,i] < 0
                satisfies_conditions = false
            end
        end
        
    end
    return C
end

function SystemCMatrix(agents, topics, p, homo)
    #= 
    agents: number of agents in the system
    topics: number of topics in the system
    homo: whether to use same logic matrix for all agents

    returns: a System Matrix of a system with the above parameters. 
    =#
    SystemC = zeros(agents*topics, agents*topics)

    if homo
        C = CMatrix(topics, p)
        for i in 1:agents
            SystemC[((i-1)*topics+1):(i*topics), ((i-1)*topics+1):(i*topics)] = C
        end

    else
        for i in 1:agents
            C = CMatrix(topics, p)
            SystemC[((i-1)*topics+1):(i*topics), ((i-1)*topics+1):(i*topics)] = C
        end
    end

    return SystemC
end

function SimilarSystemCMatrix(agents, topics, p)
    #= 
    p: probability of repeated eigenvalues after the first eigenvalue (0 <= p <= 1)

    returns: a System Matrix of a system with the above parameters, all of whose agents have similar logic matrices.
    =#

    SystemC = zeros(agents*topics, agents*topics)
    baseC = CMatrix(topics, p)
    SystemC[1:topics, 1:topics] = baseC
    for agent in 2:agents
        satisfies_conditions = false

        # Repeatedly generate matrices until Assumptions are satisfied.
        while(!satisfies_conditions)
            satisfies_conditions = true
            S = rand(topics,topics)
            C = S * baseC * inv(S)

            # Normalizes the eigenvector product according to Assumption 1.
            eig = eigvecs(C)
            eigT = eigvecs(transpose(C))
            eig1 = real(eig[:,topics][:,1])
            eig2 = real(eigT[:,topics][:,1])
            norm = sqrt(abs(transpose(eig2) * eig1))
            
            for i in 1:topics
                C[i,i] -= 1
            end
            C = C ./ norm
            for i in 1:topics
                C[i,i] += 1
            end
            
            for i in 1:topics
                if C[i,i] < 0
                    satisfies_conditions = false
                end
            end
            
            if(satisfies_conditions)
                SystemC[((agent-1)*topics+1):(agent*topics), ((agent-1)*topics+1):(agent*topics)] = C
            end
        end
    end
    return SystemC
end

function Laplacian(agents)
    #=
    returns: a matrix representing hardcoded Laplacian value.
    =#
    return [1 0 -1 0 0 0 0 0; -1 1 0 0 0 0 0 0; 0 -0.8 1 -0.2 0 0 0 0; 0 0 -1 1 0 0 0 0; 0 0 0 -0.4 1 0 -0.6 0; 0 0 -0.2 0 -0.8 1 0 0; 0 0 0 0 0 -1 1 0; -0.3 -0.7 0 0 0 0 0 1]
end

function BMatrix(agents, stubborn)
    #=
    stubborn: Boolean; whether or not to include stubborn agents.
    
    returns: a matrix containing stubbornness values of agents in a system on the diagonal.
    =#
    B = zeros(agents, agents)
    if stubborn
        for i in 1:agents
            B[i,i] = rand()
        end
    end

    return B
end

function SystemMatrixSimilarC(agents, topics, p, homo, stubborn)
    #=
    p: probability of repeated eigenvalues after the first eigenvalue (0 <= p <= 1)
    homo: whether to use same logic matrix for all agents
    stubborn: whether or not to use stubborn agents

    returns: System matrix describing a system with the given inputs, as well as stubbornness and text representation for future usage.
    The system will be generated assuming similar logic matrices between the agents in the system.
    =#

    #system matrix is Kronecker(L+B, I_n) + I_nk - SystemC
    L = Laplacian(agents)
    I = identityMatrix(topics)
    B = BMatrix(agents, stubborn)
    C = SimilarSystemCMatrix(agents, topics, p)
    Sys = -kron(L+B,I) - identityMatrix(agents*topics) + C

    #writes system information to string so it can be written to txt later
    txt = string("Laplacian:", L, ".\n")
    ctxt = ""
    for i in 1:agents
        newTxt = "C" * string.(i) * ":\n"
        for j in 1:topics
            newTxt = newTxt * join(string(C[topics*(i-1)+j,topics*(i-1)+1:topics*i])) * ",\n"
        end
        ctxt = ctxt * newTxt
    end
    txt = string(txt, ctxt)
    txt = string(txt, "Stubbornness:", B, ".\n")
    return(Sys,B,txt)
end   

function SystemMatrix(agents, topics, p, homo, stubborn, similar)
    #=
    agents: number of agents
    topics: number of topics
    p: probability of repeated eigenvalues after the first eigenvalue (0 <= p <= 1)
    homo: whether to use same logic matrix for all agents
    stubborn: whether or not to use stubborn agents.
    similar: whether or not all agents' logic matrices will be similar.

    returns: System matrix describing a system with the given inputs, as well as stubbornness and text representation for future usage.
    =#

    # if similar, use output of above function.
    if similar
        return(SystemMatrixSimilarC(agents, topics, p, homo, stubborn))
    end

    #system matrix is Kronecker(L+B, I_n) + I_nk - SystemC
    L = Laplacian(agents)
    I = identityMatrix(topics)
    C = SystemCMatrix(agents, topics, p, homo)
    B = BMatrix(agents, stubborn)
    Sys = -kron(L+B,I) - identityMatrix(agents*topics) + C
    
    #writes system information to string so it can be written to txt later
    txt = string("Laplacian:", L, ".\n")
    ctxt = ""
    for i in 1:agents
        newTxt = "C" * string.(i) * ":\n"
        for j in 1:topics
            newTxt = newTxt * join(string(C[topics*(i-1)+j,topics*(i-1)+1:topics*i])) * ",\n"
        end
        ctxt = ctxt * newTxt
    end
    txt = string(txt, ctxt)
    txt = string(txt, "Stubbornness:", B, ".\n")
    return(Sys,B,txt)
end

