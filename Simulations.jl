#=
Simulations.jl
---
Contains functions used for simulating opinion dynamics on generated networks of agents with heterogeneous belief preferences.
=#

using DifferentialEquations
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Distributions
using Random
include("./SystemMatrix.jl")

function x0(agents, topics)
    #= 
    agents: number of agents
    topics: number of topics

    returns: A vector ∈ [-1,1]^(agents*topics) representing the initial distribution of opinions in a system.
    =#
    return 2*rand(agents*topics) .- 1
end

function solveSystem(A, u0, B, topics, maxT) 
    #=
    A: system matrix
    u0: initial distribution of opinions in the system
    B: stubbornness values for the system
    topics: number of topics
    maxT: number of timesteps to render for the system

    returns: an ODE solution to a system given the previous conditions.
    =#
    f(u,p,t) = A*u + kron(B,identityMatrix(topics))*u0
    tspan = (0.0,maxT)
    prob = ODEProblem(f, u0, tspan) 
    sol = solve(prob) 
    return(sol) 
end

function example(agents, topics, p, homo, stubborn, maxT, similar)
    #=
    p: probability of repeated eigenvalues (0 <= p <= 1)
    homo: whether to use same logic matrix for all agents
    stubborn: whether or not to use stubborn agents
    maxT: number of timesteps to render for the system
    similar: whether or not all agents' logic matrices will be similar

    returns: an ODE solution to a system that is defined by the input parameters.
    =#
    S, B, txt = SystemMatrix(agents, topics, p, homo, stubborn, similar)
    x = x0(agents, topics)
    txt = string(txt, "x0 (initial state):", x, ".\n")
    sol = solveSystem(S,x,B,topics,maxT)
    return(sol, txt)
end

function isConsensus(sol, agents, topics, maxT, eps)
    #=
    sol: an ODE solution object to a system
    maxT: number of timesteps to render for the system
    eps: how close two floating-point opinions must be to be considered equivalent

    returns: either "partial-consensus", "consensus", or "no-consensus", depending on the final state of the system.
    =#
    topic_consensus = []
    final = sol[end]
    nonzero_consensus = false

    # iterate through all topics
    for topic in 1:topics
        consensus_val = 1
        max_val = nothing
        min_val = nothing
        # used to avoid cases where we diverge
        large_constant = 99 

        # iterate through agents to ensure they're within a certain bound
        for agent in 1:agents
            final_agent = final[topic+((agent-1)*topics)]
            if isnothing(max_val)
                max_val = final_agent + eps
                min_val = final_agent - eps

            # consensus is not found
            elseif (final_agent > max_val) || (final_agent < min_val) || (final_agent > large_constant)
                consensus_val = 0 
                break

            elseif ((agent == agents) && (consensus_val == 1) && (final_agent > eps || final_agent < -eps))
                nonzero_consensus = true
            end
        end

        append!(topic_consensus, consensus_val) # add consensus state
    end

    # parse topic_consensus list, return consensus/no-consensus/partial-consensus
    final_state = ""
    if (all(x->x==1, topic_consensus))
        final_state = "consensus"
    elseif (all(x->x==0, topic_consensus))
        final_state = "no-consensus"
    else
        final_state = "partial-consensus"
    end

    if(nonzero_consensus)
        final_state = final_state * "-nonzero"
    else
    end
    return(final_state)
end

function plotTopics(root, agents, topics, p, homo, stubborn, maxT, similar, eps, fn)
    #=
    root: filepath to save plots to
    fn: filename to save plot to

    Calls example and plots output, saving to the appropriate location.
    =#

    sol, txt = example(agents, topics, p, homo, stubborn, maxT, similar)
    consensus = isConsensus(sol, agents, topics, maxT, eps)
    # sample up to maxT to use in plotting
    # solT = trunc(Int, (maxT/100)*length(sol))
    seriesList = []

    # add all subseries onto list for later plotting
    for i in 1:topics
        subList = []
        for j in 1:agents
            push!(subList, sol[(i-1)*agents + j,:])
        end
        push!(seriesList, subList)
    end
   
    # save whether logic matrix is homogenous
    homoStr = "hetero"
    if homo
        homoStr = "homo"
    end 
    
    # check if target directory exists, make directories if needed
    if !isdir(root)
        mkdir(root)
    end
    if !isdir(root * string(homoStr) * "/")
        mkdir(root * string(homoStr) * "/")
    end
    path = root * string(homoStr) * "/" * consensus
    if !isdir(path)
        mkdir(path)
    end
    path = path * "/" * string()

    # write plot to appropriate subfolder and path
    plot(sol.t, seriesList, layout=(1,topics), legend=false, xlabel="Number of timesteps", ylabel="Opinion Value", size=(1350,400), margin=7*Plots.mm, grid=false)
    png(path * "/" * fn)

    # write system matrices to file
    open(path * "/" * fn * ".txt", "w") do file
        write(file, txt)
    end
end

function experiment(path, agents, topics, p, homo, stubborn, maxT, similar, eps, num_attempts)
    #=
    root: filepath to save plots to
    p: probability of repeated eigenvalues (0 <= p <= 1)
    homo: whether to use same logic matrix for all agents
    stubborn: whether or not to use stubborn agents
    maxT: number of timesteps to render for the system
    similar: whether or not all agents' logic matrices will be similar
    eps: how close two floating-point opinions must be to be considered equivalent
    fn: filename to save plot to
    num_attempts: total number of plots to randomly generate
    Generates a number of plots from randomly generated systems with the input parameters.
    =#
    for i in 1:num_attempts
        try
            fn = "attempt" * string(i)
            plotTopics(path, agents, topics, p, homo, stubborn, maxT, similar, eps, fn)
        catch e
            println(e)
            display(stacktrace(catch_backtrace()))
        end
    end
end