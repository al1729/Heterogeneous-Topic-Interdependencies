#=
Main.jl
---
Using the experiment function located in Simulations.jl, 
replicates the simulations run in the paper "Heterogeneous Topic Interdependencies in Friedkin-Johnsen Models of Opinion Dynamics".
=#

using DifferentialEquations
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Distributions
using Random
include("./SystemMatrix.jl")
include("./Simulations.jl")

# experiment: homogenous case, no stubbornness
experiment("output/test-homo-nostubborn/",8,3,0,true,false,100,false,0.001,100)
# experment: hetero case, no stubbornness
experiment("output/test-hetero-nostubborn/",8,3,0.2,false,false,100,false,0.001,100)
# experment: homo case, stubbornness
experiment("output/test-homo-stubborn/",8,3,0.2,true,true,100,false,0.001,100)
# experment: hetero case, stubbornness
experiment("output/test-hetero-stubborn/",8,3,0.2,false,true,100,false,0.001,100)
# experiment: hetero case, similar matrices, no stubbornness
experiment("output/test-similar-nostubborn/",8,3,0.2,false,false,100,true,0.001,10)