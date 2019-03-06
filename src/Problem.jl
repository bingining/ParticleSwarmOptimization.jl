using Knet

abstract type Data end

struct Single <: Data
    x
    y
end

struct MiniBatch <: Data
    trn
    dev
    tst
    MiniBatch(trn) = new(trn, Any[], Any[])
    MiniBatch(trn, tst) = new(trn, Any[], tst)
end

abstract type Problem end

struct RegularProblem <: Problem
    f::Function
    dim::Int
    l::Array{Float64, 1}
    h::Array{Float64, 1}
    RegularProblem(f, dim, l::Float64, h::Float64) = new(f, dim, ones(Float64, dim) * l, ones(Float64, dim) * h)
    RegularProblem(f, dim, l::Array{Float64,1}, h::Array{Float64,1}) = new(f, dim, l, h)
end

struct NNProblem <: Problem
    f::Symbol
    l::Float64
    h::Float64
    data::Data
    loss
end