using FFTW, LinearAlgebra, Random, Images, Optim, StatsBase

dct2(A::Array{T,2}) where T <: Number = dct(dct(A,2),1)
idct2(A::Array{T,2}) where T <: Number = idct(idct(A,2),1)

# read original image and downsize for speed
Xorig = Gray.(load("example/cat.jpeg")) # read in grayscale
X = imresize(Xorig, ratio=0.2)
ny, nx = size(X)

k = round(Int, nx * ny * 0.5)
idx = sample(1:nx*ny, k, replace=false)
b = vec(X)[idx]

using LinearAlgebra

A = kron(idct(Matrix{Float64}(I, nx, nx), 1), idct(Matrix{Float64}(I, ny, ny), 1))
A = A[idx, :]

# 执行L1优化
vx = Variable(n)
problem = minimize(norm(vx, 1), A * vx == b)
result = solve!(problem, SCS.Optimizer; silent=false)
reyw = evaluate(vx)
