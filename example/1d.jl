using FFTW, LinearAlgebra, Random, Convex, SCS

n = 5000
t = range(0, stop=1/8, length=n)
yt = sin.(1394 * π .* t) .+ sin.(3266 * π .* t)

# compute the DCT-II
yw = dct(yt)

# 抽取信号的小样本
m = 500                    # 10% 的样本量
idx = rand(1:n,m)     # 随机抽取 m 个索引（1 到 n 之间）
sort!(idx)                  # 原地排序（方便绘图，但非必须）

t_sp = t[idx]                 # 根据索引 ri 对时间向量做下采样
y_sp = yt[idx] 

# 创建DCT基矩阵并进行采样
# 创建单位矩阵
I_n = Matrix{Float64}(I, n, n)

# 对每一列进行DCT变换（相当于axis=0）
A = zeros(n, n)
for j in 1:n
    A[:, j] = idct(I_n[:, j])
end

# 根据采样索引选择行
A = A[idx, :]

# 执行L1优化
vx = Variable(n)
problem = minimize(norm(vx, 1), A * vx == y_sp)
result = solve!(problem, SCS.Optimizer; silent=false)

# 获取优化结果
reyw = evaluate(vx)

using Plots
n1 = n÷5
plot(1:n1, yw[1:n1], label="original")
plot!(1:n1, reyw[1:n1], label="recovered")

plot(t[1:n1], yt[1:n1], label="original")
plot!(t[1:n1], idct(reyw)[1:n1], label="recovered")