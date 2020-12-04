"""
    HopperTO
    	model from "Dynamically Stable Legged Locomotion"
		x = (px, py, t, r)
"""
struct HopperTO <: TO.AbstractModel
    n::Int
    m::Int
    d::Int

    mb # mass of body
    ml # mass of leg
    Jb # inertia of body
    Jl # inertia of leg

    μ  # coefficient of friction
    g  # gravity

    qL::Vector
    qU::Vector

    nq
    nu
    nc
    nf
    nb
	ns

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s

end

# Dimensions
nq = 4 # configuration dimension
nu = 2 # control dimension
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone
nb = nc * nf
ns = 1

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 10.0 # body mass
ml = 1.0  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

_n = 2 * nq
_m = nu + nc + nb #+ nc + nb + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = (1:0)#nu + nc + nb .+ (1:nc)
idx_η = (1:0)#nu + nc + nb + nc .+ (1:nb)
idx_s = (1:0)#nu + nc + nb + nc + nb .+ (1:ns)

# Kinematics
kinematics(::HopperTO, q) = [q[1] + q[4] * sin(q[3]), q[2] - q[4] * cos(q[3])]

# Methods
M_func(model::HopperTO, q) = Diagonal(@SVector [
								 model.mb + model.ml,
								 model.mb + model.ml,
								 model.Jb + model.Jl,
								 model.ml])

G_func(model::HopperTO, q) = @SVector [0.0,
									 (model.mb + model.ml) * model.g,
									 0.0,
									 0.0]

function ϕ_func(::HopperTO, q)
    @SVector [q[2] - q[4] * cos(q[3])]
end

N_func(::HopperTO, q) = @SMatrix [0.0 1.0 (q[4] * sin(q[3])) (-1.0 * cos(q[3]))]

function _P_func(model, q)
	@SMatrix [1.0 0.0 (q[4] * cos(q[3])) sin(q[3])]
end

function P_func(::HopperTO, q)
    @SMatrix [1.0 0.0 (q[4] * cos(q[3])) sin(q[3]);
              -1.0 0.0 (-1.0 * q[4] * cos(q[3])) -1.0 * sin(q[3])]
end

B_func(::HopperTO, q) = @SMatrix [0.0 0.0 1.0 0.0;
                                -sin(q[3]) cos(q[3]) 0.0 1.0]

function fd(model::HopperTO, x⁺, x, u, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
    - M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
    + transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{2}(b)
    - h * G_func(model, q2⁺))]
end

function maximum_dissipation(model::HopperTO, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nb)
	η = u[model.idx_η]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::HopperTO, x⁺, u, h)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2 = view(x⁺, 1:model.nq)
	λ = view(u, model.idx_λ)
	s = view(u, model.idx_s)

	return s[1] - (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function friction_cone(model::HopperTO, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]
	return @SVector [model.μ * λ[1] - sum(b)]
end

r = 0.5
qL = -Inf * ones(nq)
qU = Inf * ones(nq)
qL[4] = 0.1
qU[4] = r

model = HopperTO(_n, _m, d,
			   mb, ml, Jb, Jl,
			   μ, g,
			   qL, qU,
			   nq,
		       nu,
		       nc,
		       nf,
		       nb,
		   	   ns,
		       idx_u,
		       idx_λ,
		       idx_b,
		       idx_ψ,
		       idx_η,
		       idx_s)


Base.size(::HopperTO) = model.n + model.nc, model.m

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::HopperTO, x::StaticVector, u::StaticVector, t, h)
	q3 = view(x, model.nq .+ (1:model.nq))
	# q2 = view(x, 1:model.nq)
	λ = view(u, model.idx_λ)

	f(z) = fd(model, z, view(x, 1:2 * model.nq), u, h, t)

	y = copy(x[1:2 * model.nq])
	r = f(y)

	iter = 0
	while norm(r, 2) > 1.0e-8 && iter < 10
	   ∇r = ForwardDiff.jacobian(f, y)

	   Δy = -1.0 * ∇r \ r

	   α = 1.0

		iter_ls = 0

	   while α > 1.0e-8 && iter_ls < 10
	       ŷ = y + α * Δy
	       r̂ = f(ŷ)

	       if norm(r̂) < norm(r)
	           y = ŷ
	           r = r̂
	           break
	       else
	           α *= 0.5
				iter_ls += 1
	       end

			if iter_ls == 10
				@warn "line search failed"
			end
	   end

	   iter += 1
	end
	return [y; λ]
end

RobotDynamics.discrete_dynamics(PassThrough, model, (@SVector rand(size(model)[1])), (@SVector rand(size(model)[2])), 1, 0.1)

function RobotDynamics.discrete_jacobian!(::Type{PassThrough}, ∇f, model::HopperTO,
		z::AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:RobotDynamics.Explicit}

	# println("discrete dynamics jacobian")
	ix,iu,idt = z._x, z._u, N+M+1

	x = state(z)[1:2 * model.nq]
	u = control(z)
	y = RobotDynamics.discrete_dynamics(PassThrough, model, state(z), control(z), z.t, z.dt)[1:2 * model.nq]

    fy(w) = fd(model, w, x, u, z.dt, z.t)
	fx(w) = fd(model, y, w, u, z.dt, z.t)
	fu(w) = fd(model, y, x, w, z.dt, z.t)

	Dy = ForwardDiff.jacobian(fy, y)
	∇f[(1:2 * model.nq), ix[(1:2 * model.nq)]] = -1.0 * Dy \ ForwardDiff.jacobian(fx, x)
	∇f[(1:2 * model.nq), iu] = -1.0 * Dy \ ForwardDiff.jacobian(fu, u)

	∇f[2 * model.nq .+ (1:model.nc), iu[model.idx_λ]] = Diagonal(ones(model.nc))

	return nothing
end

function state_to_configuration(X)
    T = length(X)
    nq = convert(Int, floor(length(X[1])/2))
    [X[1][1:nq], [X[t][nq .+ (1:nq)] for t = 1:T]...]
end

# Visualization
using MeshCat, GeometryBasics, Colors, CoordinateTransformations

function visualize!(vis, model::HopperTO, q; Δt = 0.1)
    r_foot = 0.05
    r_leg = 0.5 * r_foot

    setobject!(vis["body"], Sphere(Point3f0(0),
        convert(Float32, 0.1)),
        MeshPhongMaterial(color = RGBA(0, 1, 0, 1.0)))

    setobject!(vis["foot"], Sphere(Point3f0(0),
        convert(Float32, r_foot)),
        MeshPhongMaterial(color = RGBA(1, 0, 0, 1.0)))

    n_leg = 100
    for i = 1:n_leg
        setobject!(vis["leg$i"], Sphere(Point3f0(0),
            convert(Float32, r_leg)),
            MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
    end

    p_leg = [zeros(3) for i = 1:n_leg]
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        p_body = [q[t][1], 0.0, q[t][2]]
        p_foot = [kinematics(model, q[t])[1], 0.0, kinematics(model, q[t])[2]]

        q_tmp = Array(copy(q[t]))
        r_range = range(0, stop = q[t][4], length = n_leg)
        for i = 1:n_leg
            q_tmp[4] = r_range[i]
            p_leg[i] = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]
        end
        q_tmp[4] = q[t][4]
        p_foot = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]

        z_shift = [0.0; 0.0; r_foot]

        MeshCat.atframe(anim, t) do
            settransform!(vis["body"], Translation(p_body + z_shift))
            settransform!(vis["foot"], Translation(p_foot + z_shift))

            for i = 1:n_leg
                settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
            end
        end
    end

    MeshCat.setanimation!(vis, anim)
end

function linear_interp(x0, xf, T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i] - x0[i]) / (T - 1) * (t - 1) + x0[i]
        end
    end
    return X
end

function configuration_to_state(Q)
    T = length(Q)
    [[t == 1 ? Q[1] : Q[t-1]; Q[t]] for t = 1:T]
end
