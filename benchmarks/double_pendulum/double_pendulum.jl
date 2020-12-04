"""
    Double pendulum
"""

struct DoublePendulum{T} <: TO.AbstractModel
    n::Int
    m::Int
    d::Int

    m1::T    # mass link 1
    J1::T    # inertia link 1
    l1::T    # length link 1
    lc1::T   # length to COM link 1

    m2::T    # mass link 2
    J2::T    # inertia link 2
    l2::T    # length link 2
    lc2::T   # length to COM link 2

    g::T     # gravity

    b1::T    # joint friction
    b2::T
end

function M(model::DoublePendulum, x)
    a = (model.J1 + model.J2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(x[2]))

    b = model.J2 + model.m2 * model.l1 * model.lc2 * cos(x[2])

    c = model.J2

    @SMatrix [a b;
              b c]
end

function τ(model::DoublePendulum, x)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    @SVector [a,
              b]
end

function C(model::DoublePendulum, x)
    a = -2.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    b = -1.0 * model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[4]
    c = model.m2 * model.l1 * model.lc2 * sin(x[2]) * x[3]
    d = 0.0

    @SMatrix [a b;
              c d]
end

function B(model::DoublePendulum, x)
    @SMatrix [1.0 0.0;
              0.0 1.0]
end

function f(model::DoublePendulum, x, u)
    q = view(x, 1:2)
    v = view(x, 3:4)
    qdd = M(model, q) \ (-1.0 * C(model, x) * v
            + τ(model, q) + B(model, q) * u[1:2] - [model.b1; model.b2] .* v)
    @SVector [x[3],
              x[4],
              qdd[1],
              qdd[2]]
end

function kinematics_mid(model::DoublePendulum, x)
    @SVector [model.l1 * sin(x[1]),
              -1.0 * model.l1 * cos(x[1])]
end

function kinematics_ee(model::DoublePendulum, x)
    @SVector [model.l1 * sin(x[1]) + model.l2 * sin(x[1] + x[2]),
              -1.0 * model.l1 * cos(x[1]) - model.l2 * cos(x[1] + x[2])]
end

n, m, d = 4, 2, 0
model = DoublePendulum(n, m, d, 1.0, 0.33, 1.0, 0.5, 1.0, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

function midpoint_implicit(model, x⁺, x, u, h)
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u))
end

function fd(model::DoublePendulum, x⁺, x, u, h, t)
    midpoint_implicit(model, x⁺, x, u, h)
end

Base.size(::DoublePendulum) = model.n, model.m

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::DoublePendulum, x::StaticVector, u::StaticVector, t, h)
	f(z) = fd(model, z, x, u, h, t)

	y = copy(x)
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
	return y
end

RobotDynamics.discrete_dynamics(PassThrough, model, (@SVector rand(size(model)[1])), (@SVector rand(size(model)[2])), 1, 0.1)

function RobotDynamics.discrete_jacobian!(::Type{PassThrough}, ∇f, model::DoublePendulum,
		z::AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:RobotDynamics.Explicit}

	# println("discrete dynamics jacobian")
	ix,iu,idt = z._x, z._u, N+M+1

	x = state(z)
	u = control(z)
	y = RobotDynamics.discrete_dynamics(PassThrough, model, state(z), control(z), z.t, z.dt)

    fy(w) = fd(model, w, x, u, z.dt, z.t)
	fx(w) = fd(model, y, w, u, z.dt, z.t)
	fu(w) = fd(model, y, x, w, z.dt, z.t)

	Dy = ForwardDiff.jacobian(fy, y)
	∇f[:, ix] = -1.0 * Dy \ ForwardDiff.jacobian(fx, x)
	∇f[:, iu] = -1.0 * Dy \ ForwardDiff.jacobian(fu, u)

	return nothing
end

# visualization
function visualize!(vis, model::DoublePendulum, x;
        color=RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)


    i = 1
    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
        convert(Float32, 0.025))
    setobject!(vis["l1$i"], l1, MeshPhongMaterial(color = color))
    l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
        convert(Float32, 0.025))
    setobject!(vis["l2$i"], l2, MeshPhongMaterial(color = color))

    setobject!(vis["elbow$i"], GeometryTypes.Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = color))
    setobject!(vis["ee$i"], GeometryTypes.Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = color))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T

        MeshCat.atframe(anim,t) do
            p_mid = [k_mid(model, x[t])[1], 0.0, k_mid(model, x[t])[2]]
            p_ee = [k_ee(model, x[t])[1], 0.0, k_ee(model, x[t])[2]]

            settransform!(vis["l1$i"], cable_transform(zeros(3), p_mid))
            settransform!(vis["l2$i"], cable_transform(p_mid, p_ee))

            settransform!(vis["elbow$i"], Translation(p_mid))
            settransform!(vis["ee$i"], Translation(p_ee))
        end
    end

    # settransform!(vis["/Cameras/default"],
    #    compose(Translation(0.0 , 0.0 , 0.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
