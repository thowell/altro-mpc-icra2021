# signed distance
struct SD <: TO.StateConstraint
	n::Int
	nc::Int
	model::TO.AbstractModel
	function SD(n::Int, nc::Int, model)
		new(n, nc, model)
	end
end
TO.state_dim(con::SD) = con.n
TO.sense(::SD) = TO.Inequality()
Base.length(con::SD) = con.nc
TO.evaluate(con::SD, x::SVector) = -1.0 * ϕ_func(con.model, x[con.model.nq .+ (1:con.model.nq)])

# complementarity (impact)
struct IC <: TO.StateConstraint
	n::Int
	nc::Int
	model::TO.AbstractModel
	function IC(n::Int, nc::Int, model)
		new(n, nc, model)
	end
end
TO.state_dim(con::IC) = con.n
TO.sense(::IC) = TO.Equality()
Base.length(con::IC) = con.nc
function TO.evaluate(con::IC, x::SVector)
	q3 = view(x, con.model.nq .+ (1:con.model.nq))
	# q2 = view(x, 1:model.nq)
	λ = view(x, 2 * con.model.nq .+ (1:con.nc))
	return min.(λ, ϕ_func(con.model, q3))
end

# friction cone
struct FC <: TO.ControlConstraint
	m::Int
	nc::Int
	model::TO.AbstractModel
	function FC(m::Int, nc::Int, model)
		new(m, nc, model)
	end
end
TO.control_dim(con::FC) = con.m
TO.sense(::FC) = TO.Inequality()
Base.length(con::FC) = con.nc
TO.evaluate(con::FC, u::SVector) = -1.0 * friction_cone(con.model, u)

# complementarity (no slip)
struct NS{T} <: TO.StateConstraint
	n::Int
	nc::Int
	model::TO.AbstractModel
	h::T
	function NS(n::Int, nc::Int, model, h::T) where T
		new{T}(n, nc, model, h)
	end
end
TO.state_dim(con::NS) = con.n
TO.sense(::NS) = TO.Equality()
Base.length(con::NS) = 1
function TO.evaluate(con::NS, x::SVector)
	q3 = view(x, con.model.nq .+ (1:con.model.nq))
	q2 = view(x, 1:con.model.nq)
	λ = view(x, 2 * con.model.nq .+ (1:con.nc))
	# return min.(λ, abs.(_P_func(con.model, q3) * (q3 - q2) / con.h))
	return [λ' * _P_func(con.model, q3) * (q3 - q2) / con.h]
end

# # Slack control
# struct Slack <: TO.ControlConstraint
# 	m::Int
# 	model
# 	function Slack(m::Int, model)
# 		new(m, model)
# 	end
# end
# TO.control_dim(con::Slack) = con.m
# TO.sense(::Slack) = TO.Equality()
# Base.length(con::Slack) = con.model.ns
# function TO.evaluate(con::Slack, u::SVector)
# 	s = view(u, con.model.idx_s)
# 	return s
# end
