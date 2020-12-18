include("/home/taylor/Research/DirectMotionPlanning/src/DirectMotionPlanning.jl")
# switch workspace
include("/home/taylor/Research/DirectMotionPlanning/src/problem.jl")
include("/home/taylor/Research/DirectMotionPlanning/models/hopper.jl")
include("/home/taylor/Research/DirectMotionPlanning/examples/development/hopper_simulator.jl")

include("/home/taylor/Research/DirectMotionPlanning/src/constraints/contact.jl")

const DMP = Main.DirectMotionPlanning
model_sim = model # DMP.no_slip_model(model)

# # Simulate contact model one step
# function step_contact(model, x1, u1, w1, h)
#     # Horizon
#     T = 2
#
#     # Objective
#     obj = DMP.PenaltyObjective(1.0e5, model.m)
#
#     # Constraints
#     con_dynamics = DMP.dynamics_constraints(model, T; w = [w1])
#     con_contact = contact_constraints(model, T)
#     con = DMP.multiple_constraints([con_dynamics, con_contact])
#
#     # Bounds
#     _uu = Inf * ones(model.m)
#     _uu[model.idx_u] .= u1
#     _ul = zeros(model.m)
#     _ul[model.idx_u] .= u1
#     ul, uu = DMP.control_bounds(model, T, _ul, _uu)
#
#     xl, xu = DMP.state_bounds(model, T, x1 = x1)
#
#     # Problem
#     prob = DMP.trajectory_optimization_problem(model,
#                    obj,
#                    T,
#                    h = h,
#                    xl = xl,
#                    xu = xu,
#                    ul = ul,
#                    uu = uu,
#                    con = con,
#                    dynamics = false) # instead, encode dynamics w/ noise
#
#     # Trajectory initialization
#     x0 = [x1 for t = 1:T]
#     u0 = [[u1; 1.0e-5 * rand(model.m - model.nu)] for t = 1:T-1] # random controls
#
#     # Pack trajectories into vector
#     z0 = DMP.pack(x0, u0, prob)
#
#     @time z = DMP.solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)
#
#     @assert DMP.check_slack(z, prob) < 1.0e-4
#     x, u = DMP.unpack(z, prob)
#
#     return x[end]
# end

step_contact(model_sim, rand(model_sim.nq), rand(model_sim.nq), rand(model_sim.nq),
    rand(model_sim.m)[1:model.nu], zeros(model_sim.nq), 0.01)

# x1 = rand(model_sim.n)
# u1 = rand(model_sim.m)[1:model.nu]
# h = 0.01
#
# # Horizon
# T = 2
#
# # Objective
# obj = DMP.PenaltyObjective(1.0e5, model_sim.m)
#
# # Constraints
# con_dynamics = DMP.dynamics_constraints(model_sim, T; w = [0.01 * ones(model.nq)])
# con_contact = contact_no_slip_constraints(model_sim, T)
# con = DMP.multiple_constraints([con_dynamics, con_contact])
#
# # Bounds
# _uu = Inf * ones(model_sim.m)
# _uu[model_sim.idx_u] .= u1
# _ul = zeros(model_sim.m)
# _ul[model_sim.idx_u] .= u1
# ul, uu = DMP.control_bounds(model_sim, T, _ul, _uu)
#
# xl, xu = DMP.state_bounds(model_sim, T, x1 = x1)
#
# # Problem
# prob = DMP.trajectory_optimization_problem(model_sim,
#                obj,
#                T,
#                h = h,
#                xl = xl,
#                xu = xu,
#                ul = ul,
#                uu = uu,
#                con = con,
#                dynamics = false)
#
# # Trajectory initialization
# x0 = [x1 for t = 1:T]
# u0 = [1.0e-5 * rand(model_sim.m) for t = 1:T-1] # random controls
#
# # Pack trajectories into vector
# z0 = DMP.pack(x0, u0, prob)
# sparsity_jacobian(prob)
# @time z = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)
#
# model_sim
# idx = prob.prob.idx
# model = prob.prob.model
# [z[idx.u[t]][model.idx_s][1] for t = 1:T-1]
# check_slack(z, prob)
# prob.prob.T
