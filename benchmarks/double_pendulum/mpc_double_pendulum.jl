import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()

using TrajectoryOptimization
using Altro
using RobotDynamics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Random
const TO = TrajectoryOptimization
const RD = RobotDynamics

using JLD2
@load "/home/taylor/Research/DirectMotionPlanning/examples/development/double_pendulum_limit_cycle.jld2" x _u _h ctg

function Altro.initialize!(solver::Altro.iLQRSolver)
	Altro.reset!(solver)
    Altro.set_verbosity!(solver)
    Altro.clear_cache!(solver)

    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0

    # Initial rollout
    # rollout!(solver)
	# @warn "skip initial rollout!"
    TO.cost!(solver.obj, solver.Z)
end

# Model and discretization
include(joinpath(@__DIR__, "double_pendulum.jl"))

n, m = size(model)

N = 1
TN = N * length(x) - (N - 1)
xN = deepcopy(x)
uN = deepcopy(_u)
for i = 1:(N - 1)
    xN = [xN..., x[2:end]...]
    uN = [uN..., _u...]
end
QN = [Diagonal(100.0 * ones(model.n)) for t = 1:TN]
RN = [Diagonal(0.1 * ones(model.m)) for t = 1:TN-1]

x1 = x[1]
xT = x[end]
X0 = deepcopy(xN)
U0 = deepcopy(uN)

# Horizon
# _T = length(x̄)
T = TN

# Time step
tf = _h[1] * (T - 1)#2 * sum(h̄)
h = _h[1] #h̄[1]


Q = QN[1] #Diagonal(@SVector ones(n))
R = RN[1] #Diagonal(0.1 * @SVector ones(m))
obj = LQRObjective(Q, R, 1.0 * Q, xT, T)

# Constraints
cons = ConstraintList(n, m, T)
add_constraint!(cons, GoalConstraint(xT), T)
add_constraint!(cons, BoundConstraint(n, m,
    u_min = -10.0 * ones(m),
    u_max = 10.0 * ones(m)), 1:T-1)

# Create and solve problem
opts = SolverOptions(
    cost_tolerance_intermediate = 1.0e-2,
    penalty_scaling = 10.0,
    penalty_initial = 1000.0,
    projected_newton = false,
    constraint_tolerance = 1.0e-3,
    iterations = 5000,
    iterations_inner = 100,
    iterations_linesearch = 100,
    iterations_outer = 500)

prob = Problem(model, obj, xT, tf,
    U0 = U0,
    X0 = X0,
    dt = h,
    x0 = x1,
	constraints = cons,
	integration = PassThrough)
# rollout!(prob)
solver = ALTROSolver(prob, opts, verbose = 2)
cost(solver)           # initial cost
rollout!(solver)
@time solve!(solver)   # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# # Get the state and control trajectories
X = states(solver)
U = controls(solver)

using Plots
plot(hcat(X[1:2:end]...)',
	labels = "")
plot(hcat(U[1:2:end]...)',
	linetype=:steppost,
	labels = "")

Z_track = TO.Traj([TO.get_trajectory(solver)[1:end]...])#,
    # TO.get_trajectory(solver)[2:end]...,
    # TO.get_trajectory(solver)[2:end]...,
    # TO.get_trajectory(solver)[2:end]...,
    # TO.get_trajectory(solver)[2:end]...])

# Z_track = TO.Traj([TO.get_trajectory(solver)[1:end]...,
#     # TO.get_trajectory(solver)[2:end]...,
#     Z_shift_init[2:end]...,
#     # Z_shift[2:end]...,
#     Z_shift[2:end]...])
X_track = states(Z_track)
U_track = controls(Z_track)

using Plots
plot(hcat(X_track[1:2:end]...)',
    labels="",
    width = 2.0, legend = :right)
plot(hcat(U_track[1:2:end]...)',
    linetype=:steppost,
    labels = "",
    width = 2.0)

include("/home/taylor/Research/DirectMotionPlanning/models/visualize.jl")
using MeshCat, GeometryBasics, Colors
vis = Visualizer()
render(vis)
visualize!(vis, model, X_track, Δt = h)

## Model-predictive control tracking problem
include("../mpc.jl")
function run_hopper_MPC(prob_mpc, opts_mpc, Z_track,
                            num_iters = length(Z_track) - prob_mpc.N)
    solver_mpc = ALTROSolver(prob_mpc, opts_mpc, verbose = 2)

    # Solve initial iteration
    Altro.solve!(solver_mpc)

	return states(solver_mpc), Dict()
    iters = zeros(Int, num_iters,2)
    times = zeros(num_iters,2)

    # Get the problem state size and control size
    n, m = size(prob_mpc)
    dt = Z_track[1].dt
    t0 = 0
    k_mpc = 1
    x0 = SVector(prob_mpc.x0)
    X_traj = [copy(x0) for k = 1:num_iters+1]

    # Begin the MPC LOOP
    for i = 1:num_iters
        # Update initial time
        t0 += dt
        k_mpc += 1
        TO.set_initial_time!(prob_mpc, t0)

        # Update initial state by using 1st control, and adding some noise
        x0 = discrete_dynamics(TO.integration(prob_mpc),
                                    prob_mpc.model, prob_mpc.Z[1])
        # if i == 150
        #     @warn "additive disturbance"
        # x0 += (1.0e-12 * [1.0; 1.0; 1.0; 1.0] .* @SVector randn(model.n) )
        # end

        # Update the initial state after the dynamics are propogated.
        TO.set_initial_state!(prob_mpc, x0)
        X_traj[i+1] = x0

        # Update tracking cost
        TO.update_trajectory!(prob_mpc.obj, Z_track, k_mpc)

        # Shift the initial trajectory
        RD.shift_fill!(prob_mpc.Z)

        # Shift the multipliers and penalties
        Altro.shift_fill!(TO.get_constraints(solver_mpc))

        # Solve the updated problem
        Altro.solve!(solver_mpc)

        if Altro.status(solver_mpc) != Altro.SOLVE_SUCCEEDED
            println(Altro.status(solver_mpc))
            @warn ("Solve not succeeded at iteration $i")
            return X_traj, Dict(:time=>times, :iter=>iters)
        end

        # Log the results and performance
        iters[i,1] = iterations(solver_mpc)

        times[i,1] = solver_mpc.stats.tsolve
    end

    @warn "solve success"
    return X_traj, Dict(:time=>times, :iter=>iters)
end

Random.seed!(1)
opts_mpc = SolverOptions(
    cost_tolerance = 1.0e-4,
    cost_tolerance_intermediate = 1.0e-4,
    constraint_tolerance = 1.0e-4,
    reset_duals = false,
    penalty_initial = 1000.0,
    penalty_scaling = 10.0,
    projected_newton = false,
    iterations = 50)
T_mpc = 100
prob_mpc = gen_tracking_problem(prob, T_mpc,
    Q = Diagonal(1000.0 * [1.0, 1.0, 1.0, 1.0]),
    R = Diagonal([1.0e-3, 1.0e-3]),
    Qf = Diagonal(1000.0 * [1.0, 1.0, 1.0, 1.0]))
solver_mpc = ALTROSolver(prob_mpc, opts_mpc, verbose = 2)

Altro.step!(solver_mpc.solver_al.solver_uncon, Inf)
X_traj = states(solver_mpc.solver_al.solver_uncon)
plot(hcat(controls(solver_mpc.solver_al.solver_uncon)...)', linetype = :steppost)
N_mpc = 50
X_traj, res = run_hopper_MPC(prob_mpc, opts_mpc, Z_track, N_mpc)

plot(hcat(X_track[1:N_mpc]...)',
    labels = "", legend = :bottomleft,
    width = 2.0, color = ["red" "green" "blue" "orange"])
plot!(hcat(X_traj[1:N_mpc]...)',
    labels = "", legend = :bottom,
    width = 1.0, color = ["red" "green" "blue" "orange"], linestyle = :dash)

vis = Visualizer()
render(vis)
visualize!(vis, model, X_traj, Δt = h)
