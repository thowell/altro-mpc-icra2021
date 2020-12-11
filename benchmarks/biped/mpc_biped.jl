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
@load joinpath(@__DIR__, "biped_gait_no_slip.jld2") x̄ ū h̄ x_proj u_proj

# Model and discretization
include(joinpath(@__DIR__, "biped.jl"))

n, m = size(model)

# Horizon
T = length(x_proj)

# Time step
tf = sum(h̄)
h = h̄[1]

# Objective
x1 = [x_proj[1]; zeros(model.nc)]
xT = [x_proj[T]; zeros(model.nc)]

X0 = [[x_proj[t]; zeros(model.nc)] for t = 1:T]
U0 = [u_proj[t][1:10] for t = 1:T-1]

Q = Diagonal(100.0 * @SVector ones(n))
R = Diagonal(100.0 * @SVector ones(m))
obj = LQRObjective(Q, R, 1.0 * Q, xT, T)

# Constraints
include(joinpath(@__DIR__, "contact_constraints.jl"))
cons = ConstraintList(n, m, T)

# add_constraint!(cons, GoalConstraint(xT, (1:2 * model.nq)), T)
add_constraint!(cons, BoundConstraint(n, m,
    x_min = [model.qL; model.qL; zeros(model.nc)],
    x_max = [model.qU; model.qU; Inf * ones(model.nc)],
    u_min = [-8.0 * ones(model.nu); zeros(m - model.nu)],
    u_max = [8.0 * ones(model.nu); Inf * ones(m - model.nu)]), 1:T-1)
add_constraint!(cons, SD(n, model.nc, model), 1:T)
add_constraint!(cons, IC(n, model.nc, model), 2:T)
add_constraint!(cons, FC(m, model.nc, model), 1:T-1)
add_constraint!(cons, NS(n, model.nc, model, h), 2:T)

# cons[end]
# TO.evaluate(cons[end], states(solver)[26])
# Create and solve problem
opts = SolverOptions(
    cost_tolerance_intermediate = 1.0e-2,
    penalty_scaling = 10.0,
    penalty_initial = 1.0e6,
    projected_newton = false,
    constraint_tolerance = 1.0e-3,
    iterations = 500,
    iterations_inner = 100,
    iterations_linesearch = 100,
    iterations_outer = 20)

prob = TO.Problem(model, obj, xT, tf,
    U0 = U0,
    X0 = X0,
    dt = h,
    x0 = copy(x1), constraints = cons, integration = PassThrough)

# FIRST PROCESSING
solver = ALTROSolver(prob, opts, verbose = 2)
cost(solver)           # initial cost
@time solve!(solver)   # solve with ALTRO
max_violation(solver)  # max constraint violation
TO.get_constraints(solver)
# TO.findmax_violation(solver)
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories

using Plots
X = states(solver)
U = controls(solver)

plot(hcat(state_to_configuration(x_proj, model.nq)...)',
	color = :red,
	width = 2.0,
	label = "")
plot!(hcat(state_to_configuration(X, model.nq)...)',
	color = :black,
	width = 1.0,
	label = "")

plot(hcat(u_proj...)[1:4, :]',
	linetype = :steppost,
	color = :red,
	width = 2.0,
	label = "")
plot!(hcat(U...)[1:4, :]',
	linetype = :steppost,
	color = :black,
	width = 1.0,
	label = "")

vis = Visualizer()
render(vis)
visualize!(vis, model_sim, state_to_configuration(X, model.nq), Δt = h)

# SHIFT TRAJECTORY
len_stride = xT[1] - x1[1]
X_track = deepcopy(states(TO.get_trajectory(solver)))
X_tmp = deepcopy(states(TO.get_trajectory(solver)))
U_track = deepcopy(controls(TO.get_trajectory(solver)))
U_tmp = deepcopy(controls(TO.get_trajectory(solver)))

for i = 1:2
	for t = 1:T-1
		_x = Array(copy(X_tmp[t+1]))
		_x[1] += i * len_stride
		_x[8] += i * len_stride
		push!(X_track, SVector{n}(_x))
		push!(U_track, U_tmp[t])
	end
end
push!(U_track, zeros(model.m))
TT = length(X_track)
t = [range(0, stop = h * (TT-1), length = TT)...]
Z_track = TO.Traj([KnotPoint(X_track[k], U_track[k], h, t[k]) for k = 1:length(U_track)])

using Plots
plot(hcat(state_to_configuration(states(Z_track)[1:1:end], model.nq)...)',
    labels="",
    width = 2.0, legend = :right)
plot(hcat(controls(Z_track)[1:1:end]...)[1:model.nu, :]',
    linetype=:steppost,
    labels = "",
    width = 2.0)

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(X_track, model.nq), Δt = h)

## Model-predictive control tracking problem
include("../mpc.jl")

function run_biped_MPC(prob_mpc, opts_mpc, Z_track,
                            num_iters = length(Z_track) - prob_mpc.N)
    solver_mpc = ALTROSolver(prob_mpc, opts_mpc, verbose = 2)

    # Solve initial iteration
    Altro.solve!(solver_mpc)

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
		w0 = (i == 101 ? 0.0 * zeros(model.nq) .* randn(model.nq) : 0.0 * randn(model.nq))
		x0 = [step_contact(model_sim,
			state(prob_mpc.Z[1])[1:2 * model.nq],
			control(prob_mpc.Z[1])[1:model.nu], w0, dt); zeros(model.nc)]

        # Update the initial state after the dynamics are propogated.
        TO.set_initial_state!(prob_mpc, x0)
        X_traj[i+1] = x0

        # Update tracking cost
		# if i >= 101
		# 	TO.update_trajectory!(prob_mpc.obj, Z_track_shift, k_mpc)
		# else
        # 	TO.update_trajectory!(prob_mpc.obj, Z_track, k_mpc)
		# end
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
    constraint_tolerance = 1.0e-2,
    reset_duals = false,
    penalty_initial = 1.0e3,
    penalty_scaling = 10.0,
    projected_newton = false,
    iterations = 500)

T_mpc = 51
prob_mpc = gen_tracking_problem(prob, T_mpc,
    Q = Diagonal(SVector{n}(10.0 * ones(n))),
    R = Diagonal(SVector{m}(1.0 * ones(m))),
    Qf = Diagonal(SVector{n}(10.0 * ones(n))))

N_mpc = 2
X_traj, res = run_biped_MPC(prob_mpc, opts_mpc, Z_track, N_mpc)

plot(hcat(X_track[1:1:N_mpc]...)[model.nq .+ (1:model.nq), :]',
    labels = "", legend = :bottomleft,
    width = 2.0, color = ["red" "green" "blue" "orange"], linestyle = :dash)
plot!(hcat(X_traj[1:1:N_mpc]...)[model.nq .+ (1:model.nq), :]',
    labels = "", legend = :bottom,
    width = 1.0, color = ["red" "green" "blue" "orange"])

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(X_traj, model.nq), Δt = h)
