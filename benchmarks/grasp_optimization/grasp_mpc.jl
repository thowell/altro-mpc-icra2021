include("src/grasp_model.jl")       # defines model
include("src/grasp_problem.jl")     # defines LQR problem
include("src/new_constraints.jl")   # defines additional constraints
include("src/grasp_mpc_helpers.jl") # defines mpc_update!() and gen_ECOS()
include("../mpc.jl")                # defines tracking problem

function run_grasp_mpc(prob_mpc, opts_mpc, Z_track, 
        num_iters = length(Z_track) - prob_mpc.N; 
        print_all=true,
        optimizer = JuMP.optimizer_with_attributes(ECOS.Optimizer, 
            "verbose"=>false,
            "feastol"=>opts_mpc.constraint_tolerance,
            "abstol"=>opts_mpc.cost_tolerance,
            "reltol"=>opts_mpc.cost_tolerance
        ),
        warmstart=false,
        warmstart_dual=false
    )
    n, m, N_mpc = size(prob_mpc)
    o = prob_mpc.model
    x0 = state(Z_track[1])
    u0 = control(Z_track[1])

    # Arrays for results
    altro_times = zeros(num_iters)
    altro_iters = zeros(num_iters)
    altro_states = [zero(x0) for i = 1:num_iters+1]
    altro_states[1] = x0
    altro_controls = [zero(u0) for i = 1:num_iters]
    ecos_times = zeros(num_iters)
    ecos_controls = copy(altro_controls)
    err_traj = zeros(num_iters,2)

    # Solve first iteration 
    altro = ALTROSolver(prob_mpc, opts_mpc)
    set_options!(altro, show_summary=false, verbose=0)
    solve!(altro)

    prob_mpc_ecos, X_ecos, U_ecos = mpc_update!(prob_mpc, o, iter, Z_track)
    set_optimizer(prob_mpc_ecos, optimizer)
    optimize!(prob_mpc_ecos)
    if warmstart_dual
        duals = [dual.(all_constraints(prob_mpc_ecos, l...)) 
            for l in list_of_constraint_types(prob_mpc_ecos)]
    end

    for iter in 1:num_iters
        # Updates prob_mpc in place, returns an equivalent ecos problem
        prob_mpc_ecos, X_ecos, U_ecos = mpc_update!(prob_mpc, o, iter, Z_track)
        Altro.shift_fill!(TO.get_constraints(altro))
        X0 = hcat(Vector.(states(altro))...)
        U0 = hcat(Vector.(controls(altro))...)

        # Solve Altro
        Altro.solve!(altro)

        # Solve Ecos
        if warmstart
            set_start_value.(X_ecos, X0)
            set_start_value.(U_ecos, U0)
        end
        if warmstart_dual
            for (i,l) in enumerate(list_of_constraint_types(prob_mpc_ecos))
                set_dual_start_value.(all_constraints(prob_mpc_ecos, l...), duals[i])
            end
        end
        JuMP.set_optimizer(prob_mpc_ecos, optimizer)
        JuMP.optimize!(prob_mpc_ecos)
        if warmstart_dual
            duals = [dual.(all_constraints(prob_mpc_ecos, l...)) 
                for l in list_of_constraint_types(prob_mpc_ecos)]
        end

        # Compute max infinity norm diff
        diffs = []
        X = [value.(X_ecos[:,i]) for i in 1:N_mpc]
        U = [value.(U_ecos[:,i]) for i in 1:N_mpc-1]
        xdiff = maximum(norm.(X - states(altro), Inf))
        udiff = maximum(norm.(U - controls(altro), Inf))
        err_traj[iter,:] = [xdiff, udiff]

        # Printouts
        if print_all
            println("Timestep $iter")
            print("ALTRO runtime: $(round(altro.stats.tsolve, digits=2)) ms")
            println("\t Max violation: $(TrajectoryOptimization.max_violation(altro))")
            print("ECOS runtime: $(round(1000*ecos.sol.solve_time, digits=2)) ms")
            println("\tStatus: ", termination_status(prob_mpc_ecos))
            println("State diff = ", round(xdiff, digits=2), "\tControl diff = ", round(udiff, digits=2))
        end

        # Update arrays
        altro_times[iter] = altro.stats.tsolve
        altro_iters[iter] = iterations(altro)
        altro_states[iter+1] = state(prob_mpc.Z[1])
        altro_controls[iter] = control(prob_mpc.Z[1])
        ecos_times[iter] = 1000*solve_time(prob_mpc_ecos)
        ecos_controls[iter] = value.(U_ecos)[:, 1]
    end

    altro_traj = Dict(:states=>altro_states, :controls=>altro_controls)
    res = Dict(:time=>[altro_times ecos_times], :iter=>altro_iters, :err_traj=>err_traj)

    # Print Solve Time Difference
    ave_diff = mean(ecos_times) - mean(altro_times)
    println("\nAverage ALTRO solve time was $(round(ave_diff, digits=2)) ms faster than that of $(solver_name(prob_mpc_ecos))\n")

    return res, altro_traj, ecos_controls
end

## MPC Setup
# o = SquareObject()
# prob_cold = GraspProblem(o,251)
# opts = SolverOptions(
#     verbose = 0,
#     projected_newton=false,
#     cost_tolerance=1e-6,
#     cost_tolerance_intermediate=1e-4,
#     constraint_tolerance=1e-6
# )
# altro = ALTROSolver(prob_cold, opts, show_summary=false)
# Altro.solve!(altro)
# Z_track = get_trajectory(altro)

# Random.seed!(1)
# opts_mpc = SolverOptions(
#     cost_tolerance=1e-4,
#     cost_tolerance_intermediate=1e-3,
#     constraint_tolerance=1e-4,
#     projected_newton=false,
#     penalty_initial=10_000.,
#     penalty_scaling=100.,
#     # reset_duals = false,
# )
# num_iters = 20 # number of MPC iterations
# N_mpc = 21 # length of the MPC horizon in number of steps


# Q = 1e3
# R = 1e0
# Qf = 10.0

# iter = 1
# prob_mpc = gen_tracking_problem(prob_cold, N_mpc, Qk = Q, Rk = R, Qfk = Qf)
# prob_mpc_ecos, X_ecos, U_ecos = mpc_update!(prob_mpc, o, iter, Z_track)
# set_start_value.(X_ecos, zeros(size(X_ecos)))

# m = prob_mpc_ecos
# set_optimizer(m, optimizer)
# optimize!(m)
# l = list_of_constraint_types(m)
# duals = [dual.(all_constraints(prob_mpc_ecos, l...)) 
#     for l in list_of_constraint_types(prob_mpc_ecos)]
# l = list_of_constraint_types(m)
# duals = [dual.(all_constraints(m, li...)) for li in l]
# set_dual_start_value.(all_constraints(m, l[1]...), duals[1])

# # Test single run
# num_iters = 100
# print_all = false 
# res, altro_traj, ecos_controls = run_grasp_mpc(prob_mpc, opts_mpc, Z_track,
#                                             num_iters, print_all=print_all)
# println(mean(res[:iter]))


## Histogram of timing results
# altro_times = res[:time][:,1]
# ecos_times = res[:time][:,2]
# bounds = extrema([altro_times; ecos_times])
# bin_min = floor(Int, bounds[1]) - 1
# bin_max = ceil(Int, bounds[2]) + 1
# bins = collect(bin_min:bin_max)
# histogram(altro_times, bins=bins, fillalpha=.5, label="ALTRO")
# histogram!(ecos_times, bins=bins, fillalpha=.5, label="ECOS")
# xlabel!("Solve Time (ms)")
# ylabel!("Counts")

# # Plot of tangential to normal force ratio
# U = altro_traj[:controls]
# normal1 = [dot(o.v[1][i+1], U[i][1:3]) for i = 1:num_iters]
# normal2 = [dot(o.v[2][i+1], U[i][4:6]) for i = 1:num_iters]
# tangent1 = [norm((I - o.v[1][i+1]*o.v[1][i+1]')*U[i][1:3]) for i = 1:num_iters]
# tangent2 = [norm((I - o.v[2][i+1]*o.v[2][i+1]')*U[i][4:6]) for i = 1:num_iters]
# friction1 = tangent1 ./ normal1
# friction2 = tangent2 ./ normal2
# plot([friction1 friction2 o.mu*ones(num_iters)],
#     xlabel="Time Step",
#     ylabel="Tangential Force/Normal Force",
#     linestyle = [:solid :solid :dash],
#     label = ["F_T1/F_N1" "F_T2/F_N2" "mu"])

# # Compare F1
# U = altro_traj[:controls]
# Ue = ecos_controls
# Uc = controls(prob_cold)
# N = num_iters

# # altro
# u1 = [U[t][1] for t = 1:N-1]
# u2 = [U[t][2] for t = 1:N-1]
# u3 = [U[t][3] for t = 1:N-1]
# # ecos
# u1e = [Ue[t][1] for t = 1:N-1]
# u2e = [Ue[t][2] for t = 1:N-1]
# u3e = [Ue[t][3] for t = 1:N-1]
# # cold solve
# u1c = [Uc[t][1] for t = 1:N-1]
# u2c = [Uc[t][2] for t = 1:N-1]
# u3c = [Uc[t][3] for t = 1:N-1]

# plot([u1 u2 u3 u1e u2e u3e u1c u2c u3c],
#     xlabel="Time Step",
#     ylabel="Coordinate",
#     label = ["u1" "u2" "u3" "u1e" "u2e" "u3e" "u1c" "u2c" "u3c"])
