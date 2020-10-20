const data = YAML.load(open(joinpath(@__DIR__, "MPC.yaml")))

if data["solver"] == "ALTRO"
	const using_altro = true

	using TrajectoryOptimization
	using RobotDynamics
	using Altro

	const TO = TrajectoryOptimization
	const RD = RobotDynamics

	include("Structs/FootstepLocation.jl")
	include("Structs/ALTROParams.jl")
	if data["linearized_friction_constraint"]
		include("Structs/LinearizedFrictionConstraint.jl")
	else
		include("Structs/FrictionConstraint.jl")
	end
	include("Structs/SwingLegParams.jl")
	include("Structs/GaitParams.jl")
	include("Structs/ControllerParams.jl")

	include("altro_solver.jl")

else
	const using_altro = false
	using OSQP
	using SparseArrays

	include("Structs/FootstepLocation.jl")
	include("Structs/QPParams.jl")
	include("Structs/SwingLegParams.jl")
	include("Structs/GaitParams.jl")
	include("Structs/ControllerParams.jl")

	include("osqp_solver.jl")
end
