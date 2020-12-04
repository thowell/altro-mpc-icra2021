function initial_torque(model, q1, h)
	x = [q1; q1]
    _fd(z) = fd(model, x, x, z, h, 0)

    y = ones(model.m)
    r = _fd(y)

    iter = 0

    while norm(r, 2) > 1.0e-8 && iter < 10
        ∇r = ForwardDiff.jacobian(_fd, y)

        Δy = -1.0 * ∇r \ r

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < 10
            ŷ = y + α * Δy
            r̂ = _fd(ŷ)

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
				# print("y: $y")
				# print("x: $x")
				# print("u: $u")
				# print("w: $w")
				# println("l2: $(norm(r))")
				# println("linf: $(norm(r, Inf))")
			end
        end

        iter += 1
    end

	if iter == 10
		@warn "newton failed"
	end
    @show norm(r)
    @show iter
    return y
end
