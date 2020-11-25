@testset "emcee.jl" begin
    @testset "example" begin
        @testset "untransformed space" begin
            # define model
            function logprob(θ)
                s, m = θ
                s > 0 || return -Inf

                mdist = Normal(0, sqrt(s))
                obsdist = Normal(m, sqrt(s))

                return logpdf(InverseGamma(2, 3), s) + logpdf(mdist, m) +
                    logpdf(obsdist, 1.5) + logpdf(obsdist, 2.0)
            end
            model = DensityModel(logprob)

            # perform stretch move and sample from prior in initial step
            Random.seed!(100)
            sampler = Ensemble(1_000, StretchProposal([InverseGamma(2, 3), Normal(0, 1)]))
            chain = sample(model, sampler, 1_000;
                           param_names = ["s", "m"], chain_type = Chains)

            @test mean(chain["s"]) ≈ 49/24 atol=0.1
            @test mean(chain["m"]) ≈ 7/6 atol=0.1
        end

        @testset "transformed space" begin
            # define model
            function logprob(θ)
                logs, m = θ
                s = exp(logs)
                sqrts = sqrt(s)

                mdist = Normal(0, sqrts)
                obsdist = Normal(m, sqrts)

                return logpdf(InverseGamma(2, 3), s) + logpdf(mdist, m) +
                    logpdf(obsdist, 1.5) + logpdf(obsdist, 2.0) + logs
            end
            model = DensityModel(logprob)

            # perform stretch move and sample from normal distribution in initial step
            Random.seed!(100)
            sampler = Ensemble(1_000, StretchProposal(MvNormal(2, 1)))
            chain = sample(model, sampler, 1_000;
                           param_names = ["logs", "m"], chain_type = Chains)

            @test mean(exp, chain["logs"]) ≈ 49/24 atol=0.1
            @test mean(chain["m"]) ≈ 7/6 atol=0.1
        end
    end
end
