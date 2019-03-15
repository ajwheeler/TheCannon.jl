module TheCannon
using Optim, Statistics, LinearAlgebra, ForwardDiff
export projected_size,
       deprojected_size,
       project_labels,
       standardize_labels,
       unstandardize_labels,
       train,
       regularized_train,
       infer,
       quad_coeff_matrix

function projected_size(nlabels; quadratic=true)
    if quadratic
        Int(1 + 2nlabels + nlabels*(nlabels-1)/2)
    else
        nlabels + 1
    end
end

function deprojected_size(nplabels; quadratic=true)
    if quadratic
        Int((-3 + sqrt(1 + 8nplabels))/2)
    else
        nplabels - 1
    end
end

function project_labels(labels::Vector{R}; quadratic=true) where R <: Real
    vec(project_labels(Matrix(transpose(labels)), quadratic=quadratic))
end
function project_labels(labels::Matrix{R}; quadratic=true) where R <: Real
    nstars, nlabels = size(labels)
    plabels = Matrix{R}(undef, nstars, projected_size(nlabels; quadratic=quadratic))
    plabels[:, 1] .= 1
    plabels[:, 2:nlabels+1] .= labels
    if quadratic
        k = 1
        for i in 1:nlabels
            for j in i:nlabels
                plabels[:, nlabels + 1 + k] .= labels[:, i] .* labels[:, j]
                k += 1
            end
        end
    end
    plabels
end

"""
Get the quadratic terms of theta as matrices.
returns an array of dimensions nlabels x nlabels x npixels

   Q = quad_coeff_matrix(theta)
   Q[:, :, 1] #quadratic coefficients for first pixel

"""
function quad_coeff_matrix(theta::Matrix{Float64}) :: Array{Float64, 3}
    nlabels = deprojected_size(size(theta, 1)) 
    npix = size(theta, 2)
    Q = Array{Float64}(undef, nlabels, nlabels, npix)
    for p in 1:npix
        k = 1
        for i in 1:nlabels
            for j in i:nlabels
                Q[i, j, p] = theta[nlabels + 1 + k, p]
                Q[j, i, p] = Q[i, j, p]
                k += 1
            end 
        end 
    end 
    Q
end

function standardize_labels(labels)
    pivot = mean(labels, dims=1)
    scale = std(labels, dims=1)
    (labels .- pivot)./scale, vec(pivot), vec(scale)
end

function unstandardize_labels(labels, pivot, scale)
    labels.*transpose(hcat(scale)) .+ transpose(hcat(pivot))
end

function train1d(flux::Matrix{Float64}, ivar::Matrix{Float64}, 
                             labels::Matrix{Float64}; Λ=10, fastmode=false)
    nstars = size(flux,1)
    npix = size(flux, 2)
    nlabels = size(labels, 2)
    labels = project_labels(labels)
    nplabels = size(labels, 2)
    println("$nstars stars, $npix pixels, $nplabels labels")

    theta = Matrix{Float64}(undef, nplabels, npix)
    scatters = Vector{Float64}(undef, npix)

    #do linear regression for each pixel
    for i in 1:npix
        println("pixel $i")
        function negative_log_likelihood(scatter) #up to constant
            Σ = Diagonal(ivar[:, i].^(-1) .+ scatter^2)
            lT_invcov_l = transpose(labels) * Σ * labels
            lT_invcov_F = transpose(labels) * Σ * flux[:,i]
            coeffs = lT_invcov_l \ lT_invcov_F
            χ = labels*coeffs - flux[:, i]
            (0.5*(transpose(χ) * inv(Σ) * χ) + #chi-squared
             0.5*sum(log.(diag(Σ))) + #normalizaion term
             Λ*sum(abs.(coeffs[4:end]))) #L1 penalty
        end
        fit = optimize(negative_log_likelihood, 0, 2)
        #println(fit)
        if ! fit.converged
            @warn "pixel $i not converged"
        end
        scatters[i] = fit.minimizer[end]
        Σ = Diagonal(ivar[:, i].^(-1) .+ scatters[i]^2)
        lT_invcov_l = transpose(labels) * Σ * labels
        lT_invcov_F = transpose(labels) * Σ * flux[:,i]
        theta[:, i] = lT_invcov_l \ lT_invcov_F
    end
    theta, scatters
end

"""
    train(flux, ivar, labels)

returns: theta, scatters
Run the training step of The Cannon, i.e. calculate coefficients for each pixel.
 - `flux` contains the spectra for each pixel in the training set.  It should be 
    `nstars x npixels` (row-vectors are spectra)
 - `ivar` contains the inverse variance for each pixel in the same shape as `flux`
 - `labels` contains the labels for each star.  It should be `nstars x nlabels`.
    It will be projected into the quadratic label space before training.
"""
function train(flux::Matrix{Float64}, ivar::Matrix{Float64}, 
                           labels::Matrix{Float64}; Λ=10, fastmode=false)
    nstars = size(flux,1)
    npix = size(flux, 2)
    nlabels = size(labels, 2)
    labels = project_labels(labels)
    nplabels = size(labels, 2)
    println("$nstars stars, $npix pixels, $nplabels labels")

    theta = Matrix{Float64}(undef, nplabels, npix)
    scatters = Vector{Float64}(undef, npix)

    #do linear regression for each pixel
    for i in 1:npix
        println("pixel $i")
        function negative_log_likelihood(coeffscatter) #up to constant
            scatter = coeffscatter[end]
            coeffs = coeffscatter[1:end-1]

            χ = labels*coeffs - flux[:, i]
            Σ = Diagonal(ivar[:, i].^(-1) .+ scatter^2)
            (0.5*(transpose(χ) * inv(Σ) * χ) + #chi-squared
             0.5*sum(log.(diag(Σ))) + #normalizaion term
             Λ*sum(abs.(coeffs[4:end]))) #L1 penalty
        end

        thetascatter0 = zeros(nplabels+1)
        lT_invcov_l = transpose(labels) * Diagonal(ivar[:, i]) * labels 
        lT_invcov_F = transpose(labels) * Diagonal(ivar[:, i]) * flux[:,i]
        #lT_invcov_l = transpose(labels) * labels 
        #lT_invcov_F = transpose(labels) * flux[:,i]
        thetascatter0[1:end-1] = lT_invcov_l \ lT_invcov_F
        thetascatter0[end] = 0.01

        #lower = vcat(fill(-1., nplabels), [0.])
        #upper = vcat(fill(2., nplabels), [1.])
        fit = optimize(negative_log_likelihood, thetascatter0,
                       iterations=fastmode ? 1000 : 100000)
        #println(fit)
        if ! fit.g_converged
            @warn "pixel $i not converged"
        end
        scatters[i] = fit.minimizer[end]
        theta[:, i] = fit.minimizer[1:end-1]
    end
    theta, scatters
end

function logπ(label::R, p) where R <: Real
    if ismissing(p)
        return 0
    else
        σ = (p[3] - p[1])/2
        return -0.5*((label-p[2])/σ)^2 - log(σ)
    end
end                                        

"""
    test(flux, ivar, theta, scatters

Run the test step of the cannon.
Given a Cannon model (from training), infer stellar parameters
"""
function infer(flux::AbstractMatrix{Float64}, ivar::AbstractMatrix{Float64},
              theta::AbstractMatrix{Float64}, scatters::Vector{Float64}, 
              prior::AbstractMatrix{Union{Tuple{Float64, Float64, Float64}, Missing}};
              quadratic=true, verbose=true)
    nstars = size(flux, 1)
    nplabels = size(theta, 1)
    nlabels = deprojected_size(nplabels; quadratic=quadratic)

    inferred_labels = Matrix{Float64}(undef, nstars, nlabels)
    chi_squared = Vector{Float64}(undef, nstars)
    information = Array{Float64, 3}(undef, nstars, nlabels, nlabels)

    thetaT = transpose(theta)
    for i in 1:nstars
        if verbose && i%100==0
            println("inferring labels for star $i")
        end

        F = flux[i, :]
        invσ2 = (ivar[i, :].^(-1) .+ scatters.^2).^(-1)
        function negative_log_post(labels)
            A2 = (thetaT * project_labels(labels; quadratic=quadratic) .- F).^2
            0.5 * sum(A2 .* invσ2) - sum(logπ.(labels, prior[:, i]))
        end
        fit = optimize(negative_log_post, zeros(nlabels), Optim.Options(g_tol=1e-6),
                      autodiff=:forward)
        
        inferred_labels[i, :] = fit.minimizer
        chi_squared[i] = sum((thetaT * project_labels(fit.minimizer; quadratic=quadratic) .- F).^2 .* invσ2)
        information[i, :, :] = ForwardDiff.hessian(negative_log_post, fit.minimizer)
    end
    inferred_labels, chi_squared, information
end
function infer(flux::AbstractMatrix{Float64}, ivar::AbstractMatrix{Float64},
              theta::AbstractMatrix{Float64}, scatters::Vector{Float64};
              kwargs...)
    prior = Matrix{Missing}(missing, nlabels, nstars)
    infer(flux, ivar, theta, scatters, prior; kwargs...)
end

end
