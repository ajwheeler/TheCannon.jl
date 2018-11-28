module TheCannon
using Optim, Statistics, LinearAlgebra
export projected_size,
       deprojected_size,
       project_labels,
       standardize_labels,
       unstandardize_labels,
       train,
       infer

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

function project_labels(labels::Vector{Float64}; quadratic=true)
    n = length(labels)
    np = projected_size(n; quadratic=quadratic)
    plabels = Vector{Float64}(undef, np)
    plabels[1] = 1
    plabels[2:n+1] .= labels
    if quadratic
        k = 1
        for i in 1:n
            for j in i:n
                plabels[n + 1 + k] = labels[i] * labels[j]
                k += 1
            end
        end
    end
    plabels
end

#kinda redundant, but I wrote this first and it's faster this way.
function project_labels(labels::Matrix; quadratic=true)
    nstars, nlabels = size(labels)
    plabels = Matrix{Float64}(undef, nstars, projected_size(nlabels; quadratic=quadratic))
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
function train(flux::Matrix{Float64}, ivar::Matrix{Float64}, labels::Matrix{Float64}; 
               quadratic=true)
    nstars = size(flux,1)
    npix = size(flux, 2)
    labels = project_labels(labels; quadratic=quadratic)
    nplabels = size(labels, 2)
    println("$nstars stars, $npix pixels, $nplabels (projected) labels")

    theta = Matrix{Float64}(undef, nplabels, npix) #insert undef as first arg for Julia 1.0
    scatters = Vector{Float64}(undef, npix)

    #do linear regression for each pixel
    for i in 1:npix
        #calculate theta
        lT_invcov_l = transpose(labels) * Diagonal(ivar) * labels #more this out of loop, do cholesky docomp, check condition number?
        lT_invcov_F = transpose(labels) * Diagonal(ivar) * flux[:,i]
        theta[:, i] = lT_invcov_l \ lT_invcov_F

        #calculate optimal scatter
        function negative_log_likelihood(scatter::Float64) #up to constant
            χ = labels*theta[:, i] - flux[:, i]
            Σ = Diagonal(ivar[:, i].^(-1) .+ scatter^2)
            (0.5*(transpose(χ) * inv(Σ) * χ) + #chi-squared
             0.5*sum(log.(diag(Σ)))) #log(det(Σ))
        end
        fit = optimize(negative_log_likelihood, 0, 1)
        scatters[i] = fit.minimizer
    end
    theta, scatters
end

function regularized_train(flux::Matrix{Float64}, ivar::Matrix{Float64}, 
                           labels::Matrix{Float64}, Λ=0.1)
    nstars = size(flux,1)
    npix = size(flux, 2)
    labels = project_labels(labels)
    nplabels = size(labels, 2)
    println("$nstars stars, $npix pixels, $nplabels labels")

    theta = Matrix{Float64}(undef, nplabels, npix) #insert undef as first arg for Julia 1.0
    scatters = Vector{Float64}(undef, npix)

    #do linear regression for each pixel
    for i in 1:npix
        println("pixel $i")
        function negative_log_likelihood(coeffscatter::Vector{Float64}) #up to constant
            scatter = coeffscatter[end]
            coeffs = coeffscatter[1:end-1]
            invσ = (ivar[:, i].^(-1) .+ scatter^2).^(-1)
            A2 = (labels*coeffs - flux[:, i]).^2
            (0.5*sum(A2.*invσ) - 0.5*sum(log.(invσ)) +  
             Λ * sum(abs.(coeffs[6:end])) #regularization term
             )
        end
        fit = optimize(negative_log_likelihood, zeros(nplabels+1), 
                       Optim.Options(g_tol=1e-2, iterations=1000))
        scatters[i] = fit.minimizer[end]
        theta[:, i] = fit.minimizer[1:end-1]
    end
    theta, abs.(scatters)
end

function logπ(label::Float64, p::Union{Tuple{Float64, Float64, Float64}, Missing})
    if ismissing(p)
        return 0
    else
        σ = (p[3] - p[1])/2
        return -0.5*((label-p[2])/σ)^2 - log(σ)
    end
end                                        

#using PyCall
#pyoptimize = pyimport("scipy.optimize")
#curve_fit = pyoptimize[:curve_fit]
"""
    test(flux, ivar, theta, scatters

Run the test step of the cannon.
"""
function infer(flux::Matrix{Float64}, ivar::Matrix{Float64},
              theta::Matrix{Float64}, scatters::Vector{Float64}, 
              prior::Matrix{Union{Tuple{Float64, Float64, Float64}, Missing}};
              quadratic=true)
    nstars = size(flux, 1)
    nplabels = size(theta, 1)
    nlabels = deprojected_size(nplabels; quadratic=quadratic)
    inferred_labels = Matrix{Float64}(undef, nstars, nlabels)
    chi_squared = Vector{Float64}(undef, nstars)
    thetaT = (transpose(theta))
    for i in 1:nstars
        i%100==0 && println("inferring labels for star $i")

        F = flux[i, :]
        invσ = (ivar[i, :].^(-1) .+ scatters.^2).^(-1)
        function negative_log_post(labels::Vector{Float64})
            A2 = (thetaT * project_labels(labels; quadratic=quadratic) .- F).^2
            0.5 * sum(A2 .* invσ) - sum(logπ.(labels, prior[:, i]))
        end
        #fit = pyoptimize[:minimize](negative_log_likelihood, zeros(nlabels))
        fit = optimize(negative_log_post, zeros(nlabels), Optim.Options(g_tol=1e-6))
        
        #σ = ivar[i, :].^(-1) + scatters.^2
        #function model_flux(theta, labels::Vector{Float64})
        #    transpose(theta)*project_labels(labels)
        #end
        #labels, labelcov = curve_fit(model_flux, theta, flux[i, :], sigma=σ, absolute_sigma=true)

        inferred_labels[i, :] = fit.minimizer
        chi_squared[i] = fit.minimum
        #inferred_labels[i, :] = fit["x"]
    end
    inferred_labels, chi_squared
end

end
