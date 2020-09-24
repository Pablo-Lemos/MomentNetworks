# MomentNetworks
Demonstration of MomentNetworks for high-dimensional probability density estimation (LFI)

## Test A
### 20 parameters, 20 data elements, and 1000000 training data realisations

#### (I)
Estimate the full (20,20) dimensional distribution $P(d | \theta ))$ with DELFI and compare with MCMC from the known likelihood. DELFI fails completely in this set-up. If the number parameters is reduced to 12, DELFI does not fail:

[/notebooks/delfi_failure_example.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/notebooks/delfi_failure_example.ipynb)

#### (II)
Estimate the variance of the marginal dimensional distributions $P(d|\theta_i))$ and compare with MCMC from the known likelihood. The Moment Network perform well --  approx. 4 per cent accuracy for the estimated variance: 

[/notebooks/delfi_moment_network.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/notebooks/delfi_moment_network.ipynb)

