# PyPID_mvn
This repository implements the PID (Partial Information Decomposition) for Multivariate Normal (MVN) systems measuring the redundancy using the Iccs measure developed by Ince (2017)  https://doi.org/10.3390/e19070318

Right now this implementation works only for the bivariate case (two sources and one target)

For a broader implementation of the Partial Information Decomposition please see the [partial-info-decomp](https://github.com/robince/partial-info-decomp) repository

How to use it:
```python
    from gcmi import copnorm
    import numpy as np
    
    def partial_information_decomposition(X,Y):
      
      '''
      Compute the partial information decomposition between the two sources in X and each target in Y using the Iccs redundancy measure for continuous gaussian variables.

      Parameters:
      X : np.array
        The matrix containing the two sources. The sources are in rows so the shape is (n_sources, n_samples)
      Y : np.array
        The matrix containing the targets. Each target is treated separately and could be e.g., a different EEG channel. The targets are in rows so the shape is (n_targtes, n_samples)
      '''

      # Check for right dimensionality
      n_sources = X.shape[0]
      if n_sources != 2:
        raise ValueError('This pid version only works in the case of two sources')
      
      # Build the empty 2D lattice
      lat = pid_mvn.Lattice2D()

      # Be sure that Y is a 2D array
      Y = np.atleast_2d(Y)

      # Copula normalization of the variables to make them gaussian
      X_copnorm = gcmi.copnorm(X)
      Y_copnorm = gcmi.copnorm(Y)

      # Prepare the data structure
      n_targets = Y.shape[0]
      PIs = np.zeros((n_targets, 4)) # e.g., (n_channels, 4) (Rdn, Unq1, Unq2, Syn)

      # Compute the sample covariance matrix
      for tar in range(n_targets):
        C = np.cov(np.vstack((X_copnorm,Y_copnorm[tar,:])))

        # Compute the 4 partial information atoms
        PIs[tar,:] = pid_mvn.calc_pi_mvn(lat, Cfull=C, varsizes=[1,1,1], Icap=pid_mvn.Iccs_mvn, forcenn=True).PI

      return PIs
```

This function uses the code in this repository and the copnorm function that can be found [here](https://github.com/robince/gcmi/tree/master/python) to compute the partial information decomposition for every channel of the Y variable.
