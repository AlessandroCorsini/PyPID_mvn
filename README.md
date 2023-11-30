# PyPID_mvn
This repository implements the PID (Partial Information Decomposition) for MultiVariate Normal (MVN) systems measuring the redundancy using the Iccs measure developed by Ince (2017)  https://doi.org/10.3390/e19070318

Right now this implementation works only for the bivariate case (two sources and one target)

For a broader implementation of the Partial Information Decomposition please see the [partial-info-decomp](https://github.com/robince/partial-info-decomp) repository

How to use it:

    from gcmi import copnorm
    import numpy as np
    
    def partial_information_decomposition(X, Y, Z):
    
      '''
      Compute the partial information decomposition between the sources X,Y and possibly multivariate Z using the Iccs redundancy measure for continuous gaussian variables.
      '''

      # Build the empty 2D lattice
      lat = pid_mvn.Lattice2D()

      # Be sure that Z is a 2D array
      Z = np.atleast_2d(Z)

      # Copula normalization of the variables to make them gaussian
      X_copnorm = gcmi.copnorm(X).squeeze()
      Y_copnorm = gcmi.copnorm(Y).squeeze()
      Z_copnorm = gcmi.copnorm(Z)

      # Prepare the data structure
      PIs = np.zeros((Z.shape[0], 4)) # (n_channels, 4) (Rdn, Unq1, Unq2, Syn)

      # Compute the empirical covariance matrix
      for ch in range(Z.shape[0]):
        C = np.cov(np.array([X_copnorm, Y_copnorm, Z_copnorm[ch,:]]))

        # Compute the 4 partial information atoms
        PIs[ch,:] = pid_mvn.calc_pi_mvn(lat, Cfull=C, varsizes=[1,1,1], Icap=pid_mvn.Iccs_mvn, forcenn=True).PI

      return PIs


This function uses the code in this repository and the copnorm function that can be found [here](https://github.com/robince/gcmi/tree/master/python) to compute the partial information decomposition for every channel of the Z variable. Z could be the EEG time-series of all the channels.
