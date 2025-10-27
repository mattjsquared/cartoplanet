import numpy as np
import pyshtools as pysh

from ctplanet import pyMoho, pyMohoRho


def MohoRho_twolayer(
    pot: pysh.SHGravCoeffs,
    topo: pysh.SHCoeffs,
    density: pysh.SHCoeffs,
    porosity: float,
    lmax: int,
    rho_m: float,
    thickave: float,
    filter_type: int = 0,
    half: float | None = None,
    nmax: int = 8,
    delta_max: float = 5.,
    lmax_calc: int | None = None,
    correction: pysh.SHGravCoeffs | None = None,
    quiet: bool = False
) -> pysh.SHCoeffs:
  """
  Calculate the relief along the crust-mantle interface assuming a
  constant density mantle and a laterally varying crustal density.

  Returns
  -------
  moho : SHCoeffs class instance
    The radius of the crust-mantle interface.

  Parameters
  ----------
  pot : SHGravCoeffs class instance
    Gravitational potential spherical harmonic coefficients.
  topo : SHCoeffs class instance
    Spherical harmonic coefficients of the surface relief.
  density : SHCoeffs class instance
    Spherical harmonic coefficients of the crustal grain density.
  porosity : float
    Crustal porosity (from 0 to 1).
  lmax : int
    Maximum spherical harmonic degree of the function, which determines the
    sampling interval of the internally computed grids.
  rho_m : float
    Mantle density in kg / m^3.
  thickave : float
    Average thickness of the crust in meters.
  filter_type : int, optional, default = 0
    0 = no filtering, 1 = minimum amplitude filter, 2 = minimum
    curvature filter.
  half : float, optional, default = None
    The spherical harmonic degree where the filter is equal to 0.5. This
    must be set when filter_type is 1 or 2.
  nmax : int, optional, default = 8
    The maximum order used in the Taylor-series expansion when calculating
    the potential coefficients.
  delta_max : float, optional, default = 5.0
    The algorithm will continue to iterate until the maximum difference in
    relief between solutions is less than this value (in meters).
  lmax_calc : int
    Maximum spherical harmonic degree when evalulating the functions.
  correction : SHGravCoeffs class instance, optional, default = None
    If present, these coefficients will be added to the Bouguer correction
    (subtracted from the Bouguer anomaly) before performing the inversion.
    This could be used to account for the pre-computed gravitational
    attraction of the polar caps of Mars, which have a different density
    than the crust.
  quiet : boolean, optional, default = False
    If True, suppress printing output during the iterations.
  """
  # If using a filter, check for `half`
  if (filter_type == 1 or filter_type == 2) and half is None:
    raise ValueError("half must be set when filter_type is either 1 or 2.")
  # Use maximum specified/available spectral resolution
  if lmax_calc is None:
    lmax_calc = lmax
  # Set the Moho reference radius
  d = topo.coeffs[0, 0, 0] - thickave
  # Determine average crustal bulk density
  rho_crust_ave = density.coeffs[0, 0, 0] * (1. - porosity)
  # Fetch the mass of the body
  mass = pot.mass
  ### STEP 0: Expand inputs to regular grids for use in computing Bouguer correction
  # Expand spectral topography and density onto a regular grid (resolution of `lmax`, not `lmax_calc`; do not duplicate edge longitude)
  topo_grid = topo.expand(grid='DH2', lmax=lmax, extend=False)
  density_grid = density.expand(grid='DH2', lmax=lmax, extend=False)
  # Optional diagnostic prints for grid ranges.
  if quiet is False:
    print("Maximum radius (km) = {:f}".format(topo_grid.data.max() / 1.e3))
    print("Minimum radius (km) = {:f}".format(topo_grid.data.min() / 1.e3))
    print("Maximum density (kg/m3) = {:f}".format(density_grid.data.max() / 1.e3))
    print("Minimum density (kg/m3) = {:f}".format(density_grid.data.min() / 1.e3))
  ### STEP 1: Compute Bouguer anomaly corrected for lateral density variations
  # Compute spectral Bouguer correction (also return C00 of re-expanded topography [instead of just using C00 of the original spectral topo, for some reason])
  bc, r0 = pysh.gravmag.CilmPlusRhoHDH(
    topo_grid.data, 
    nmax, 
    mass, 
    density_grid.data * (1. - porosity),
    lmax=lmax_calc
  )
  # Account for additional pre-computed gravitational sources other than Moho relief [e.g., polar caps]
  if correction is not None:
    bc += correction.change_ref(r0=r0).to_array(lmax=lmax_calc)
  # Shift the input potential to the new reference radius r0 and compute spectral Bouguer anomaly
  pot2 = pot.change_ref(r0=r0)
  ba = pot2.to_array(lmax=lmax_calc, errors=False) - bc
  # Remove lateral variations in crustal density from the Bouguer anomaly [note: the algorithm here removes the effect of topography SEPARATELY AND BEFORE accounting for lateral density variation]
  for l in range(1, lmax_calc + 1):
    ba[:, l, :l + 1] = ba[:, l, :l + 1] \
                        - 4. * np.pi * density.coeffs[:, l, :l + 1] \
                        * (1. - porosity) \
                        * (r0**3 - (d**3)*(d/r0)**l) \
                        / (2 * l + 1) / (l + 3) / mass
  ### STEP 2: Compute first crustal thickness guess [TODO: this is the n = 1–only term?]
  # Initialize Moho coefficients and set the mean value to 'd'.
  moho = pysh.SHCoeffs.from_zeros(lmax=lmax_calc)
  moho.coeffs[0, 0, 0] = d
  # Compute an initial estimate for spectral Moho relief (with optional downward-continuation filtering)
  for l in range(1, lmax_calc + 1):
    if filter_type == 0:
      moho.coeffs[:, l, :l + 1] = ba[:, l, :l + 1] * mass * \
          (2 * l + 1) * ((r0 / d)**l) / \
          (4. * np.pi * (rho_m - rho_crust_ave) * d**2)
    elif filter_type == 1:
      moho.coeffs[:, l, :l + 1] = pysh.gravmag.DownContFilterMA(
          l, half, r0, d) * ba[:, l, :l + 1] * mass * \
          (2 * l + 1) * ((r0 / d)**l) / \
          (4. * np.pi * (rho_m - rho_crust_ave) * d**2)
    else:
      moho.coeffs[:, l, :l + 1] = pysh.gravmag.DownContFilterMC(
          l, half, r0, d) * ba[:, l, :l + 1] * mass * \
          (2 * l + 1) * ((r0 / d)**l) / \
          (4.0 * np.pi * (rho_m - rho_crust_ave) * d**2)
  # Expand the initial Moho estimate to a regular grid
  moho_grid3 = moho.expand(
    grid='DH2', 
    lmax=lmax, 
    lmax_calc=lmax_calc,
    extend=False
  )
  # Compute the crustal density contrast grid
  drho_grid = rho_m - density_grid * (1. - porosity)
  # Optional diagnostic prints for initial crustal thickness
  temp_grid = topo_grid - moho_grid3
  if quiet is False:
    print('Maximum Crustal thickness (km) = {:f}'.format(temp_grid.data.max() / 1.e3))
    print('Minimum Crustal thickness (km) = {:f}'.format(temp_grid.data.min() / 1.e3))
  ### STEP 3: Re-estimate the Moho using the full nmax solution
  # Compute the next spectral Moho estimate, accounting for laterally varying density [TODO: isn't this double-dipping on the density variations, since the density-corrected `ba` is passed as an argument??]
  moho.coeffs = pysh.gravmag.BAtoHilmRhoHDH(
    ba, 
    moho_grid3.data, 
    drho_grid.data, 
    nmax, 
    mass, 
    r0,
    lmax=lmax, 
    filter_type=filter_type, 
    filter_deg=half,
    lmax_calc=lmax_calc
  )
  # Expand the updated coefficients and compute another grid-based crustal thickness estimate to assess convergence.
  moho_grid2 = moho.expand(
    grid='DH2', 
    lmax=lmax, 
    lmax_calc=lmax_calc,
    extend=False
  )
  temp_grid = topo_grid - moho_grid2
  if quiet is False:
    print('Delta (km) = {:e}'.format(abs(moho_grid3.data - moho_grid2.data).max() / 1.e3))
    print('Maximum Crustal thickness (km) = {:f}'.format(temp_grid.data.max() / 1.e3))
    print('Minimum Crustal thickness (km) = {:f}'.format(temp_grid.data.min() / 1.e3))
  ### STEP 4: Iterate the Moho solution until it converges
  # Initialize the iterator and residual
  iter = 0
  delta = 1.0e9
  # Iterate
  while delta > delta_max:
    iter += 1
    if quiet is False:
      print('Iteration {:d}'.format(iter))
    # Prepare the next guess (average of the previous two)
    moho_grid = (moho_grid2 + moho_grid3) / 2.
    temp_grid = topo_grid - moho_grid
    if quiet is False:
      print("Delta (km) = {:e}".format(abs(moho_grid.data - moho_grid2.data).max() / 1.e3))
      print('Maximum Crustal thickness (km) = {:e}'.format(temp_grid.data.max() / 1.e3))
      print('Minimum Crustal thickness (km) = {:e}'.format(temp_grid.data.min() / 1.e3))
    # Shift history and perform another BA->Hilm inversion on the current gridded guess.
    moho_grid3 = moho_grid2
    moho_grid2 = moho_grid
    iter += 1 # TODO: why is the iterator incremented twice per loop?
    if quiet is False:
        print('Iteration {:d}'.format(iter))
    moho.coeffs = pysh.gravmag.BAtoHilmRhoHDH(
      ba, 
      moho_grid2.data, 
      drho_grid.data, 
      nmax, 
      mass, 
      r0,
      lmax=lmax, 
      filter_type=filter_type, 
      filter_deg=half,
      lmax_calc=lmax_calc
    )
    moho_grid = moho.expand(
      grid='DH2', 
      lmax=lmax, 
      lmax_calc=lmax_calc, 
      extend=False
    )
    # Compute convergence metric and diagnostics
    delta = abs(moho_grid.data - moho_grid2.data).max()
    temp_grid = topo_grid - moho_grid
    if quiet is False:
      print('Delta (km) = {:e}'.format(delta / 1.e3))
      print('Maximum Crustal thickness (km) = {:f}'.format(temp_grid.data.max() / 1.e3))
      print('Minimum Crustal thickness (km) = {:f}'.format(temp_grid.data.min() / 1.e3))
    # Reset the history for the next iteration
    moho_grid3 = moho_grid2
    moho_grid2 = moho_grid
    # Abort if solution diverges significantly
    if abs(temp_grid.data).max() > 500.e3:
      print('Not converging')
      exit(1)
  # Return the converged Moho coefficients.
  return moho
