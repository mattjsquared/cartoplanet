import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import copy, warnings, functools

from scipy.integrate import quad, cumulative_simpson
from scipy.optimize import root_scalar
from tqdm import tqdm
from cartoplanet import config

body = config['BODY']['body']

DEBUG = True




def great_circle_distance(lat1, lon1, lat2, lon2, R=config.getfloat(body, "R0")/1e3, in_degrees=True):
  """
  Compute great-circle distance on a sphere.

  Parameters
  ----------
  lat1, lon1, lat2, lon2 : array-like
    Coordinates of the two points (defaults in degrees if in_degrees=True).
  R : float
    Sphere radius in desired linear units (default = Rmoon/1e3 for km).
  in_degrees : bool
    If True, inputs are converted from degrees to radians.

  Returns
  -------
  d : array-like
    Great-circle distance in same units as R.
  """
  
  if in_degrees:
      lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))

  dlat = lat2 - lat1
  dlon = (lon2 - lon1 + np.pi) % (2*np.pi) - np.pi

  a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
  
  return 2 * R * np.arcsin(np.sqrt(a))


def compute_ejecta_thickness_and_mixing_xarray(
    Rat: float,
    r_soi_center: float,
    elevation: np.ndarray,
    abundances: np.ndarray = None,
    T_Rat: float = None,
    bt: float = 3,
    theta_degree: float = 45,
    Y: float = 10e6,
    K1: float = 1.03,
    mu: float = 0.41,
    nu: float = 0.4,
    R0: float = 1737.4,
    g: float = 1.622,
    Dsc: float = 19,
    mixing: bool = True,
    return_secondary: bool = False,
):
  """
  The code of the ballistic sedimentation model of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Originally created with MATLAB R2016b: 
  Xie, M. Liu, T. and Xu, A. (2020), Ballistic sedimentaiton model [Code], Zenodo. https://doi.org/10.5281/zenodo.3692887.

  Adapted to Python by Matt Jones 2026.

  Parameters
  ----------
  Rat : float
    The apparent radius of a transient crater at preimpact surface in kilometers.
  r_soi_center : float
    The great-circle distance from SOI center to the parent crater center in kilometers.
  elevation : ndarray or list of float
    Depth from the surface of pre-existing local material. Note that local material often consists of earlier basin ejecta and pre-basin materials
  abundances : ndarray or list of float, optional
    The abundance of previously emplaced crater ejecta in ejecta deposits versus depth from the surface of the ejecta deposits. The ejecta deposits are the pre-existing local material for the emplacement of the ejecta of the current crater of interest for the earliest crater, abundances should be an empty array.
  T_Rat : float, optional
    The thickness of ejecta at Rat with default value of T_Rat=0.068Rat.
  bt : float, optional
    The power-law index of ejecta thickness distribution with default value of bt=3.
  theta_degree : float, optional
    The launch/impact angle of ejecta in degrees with default value of theta_degree=45.
  Y : float, optional
    The strength of target material.
  K1, mu, nu : float, optional
    Constants in crater scaling laws (Equation 12 see also Table 2).
  R0 : float, optional
    The radius of the target body in kilometers (default is the mean radius of the Moon).
  g : float, optional
    The gravitational acceleration of the target body in m/s^2 (default is the Moon's gravity).
  Dsc : float, optional
    The simple-to-complex crater transition diameter in km (default is 19 km for the Moon).
  mixing : bool, optional
    Whether to compute the mixing of ejecta with local material. If False, the abundance of ejecta in ejecta deposits is not updated by mixing and the abundance of ejecta in ejecta deposits is the same as the abundance of ejecta in excavated ejecta deposits. Default is True.
  return_secondary : bool, optional
    Whether to return the information of secondary craters. Default is False.
  
  Returns
  -------
  Tlocal_med : 
    The median thickness of excavated local material.
  Tprimary : 
    The thickness of primary ejecta.
  new_abundances : 
    The abundance of ejecta in ejecta deposits as a function of depth (i.e., `elevation`).
  secondary_craters : 
    A struct recording the information of secondary craters.
  fraction_excavated : 
    Fraction of excavated local material as a function of depth (i.e., `elevation`).
  """
  ### //Standardize inputs//
  secondary_craters = {}                  #
  r_soi_center = np.asarray(r_soi_center) #m
  elevation    = np.asarray(elevation)    #m
  abundances   = np.asarray(abundances)   #fraction
  Dsc = Dsc * 1000                        #m -- simple to complex transition diameter -- Xie et al. (2020) (from Pike (1980) LPSC, p27)
  ### //Define helper function for converting apparent diameter to final diameter//
  def Dat2D(Dat_m=None):
    if Dat_m is None:
      Dat_m = np.logspace(3, 5, 1000)                     #m -- apparent transient diameter
    D_m   = Dat_m * 1.43                                  #m -- final diameter for simple craters
    index = D_m > Dsc                                     #indices of complex craters
    D_m[index] = 1.52 * Dsc**(-0.18) * Dat_m[index]**1.18 #m -- final crater diameter
    return D_m
  ### //Define interpolation helper function//
  def safe_interp(x, xp, fp):
    result = np.interp(x, xp, fp)
    if any(x > 1 for x in result) or any(x < 0 for x in result):
      raise ValueError("Interpolation error: Unexpected extrapolation, or data values are not between 0 and 1.")
    return result
  ### //Set parameters//
  if T_Rat is None:
    T_Rat         = 0.068 * Rat                                                                     #m
  Cex             = 3.5                                                                             #
  rhot            = 3e6                                                                             #g/m^3 -- target density
  rhoe            = rhot                                                                            #g/m^3 -- ejecta density
  L_soi           = 2 * r_soi_center**0.5                                                           #km
  launch_position = Rat                                                                             #km
  S_soi           = L_soi**2                                                                        #km^2
  g_km            = g / 1000                                                                        #km/s^2
  theta           = np.radians(theta_degree)                                                        #radians
  Cmh             = 0.013                                                                           #
  b_MT            = 0.91                                                                            #
  b               = 0.98                                                                            #
  # Validate SOI geometry
  if np.any(r_soi_center <= (Rat + L_soi/2)):
    raise ValueError("SOI must be outside crater rim.\n\n")
  # Coefficients in Equation (10) derived from the work of Allen (1979).
  a_rLSC          = 7.209219501948585                                                               #
  b_rLSC          = 0.935213298326370                                                               #
  # SOI characteristics
  N_soi           = len(r_soi_center)                                                               #number of SOI
  r_soi_outer     = r_soi_center + L_soi/2                                                          #km, the outer boundary of the SOI
  r_soi_inner     = r_soi_center - L_soi/2                                                          #km, the inner boundary of the SOI
  # Surface area of the SOI ring accounting for spherical target (Equation (4))
  S_ring          = np.abs( 2*np.pi * (R0**2) * (np.cos(r_soi_inner/R0) - np.cos(r_soi_outer/R0)) ) #km^2 -- if r_soi_outer/R0 > np.pi, S_ring < 0 [**this is a notable simplification!]
  ### //Initialize abundances//
  if not abundances.size:
    m_depth        = len(elevation)                                      #
    n_component    = 1                                                   #the total number of ejecta sources.
    abundances     = np.zeros((m_depth, n_component, N_soi))             #[depth, component, SOI]
    new_abundances = np.zeros(abundances.shape)                          #
  else:
    abundances     = np.atleast_3d(abundances)                           #[depth, component, SOI]
    m_depth, n_component, k = abundances.shape                           #the total number of differently sourced ejecta in local material.
    n_component   += 1                                                   #we add one more component for the ejecta of the current crater of interest
    new_component  = np.zeros((m_depth, 1, k))                           #initialize new abundances to zero
    abundances     = np.concatenate((abundances, new_component), axis=1) #[depth, component, SOI]
    new_abundances = np.zeros(abundances.shape)                          #
  idx_newcomponent = n_component - 1                                     #the index of the component for the ejecta of the current crater of interest in the `abundances` array
  ### //Initialize outputs//
  Tlocal_med         = np.zeros(r_soi_center.shape)          #m -- median local material thickness
  Tprimary           = np.zeros(r_soi_center.shape)          #m -- primary ejecta thickness
  fraction_excavated = np.zeros((len(elevation), N_soi))     #
  if return_secondary:
    if not mixing:
      raise ValueError("`mixing` must be True to return secondary crater information..")
    for key in ['D', 'largest', 'S_SOI', 'S_ring', 'density', 'density_cumulative', 'Num_SOI', 'Num_SOI_cumulative']:
      secondary_craters[key] = np.empty(N_soi, dtype=object) #
  ### //Define velocity-distance helper functions and values//
  # Spherical target
  f_Rs              = lambda r_gc: r_gc - launch_position                                                                                      #
  f_X               = lambda r_gc: f_Rs(r_gc) / (2 * R0)                                                                                       #
  rgc2velocity      = lambda r_gc: np.sqrt(  R0*g_km*np.tan(f_X(r_gc)) / (np.tan(f_X(r_gc))*np.cos(theta)**2 + np.sin(theta)*np.cos(theta))  ) #
  v_soi_inner       = rgc2velocity(r_soi_inner)                                                                                                #km/s
  v_soi_outer       = rgc2velocity(r_soi_outer)                                                                                                #km/s
  v_soi_center      = rgc2velocity(r_soi_center)                                                                                               #km/s
  # Flat-surface approximation
  v2r_flat          = lambda v: launch_position + v**2 * np.sin(2*theta) / (g_km)                                                              #
  r_soi_inner_flat  = v2r_flat(v_soi_inner)                                                                                                    #km
  r_soi_outer_flat  = v2r_flat(v_soi_outer)                                                                                                    #km
  r_soi_center_flat = v2r_flat(v_soi_center)                                                                                                   #km
  S_ring_flat       = np.pi * (r_soi_outer_flat**2 - r_soi_inner_flat**2)                                                                      #km^2
  ### //Calculate primary ejecta thickness (Equation 4)//
  Thickness_flat                 = T_Rat * (r_soi_center_flat / Rat)**(-bt)    #km -- the thickness of ejecta on flat surface at distance r_soi_center_flat
  Tprimary[:]                    = Thickness_flat * S_ring_flat / S_ring * 1e3 #m
  Tprimary[v_soi_center >= 2.38] = 0                                           #m -- [??? from Xie et al. (2020) code -- not sure of the motivation here]
  # Mass of primary ejecta (Equation 5)
  M_soi                          = rhoe * Tprimary * S_soi*1e6                 #g
  ### //Calculate position and velocity of largest secondary crater (Equation 8)//
  r_LSC = a_rLSC * Rat**b_rLSC #
  v_LSC = rgc2velocity(r_LSC)  #km/s
  ### //Scaling for primary ejecta fragment mass bounds (Equation 7)//
  M_tot                      = 0.09 * np.pi * rhot * (Rat*1000)**3                    #g
  mh                         = np.ones_like(v_soi_center) * Cmh * M_tot**b_MT         #g -- upper limit of ejecta mass
  mh[v_soi_center >= v_LSC] *= (v_soi_center[v_soi_center >= v_LSC]/v_LSC)**(-5.7)    #g -- upper limit of ejecta mass
  ml                         = mh * 1e-23                                             #g -- lower limit of ejecta mass
  C_soi                      = M_soi * (1-b)/b / (mh**(1-b) - ml**(1-b))              #
  N_massbins                 = np.round(np.log(mh/ml) / np.log(1.05)).astype(int) - 1 #mh=ml*q**(n+1) => n=log(mh/ml)/log(q)-1, q=1.05
  N_massbins                 = N_massbins[0]                                          #NOTE: this is because ml is a constant fraction of mh, thus the N_massbins formula gives the same value for all SOI
  if any(C_soi < 0):
    raise ValueError("C_soi < 0\n\n")
  ### //Compute discrete mass bins//
  mL       = np.logspace(np.log10(ml), np.log10(mh), N_massbins+1, axis=1) #
  mR       = mL[:, 1:]                                                     #right boundary of mass bin.
  mL       = mL[:, :-1]                                                    #left boundary of mass bin.
  mass     = np.sqrt(mL * mR)                                              #g
  N_ejecta = C_soi[:, None] * (mL**(-b) - mR**(-b))                        #number of ejecta framents in each mass bin
  ### //Scaling for characteristic secondary crater sizes//
  Vp      = v_soi_center * 1000 * np.sin(theta)                                                                                                      #m/s -- vertical component
  Vp2     = Vp**2                                                                                                                                    #m^2/s^2
  ai      = ((3 * mass) / (4 * np.pi * rhoe))**(1/3)                                                                                                 #m -- radius of projectile
  Rat_sec = K1 * ai * (  (g*ai/Vp2[:, None])*(rhot/rhoe)**(2*nu/mu) + (Y/(rhot*Vp2[:, None]))**(1+mu/2)*(rhot/rhoe)**(nu*(2+mu)/mu)  )**(-mu/(2+mu)) #m
  d_ex    = 0.0134 * Rat_sec * (v_soi_center[:, None] * 1e3)**0.38                                                                                   #Equation 16
  ### //Initialize primary ejecta deposit layers//
  Tprimary_perlayer      = np.maximum(Cex*d_ex[:, -1]/5000, Tprimary/100)                                          #m
  N_layers               = np.ceil(Tprimary / Tprimary_perlayer).astype(int)                                       #
  Tprimary_perlayer      = np.divide(Tprimary, N_layers, out=np.zeros_like(Tprimary), where=(N_layers > 0))        #m
  N_layers_max           = N_layers.max()                                                                          #
  k                      = np.arange(N_layers_max)[None, :]                                                        #(1, N_layers_max)
  idx_mask               = k < N_layers[:, None]                                                                   #(N_soi, N_layers_max)
  elevation_layerbottoms = k * Tprimary_perlayer[:, None]                                                          #(N_soi, N_layers_max) -- m, the elevation of the lower boundary of a layer with respect to the surface of local material
  elevation_layerbottoms = np.where(idx_mask, elevation_layerbottoms, np.nan)                                      #
  ### //Compute thickness of local material excavated by secondary impacts (Equation 19)//
  Tlocal_per_massbin                = np.zeros((N_soi, N_massbins))                #
  coverage_exponent                 = np.zeros((N_soi, N_massbins, N_layers_max))  #the exponent in Equation 19
  # Calculate density of secondary craters per layer of primary ejecta
  S_secondarycraters                = np.pi * Rat_sec**2                           #m^2, the area of secondary craters at surface
  density_secondarycraters          = N_ejecta / S_soi[:, None] * 1e-6             #m^-2, density of secondary craters
  d_eff                             = Cex * d_ex                                   #m, effective excavation depth
  density_secondarycraters_onelayer = density_secondarycraters / N_layers[:, None] #m^-2,the density of secondary craters formed by a single layer of ejecta
  # Compute local material thickness added to deposit by each primary ejecta layer and mass bin
  Tlocal_per_massbin[:, :]          = d_eff[:, :]                                  #
  ### //Vectorized calculation of Equation (19) terms over [SOI, massbin_i, massbin_j, layer]//
  # -----------------------------------------------------------------------------
  # Shape guide:
  #   s = SOI index
  #   i = "query" mass-bin index (the Tlocal threshold being evaluated)
  #   j = contributing mass-bin index (secondary craters summed into i)
  #   k = primary-ejecta layer index
  #
  # This block is equivalent to the old nested loops:
  #   for s in SOI:
  #     for k in layers:
  #       for i in massbins:
  #         sum over valid j>=i of:
  #           S_secondary(j)
  #           * (1 - Tlocal(i)/d_eff(j))
  #           * (1 - z_layer(k)/d_eff(j))
  #           * density_onelayer(j)
  # where validity requires Tlocal(i) + z_layer(k) < d_eff(j).
  # -----------------------------------------------------------------------------
  ### //Define mass indices, masks, and views//
  # Enforce j >= i (upper-triangular in [i, j]) to match original index = np.arange(i, N_massbins)
  range_massbin = np.arange(N_massbins)                            #
  mask_uppertri = range_massbin[None, :] >= range_massbin[:, None] #(N_massbins, N_massbins)
  # Broadcast d_eff as both an i-indexed view and a j-indexed view
  d_eff_i       = d_eff[:, :,    None, None]                       #(N_soi, N_massbins, 1,          1)
  d_eff_j       = d_eff[:, None, :,    None]                       #(N_soi, 1,          N_massbins, 1)
  ### //Define depth mask//
  # Broadcast layer-bottom elevations along [i, j]
  elev_layers     = elevation_layerbottoms[:, None, None, :]                       #(N_soi, 1,          1,          N_layers_max)
  # Valid depth mask (ignores nan's from N_layer padding)
  idx_validlayers = ((d_eff_i + elev_layers) < d_eff_j) & np.isfinite(elev_layers) #(N_soi, N_massbins, N_massbins, N_layers_max)
  # Final mask: valid depth condition AND valid layer AND j>=i ordering
  mask_total      = idx_validlayers & mask_uppertri[None, :, :, None]              #
  ### //Compute coverage exponent (Eq. 19)//
  # Geometry/correction term in Eq. 19 before multiplying by crater density
  area_term                = (S_secondarycraters[:, None, :, None] * (1 - d_eff_i / d_eff_j) * (1 - elev_layers / d_eff_j)) #(N_soi, N_massbins, N_massbins, N_layers_max)
  # Per-layer crater density contribution for each contributing mass bin j
  density_term             = density_secondarycraters_onelayer[:, None, :, None]                                            #(N_soi, 1,          N_massbins, 1)
  # Sum over contributing j-axis to recover E[s, i, k] (the Eq. 19 exponent term)
  coverage_exponent        = np.sum(np.where(mask_total, area_term * density_term, 0.0), axis=2)                            #
  # Layer-wise accumulation (negative cumulative exponent from original algorithm)
  coverage_exponent_cumsum = -np.cumsum(coverage_exponent, axis=2)                                                          #
  ### //Compute coverage fraction//
  # Coverage fraction after k layers at each (s, i): W = 1 - exp(E)
  W_PIS_nLayers   = 1 - np.exp(coverage_exponent_cumsum)                                                                                        #(N_soi, N_massbins, N_layers_max)
  if np.any((W_PIS_nLayers < 0) | (W_PIS_nLayers > 1)):
    raise ValueError("Coverage fraction (W) should be between 0 and 1.")
  # Select each SOI's final valid layer (k = N_layers[s]-1) from the padded layer axis
  idx_lastlayer   = np.maximum(N_layers - 1, 0)                                                                                                 #
  W_PIS_allLayers = np.take_along_axis(W_PIS_nLayers, np.broadcast_to(idx_lastlayer[:, None, None], (N_soi, N_massbins, 1)), axis=2).squeeze(2) #
  # Determine median (W = 0.5) local material excavation depth: excavation depth of first mass bin where final-layer W drops below 0.5
  idx_med_candidates = W_PIS_allLayers < 0.5
  if np.any((Tprimary > 0) & (~np.any(idx_med_candidates, axis=1))):
    raise ValueError("Could not determine median local excavation thickness for one or more SOIs.")
  idx_med = np.argmax(idx_med_candidates, axis=1)
  Tlocal_med[:] = Tlocal_per_massbin[np.arange(N_soi), idx_med]
  Tlocal_med[Tprimary == 0] = 0
  ### //----------MIXING----------//
  if not mixing:
    return Tlocal_med, Tprimary, None, None, None
  ### //Begin SOI loop//
  for i_soi in range(N_soi):
    ### //Fetch pre-computed values//
    Tprimary_soi = Tprimary[i_soi]                            #m
    if Tprimary_soi == 0:
      continue
    N_ejecta = N_ejecta[i_soi, :]                             #number of ejecta fragments in each mass bin
    Tprimary_perlayer = Tprimary_perlayer[i_soi]              #m
    N_layers = N_layers[i_soi]                                #
    elevation_layerbottoms = elevation_layerbottoms[i_soi, :] #m
    d_eff = d_eff[i_soi, :]                                   #m
    W_PIS_nLayers = W_PIS_nLayers[i_soi, :, :N_layers]        #[all mass bins, all valid layers]
    ### //Interpolate final W-with-depth onto `elevation` grid//
    fraction_excavated[:, i_soi] = np.interp(elevation, np.flip(-Tlocal_per_massbin[i_soi, :]), np.flip(W_PIS_allLayers[i_soi, :]))
    ### //Mixing depth information -- agnostic of surface elevation//
    zmax_mixingzone      = max(d_eff)                                                            #maximum depth of mixing zone from the surface
    elevation_mixingzone = np.arange(-Tprimary_perlayer/2, -zmax_mixingzone, -Tprimary_perlayer) #depth grid for mixing zone
    dz_mixingzone        = elevation_mixingzone[0] - elevation_mixingzone[1]                     #
    ### //Construct W-with-depth kernel//
    # Interpolate W-with-depth for one layer onto mixing zone grid
    W_onelayer          = W_PIS_nLayers[:, 0]                                                     #[all mass bins, first layer]
    Wmz_onelayer        = safe_interp(elevation_mixingzone, np.flip(-d_eff), np.flip(W_onelayer)) #
    Texcavated_onelayer = np.sum(dz_mixingzone * Wmz_onelayer)                                    #effective total thickness excavated by one layer of ejecta
    ### //Actual depth information -- completed deposition//
    elevation_toplayer    = Tprimary_soi - (dz_mixingzone/2)                                              #elevation of the center of the top layer discretized by `dz_mixingzone`
    n_depth_fill          = int(abs( elevation_toplayer / dz_mixingzone ))                                #
    elevation_deposit     = np.flip(  np.linspace(dz_mixingzone/2, elevation_toplayer, n_depth_fill+1)  ) #depth grid for the ejecta deposit only
    elevation_mixinggrid  = np.concatenate((elevation_deposit, elevation_mixingzone), axis=0)             #elevations from top of ejecta deposit to depth of `zmax_mixingzone` with uniform grid spacing
    abundances_mixinggrid = np.zeros((len(elevation_mixinggrid), n_component))                            #
    ### //Mixing computation -- note that excavated materials could be ejecta deposits formed by earlier ejecta and/or local material//
    for i in range(N_layers):
      surface_elevation = elevation_layerbottoms[i]                                                                                                                                            #
      # Set up reference indices for the layer of ejecta being deposited and compute first mixing step
      if i == 0:
        idx_mixingelevations  = np.where( (elevation_mixinggrid>(surface_elevation-zmax_mixingzone)) & (elevation_mixinggrid<surface_elevation) )[0]                                           #select elevations for deposit i's mixing zone
        ## TODO: idx_mixingdepths should be invariant, no? like, shouldn't this always just be equivalent to `range(len(Wmz_onelayer))`?
        idx_mixingdepths      = np.floor((surface_elevation - elevation_mixinggrid[idx_mixingelevations]) / dz_mixingzone).astype(int)                                                         #map elevations in the mixing zone to layer indices for Wmz_onelayer
        # Interpolate preexisting components' abundances onto the mixing grid
        for j in range(0, n_component-1):
          abundances_mixinggrid[idx_mixingelevations, j] = safe_interp(elevation_mixinggrid[idx_mixingelevations], np.flip(elevation), np.flip(abundances[:, j, i_soi]))                       #
      else:
        idx_mixingelevations -= 1                                                                                                                                                              #
      # Calculate the fraction of primary ejecta in the layer being deposited (including materials excavated from earlier deposits)
      Tprimary_excavated                                                 = sum(dz_mixingzone * abundances_mixinggrid[idx_mixingelevations, idx_newcomponent] * Wmz_onelayer[idx_mixingdepths]) #ejecta in excavated ejecta deposits
      abundance_primary_thislayer                                        = (Tprimary_perlayer + Tprimary_excavated) / (Tprimary_perlayer + Texcavated_onelayer)                                #
      # Each layer's new primary ejecta abundance is now the weighted average of (1) non-excavated material from that layer and (2) the homogenized current deposit (`abundance_primary_thislayer`) that fills in holes
      abundances_mixinggrid[idx_mixingelevations,      idx_newcomponent] = abundances_mixinggrid[idx_mixingelevations, idx_newcomponent]*(1-Wmz_onelayer[idx_mixingdepths]) + abundance_primary_thislayer*Wmz_onelayer[idx_mixingdepths]
      abundances_mixinggrid[idx_mixingelevations[0]-1, idx_newcomponent] = abundance_primary_thislayer                                                                                         #deposit the current layer
      for k in range(n_component-1): #the jth component reworks preexisting materials, reducing the abundance of preexisting materials in ejecta deposits
        # Calculate the fraction of component k in the layer being deposited (excavated from earlier deposits and/or beneath pre-impact surface)
        Tk_excavated                                                     = sum(dz_mixingzone * abundances_mixinggrid[idx_mixingelevations, k] * Wmz_onelayer[idx_mixingdepths])                #ejecta in excavated ejecta deposits
        abundance_k_thislayer                                            = Tk_excavated / (Texcavated_onelayer + Tprimary_perlayer)                                                            #
        # Each layer's new component k abundance is now the weighted average of (1) non-excavated material from that layer and (2) the homogenized current deposit (`abundance_k_thislayer`) that fills in holes
        abundances_mixinggrid[idx_mixingelevations,      k]              = abundances_mixinggrid[idx_mixingelevations,      k]*(1-Wmz_onelayer[idx_mixingdepths]) + abundance_k_thislayer*Wmz_onelayer[idx_mixingdepths] #
        abundances_mixinggrid[idx_mixingelevations[0]-1, k]              = abundances_mixinggrid[idx_mixingelevations[0]-1, k] + abundance_k_thislayer                                         #
    ### //Interpolate new abundances onto new elevation grid that includes the deposit//
    ## TODO: Hmm... this should probably include extrapolation?
    elevation_afterdeposit      = np.concatenate((np.flip(elevation_layerbottoms)+Tprimary_perlayer/2, elevation), axis=0) #new elevation grid that includes the deposit
    new_abundances_afterdeposit = np.zeros((len(elevation_afterdeposit), n_component))                                     #
    for j in range(n_component):
      new_abundances_afterdeposit[:, j] = safe_interp(elevation_afterdeposit, np.flip(elevation_mixinggrid), np.flip(abundances_mixinggrid[:, j])) #
    ### //Interpolate new abundances onto original elevation grid for output//
    for k in range(n_component):
      new_abundances[:, k, i_soi]    = np.interp(elevation, np.flip(elevation_afterdeposit-Tprimary_soi), np.flip(new_abundances_afterdeposit[:, k])) #
    ### //Record secondary crater information//
    if return_secondary:
      Dat         = Rat_sec * 2 #m
      D_secondary = Dat2D(Dat)  #m
      secondary_craters['D'][i_soi]                  = D_secondary                                                      #m
      secondary_craters['largest'][i_soi]            = D_secondary[-1]                                                  #m
      secondary_craters['S_SOI'][i_soi]              = S_soi[i_soi]                                                     #km^2
      secondary_craters['S_ring'][i_soi]             = S_ring[i_soi]                                                    #km^2
      secondary_craters['density'][i_soi]            = N_ejecta / S_soi[i_soi]                                          #km^-2
      secondary_craters['density_cumulative'][i_soi] = np.flip(np.cumsum(np.flip(secondary_craters['density'][i_soi]))) #
      secondary_craters['Num_SOI'][i_soi]            = N_ejecta                                                         #
      secondary_craters['Num_SOI_cumulative'][i_soi] = np.flip(np.cumsum(np.flip(secondary_craters['Num_SOI'][i_soi]))) #
  ### //End SOI loop//
  return Tlocal_med, Tprimary, new_abundances, secondary_craters, fraction_excavated


def compute_ejecta_thickness_and_mixing(
    Rat,
    r_SOI_center,
    Elevation_from_Local_Surface,
    abundance_local_PE = None,
    T_Rat = None,
    bt = 3,
    theta_degree = 45,
    Y = 10e6,
    K1 = 1.03,
    mu = 0.41,
    nu = 0.4,
    mixing = True
):
  """
  The code of the ballistic sedimentation model of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Originally created with MATLAB R2016b: 
  Xie, M. Liu, T. and Xu, A. (2020), Ballistic sedimentaiton model [Code], Zenodo. https://doi.org/10.5281/zenodo.3692887.

  Adapted to Python by Matt Jones 2026.

  Parameters
  ----------
  Rat : float
    The apparent radius of a transient crater at preimpact surface in kilometers.
  r_SOI_center : float
    The great-circle distance from SOI center to the parent crater center in kilometers.
  Elevation_from_Local_Surface : ndarray or list of float
    Depth from the surface of pre-existing local material. Note that local material often consists of earlier basin ejecta and pre-basin materials
  abundance_local_PE : ndarray or list of float, optional
    The abundance of previously emplaced crater ejecta in ejecta deposits versus depth from the surface of the ejecta deposits. The ejecta deposits are the pre-existing local material for the emplacement of the ejecta of the current crater of interest for the earliest crater, abundance_local_PE should be an empty array.
  T_Rat : float, optional
    The thickness of ejecta at Rat with default value of T_Rat=0.068Rat.
  bt : float, optional
    The power-law index of ejecta thickness distribution with default value of bt=3.
  theta_degree : float, optional
    The launch/impact angle of ejecta in degrees with default value of theta_degree=45.
  Y : float, optional
    The strength of target material.
  K1, mu, nu : float, optional
    Constants in crater scaling laws (Equation 12 see also Table 2).
  
  Returns
  -------
  T_LM_med : 
    The median thickness of excavated local material.
  T_PE : 
    The thickness of primary ejecta.
  abundance_PEinNewED : 
    The abundance of ejecta in ejecta deposits as a function of depth (i.e., Elevation_from_Local_Surface).
  secondary_craters : 
    A struct recording the information of secondary craters.
  Fraction_ExcavatedLM : 
    Fraction of excavated local material as a function of depth (i.e., Elevation_from_Local_Surface).
  """
  def Dat2D(Dat_m=None):
    #D_m,in meters,final diameter
    #Dat_m,in meters,apparent diameter
    if Dat_m is None:
      Dat_m = np.logspace(3, 5, 1000)
    D_m = Dat_m * 1.43
    Dsc = 19000 #m, Simple to complex transition ~19km (Pike 1980 LPSC, p27)
    index = D_m > Dsc
    D_m[index] = 1.52 * Dsc**(-0.18) * Dat_m[index]**1.18
    return D_m
  secondary_craters = {}
  Elevation_from_Local_Surface = np.asarray(Elevation_from_Local_Surface)
  abundance_local_PE           = np.asarray(abundance_local_PE)
  ### Set parameters
  if T_Rat is None:
      T_Rat  = 0.068 * Rat
  Cex        = 3.5
  Rm         = 1737.4   #km
  rhot       = 3e6      #target density  g/m**3
  rhoe       = rhot     #ejecta density  g/m**3
  L_SOI      = 2 * r_SOI_center**0.5 #km
  launch_position = Rat #km
  S_SOI      = L_SOI**2 #km**2
  g          = 1.622    #m/s**2
  g_km       = g / 1000 #km/s**2
  theta      = np.radians(theta_degree)
  Cmh        = 0.013
  b_MT       = 0.91
  # Coefficients in Equation (10) derived from the work of Allen (1979).
  a_rLSC     = 7.209219501948585
  b_rLSC     = 0.935213298326370
  #---------------------------------------  
  N_SOI = len(r_SOI_center) #Number of SOI
  r_SOI_outer = r_SOI_center + L_SOI/2 #km, the outer boundary of the SOI
  r_SOI_inner = r_SOI_center - L_SOI/2 #km, the inner boundary of the SOI
  # -------the area of a ring for spherical target in Equation (4)-------- #
  S_ring   = 2*np.pi * (Rm**2) * (np.cos(r_SOI_inner/Rm) - np.cos(r_SOI_outer/Rm)) #km2
  S_ring   = abs(S_ring) #if r_SOI_outer/Rm > np.pi, S_ring < 0
  #-------------------------------------------------------------------------
  if not abundance_local_PE.size:
    m_depth = len(Elevation_from_Local_Surface)
    n_component = 1 #the total number of ejecta sources.
    abundance_local_PE = np.zeros((m_depth, n_component, N_SOI)) #[depth,component,SOI]
    abundance_PEinNewED = np.zeros(abundance_local_PE.shape)
  else:
    abundance_local_PE = np.atleast_3d(abundance_local_PE) #[depth,component,SOI]
    m_depth, n_component, k = abundance_local_PE.shape #the total number of differently sourced ejecta in local material.
    n_component += 1 #we add one more component for the ejecta of the current crater of interest.
    new_component = np.zeros((m_depth, 1, k)) #the abundance of ejecta of the current crater of interest in local material is zero before the emplacement of the ejecta of the current crater of interest.
    abundance_local_PE = np.concatenate((abundance_local_PE, new_component), axis=1) #[depth,component,SOI]
    
    abundance_PEinNewED = np.zeros(abundance_local_PE.shape)
  
  
  T_LM_med = np.zeros(r_SOI_center.shape) #m -- median local material thickness
  T_PE     = np.zeros(r_SOI_center.shape) #m -- primary ejecta thickness
  Fraction_ExcavatedLM = np.zeros((len(Elevation_from_Local_Surface), N_SOI))
  if mixing:
    for key in ['D', 'largest', 'S_SOI', 'S_ring', 'density', 'density_cumulative', 'Num_SOI', 'Num_SOI_cumulative']:
      secondary_craters[key] = np.empty(N_SOI, dtype=object)
  for i_SOI in range(N_SOI):
    if r_SOI_center[i_SOI] <= (Rat + L_SOI[i_SOI]/2):
      raise ValueError("SOI has to be outside crater rim.\n\n")
    
    Rs           = lambda r_gc: r_gc - launch_position
    X            = lambda r_gc: Rs(r_gc) / (2 * Rm)
    rgc2velocity = lambda r_gc: np.sqrt(Rm * g_km * np.tan(X(r_gc)) / (np.tan(X(r_gc)) * np.cos(theta)**2 + np.sin(theta) * np.cos(theta)))

    v_SOI_inner  = rgc2velocity(r_SOI_inner[i_SOI])  #km/s
    v_SOI_outer  = rgc2velocity(r_SOI_outer[i_SOI])  #km/s
    v_SOI_center = rgc2velocity(r_SOI_center[i_SOI]) #km/s

    if v_SOI_center >= 2.38:
      T_PE[i_SOI]     = 0
      T_LM_med[i_SOI] = 0
      continue
    
    
    v2r_flat = lambda v: launch_position + v**2 * np.sin(2*theta) / (g_km)
    
    r_SOI_inner_flat     = v2r_flat(v_SOI_inner)  #km
    r_SOI_outer_flat     = v2r_flat(v_SOI_outer)  #km
    r_SOI_center_flat    = v2r_flat(v_SOI_center) #km
    
    S_ring_flat    = np.pi*(r_SOI_outer_flat**2 - r_SOI_inner_flat**2) #km**2
    #------Equation (4)------------------------------
    Thickness_flat    = T_Rat*(r_SOI_center_flat/Rat)**-bt #km,  the thickness of ejecta on flat surface at distance r_SOI_center_flat
    Thickness_SOI_km  = Thickness_flat*S_ring_flat/S_ring[i_SOI]  # km
    T_PE[i_SOI]       = Thickness_SOI_km * 1000 #m
    #-------Equation (5)-----------------------------
    M_SOI = rhoe * T_PE[i_SOI] * S_SOI[i_SOI] * 1e6 #g
    #------v_LSC,Equation (8)-----------------------
    r_LSC = a_rLSC * Rat**b_rLSC
    v_LSC = rgc2velocity(r_LSC) #km/s
    #-------equation (7)-----------------------------
    MT = 0.09 * np.pi * rhot * (Rat*1000)**3 #g
    if v_SOI_center >= v_LSC:
      mh = Cmh * MT**b_MT * (v_SOI_center/v_LSC)**(-5.7) #g, upper limit of ejecta mass
    else:
      mh = Cmh * MT**b_MT #g, upper limit of ejecta mass
    

    b = 0.98 #

    ml = mh * 1e-23 #lower limit of ejecta mass
    C_SOI = M_SOI * (1-b)/b / (mh**(1-b) - ml**(1-b)) # 2
    if C_SOI < 0:
      raise ValueError("C_SOI < 0\n\n")
    
    #--------------------------cumulative M_SOI larger than or equal to M_SOI m  within SOI-----------
    N_mass_bin = round(np.log(mh/ml) / np.log(1.05)) - 1 #mh=ml*q**(n+1) => n=log(mh/ml)/log(q)-1, q=1.05
    mL = np.logspace(np.log10(ml), np.log10(mh), N_mass_bin+1)
    
    mR = mL[1:] #right boundary of mass bin.
    mL = mL[:-1] #left boundary of mass bin.
    mass = (mL * mR)**0.5 #g
    
    N_ejecta = C_SOI*mL**(-b) - C_SOI*mR**(-b) #number of ejecta framents in each mass bin.
    #------------------scaling law--------------------------------------
    Vp  = v_SOI_center * 1000 * np.sin(theta) #m/s, vertical component
    Vp2 = Vp**2
    ai = ((3 * mass) / (4 * np.pi * rhoe))**(1/3) #m,radius of projectile
    Rat_sec = K1 * ai* ((g*ai/Vp2)*(rhot/rhoe)**(2*nu/mu)+(Y/(rhot*Vp2))**(1+mu/2)*(rhot/rhoe)**(nu*(2+mu)/mu))**(-mu/(2+mu)) #m
    Dat = Rat_sec * 2
    #----------convert Dat to final diameter D (Equation (13))---------
    D = Dat2D(Dat)
    #-------------------------------------------------------------------
    d_ex  = 0.0134 * Rat_sec * (v_SOI_center * 1000)**0.38 # Equation (16)


    T_onePE_Layer = max([Cex*d_ex[-1]/5000, T_PE[i_SOI]/100]) #m
    N_layers      = int(np.ceil(T_PE[i_SOI] / T_onePE_Layer))
    T_onePE_Layer = T_PE[i_SOI] / N_layers
    
    elevation_bottom_layer = np.linspace(0, T_PE[i_SOI], N_layers+1) #the elevation of the lower boundary of a layer with respect to the surface of local material
    if len(elevation_bottom_layer) <= 2:
      elevation_bottom_layer = 0
    else:
      elevation_bottom_layer = elevation_bottom_layer[:-1] #the elevation of the bottom of each layer
    
    # ------------------------------------------------------local material (Equation 19)-----------------------------------------------------------------------------------------
    S_secondary_craters          = np.pi * Rat_sec**2    #m**2, the area of secondary craters at surface
    density_secondary_craters    = N_ejecta / S_SOI[i_SOI] * 1E-06  #m**-2, density of secondary craters
    
    T_LM = np.zeros(N_mass_bin)
    d_eff = Cex * d_ex #effective excavation depth.
    E = np.zeros((N_mass_bin, N_layers)) #the exponent in Equation (19)
    density_SCs_onelayer = density_secondary_craters / N_layers #m**-2,the density of secondary craters formed by a layer of ejecta
    for k in range(N_layers):
      for i in range(N_mass_bin):
        dmin = d_ex[i] #the effective depth of the ith crater is dmin
        T_LM[i] = Cex * dmin
        
        index = np.arange(i, N_mass_bin)
        index = index[(T_LM[i] + elevation_bottom_layer[k]) < d_eff[index]]
        S_PIS = S_secondary_craters[index] * (1 - T_LM[i]/d_eff[index]) * (1 - elevation_bottom_layer[k]/d_eff[index])
        E[i, k] = E[i, k] + sum(S_PIS * density_SCs_onelayer[index])
          
    
    E = -np.cumsum(E, 1)
    
    W_PIS_nLayers = 1 - np.exp(E) #Equation (19)
    W_PIS_allLayers = W_PIS_nLayers[:, -1]
    
    T_LM_med[i_SOI] = T_LM[np.where(W_PIS_allLayers < 0.5)[0][0]]
    if not mixing:
      continue
    
    Fraction_ExcavatedLM[:, i_SOI] = np.interp(Elevation_from_Local_Surface, np.flip(-T_LM), np.flip(W_PIS_allLayers)) #Figure 4a
    # ------------------------------------------------------Mixing-----------------------------------------------------------------------------------------
    Depth_Deposits = np.concatenate((np.flip(elevation_bottom_layer)+T_onePE_Layer/2, Elevation_from_Local_Surface), axis=0)
    #---------------One component Mixed by ejecta--------------
    d_mz     = Cex * d_ex
    dmz_max  = -max(d_mz) #maximum depth of mixing zone
    dmz      = np.arange(-T_onePE_Layer/2, dmz_max, -T_onePE_Layer)
    step_dmz = dmz[0] - dmz[1]

    W_onelayer                     = W_PIS_nLayers[:, 0]
    Wmz_onelayer                   = np.interp(dmz, np.flip(-d_mz), np.flip(W_onelayer))
    Wmz_onelayer[Wmz_onelayer > 1] = 1 #correct inappropriate values derived from extrapolation if exist
    Wmz_onelayer[Wmz_onelayer < 0] = 0 #correct inappropriate values derived from extrapolation if exist

    T_ED_ex_by_onelayer = np.sum(step_dmz * Wmz_onelayer)

    n_depth_fill = int(abs( (T_PE[i_SOI]-(step_dmz/2)) / step_dmz ))
    a_temp = np.flip(np.linspace(step_dmz/2, T_PE[i_SOI]-step_dmz/2, n_depth_fill+1))
    Depth_Deposits_samebinsize = np.concatenate((a_temp, dmz), axis=0) #???????????????????????????
    binSize_one_ED_Layer = np.ones(Depth_Deposits_samebinsize.shape) * step_dmz

    abundance_PEinED_samebinsize = np.zeros((len(Depth_Deposits_samebinsize), n_component))
    for i in range(N_layers): #mixing ejecta with excavated materials. Note that the excavated materials could be ejecta deposits formed by earlier ejecta and/or local material.
      if i == 0:
        index_PreED = np.where((Depth_Deposits_samebinsize > (dmz_max + elevation_bottom_layer[i])) & (Depth_Deposits_samebinsize < elevation_bottom_layer[i]))[0]
        index_mz = np.floor((elevation_bottom_layer[i] - Depth_Deposits_samebinsize[index_PreED]) / step_dmz).astype(int)
        for j in range(0, n_component):
          abundance_PEinED_samebinsize[index_PreED, j] = np.interp(Depth_Deposits_samebinsize[index_PreED], np.flip(Elevation_from_Local_Surface), np.flip(abundance_local_PE[:, j, i_SOI]))
          # figuresemilogx(-Elevation_from_Local_Surface,abundance_local_PE(:,j,i_SOI),'r-',-Depth_Deposits_samebinsize(index_PreED),abundance_PEinED_samebinsize(index_PreED,j),'k--')
          abundance_PEinED_samebinsize[abundance_PEinED_samebinsize[:,j] > 1, j] = 1 #correct inappropriate values derived from extrapolation if exist
          abundance_PEinED_samebinsize[abundance_PEinED_samebinsize[:,j] < 0, j] = 0 #correct inappropriate values derived from extrapolation if exist
          
      else:
        index_PreED = index_PreED - 1
      
      #-------------------------ejecta abundance---------------------------
      idx_new_component = n_component - 1
      T_PE_in_exED = sum(binSize_one_ED_Layer[index_PreED] * abundance_PEinED_samebinsize[index_PreED, idx_new_component] * Wmz_onelayer[index_mz])#ejecta in excavated ejecta deposits
      abundance_PE_in_excavatedMaterials = (T_onePE_Layer + T_PE_in_exED) / (T_ED_ex_by_onelayer + T_onePE_Layer)
      
      abundance_PEinED_samebinsize[index_PreED,      idx_new_component] = abundance_PEinED_samebinsize[index_PreED, idx_new_component] * (1 - Wmz_onelayer[index_mz]) + Wmz_onelayer[index_mz] * abundance_PE_in_excavatedMaterials
      abundance_PEinED_samebinsize[index_PreED[0]-1, idx_new_component] = abundance_PE_in_excavatedMaterials
      for k in range(n_component-1): #the jth component reworks preexisting materials, reducing the abundance of preexisting materials in ejecta deposits
        T_PE_in_exED = sum(binSize_one_ED_Layer[index_PreED] * abundance_PEinED_samebinsize[index_PreED, k] * Wmz_onelayer[index_mz]) #ejecta in excavated ejecta deposits
        abundance_PE_in_excavatedMaterials = T_PE_in_exED / (T_ED_ex_by_onelayer + T_onePE_Layer)

        abundance_PEinED_samebinsize[index_PreED,      k]       = abundance_PEinED_samebinsize[index_PreED, k] * (1 - Wmz_onelayer[index_mz]) + Wmz_onelayer[index_mz] * abundance_PE_in_excavatedMaterials
        abundance_PEinED_samebinsize[index_PreED[0]-1, k]       = abundance_PEinED_samebinsize[index_PreED[0]-1, k] + abundance_PE_in_excavatedMaterials
        
    

    abundance_PEinNewED_oneSOI                                              = np.zeros((len(Depth_Deposits), n_component))
    for j in range(n_component):
      abundance_PEinNewED_oneSOI[:, j]                                    = np.interp(Depth_Deposits, np.flip(Depth_Deposits_samebinsize), np.flip(abundance_PEinED_samebinsize[:, j]))
      abundance_PEinNewED_oneSOI[abundance_PEinNewED_oneSOI[:, j]>1, j] = 1 #correct inappropriate values derived from extrapolation if exist
      abundance_PEinNewED_oneSOI[abundance_PEinNewED_oneSOI[:, j]<0, j] = 0 #correct inappropriate values derived from extrapolation if exist
    
    
    #---------------------------------------------------------------------------------------------------
    Fraction_PE_new1           = np.zeros((len(Elevation_from_Local_Surface), n_component))
    for k in range(n_component):
      Fraction_PE_new1[:, k] = np.interp(Elevation_from_Local_Surface, np.flip(Depth_Deposits-T_PE[i_SOI]), np.flip(abundance_PEinNewED_oneSOI[:, k]))
    
    #--------------------------------------
    abundance_PEinNewED[:, :, i_SOI] = Fraction_PE_new1

    if mixing:
      secondary_craters['D'][i_SOI]                  = D                       #m
      secondary_craters['largest'][i_SOI]            = D[-1]                   #m
      secondary_craters['S_SOI'][i_SOI]              = S_SOI[i_SOI]            #km**2
      secondary_craters['S_ring'][i_SOI]             = S_ring[i_SOI]           #km**2
      secondary_craters['density'][i_SOI]            = N_ejecta / S_SOI[i_SOI] #km**-2
      secondary_craters['density_cumulative'][i_SOI] = np.flip(np.cumsum(np.flip(secondary_craters['density'][i_SOI])))
      secondary_craters['Num_SOI'][i_SOI]            = N_ejecta
      secondary_craters['Num_SOI_cumulative'][i_SOI] = np.flip(np.cumsum(np.flip(secondary_craters['Num_SOI'][i_SOI])))
      

  return T_LM_med, T_PE, abundance_PEinNewED, secondary_craters, Fraction_ExcavatedLM


def ballistic_sedimentation_Xie_xarray(
    xr_basin: xr.Dataset,
    r_SOI: np.ndarray = None,
    coord_SOI: np.ndarray = None,
    dz: float = 0.05,
    max_depth: float = 1e4,
    R0: float = 1737.4,
    **kw_args
) -> tuple:
  """
  
  """
  if r_SOI is None and coord_SOI is None:
    raise ValueError("Either r_SOI or coord_SOI must be provided.")
  elif r_SOI is not None and coord_SOI is not None:
    raise ValueError("Only one of r_SOI or coord_SOI should be provided.")
  nsteps_Elevation = int(max_depth / dz) + 1
  elevation = np.linspace(dz/2, -max_depth+(dz/2), nsteps_Elevation)
  coord_basin = df_basin[['lat', 'lon']].values
  Rat = df_basin['Rat'].values
  if coord_SOI is not None:
    r_SOI = []
    for c in coord_SOI:
      r_SOI.append(great_circle_distance(c[0], c[1], coord_basin[:, 0], coord_basin[:, 1], R=R0))
    r_SOI = np.asarray(r_SOI)
  T_LM_med                                    = np.zeros((len(Rat), r_SOI.shape[0]))
  T_PE                                        = np.zeros((len(Rat), r_SOI.shape[0]))
  abundance_PEinNewED = []
  secondary_craters                           = np.empty(len(Rat), dtype=object)
  Fraction_ExcavatedLM                        = np.zeros((len(elevation), len(Rat), r_SOI.shape[0]))
  abundance_PEinED_withoutMixingbyLaterEjecta = np.zeros((len(elevation), len(Rat), r_SOI.shape[0]))
  for i in tqdm(range(len(Rat)), desc="Running ballistic sedimentation model for each basin"):
    T_LM_med[i], T_PE[i], abundance_PEinNewED, secondary_craters[i], Fraction_ExcavatedLM[:, i, :] = compute_ejecta_thickness_and_mixing(
        Rat=Rat[i],
        r_SOI_center=r_SOI[:, i],
        Elevation_from_Local_Surface=elevation,
        abundance_local_PE=abundance_PEinNewED,
        **kw_args
    )
    abundance_PEinED_withoutMixingbyLaterEjecta[:, i, :] = abundance_PEinNewED[:, -1, :]
  return T_LM_med, T_PE, abundance_PEinNewED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta


def ballistic_sedimentation_Xie(
    df_basin: pd.DataFrame,
    r_SOI: np.ndarray = None,
    coord_SOI: np.ndarray = None,
    dz: float = 0.05,
    max_depth: float = 1e4,
    R0: float = 1737.4,
    **kw_args
) -> tuple:
  """
  
  """
  if r_SOI is None and coord_SOI is None:
    raise ValueError("Either r_SOI or coord_SOI must be provided.")
  elif r_SOI is not None and coord_SOI is not None:
    raise ValueError("Only one of r_SOI or coord_SOI should be provided.")
  nsteps_Elevation = int(max_depth / dz) + 1
  elevation = np.linspace(dz/2, -max_depth+(dz/2), nsteps_Elevation)
  coord_basin = df_basin[['lat', 'lon']].values
  Rat = df_basin['Rat'].values
  if coord_SOI is not None:
    r_SOI = []
    for c in coord_SOI:
      r_SOI.append(great_circle_distance(c[0], c[1], coord_basin[:, 0], coord_basin[:, 1], R=R0))
    r_SOI = np.asarray(r_SOI)
  T_LM_med                                    = np.zeros((len(Rat), r_SOI.shape[0]))
  T_PE                                        = np.zeros((len(Rat), r_SOI.shape[0]))
  abundance_PEinNewED = []
  secondary_craters                           = np.empty(len(Rat), dtype=object)
  Fraction_ExcavatedLM                        = np.zeros((len(elevation), len(Rat), r_SOI.shape[0]))
  abundance_PEinED_withoutMixingbyLaterEjecta = np.zeros((len(elevation), len(Rat), r_SOI.shape[0]))
  for i in tqdm(range(len(Rat)), desc="Running ballistic sedimentation model for each basin"):
    # T_LM_med[i], T_PE[i], abundance_PEinNewED, secondary_craters[i], Fraction_ExcavatedLM[:, i, :] = compute_ejecta_thickness_and_mixing(
    T_LM_med[i], T_PE[i], abundance_PEinNewED, secondary_craters[i], Fraction_ExcavatedLM[:, i, :] = compute_ejecta_thickness_and_mixing_xarray(
        Rat=Rat[i],
        r_soi_center=r_SOI[:, i],
        elevation=elevation,
        abundances=abundance_PEinNewED,
        **kw_args
    )
    abundance_PEinED_withoutMixingbyLaterEjecta[:, i, :] = abundance_PEinNewED[:, -1, :]
  return T_LM_med, T_PE, abundance_PEinNewED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta


def Xie_figure10c() -> tuple:
  """
  Code to reproduce Figure 10c of Xie et al. (2020) using the ballistic sedimentation model of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Originally created with MATLAB R2016b:
  Xie, M. Liu, T. and Xu, A. (2020), Ballistic sedimentaiton model [Code], Zenodo. https://doi.org/10.5281/zenodo.3692887.

  Adapted to Python by Matt Jones 2026.
  """
  df_basin = pd.DataFrame(
    {
      'lat': [-16.15, -24.28, 18.03, 26.57,  34.71, -19.83],
      'lon': [ 34.59, -39.35, 60.12, 18.05, -17.07, -94.58],
      'Dat': [339, 300, 370, 350, 402, 418]
    },
    index = ["Nectaris", "Humorum", "Crisium", "Serenitatis", "Imbrium", "Orientale"]
  )
  df_basin['Rat'] = df_basin['Dat'] / 2
  coord_A16 = [-8.973, 15.5]
  Tlm, Tpe, abundance_PEinED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta = ballistic_sedimentation_Xie(df_basin, coord_SOI=np.array([coord_A16]))
  Tlm = Tlm.squeeze()
  Tpe = Tpe.squeeze()
  Rat = df_basin['Rat'].values
  Name_basin = df_basin.index.values
  with plt.rc_context({
      'font.family': 'Myriad Pro',
      'figure.dpi': 300,
      'lines.linewidth': 1.5,
  }):
    fig, ax = plt.subplots(1, 1, figsize=(6.67, 4))
    C1 = [
        '#ff0000', 
        '#00ffff', 
        '#0000ff', 
        '#000000', 
        '#ff00ff', 
        '#00ff00'
    ]
    for i in range(len(Rat)+1):
        if i < len(Rat):
            ax.loglog(-elevation+sum(Tpe[i+1:]), abundance_PEinED_withoutMixingbyLaterEjecta[:,i]*100, color=C1[i], linestyle='--')
            ax.loglog(-elevation, abundance_PEinED[:, i]*100, color=C1[i])
            ax.text(1800, 7*2**(0.5*i), Name_basin[i], color=C1[i])
        else:
            abundance_PreNectarianMaterials = 100 - abundance_PEinED_withoutMixingbyLaterEjecta[:, 0]*100
            ax.loglog(-elevation+sum(Tpe[1:]), abundance_PreNectarianMaterials, linestyle='--', color=[0.5, 0.5, 0.5])
            abundance_PreNectarianMaterials = (1 - np.sum(abundance_PEinED, axis=1)) * 100 #abundance_PreNectarianMaterials=(1-sum(abundance_PEinED(:,1:end),2))*100
            ax.loglog(-elevation, abundance_PreNectarianMaterials, color=[0.5, 0.5, 0.5])
            ax.text(1800, 4, "Pre-Nectarian\nmaterials", color=[0.5, 0.5, 0.5])
    ax.set_ylim([0.1, 100])
    ax.set_xlim([0.1, 20000])
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], labels=['1', '2', '5', '10', '20', '50', '100'])
    ax.set_xlabel("Depth from surface (m)")
    ax.set_ylabel("Abundance of basin ejecta in deposits (%)")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.show()
  return Tlm, Tpe, abundance_PEinED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta


def Xie_figure5() -> tuple:
  """
  Code to reproduce Figure 5 of Xie et al. (2020) using the ballistic sedimentation model of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Reproduced in Python by Matt Jones 2026.
  """
  df_basin = pd.DataFrame(
     {
        'lat': [-19.83],
        'lon': [-94.58],
        'Dat': [418]
     },
     index = ["Orientale"]
  )
  df_basin['Rat'] = df_basin['Dat'] / 2
  coord_A16 = [-8.973, 15.5]
  Tlm, Tpe, abundance_PEinED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta = ballistic_sedimentation_Xie(df_basin, coord_SOI=np.array([coord_A16]))
  Tlm = Tlm.squeeze()
  Tpe = Tpe.squeeze()
  Rat = df_basin['Rat'].values
  with plt.rc_context({
      'font.family': 'Myriad Pro',
      'figure.dpi': 300,
      'lines.linewidth': 1.5,
  }):
    fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    ax[0].semilogx(-elevation, Fraction_ExcavatedLM[:, 0, 0]*100, color=[0, 0, 0])
    ax[0].set_xlabel("Depth from pre-impact surface (m)")
    ax[0].set_ylabel("Fraction of excavated local materials (%)")
    ax[0].set_xlim((1e0, 2e3))
    ax[0].set_ylim((0, 100))
    ax[1].semilogx(-(elevation+Tpe), abundance_PEinED[:, 0, 0]*100, color=[1, 0, 0], linewidth=2, label="Orientale ejecta")
    ax[1].semilogx(-(elevation+Tpe), (1-abundance_PEinED[:, 0, 0])*100, color=[0, 0, 0], linewidth=2, label="Pre-Orientale materials")
    ax[1].vlines(Tpe, 0, 100, color=[0, 0, 1], linestyle='--', linewidth=.7, label="Pre-impact surface")
    ax[1].set_xlabel("Depth from surface (m)")
    ax[1].set_ylabel("Abundance of materials in deposits (%)")
    ax[1].set_xlim((1e0, 2e3))
    ax[1].set_ylim((0, 100))
    ax[1].legend()
    for a in ax:
      a.set_yticks(np.linspace(0, 100, 11))
      a.minorticks_on()
      a.tick_params(which='both', direction='in', right=True, top=True)
    plt.show()
  return Tlm, Tpe, abundance_PEinED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta

def Xie_figure10c_test() -> tuple:
  """
  Reproduce Figure 10c of Xie et al. (2020) using the new analytical-thickness
  + mixing-kernel pipeline (``precompute_SOI`` → ``compute_mixing_kernel`` →
  ``compute_ejecta_mixing``), for comparison with ``Xie_figure10c`` which uses
  the original ``ballistic_sedimentation_Xie`` workflow.
  """
  ### //Basin definitions//
  basin_names = ["Nectaris", "Humorum", "Crisium", "Serenitatis", "Imbrium", "Orientale"]
  basin_lats  = [-16.15, -24.28, 18.03, 26.57,  34.71, -19.83]
  basin_lons  = [ 34.59, -39.35, 60.12, 18.05, -17.07, -94.58]
  basin_Dats  = [339, 300, 370, 350, 402, 418]
  coord_A16   = (-8.973, 15.5)
  ### //Initialize variables//
  # Elevation grid
  dz        = 0.05
  max_depth = 1e4
  nz        = int(max_depth / dz) + 1
  elevation = np.linspace(dz/2, -max_depth + (dz/2), nz)
  # Ejecta model
  ejecta_model   = EjectaModel(material="sand")
  ejecta_model.Y = 10e6
  n_basins       = len(basin_names)
  Tprimary       = np.zeros(n_basins)
  abundances              = None
  abundance_without_later = np.zeros((len(elevation), n_basins))
  ### //Emplace basins chronologically//
  for i in range(n_basins):
    Rat_km = basin_Dats[i] / 2
    # Build minimal xr.Dataset expected by precompute_SOI
    ds = xr.Dataset({
      'R':     Rat_km * 1.3,
      'Rat':   Rat_km,
      'basin': basin_names[i],
    })
    rSOI        = great_circle_distance(basin_lats[i], basin_lons[i], coord_A16[0], coord_A16[1])
    cache       = precompute_SOI(ds, rSOI=[rSOI], ejecta_model=ejecta_model)
    Tprimary[i] = cache['thickness_primary'][0]
    kernel      = compute_mixing_kernel(cache, 0)
    # Handle case of no primary ejecta, or compute mixing
    if kernel is None:
      if abundances is None:
        abundances = np.zeros((len(elevation), 1))
      else:
        abundances = np.column_stack([abundances, np.zeros(len(elevation))])
      continue
    else:
      abundances                    = compute_ejecta_mixing(kernel, elevation, abundances)
      abundance_without_later[:, i] = abundances[:, -1]
  ### //Plot//
  with plt.rc_context({
      'font.family': 'Myriad Pro',
      'figure.dpi': 300,
      'lines.linewidth': 1.5,
  }):
    fig, ax = plt.subplots(1, 1, figsize=(6.67, 4))
    C1 = ['#ff0000', '#00ffff', '#0000ff', '#000000', '#ff00ff', '#00ff00']
    # Plot basins
    for i in range(n_basins):
      ax.loglog(
        -elevation + sum(Tprimary[i+1:]),
        abundance_without_later[:, i] * 100,
        color = C1[i], 
        linestyle = '--'
      )
      ax.loglog(
        -elevation,
        abundances[:, i] * 100,
        color = C1[i]
      )
      ax.text(1800, 7 * 2**(0.5 * i), basin_names[i], color=C1[i])
    # Plot pre-Nectarian materials
    abundance_PreNect = (1 - abundance_without_later[:, 0]) * 100
    ax.loglog(
      -elevation + sum(Tprimary[1:]),
      abundance_PreNect,
      linestyle = '--', 
      color = [0.5, 0.5, 0.5]
    )
    abundance_PreNect = (1 - np.sum(abundances, axis=1)) * 100
    ax.loglog(
      -elevation,
      abundance_PreNect,
      color = [0.5, 0.5, 0.5]
    )
    ax.text(1800, 4, "Pre-Nectarian\nmaterials", color=[0.5, 0.5, 0.5])
    # Formatting
    ax.set_ylim([0.1, 100])
    ax.set_xlim([0.1, 20000])
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], labels=['1', '2', '5', '10', '20', '50', '100'])
    ax.set_xlabel("Depth from surface (m)")
    ax.set_ylabel("Abundance of basin ejecta in deposits (%)")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.show()
  return Tprimary, abundances, abundance_without_later, elevation


class EjectaModel:
  def __init__(
    self,
    theta0: float = 45,
    rho_e: float = 3000,
    rho_t: float = 3000,
    b: float = 0.98,
    cov: float = 0.5,
    R0: float = 1737.4e3,
    g: float = 1.622,
    R_sc: float = 9.5e3,
    C_mh: float = 0.013,
    b_mt: float = 0.91,
    b_v: float = -5.7,
    material: str = "sand"
  ):
    """
    Initialize ejecta model parameters. Defaults are from Xie et al. (2020).

    Parameters
    ----------
    ds_basin : xr.Dataset
      Must contain:
        - R   : basin rim radius (km)
        - Rat : apparent transient crater radius (km); if NaN, will be estimated.
    theta0 : float
      Ejection angle of primary fragments (degrees).
    rho_e : float
      Density of primary ejected material, kg m⁻³.
    rho_t : float
      Density of target/local material, kg m⁻³.
    b : float
      Fragment-mass - frequency exponent.
    cov : float
      Desired coverage fraction (0-1) for local excavation.
    R0 : float
      Planetary radius, m.
    g : float
      Surface gravity, m s⁻².
    R_sc : float
      Simple-complex crater transition radius, m. Default is 9.5e3 m for the Moon [from Croft (1985), reported in Xie et al. (2020)].
    nSOI : int
      Number of radial bins (Squares of Interest) to compute.
    rSOI : float or list of floats
      If set, overrides `nSOI` to compute specific distances from the basin center (in km).
    radius_cutoff : float or None
      If set, maximum distance = radius_cutoff * Rt_km.
    C_mh, b_mt, b_v : float
      Parameters controlling upper/lower fragment-mass scalings (see below Eq. 7 of Xie et al. (2020)).
    material : {"sand", "hard rock", "soft rock"}
      Controls strength and scaling constants K1, μ, ν, Y.
    """
    self.theta0 = np.radians(theta0)
    self.rho_e = rho_e
    self.rho_t = rho_t
    self.b = b
    self.cov = cov
    self.R0 = R0
    self.g = g
    self.R_sc = R_sc
    self.C_mh = C_mh
    self.b_mt = b_mt
    self.b_v = b_v
    if material not in ["sand", "hard rock", "soft rock"]:
      raise ValueError("`material` must be 'sand', 'hard rock', or 'soft rock'.")
    self.material = material
    self._set_material_properties()
  
  def _set_material_properties(self):
    """
    Initialize material properties K1, nu, mu, Y based on the `material` parameter. Values are from Xie et al. (2020).
    """
    if self.material == "sand":
      K1 = 1.03
      nu = 0.4
      mu = 0.41
      Y  = 10e3
    elif self.material == "hard rock":
      K1 = 0.93
      nu = 0.4
      mu = 0.55
      Y  = 10e6
    elif self.material == "soft rock":
      K1 = 0.93
      nu = 0.4
      mu = 0.55
      Y  = 1e6
    self.K1 = K1
    self.nu = nu
    self.mu = mu
    self.Y = Y
    return


def precompute_SOI(
  ds_basin: xr.Dataset,
  nSOI: int = 20,
  rSOI: float | list[float] = None,
  radius_cutoff: float = None,
  ejecta_model: dict | EjectaModel = None
):
  """
  Compute per-SOI parameters needed for ejecta thickness calculations.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain:
      - R   : basin rim radius (km)
      - Rat : apparent transient crater radius (km); if NaN, will be estimated.
  nSOI : int
    Number of radial bins (Squares of Interest) to compute.
  rSOI : float or list of floats
    If set, overrides `nSOI` to compute specific distances from the basin center (in km).
  radius_cutoff : float or None
    If set, maximum distance = radius_cutoff * R_km.
  ejecta_model : dict or EjectaModel
    If dict, must contain the following keys (with values as described in the EjectaModel class):
    - theta0 : float
    - rho_e : float
    - rho_t : float
    - b : float
    - cov : float
    - R0 : float
    - g : float
    - R_sc : float
    - C_mh : float
    - b_mt : float
    - b_v : float
    - material : {"sand", "hard rock", "soft rock"}

  Returns
  -------
  cache : dict
    Dictionary containing pre-computed per-SOI parameters for ejecta thickness calculations. Cached parameters are:
      - pre_frag_radius : float
      - exp_sec_transient_radius : float
      - dist_km : ndarray, shape (nSOI,)
      - mass_lower : ndarray, shape (nSOI,)
      - mass_upper : ndarray, shape (nSOI,)
      - mass_norm_constant : ndarray, shape (nSOI,)
      - soi_area : ndarray, shape (nSOI,)
      - thickness_primary : ndarray, shape (nSOI,)
      - pre_sec_transient_radius1 : ndarray, shape (nSOI,)
      - pre_sec_transient_radius2 : ndarray, shape (nSOI,)
      - pre_central_effective_depth : ndarray, shape (nSOI,)
      - ejecta_model : EjectaModel
  """
  ### //Initialize the ejecta model parameters// ###
  # Define the ejecta model object, if needed
  if ejecta_model is None:
    ejecta_model = EjectaModel()
  elif isinstance(ejecta_model, dict):
    ejecta_model = EjectaModel(**ejecta_model)
  else:
    if not isinstance(ejecta_model, EjectaModel):
      raise ValueError("`ejecta_model` must be either a dict of parameters or an instance of EjectaModel.")
  # Extract parameters from the ejecta model object
  theta0 = ejecta_model.theta0 #[radians]
  rho_e  = ejecta_model.rho_e * 1e3  #[g/m^3] -- convert from kg/m^3 to match methods of Xie et al. (2020)
  rho_t  = ejecta_model.rho_t * 1e3  #[g/m^3]
  b      = ejecta_model.b
  cov    = ejecta_model.cov    #[area fraction]
  R0     = ejecta_model.R0     #[m]
  g      = ejecta_model.g      #[m/s^2]
  R_sc   = ejecta_model.R_sc   #[m]
  C_mh   = ejecta_model.C_mh
  b_mt   = ejecta_model.b_mt
  b_v    = ejecta_model.b_v
  K1     = ejecta_model.K1
  nu     = ejecta_model.nu
  mu     = ejecta_model.mu
  Y      = ejecta_model.Y
  
  ### //Read in basin parameters// ###
  # Present-day rim radius of basin
  R_km = ds_basin.R.item() #[km]
  R_m  = R_km * 1e3        #[m]
  # Apparent transient radius of basin (radius of the transient crater **at the level of the pre-impact surface**)
  Rat_km = ds_basin.Rat.item() #[km]
  if (Rat_km is None) or (not np.isfinite(Rat_km)):
    # Empirical relationship derived from D's of Neumann et al. (2015), Dat's of
    # Miljković et al. (2016), and SPA Dat from Rajšić (2025, personal communication)
    x1_temp, x0_temp = [0.38, 38.18] #[dimensionless, km] -- empirical relationship INCLUDING SPA
    Rat_km = x1_temp*R_km + x0_temp
    del x1_temp, x0_temp
  Rat_m = Rat_km * 1e3 #[m]
  # Transient rim radius of basin (included because it's used in Xie et al. (2020), but this implementation does not actually use it for anything)
  Rt_km = Rat_km * 1.2 #[km]
  Rt_m  = Rt_km * 1e3  #[m]
  
  ### //Set up distance array// ###
  # Set lower distance bound based on `R` and `Rat`
  min_dist = R_km #[km] -- ejecta is deposited beginning roughly at R [loose interpretation of reference to Melosh (1989) in Xie et al. (2020) Section 2.1.2]
  safe_eps = 1e-3 #[km] -- small epsilon to avoid numerical issues near `Rat`
  min_dist_safe = ( 0.5 * (1 + np.sqrt(1 + 4*Rat_km)) )**2 + safe_eps #[km] -- minimum distance at which the innermost SOI boundary abutts Rat -- anything smaller leads to an undefined velocity and ejecta thickness
  if min_dist < min_dist_safe:
    warnings.warn(f"\n'{ds_basin.basin.item()}' (R = {R_km:.2f} km) has Rat ({Rat_km:.2f} km) that is too large to use R as minimum distance. Adjusting to minimum safe distance of {min_dist_safe:.2f} km.\nNote that this will accentuate {ds_basin.basin.item()}'s innermost ejecta thickness compared to other basins.\n")
    min_dist = min_dist_safe #[km]
  # Set upper distance bound based on `radius_cutoff` argument
  dist_limit = np.pi*(R0/1e3) * .99 #[km] -- model yields errors very close to the antipode, so cut off profile just shy of π
  if radius_cutoff is None:
    max_dist = dist_limit #[km]
  else:
    max_dist = min([radius_cutoff*R_km, dist_limit]) #[km]
  # Generate distance array (each point is the great circle distance of center of a square of interest (SOI) from the basin center)
  if rSOI is None:
    dist_km = np.linspace(min_dist, max_dist, nSOI) #[km]
    dist_m  = dist_km * 1e3 #[m]
  else:
    if nSOI is not None:
      warnings.warn("`rSOI` is set, so `nSOI` will be ignored.")
    dist_km = np.asarray(rSOI) #[km]
    if any(dist_km < min_dist) or any(dist_km > max_dist):
      raise ValueError(f"All values in `rSOI` must be between {min_dist:.2f} km and {max_dist:.2f} km.")
    dist_m  = dist_km * 1e3 #[m]
  
  ### //Define size of each SOI -- Section 2.1.1 of Xie et al. (2020)// ###
  soi_side_km  = 2 * np.sqrt(dist_km)                  #L_SOI [km]
  soi_side_m   = soi_side_km * 1e3                     #L_SOI [m]
  soi_area     = soi_side_m**2                         #S [m^2] -- see text after Eq. 5 of Xie et al. (2020)

  ### //Define velocity parameters for each SOI -- Eq. 1 of Xie et al. (2020)// ###
  # Helper function (convert great circle distance to ejecta velocity at that distance)
  _launch_offset = lambda d: d - Rat_m                 #R_s [see text above Eq. 1 of Xie et al. (2020)]
  _X_term = lambda d: _launch_offset(d) / (2*R0)       #X [see text below Eq. 1]
  _ejecta_velocity = lambda d: np.sqrt(R0*g*np.tan(_X_term(d))) / (np.sqrt( np.tan(_X_term(d))*np.cos(theta0)**2 + np.sin(theta0)*np.cos(theta0) )) #v(r_gc)
  # Velocity parameters for each SOI
  velocity_soi = _ejecta_velocity(dist_m)              #v(r_gc) [m/s] -- velocity of primary ejecta in each SOI
  vertical_velocity = velocity_soi * np.sin(theta0)    #v_⊥ [m/s] -- ground-perpendicular velocity
  velocity_soi_e = velocity_soi**.38                   #v^0.38 [m^0.38/s^0.38] -- see Eq. 16 of Xie et al. (2020)

  ### //Define extents of SOIs// ###
  # Helper function (convert great circle distance to distance on a flat target) -- Eq. 2 of Xie et al. (2020)
  _flat_radius = lambda d: Rat_m + _ejecta_velocity(d)**2 * np.sin(2*theta0) / g #r(r_gc)
  # Inner, outer, and mean radial distance of each SOI from the basin center -- see text after Eq. 3 of Xie et al. (2020)
  inner_radius_gc = dist_m - (soi_side_m/2)            #r_gc [inner] [m] -- great circle distance
  inner_radius = _flat_radius(inner_radius_gc)         #r_inner [m] -- flat target distance
  outer_radius_gc = dist_m + (soi_side_m/2)            #r_gc [outer] [m]
  outer_radius = _flat_radius(outer_radius_gc)         #r_outer [m]
  mean_radius  = np.sqrt(inner_radius * outer_radius)  #r-bar [m] -- geometric mean to account for spherical surface
  # Area of basin-concentric ring that encompasses each SOI -- see text below Eq. 4 of Xie et al. (2020)
  sphere_ring_area = 2*np.pi*(R0**2) * (np.cos(inner_radius_gc / R0) - np.cos(outer_radius_gc / R0)) #S_ring [m^2]
  flat_ring_area   = np.pi * (outer_radius**2 - inner_radius**2) #S_ring_flat [m^2]
  
  ### //Calculate primary ejecta distribution -- Eqs. 4 & 5 of Xie et al. (2020)// ###
  ##TODO: Make T_Rat = 0.068*Rat_m and bt = 3 adjustable as arguments of EjectaModel
  thickness_primary = (0.068*Rat_m * (mean_radius / Rat_m)**(-3) * (flat_ring_area / sphere_ring_area)) #δ_SOI [m] -- thickness of primary ejecta per SOI
  mass_primary = thickness_primary * rho_e * soi_area  #M_SOI [g] -- mass of primary ejecta per SOI
  # Total mass of material ejected from primary transient crater -- see text below Eq. 7 of Xie et al. (2020))
  total_mass_primary = 0.09 * rho_e * np.pi * Rat_m**3 #M_T [g]
  
  ### //Largest secondary crater (LSC) parameters// ###
  # Coefficients in Equation (10) derived from the work of Allen (1979).
  a_rLSC          = 7.209219501948585                #Xie et al. (2020) derived this from Allen (1979)
  b_rLSC          = 0.935213298326370                #Xie et al. (2020) derived this from Allen (1979)
  lsc_distance_m = (a_rLSC * Rat_km**b_rLSC) * 1e3   #r_LSC [m] -- Eq. 10 of Xie et al. (2020) [**this is only valid for complex and larger craters**]
  velocity_lsc    = _ejecta_velocity(lsc_distance_m) #v_LSC [m/s] -- Eq. 8 of Xie et al. (2020)

  ### //Fragment mass parameters// ###
  # Upper and lower bounds on fragment mass -- see text below Eq. 7 of Xie et al. (2020)
  mass_upper = np.full_like(velocity_soi, C_mh*(total_mass_primary**b_mt)) #[g]
  mask = velocity_soi >= velocity_lsc
  # NOTE: b_v is specified as -5.7 in Xie et al. (2020), but Xie et al. (2020)'s equation for m_h uses -b_v; the exponent should be -5.7 to match the original Xie et al. (2020) model, so here we change the equation to use b_v, not -b_v.
  mass_upper[mask] *= (velocity_soi[mask] / velocity_lsc)**(b_v) #m_h [g]
  # mass_lower = 1e-18 * mass_upper                      #m_l [g]
  mass_lower = 1e-23 * mass_upper                      #m_l [g]
  # Empirical factor for fragment mass-frequency relationship -- Eq. 7 of Xie et al. (2020)
  mass_norm_constant = mass_primary * (1 - b) / (b * (mass_upper**(1 - b) - mass_lower**(1 - b))) #C_SOI
  
  ### //Begin ballistic sedimentation model – compute thickness of local material mixed into total ejecta deposit// ###
  # Define excavation scaling parameter C_ex from Xie et al. (2020)
  C_ex = 3.5
  # Pre-compute unchanging factors for SOI-loop calculations
  pre_frag_radius = (3 / (4*np.pi*rho_e))**(1/3)
  pre_sec_transient_radius1 = K1**(-(2+mu)/mu) * (g/vertical_velocity**2) * (rho_t/rho_e)**(2*nu/mu)
  pre_sec_transient_radius2 = K1**(-(2+mu)/mu) * (Y/(rho_t*vertical_velocity**2))**((2+mu)/2) * (rho_t/rho_e)**(nu*(2+mu)/mu)
  exp_sec_transient_radius = -mu / (2+mu)
  pre_central_effective_depth = C_ex * 0.0134 * velocity_soi_e

  cache = {
    # Scalars
    'pre_frag_radius':             pre_frag_radius,
    'exp_sec_transient_radius':    exp_sec_transient_radius,
    # Arrays
    'dist_km':                     dist_km,
    'mass_lower':                  mass_lower,
    'mass_upper':                  mass_upper,
    'mass_norm_constant':          mass_norm_constant,
    'soi_area':                    soi_area,
    'thickness_primary':           thickness_primary,
    'pre_sec_transient_radius1':   pre_sec_transient_radius1,
    'pre_sec_transient_radius2':   pre_sec_transient_radius2,
    'pre_central_effective_depth': pre_central_effective_depth,
    # Objects
    'ejecta_model':                ejecta_model
  }
  return cache


def compute_central_effective_depth(                                 #d_eff (Eq. 17 of Xie et al. (2020))
    m: float,
    pre_frag_radius: float,
    pre_sec1: float,
    pre_sec2: float,
    exp_sec_transient_radius: float,
    pre_deff: float
) -> float:
  """
  Eq. 17 of Xie et al. (2020)
  """
  a = pre_frag_radius * m**(1/3)
  sec_transient_radius = a * (pre_sec1*a + pre_sec2)**(exp_sec_transient_radius)
  return pre_deff * sec_transient_radius


def compute_coverage_fraction(
    T_LM: float,
    ejecta_model: EjectaModel,
    pre_frag_radius: float,
    exp_sec_transient_radius: float,
    ml: float,
    mh: float,
    C: float,
    S: float,
    pthick: float,
    pre_sec1: float,
    pre_sec2: float,
    pre_deff: float,
) -> float:
  """
  Eq. 19 of Xie et al. (2020)—computes coverage fraction W(>T_LM).

  Parameters
  ----------
  T_LM : float
    Local excavation thickness [m].
  ejecta_model : EjectaModel
    Ejecta model parameters.
  pre_frag_radius : float
    Pre-computed factor for fragment radius (see `precompute_SOI`).
  exp_sec_transient_radius : float
    Pre-computed exponent for secondary transient crater radius (see `precompute_SOI`).
  ml : float
    Minimum fragment mass [kg] (see `precompute_SOI`).
  mh : float
    Maximum fragment mass [kg] (see `precompute_SOI`).
  C : float
    Normalization constant for fragment mass-frequency distribution (see `precompute_SOI`).
  S : float
    SOI surface area [m²] (see `precompute_SOI`).
  pthick : float
    Primary ejecta thickness in the SOI area [m] (see `precompute_SOI`).
  pre_sec1 : float
    Pre-computed factor for secondary transient crater radius (see `precompute_SOI`).
  pre_sec2 : float
    Pre-computed factor for secondary transient crater radius (see `precompute_SOI`).
  pre_deff : float
    Pre-computed factor for central effective depth (see `precompute_SOI`).
  
  Returns
  -------
  W : float
    Coverage fraction W(>T_LM).
  """
  b = ejecta_model.b

  _central_effective_depth = functools.partial(
    compute_central_effective_depth,
    pre_frag_radius = pre_frag_radius,
    pre_sec1 = pre_sec1,
    pre_sec2 = pre_sec2,
    exp_sec_transient_radius = exp_sec_transient_radius,
    pre_deff = pre_deff
  )
  
  deff_min = _central_effective_depth(ml) #d_eff for the smallest fragment mass
  deff_max = _central_effective_depth(mh) #d_eff for the largest fragment mass
  if T_LM <= deff_min: #maximum coverage case
    m0 = ml
  elif T_LM >= deff_max: #minimum coverage case
    return 0
  else:
    sol = root_scalar(
      lambda m: _central_effective_depth(m) - T_LM,
      bracket=[ml, mh],
      method='bisect'
    )
    m0 = sol.root
  
  def _f(m):
    """
    Analytical equivalent of Riemann sum in Eq. 19 of Xie et al. (2020)
    """
    a = pre_frag_radius * m**(1/3) # not using _central_effective_depth because we need R_at for `uncorrected_area`
    R_at = a * (pre_sec1*a + pre_sec2)**(exp_sec_transient_radius)
    d_eff = pre_deff * R_at
    if pthick <= d_eff:                                              #1/N_layers * sum( ( (j-1)*δ_SOI / (N_layers*C_ex*d_ex(R_at(m))) ) ) [analytical replacement]
      correction2 = 1 - pthick/(2*d_eff)
    else:
      correction2 = d_eff/(2*pthick)
    uncorrected_area = np.pi * R_at**2                               #S_PIS [uncorrected] -- π*R_at(m)^2
    correction1 = max(0, 1-(T_LM/d_eff))                             #( 1 - T_LM/( C_ex*d_ex(R_at(m)) ) )
    DN = C * b*m**(-b-1)                                             #ΔN [analytical replacement]
    return uncorrected_area * correction1 * correction2 * DN / S
  
  def _f_log(x):
    """
    Convert intermediate solution to logspace for numerical stability in integration
    """
    m = np.exp(x)
    return _f(m) * m
  
  integral = quad(_f_log, np.log(m0), np.log(mh), epsabs=0, epsrel=1e-2)[0]
  W = 1 - np.exp(-integral)
  return W                                                           #W(>T_LM)


def compute_coverage_fraction_onelayer(
    T_LM: float,
    ejecta_model: EjectaModel,
    nlayers: int,
    pre_frag_radius: float,
    exp_sec_transient_radius: float,
    ml: float,
    mh: float,
    C: float,
    S: float,
    pthick: float,
    pre_sec1: float,
    pre_sec2: float,
    pre_deff: float,
) -> float:
  """
  Eq. 19 of Xie et al. (2020)—computes coverage fraction W(>T_LM).

  Parameters
  ----------
  T_LM : float
    Local excavation thickness [m].
  ejecta_model : EjectaModel
    Ejecta model parameters.
  pre_frag_radius : float
    Pre-computed factor for fragment radius (see `precompute_SOI`).
  exp_sec_transient_radius : float
    Pre-computed exponent for secondary transient crater radius (see `precompute_SOI`).
  ml : float
    Minimum fragment mass [kg] (see `precompute_SOI`).
  mh : float
    Maximum fragment mass [kg] (see `precompute_SOI`).
  C : float
    Normalization constant for fragment mass-frequency distribution (see `precompute_SOI`).
  S : float
    SOI surface area [m²] (see `precompute_SOI`).
  pthick : float
    Primary ejecta thickness in the SOI area [m] (see `precompute_SOI`).
  pre_sec1 : float
    Pre-computed factor for secondary transient crater radius (see `precompute_SOI`).
  pre_sec2 : float
    Pre-computed factor for secondary transient crater radius (see `precompute_SOI`).
  pre_deff : float
    Pre-computed factor for central effective depth (see `precompute_SOI`).
  
  Returns
  -------
  W : float
    Coverage fraction W(>T_LM).
  """
  ### Fetch `b` from ejecta model for calculating `DN`
  b = ejecta_model.b

  ### 
  mspace = np.logspace(np.log10(ml), np.log10(mh), 1000)
  # Calculate `R_at`, `d_eff`, and uncorrected crater area over fragment mass space
  a = pre_frag_radius * mspace**(1/3)
  R_at = a * (pre_sec1*a + pre_sec2)**(exp_sec_transient_radius)
  deff_space = pre_deff * R_at
  uncorrected_area = np.pi * R_at**2                                 #S_PIS [uncorrected] -- π*R_at(m)^2
  # Compute `DN` over fragment mass space
  DN = C * b*mspace**(-b-1)                                          #ΔN [analytical replacement]

  partial = uncorrected_area * DN / S

  ### Correct crater area for `T_LM`
  correction1 = np.maximum(0, 1 - (T_LM / deff_space))               #( 1 - T_LM/( C_ex*d_ex(R_at(m)) ) )
  partial *= correction1

  # Find m0 such that d_eff(m0) = T_LM
  m0 = np.interp(T_LM, deff_space, mspace, left=ml, right=mh)
  
  integral = cumulative_simpson(partial, mspace)
  W = 1 - np.exp(-integral)
  return W                                                           #W(>T_LM)


def compute_mixing_kernel(
    cache: dict,
    i_soi: int,
) -> dict | None:
  """
  Build the per-layer depth-excavation coverage kernel for a single SOI and basin.
  The kernel can be referenced when computing layer-by-layer vertical ejecta mixing.

  Analytical equivalent of the Riemann sum in Eq. 20 of Xie et al. (2020), discretized over depth.

  Parameters
  ----------
  cache : dict
    Output of `precompute_SOI`.
  i_soi : int
    Index for the SOI arrays in `cache`.

  Returns
  -------
  kernel : dict or None
    `None` when primary ejecta thickness is zero at this SOI.
    Otherwise a dict with keys:
    - primary_thickness      : float   - total thickness of primary ejecta in this SOI
    - Wmz_onelayer           : ndarray - one-layer coverage fraction at each depth point
    - mixing_grid            : ndarray - elevations of mixing grid points
    - dz                     : float   - mixing grid spacing
    - Texcavated_onelayer    : float   — thickness of preexisting material excavated by deposition of one layer
    - Tprimary_perlayer      : float   — thickness of one primary ejecta layer
    - N_layers               : int     — number of primary ejecta layers
    - elevation_layerbottoms : ndarray — bottom elevation of each deposit layer (w.r.t. pre-impact surface)
    - zmax                   : float   — maximum depth of mixing for one layer
  """
  ### //Fetch parameters from cache for this SOI//
  ejecta_model    = cache['ejecta_model']
  b               = ejecta_model.b
  pre_frag_radius = cache['pre_frag_radius']
  exp_sec         = cache['exp_sec_transient_radius']
  ml              = cache['mass_lower'][i_soi]        #[g]
  mh              = cache['mass_upper'][i_soi]        #[g]
  C_soi           = cache['mass_norm_constant'][i_soi]
  S_soi           = cache['soi_area'][i_soi]          #[m^2]
  pthick          = cache['thickness_primary'][i_soi] #[m]
  pre_sec1        = cache['pre_sec_transient_radius1'][i_soi]
  pre_sec2        = cache['pre_sec_transient_radius2'][i_soi]
  pre_deff        = cache['pre_central_effective_depth'][i_soi]
  # No mixing if no primary ejecta is deposited
  if pthick == 0:
    return None
  elif pthick < 0:
    raise ValueError(f"Primary ejecta thickness is <0 ({pthick} m).")
  ### //Predefine d_eff function for this SOI's parameters//
  _deff = functools.partial(
    compute_central_effective_depth,
    pre_frag_radius = pre_frag_radius,
    pre_sec1 = pre_sec1,
    pre_sec2 = pre_sec2,
    exp_sec_transient_radius = exp_sec,
    pre_deff = pre_deff,
  )
  deff_max = _deff(mh) #[m]
  deff_min = _deff(ml) #[m]
  ### //Calculate primary ejecta deposit layer parameters//
  Tprimary_perlayer = max(deff_max / 5000, pthick / 100)           #[m]
  N_layers = int(np.ceil(pthick / Tprimary_perlayer))
  Tprimary_perlayer = pthick / N_layers                            #[m]
  elevation_layerbottoms = np.arange(N_layers) * Tprimary_perlayer #[m]
  ### // Define depth grid//
  zmax        = deff_max                                                     #[m]
  mixing_grid = np.arange(-Tprimary_perlayer / 2, -zmax, -Tprimary_perlayer) #[m]
  dz          = Tprimary_perlayer                                            #[m]
  ### //Compute mixing kernel//
  T_LM_values  = -mixing_grid               #[m] -- make values positive
  Wmz_onelayer = np.zeros_like(T_LM_values) #[area fraction]
  for idx, T_LM in enumerate(T_LM_values):
    if T_LM >= deff_max:
      continue
    # Define lower search bound for integral
    if T_LM <= deff_min:
      m0 = ml
    else:
      m0 = root_scalar(
        lambda m: _deff(m) - T_LM,
        bracket=[ml, mh],
        method='bisect',
      ).root
    # Define helper function for integration (Equation 19 of Xie et al. (2020)) (in log space, to improve numerical stability)
    def _f_log(x, _T=T_LM):
      m    = np.exp(x)
      a    = pre_frag_radius * m**(1/3)
      R_at = a * (pre_sec1 * a + pre_sec2)**exp_sec
      d_eff_m = pre_deff * R_at
      area = np.pi * R_at**2
      corr = max(0.0, 1.0 - _T/d_eff_m)
      DN   = C_soi * b * m**(-b - 1)
      return area * corr * DN / (S_soi * N_layers) * m                            # `* m` is from log substitution
    # Integrate for total coverage fraction at this depth
    integral_val = quad(_f_log, np.log(m0), np.log(mh), epsabs=0, epsrel=1e-2)[0] #exponent of Eq. 20 of Xie et al. (2020)
    Wmz_onelayer[idx] = 1 - np.exp(-integral_val)                                 #[area fraction] -- W(>T_LM) for this layer
  # Ensure no invalid values
  if np.any(Wmz_onelayer < 0) or np.any(Wmz_onelayer > 1):
    raise ValueError("Computed coverage fractions are out of bounds [0, 1]. Check the integration and input parameters.")
  ### //Return the kernel//
  Texcavated_alllayers = np.sum(dz * Wmz_onelayer) #[m]
  return {
    'primary_thickness':      pthick,
    'Wmz_onelayer':           Wmz_onelayer,
    'mixing_grid':            mixing_grid,
    'dz':                     dz,
    'Texcavated_alllayers':   Texcavated_alllayers,
    'Tprimary_perlayer':      Tprimary_perlayer,
    'N_layers':               N_layers,
    'elevation_layerbottoms': elevation_layerbottoms,
    'zmax':                   zmax,
  }

def compute_ejecta_mixing(
    kernel: dict,
    elevation: np.ndarray,
    abundances: np.ndarray = None,
) -> np.ndarray:
  """
  Compute vertical mixing of primary ejecta with local material for a single SOI using a pre-built 
  kernel from `compute_mixing_kernel`. Accounts for multiple components if `abundances` is 2-D.

  This is a semi-analytical pipeline for the layer-by-layer mixing algorithm of Xie et al. (2020).

  Parameters
  ----------
  kernel : dict
    Output of `compute_mixing_kernel`.
  elevation : ndarray, shape (m_elev,)
    Grid of elevation w.r.t. pre-impact surface.
  abundances : ndarray, shape (m_elev, n_existing) or None
    Abundance of each pre-existing ejecta component versus `elevation`.
    Passing `None` (or an empty array) computes only this basin's mixing profile.

  Returns
  -------
  new_abundances : ndarray, shape (m_elev, n_existing + 1)
    Updated abundances after mixing.  The last column is the newly emplaced
    primary ejecta component.
  """
  ### //Fetch parameters from kernel//
  Tprimary               = kernel['primary_thickness']      #[m]
  Wmz_onelayer           = kernel['Wmz_onelayer']           #[area fraction]
  dz                     = kernel['dz']                     #[m]
  Texcavated_alllayers   = kernel['Texcavated_alllayers']   #[m]
  Tprimary_perlayer      = kernel['Tprimary_perlayer']      #[m]
  N_layers               = kernel['N_layers']
  elevation_layerbottoms = kernel['elevation_layerbottoms'] #[m]
  zmax                   = kernel['zmax']                   #[m]
  ### //Define interpolation helper function to prevent invalid values//
  def _safe_interp(x, xp, fp):
    result = np.interp(x, xp, fp)
    if any(x > 1 for x in result) or any(x < 0 for x in result):
      raise ValueError("Interpolation error: Unexpected extrapolation, or data values are not between 0 and 1.")
    return result
  ### //Initialize mixing components//
  elevation = np.asarray(elevation) #[m]
  if abundances is None or (hasattr(abundances, 'size') and abundances.size == 0):
    n_component    = 1
    abundances_2d  = np.zeros((len(elevation), 1))
  else:
    abundances_2d  = np.atleast_2d(abundances)
    n_component    = abundances_2d.shape[1] + 1
  idx_newcomponent = n_component - 1
  ### //Initialize mixing grid//
  elevation_toplayer    = Tprimary - (dz / 2)                                                #[m]
  n_depth_fill          = int(abs(elevation_toplayer / dz))
  elevation_deposit     = np.flip(np.linspace(dz / 2, elevation_toplayer, n_depth_fill + 1)) #[m]
  elevation_mixinggrid  = np.concatenate((elevation_deposit, kernel['mixing_grid']))         #[m]
  ### //Initialize preexisting abundances on the mixing grid//
  elevation_initialsurface                   = elevation_layerbottoms[0]                          #[m] -- elevation of the surface before any of this basin's ejecta is deposited
  idx_mixingzone                             = np.where((elevation_mixinggrid > (elevation_initialsurface - zmax)) & (elevation_mixinggrid < elevation_initialsurface))[0] #starting indices of the moving mixing-zone window for `elevation_mixinggrid`
  abundances_mixinggrid                      = np.zeros((len(elevation_mixinggrid), n_component)) #[area fraction]
  for k in range(idx_newcomponent): #initialize the first layer's mixing zone with preexisting abundances
    abundances_mixinggrid[idx_mixingzone, k] = _safe_interp(elevation_mixinggrid[idx_mixingzone], np.flip(elevation), np.flip(abundances_2d[:, k])) #[area fraction] -- abundances of preexisting components within mixing zone for the first layer
  ### //Compute vertical mixing//
  for i in range(N_layers): #emplace primary ejecta layer-by-layer
    ### First handle just primary ejecta for this layer...
    Texcavated = np.sum(dz * abundances_mixinggrid[idx_mixingzone, idx_newcomponent] * Wmz_onelayer) #[m] -- thickness of preexisting material excavated by deposition of this layer
    frac_new   = (Tprimary_perlayer + Texcavated) / (Tprimary_perlayer + Texcavated_alllayers)       #[volume fraction] -- fraction of this layer's mixed material that is new primary ejecta
    abundances_mixinggrid[idx_mixingzone,        idx_newcomponent] = abundances_mixinggrid[idx_mixingzone, idx_newcomponent]*(1 - Wmz_onelayer) + frac_new*Wmz_onelayer
    abundances_mixinggrid[idx_mixingzone[0]-1, idx_newcomponent] = frac_new
    ### ...then loop to redistribute preexisting components
    for k in range(idx_newcomponent):
      Tk_excavated = np.sum(dz * abundances_mixinggrid[idx_mixingzone, k] * Wmz_onelayer)
      frac_k       = Tk_excavated / (Texcavated_alllayers + Tprimary_perlayer)
      abundances_mixinggrid[idx_mixingzone,      k]  = abundances_mixinggrid[idx_mixingzone, k]*(1 - Wmz_onelayer) + frac_k*Wmz_onelayer
      abundances_mixinggrid[idx_mixingzone[0]-1, k] += frac_k
    ### Shift the mixing zone up for the next layer
    idx_mixingzone -= 1
  ### //Stack the new/mixed deposit on top of the original elevation grid and interpolate results onto that grid//
  elevation_afterdeposit  = np.concatenate((np.flip(elevation_layerbottoms) + Tprimary_perlayer / 2, elevation)) #[m]
  abundances_afterdeposit = np.zeros((len(elevation_afterdeposit), n_component))                                 #[area fraction]
  new_abundances          = np.zeros((len(elevation), n_component))                                              #[area fraction]
  for k in range(n_component):
    abundances_afterdeposit[:, k] = _safe_interp(elevation_afterdeposit, np.flip(elevation_mixinggrid), np.flip(abundances_mixinggrid[:, k]))   #[area fraction]
    new_abundances[:, k]          = _safe_interp(elevation, np.flip(elevation_afterdeposit - Tprimary), np.flip(abundances_afterdeposit[:, k])) #[area fraction]
  return new_abundances


def compute_thickness_1D(
  ds_basin: xr.Dataset,
  nSOI: int = 20,
  rSOI: float | list[float] = None,
  radius_cutoff: float = None,
  ejecta_model: dict | EjectaModel = None
):
  """
  Compute 1D radial profiles of primary ejecta thickness, local excavation thickness,
  and total ejecta thickness for a lunar basin.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain:
      - R   : basin rim radius (km)
      - Rat : apparent transient crater radius (km); if NaN, will be estimated.
  nSOI : int
    Number of radial bins (Squares of Interest) to compute.
  rSOI : float or list of floats
    If set, overrides `nSOI` to compute specific distances from the basin center (in km).
  radius_cutoff : float or None
    If set, maximum distance = radius_cutoff * Rt_km.
  ejecta_model : dict or EjectaModel
    If dict, must contain the following keys (with values as described in the EjectaModel class):
    - theta0 : float
    - rho_e : float
    - rho_t : float
    - b : float
    - cov : float
    - R0 : float
    - g : float
    - R_sc : float
    - C_mh : float
    - b_mt : float
    - b_v : float
    - material : {"sand", "hard rock", "soft rock"}

  Returns
  -------
  dist_km : ndarray, shape (nSOI,)
    Great-circle distances of SOI centers, in km.
  thickness_primary : ndarray, shape (nSOI,)
    Primary ejecta thickness per SOI, in m.
  thickness_local : ndarray, shape (nSOI,)
    Excavation thickness of local target material due to secondary craters, in m.
  thickness_total : ndarray, shape (nSOI,)
    Sum of primary ejecta + local excavation thickness, in m.
  """
  # Precompute per-SOI parameters
  cache = precompute_SOI(
    ds_basin = ds_basin,
    nSOI = nSOI,
    rSOI = rSOI,
    radius_cutoff = radius_cutoff,
    ejecta_model = ejecta_model
  )
  ejecta_model = cache['ejecta_model']
  dist_km = cache['dist_km']
  thickness_primary = cache['thickness_primary']
  thickness_local = np.zeros_like(dist_km)
  for i in range(len(dist_km)):
    # Cache the d_eff and W routines with per-SOI parameters
    _central_effective_depth_i = functools.partial(
      compute_central_effective_depth,
      pre_frag_radius = cache['pre_frag_radius'],
      pre_sec1 = cache['pre_sec_transient_radius1'][i],
      pre_sec2 = cache['pre_sec_transient_radius2'][i],
      exp_sec_transient_radius = cache['exp_sec_transient_radius'],
      pre_deff = cache['pre_central_effective_depth'][i],
    )
    _coverage_frac_i = functools.partial(
      compute_coverage_fraction,
      ejecta_model = ejecta_model,
      pre_frag_radius = cache['pre_frag_radius'],
      exp_sec_transient_radius = cache['exp_sec_transient_radius'],
      ml = cache['mass_lower'][i],
      mh = cache['mass_upper'][i],
      C = cache['mass_norm_constant'][i],
      S = cache['soi_area'][i],
      pthick = thickness_primary[i],
      pre_sec1 = cache['pre_sec_transient_radius1'][i],
      pre_sec2 = cache['pre_sec_transient_radius2'][i],
      pre_deff = cache['pre_central_effective_depth'][i],
    )
    # Check for valid solution
    if (_coverage_frac_i(0) <= ejecta_model.cov):
      raise ValueError(f"Maximum coverage {_coverage_frac_i(0)} is not greater than `cov` ({ejecta_model.cov}) at distance {dist_km[i]:.2f} km; cannot compute local excavation thickness.")
    # Solve for local excavation thickness that yields desired coverage fraction
    Tmax = _central_effective_depth_i(cache['mass_upper'][i])
    sol = root_scalar(
      lambda T: _coverage_frac_i(T) - ejecta_model.cov,
      bracket=[0, Tmax],
      method='bisect'
    )
    thickness_local[i] = sol.root                                    #T_LM_med
  thickness_total = thickness_primary + thickness_local #T_ED_med [*median only if `cov=0.5] -- see text below Eq. 19
  return dist_km, thickness_primary, thickness_local, thickness_total


# Analytical solution for ballistic sedimentation model of Xie et al. (2020)
def compute_thickness_1D_OBSOLETE(
  ds_basin: xr.Dataset,
  theta_0: float = 45,
  rho_e: float = 3000,
  rho_t: float = 3000,
  b: float = 0.98,
  cov: float = 0.5,
  R0: float = 1737e3,
  g: float = 1.62,
  R_sc: float = 9.5e3,
  nSOI: int = 20,
  radius_cutoff: float = None,
  C_mh: float = 0.013,
  b_mt: float = 0.91,
  b_v: float = -5.7,
  material: str = "sand"
):
  """
  Compute 1D radial profiles of primary ejecta thickness, local excavation thickness,
  and total ejecta thickness for a lunar basin.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain:
      - R   : basin rim radius (km)
      - Rat : apparent transient crater radius (km); if NaN, will be estimated.
  theta_0 : float
    Ejection angle of primary fragments (degrees).
  rho_e : float
    Density of primary ejected material, kg m⁻³.
  rho_t : float
    Density of target/local material, kg m⁻³.
  b : float
    Fragment-mass - frequency exponent.
  cov : float
    Desired coverage fraction (0-1) for local excavation.
  R0 : float
    Planetary radius, m.
  g : float
    Surface gravity, m s⁻².
  R_sc : float
    Simple-complex crater transition radius, m. Default is 9.5e3 m for the Moon [from Croft (1985), reported in Xie et al. (2020)].
  nSOI : int
    Number of radial bins (Squares of Interest) to compute.
  radius_cutoff : float or None
    If set, maximum distance = radius_cutoff * Rt_km.
  C_mh, b_mt, b_v : float
    Parameters controlling upper/lower fragment-mass scalings (see below Eq. 7 of Xie et al. (2020)).
  material : {"sand", "hard rock", "soft rock"}
    Controls strength and scaling constants K1, μ, ν, Y.

  Returns
  -------
  dist_km : ndarray, shape (nSOI,)
    Great-circle distances of SOI centers, in km.
  thickness_primary : ndarray, shape (nSOI,)
    Primary ejecta thickness per SOI, in m.
  thickness_local : ndarray, shape (nSOI,)
    Excavation thickness of local target material due to secondary craters, in m.
  thickness_total : ndarray, shape (nSOI,)
    Sum of primary ejecta + local excavation thickness, in m.
  """
  
  ### //Set material parameters [all reported in Xie et al. (2020)]// ###
  if material == "sand":
    K1 = 1.03
    nu= 0.4
    mu = 0.41
    Y = 10e3
  elif material == "hard rock":
    K1 = 0.93
    nu= 0.4
    mu = 0.55
    Y = 10e6
  elif material == "soft rock":
    K1 = 0.93
    nu= 0.4
    mu = 0.55
    Y = 1e6
  else:
    raise ValueError("`material` must be 'sand', 'hard rock', or 'soft rock'.")
  
  ### //Convert ejection angle to radians// ###
  theta_rad    = np.deg2rad(theta_0)
  
  ### //Read in basin parameters// ###
  # Present-day rim radius of basin
  R_km = ds_basin.R.item()
  R_m  = R_km * 1e3
  # Apparent transient radius of basin
  Rat_km = ds_basin.Rat.item()
  if (Rat_km is None) or (not np.isfinite(Rat_km)):
    # Empirical relationship derived from D's of Neumann et al. (2015), Dat's of
    # Miljković et al. (2016), and SPA Dat from Rajšić (2025, personal communication)
    x1_temp, x0_temp = [0.38, 38.18] #empirical relationship INCLUDING SPA
    Rat_km = x1_temp*R_km + x0_temp
    del x1_temp, x0_temp
  Rat_m = Rat_km * 1e3
  # Transient rim radius of basin
  Rt_km = Rat_km * 1.2
  Rt_m  = Rt_km * 1e3
  
  ### //Set up distance array// ###
  # Lower and upper bounds on distance from basin center
  min_dist = R_km #ejecta is deposited beginning roughly at R [loose interpretation of reference to Melosh (1989) in Xie et al. (2020) Section 2.1.2]
  safe_eps = 1e-3 #small epsilon to avoid numerical issues near `Rat`
  min_dist_safe = ( 0.5 * (1 + np.sqrt(1 + 4*Rat_km)) )**2 + safe_eps #minimum distance at which the innermost SOI boundary abutts Rat -- anything smaller leads to an undefined velocity and ejecta thickness
  if min_dist < min_dist_safe:
    warnings.warn(f"\n'{ds_basin.basin.item()}' (R = {R_km:.2f} km) has Rat ({Rat_km:.2f} km) that is too large to use R as minimum distance. Adjusting to minimum safe distance of {min_dist_safe:.2f} km.\nNote that this will accentuate {ds_basin.basin.item()}'s innermost ejecta thickness compared to other basins.\n")
    min_dist = min_dist_safe
  dist_limit = np.pi*(R0/1e3) * .99 #model yields errors very close to the antipode, so cut off profile just shy of π
  # Set upper distance bound based on `radius_cutoff` argument
  if radius_cutoff is None:
    max_dist = dist_limit
  else:
    max_dist = min([radius_cutoff*Rt_km, dist_limit])
  # Generate distance array (each point is the great circle distance of center of a square of interest (SOI) from the basin center)
  dist_km = np.linspace(min_dist, max_dist, nSOI)
  dist_m  = dist_km * 1e3
  
  ### //Define size of each SOI -- Section 2.1.1 of Xie et al. (2020)// ###
  soi_side_km  = 2 * np.sqrt(dist_km)                  #L_SOI [km]
  soi_side_m   = soi_side_km * 1e3                     #L_SOI [m]
  soi_area     = soi_side_m**2                         #S -- see text after Eq. 5 of Xie et al. (2020)

  ### //Define velocity parameters for each SOI -- Eq. 1 of Xie et al. (2020)// ###
  # Helper function (convert great circle distance to ejecta velocity at that distance)
  _launch_offset = lambda d: d - Rat_m                 #R_s [see text above Eq. 1 of Xie et al. (2020)]
  _X_term = lambda d: _launch_offset(d) / (2*R0)       #X [see text below Eq. 1]
  _ejecta_velocity = lambda d: np.sqrt(R0*g*np.tan(_X_term(d))) / (np.sqrt( np.tan(_X_term(d))*np.cos(theta_rad)**2 + np.sin(theta_rad)*np.cos(theta_rad) )) #v(r_gc)
  # Velocity parameters for each SOI
  velocity_soi = _ejecta_velocity(dist_m)              #v(r_gc) -- velocity of primary ejecta in each SOI
  vertical_velocity = velocity_soi * np.sin(theta_rad) #v_⊥ -- ground-perpendicular velocity
  velocity_soi_e = velocity_soi**.38                   #v^0.38 -- see Eq. 16 of Xie et al. (2020)

  ### //Define extents of SOIs// ###
  # Helper function (convert great circle distance to distance on a flat target) -- Eq. 2 of Xie et al. (2020)
  _flat_radius = lambda d: Rat_m + _ejecta_velocity(d)**2 * np.sin(2*theta_rad) / g #r(r_gc)
  # Inner, outer, and mean radial distance of each SOI from the basin center -- see text after Eq. 3 of Xie et al. (2020)
  inner_radius_gc = dist_m - (soi_side_m/2)            #r_gc [inner] -- great circle distance
  inner_radius = _flat_radius(inner_radius_gc)         #r_inner -- flat target distance
  outer_radius_gc = dist_m + (soi_side_m/2)            #r_gc [outer]
  outer_radius = _flat_radius(outer_radius_gc)         #r_outer
  mean_radius  = np.sqrt(inner_radius * outer_radius)  #r-bar -- geometric mean to account for spherical surface
  # Area of basin-concentric ring that encompasses each SOI -- see text below Eq. 4 of Xie et al. (2020)
  sphere_ring_area = 2*np.pi*(R0**2) * (np.cos(inner_radius_gc / R0) - np.cos(outer_radius_gc / R0)) #S_ring
  flat_ring_area   = np.pi * (outer_radius**2 - inner_radius**2) #S_ring_flat
  
  ### //Calculate primary ejecta distribution -- Eqs. 4 & 5 of Xie et al. (2020)// ###
  thickness_primary = (0.068 * Rat_m * (mean_radius / Rat_m)**(-3) * (flat_ring_area / sphere_ring_area)) #δ_SOI -- thickness of primary ejecta per SOI
  mass_primary = thickness_primary * rho_e * soi_area  #M_SOI -- mass of primary ejecta per SOI
  # Total mass of material ejected from primary transient crater -- see text below Eq. 7 of Xie et al. (2020))
  total_mass_primary = 0.09 * rho_e * np.pi * Rat_m**3 #M_T
  
  ### //Largest secondary crater (LSC) parameters// ###
  lsc_distance_m = (7.21 * Rat_km**0.94) * 1e3         #r_LSC -- Eq. 10 of Xie et al. (2020) [**this is only valid for complex and larger craters**]
  velocity_lsc    = _ejecta_velocity(lsc_distance_m)   #v_LSC -- Eq. 8 of Xie et al. (2020)

  ### //Fragment mass parameters// ###
  # Upper and lower bounds on fragment mass -- see text below Eq. 7 of Xie et al. (2020)
  mass_upper = np.full_like(velocity_soi, C_mh*(total_mass_primary**b_mt))
  mask = velocity_soi >= velocity_lsc
  mass_upper[mask] *= (velocity_soi[mask] / velocity_lsc)**(-b_v) #m_h
  mass_lower = 1e-18 * mass_upper                      #m_l
  # Empirical factor for fragment mass-frequency relationship -- Eq. 7 of Xie et al. (2020)
  mass_norm_constant = mass_primary * (1 - b) / (b * (mass_upper**(1 - b) - mass_lower**(1 - b))) #C_SOI
  
  ### //Begin ballistic sedimentation model – compute thickness of local material mixed into total ejecta deposit// ###
  # Initialize array for local material thickness
  thickness_local = np.zeros_like(thickness_primary)   #T_LM
  # Define excavation scaling parameter C_ex from Xie et al. (2020)
  C_ex = 3.5
  # Pre-compute unchanging factors for SOI-loop calculations
  pre_frag_radius = (3 / (4*np.pi*rho_e))**(1/3)
  pre_sec_transient_radius1 = K1**(-(2+mu)/mu) * (g/vertical_velocity**2) * (rho_t/rho_e)**(2*nu/mu)
  pre_sec_transient_radius2 = K1**(-(2+mu)/mu) * (Y/(rho_t*vertical_velocity**2))**((2+mu)/mu) * (rho_t/rho_e)**(nu*(2+mu)/mu)
  exp_sec_transient_radius = -mu / (2+mu)
  pre_central_effective_depth = C_ex * 0.0134 * velocity_soi_e
  # Loop over SOIs
  for i in range(len(dist_m)):
    vel_vert_i = vertical_velocity[i]
    vel_e_i = velocity_soi_e[i]
    ml_i = mass_lower[i]
    mh_i = mass_upper[i]
    C_i = mass_norm_constant[i]
    S_i = soi_area[i]
    pthick_i = thickness_primary[i]
    pre_sec1_i = pre_sec_transient_radius1[i]
    pre_sec2_i = pre_sec_transient_radius2[i]
    pre_deff_i = pre_central_effective_depth[i]

    def _central_effective_depth_i(m):                               #d_eff (Eq. 17 of Xie et al. (2020))
      a = pre_frag_radius * m**(1/3)
      sec_transient_radius = a * (pre_sec1_i*a + pre_sec2_i)**(exp_sec_transient_radius)
      return pre_deff_i * sec_transient_radius
    
    def _coverage_frac_i(T_LM):
      '''
      Eq. 19 of Xie et al. (2020)
      '''
      # Compute the mass where d_eff(m) == T_LM or select the upper/lower mass bound
      deff_min = _central_effective_depth_i(ml_i) #d_eff for the smallest fragment mass
      deff_max = _central_effective_depth_i(mh_i) #d_eff for the largest fragment mass
      if T_LM <= deff_min: #maximum coverage case
        m0 = ml_i
      elif T_LM >= deff_max: #zero coverage case
        return 0
      else:
        # Compute mass `m0` where d_eff(m) == T_LM
        sol = root_scalar(
          lambda m: _central_effective_depth_i(m) - T_LM,
          bracket=[ml_i, mh_i],
          method='bisect'
        )
        m0 = sol.root
      
      def _f(m):
        '''
        Analytical equivalent of Riemann sum in Eq. 19 of Xie et al. (2020)
        '''
        a = pre_frag_radius * m**(1/3)
        R_at = a * (pre_sec1_i*a + pre_sec2_i)**(exp_sec_transient_radius)
        d_eff = pre_deff_i * R_at
        if pthick_i <= d_eff:                                        #1/N_layers * sum( ( (j-1)*δ_SOI / ( N_layers*C_ex*d_ex(R_at(m)) ) ) ) [analytical replacement]
          correction2 = 1 - pthick_i/(2*d_eff)
        else:
          correction2 = d_eff/(2*pthick_i)
        uncorrected_area = np.pi * R_at**2                           #S_PIS [uncorrected] -- π*R_at(m)^2
        correction1 = max(0, 1-(T_LM/d_eff))                         #( 1 - T_LM/( C_ex*d_ex(R_at(m)) ) )
        DN = C_i * b*m**(-b-1)                                       #ΔN [analytical replacement]
        return uncorrected_area * correction1 * correction2 * DN / S_i
      
      def _f_log(x):
        '''
        Convert intermediate solution to logspace for numerical stability in integration
        '''
        m = np.exp(x)
        return _f(m) * m
      
      integral = quad(_f_log, np.log(m0), np.log(mh_i), epsabs=0, epsrel=1e-2)[0]
      W = 1 - np.exp(-integral)
      return W                                                       #W(>T_LM)
    
    if (_coverage_frac_i(0) <= cov): raise ValueError(f"Maximum coverage {_coverage_frac_i(0)} is not greater than `cov` ({cov}).")
    Tmax = _central_effective_depth_i(mh_i) # using max possible excavation depth as upper bound for T_LM -- this is the maximum thickness of local material that could be excavated by any fragment mass in this SOI
    # Integrate Eq. 19 of Xie et al. (2020)
    sol = root_scalar(
      lambda T: _coverage_frac_i(T) - cov,
      bracket=[0, Tmax],
      method='bisect'
    )
    thickness_local[i] = sol.root                                    #T_LM_med
    
    ### *****TODO: implement Eq. 20 of Xie et al. (2020)*****
    
  thickness_total = thickness_primary + thickness_local #T_ED_med [*median only if `cov`=0.5] -- see text below Eq. 19
  return dist_km, thickness_primary, thickness_local, thickness_total


### **2D interpolation of 1D results**
def compute_thickness_2D(clat, clon, d1, p1, l1, t1, lat_grid, lon_grid, R0=config.getfloat(body, 'R0')):
  """
  
  """
  d2 = great_circle_distance(clat, clon, lat_grid, lon_grid, R=R0/1e3)
  p2 = np.interp(d2, d1, p1, left=0.0)
  l2 = np.interp(d2, d1, l1, left=0.0)
  t2 = np.interp(d2, d1, t1, left=0.0)
  return d2, p2, l2, t2


# Multi-basin processing utility
def build_global_ejecta_dataset(
  ds_in: xr.Dataset,
  nSOI: int = 20,
  nlat: int = 200,
  radius_cutoff: float = None,
  R0: float = config.getfloat(body, 'R0'),
  verbose: int = 0
):
  """
  Given a per-basin Dataset `ds_in` with coords: basin, lat, lon, and data_vars clat, clon, R, Rat,
  run compute_thickness_1D() and compute_thickness_2D() for each basin, assemble the results into 
  new 1D and 2D DataArrays, and return an updated Dataset including the global summed thickness.

  Parameters
  ----------
  ds_in : xarray.Dataset
    Must contain coords 'basin' and data_vars 'clat', 'clon', 'R', 'Rat'.
  nSOI : int
    Number of SOI points for the 1D profile.
  nlat : int
    Number of latitude points for the 2D profile (number of longitude points is 2*nlat).
  radius_cutoff : float or None
    If provided, passed to compute_thickness_1D for outer cutoff.
  R0 : float
    Planetary radius (km) for great-circle distance.
      

  Returns
  -------
  ds : xarray.Dataset
    Original `ds_in` with added dims 'lat', 'lon', 'profile_pt' and data_vars:
    dists1d, primary1d, local1d, total1d, dists2d, primary2d, local2d, total2d, global_total.
  """
  
  # Copy Dataset and initialize 2D coordinates
  ds = copy.deepcopy(ds_in)
  ds = ds.assign_coords(lat=np.linspace(-90, 90, nlat), lon=np.linspace(-180+(180/nlat), 180, nlat*2))
  
  # Dimensions
  NB   = ds.sizes["basin"]
  NLAT = ds.sizes["lat"]
  NLON = ds.sizes["lon"]

  # Build lat/lon mesh
  lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")

  # Sample first basin for ND
  example = ds.isel(basin=0)
  d0, p0, l0, t0 = compute_thickness_1D(example, nSOI=nSOI, radius_cutoff=radius_cutoff)
  ND = d0.size

  # Pre-allocate
  dist1d    = np.zeros((NB, ND))
  primary1d = np.zeros((NB, ND))
  local1d   = np.zeros((NB, ND))
  total1d   = np.zeros((NB, ND))
  dist2d    = np.zeros((NB, NLAT, NLON))
  primary2d = np.zeros((NB, NLAT, NLON))
  local2d   = np.zeros((NB, NLAT, NLON))
  total2d   = np.zeros((NB, NLAT, NLON))

  # Loop
  for i, basin_name in enumerate(tqdm(ds.basin.values, desc="Basins")):
    b = ds.sel(basin=basin_name)
    # 1D
    d1, p1, l1, t1 = compute_thickness_1D(b, nSOI=nSOI, radius_cutoff=radius_cutoff)
    dist1d[i]    = d1
    primary1d[i] = p1
    local1d[i]   = l1
    total1d[i]   = t1
    # 2D
    clat = b.clat.item()
    clon = b.clon.item()
    d2, p2, l2, t2 = compute_thickness_2D(clat, clon, d1, p1, l1, t1, lat2d, lon2d, R0=R0)
    if radius_cutoff is not None:
      radius_cutoff_km = radius_cutoff * b.Rat.item()*1.2
      outside_cutoff = d2 > radius_cutoff_km
      p2[outside_cutoff] = 0.0
      l2[outside_cutoff] = 0.0
      t2[outside_cutoff] = 0.0
    dist2d[i]    = d2
    primary2d[i] = p2
    local2d[i]   = l2
    total2d[i]   = t2

  # Build new coords and data_vars
  ds = ds.assign_coords(profile_pt=np.arange(ND))
  ds = ds.assign(
    dists1d    = (("basin","profile_pt"), dist1d),
    primary1d  = (("basin","profile_pt"), primary1d),
    local1d    = (("basin","profile_pt"), local1d),
    total1d    = (("basin","profile_pt"), total1d),
    dists2d    = (("basin","lat","lon"),    dist2d),
    primary2d  = (("basin","lat","lon"),    primary2d),
    local2d    = (("basin","lat","lon"),    local2d),
    total2d    = (("basin","lat","lon"),    total2d),
  )

  # Global sum
  global_total = ds.total2d.sum(dim="basin").values

  
  ds = ds.assign(
    global_total = (('lat', 'lon'), global_total)
  )

  return ds


# Post-process a global multi-basin dataset to simulate megaregolith compaction
def global_ejecta_compacted_ordered(
    ds: xr.Dataset,
    ordered_basins: list[str],
    verbose: bool = False
) -> xr.Dataset:
  """
  Build a “compacted” global ejecta-thickness map by layering basins in a user-specified order.
  For each basin in `ordered_basins`, its 2D thickness is added everywhere, then any pixel
  lying within that basin's rim is zeroed—so that interior regions of later basins
  destroy/excavate previous ejecta.

  Parameters
  ----------
  ds : xarray.Dataset
    Must contain all of:
    - coords:
      * 'basin'   : the basin names
      * 'lat', 'lon' : spatial grid
    - data_vars:
      * 'total2d' : (basin, lat, lon) array of per-basin ejecta thickness
      * 'dists2d' : (basin, lat, lon) array of great-circle distance from each basin center
      * 'R'       : (basin,) basin radius (same units as dists2d)
      * 'global_total' : (lat, lon) existing global sum (will be replaced)
  ordered_basins : iterable of str
    The sequence of basin names (strings) in the exact order you want to “compact” them.
    Basins later in the list will erase any ejecta under their rims from prior basins.

  Returns
  -------
  xarray.Dataset
    A copy of `ds` with its `'global_total'` data variable replaced by the new compacted
    thickness map (shape `(lat, lon)`).
  """
  
  # start with a blank lat×lon grid
  total_compacted = np.zeros_like(ds.global_total.values)

  for b in tqdm(ordered_basins, desc="Compacting basins"):
    ### TODO: Account for (1) "local" material that is actually preexisting megaregolith (2) "primary" material that is actually preexisting megaregolith

    # 1) fetch this basin’s ejecta thickness and distance grids
    # basin_thick  = ds.total2d.sel(basin=b).values #m (lat, lon) -- thickness of basin's ejecta at each grid point
    basin_primary = ds.primary2d.sel(basin=b).values #m (lat, lon) -- thickness of basin's primary ejecta at each grid point
    basin_local   = ds.local2d.sel(basin=b).values #m (lat, lon) -- thickness of local material in basin's deposit at each grid point
    basin_dists  = ds.dists2d.sel(basin=b).values #km (lat, lon) -- distance of each grid point from basin center
    basin_radius = ds.R.sel(basin=b).item()       #km -- final basin rim radius

    # 2) add the primary material
    total_compacted += basin_primary

    # 3) account for "local" material that is actually preexisting megaregolith
    added_local = basin_local - total_compacted
    added_local[added_local < 0] = 0.0
    if verbose and np.sum(added_local) == 0:
      print(f"Basin {b} did not add local material.")
    total_compacted += added_local

    # 4) account for "primary" material that is actually preexisting megaregolith
    ### TODO: need to determine (1) how much of the primary ejecta is preexisting megaregolith (2) how to spatially distribute that
    warnings.warn("`global_ejecta_compacted_ordered()` does not yet account for preexisting megaregolith in primary ejecta.")

    # 5) zero out points whose distance ≤ basin_radius
    mask = (basin_dists <= basin_radius)
    total_compacted[mask] = 0.0

  # write the new global_total back into a copy of ds
  return ds.assign(global_total=(('lat', 'lon'), total_compacted))
