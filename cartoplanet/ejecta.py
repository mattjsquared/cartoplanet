import numpy as np
import xarray as xr
import copy, warnings, tqdm

from scipy.integrate import quad
from scipy.optimize import root_scalar
from cartoplanet import config

body = config['BODY']['body']




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


# Analytical solution for ballistic sedimentation model of Xie et al. (2020)
def compute_thickness_1D(
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
    If set, maximum distance = radius_cutoff × Rt_km.
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
#     x1_temp, x0_temp = [0.37, 39.74] #empirical relationship EXCLUDING SPA
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
      deff_min = _central_effective_depth_i(ml_i) #effective depth for the smallest fragment mass
      deff_max = _central_effective_depth_i(mh_i) #effective depth for the largest fragment mass
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
        if pthick_i <= d_eff:                                       #1/N_layers * sum( ( (j-1)*δ_SOI / ( N_layers*C_ex*d_ex(R_at(m)) ) ) )
          correction2 = 1 - pthick_i/(2*d_eff)
        else:
          correction2 = d_eff/(2*pthick_i)
        uncorrected_area = np.pi * R_at**2                           #S_PIS [uncorrected] -- π*R_at(m)^2
        correction1 = max(0, 1-(T_LM/d_eff))                         #( 1 - T_LM/( C_ex*d_ex(R_at(m)) ) )
        DN = C_i * b*m**(-b-1)                                       #ΔN
        return uncorrected_area * correction1 * correction2 * DN / S_i
      
      def _f_log(x):
        '''
        Convert intermediate solution to logspace for numerical stability in integration
        '''
        m = np.exp(x)
        return _f(m) * m
      
      integral = quad(_f_log, np.log(m0), np.log(mh_i), epsabs=0, epsrel=1e-2)[0]
      W = 1 - np.exp(-integral)
      return W                                                      #W(>T_LM)
    
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
  for i, basin_name in enumerate(tqdm.tqdm(ds.basin.values, desc="Basins")):
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

  for b in tqdm.tqdm(ordered_basins, desc="Compacting basins"):
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
