import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import copy, warnings, functools, os

from scipy.integrate import quad, cumulative_simpson
from scipy.optimize import root_scalar
from tqdm import tqdm
from cartoplanet import config

body = config['BODY']['body']
TEST = True


### General utility
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


def Xie_figure5(grid=False) -> tuple:
  """
  Code to reproduce Figure 5 of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Reproduced using an adapted semi-analytical approach in Python by Matt J. Jones 2026.
  """
  ### //Basin definition//
  basin_name = "Orientale"
  basin_lat = -19.83
  basin_lon = -94.58
  basin_Dat = 418
  basin_Rat = basin_Dat / 2
  ds = xr.Dataset(
    {
      'R':   (('basin',), [basin_Rat * 1.3]),
      'Rat': (('basin',), [basin_Rat]),
      'clat': (('basin',), [basin_lat]),
      'clon': (('basin',), [basin_lon]),
      'order': (('basin',), [0])
    },
    coords = {
      'basin': [basin_name],
    }
  )
  coord_A16 = (-8.973, 15.5)
  ### //Initialize for ejecta emplacement//
  # Ejecta model
  ejecta_model   = EjectaModel()
  ejecta_model.Y = 10e6
  # Prep SOI conditions
  rSOI     = great_circle_distance(basin_lat, basin_lon, coord_A16[0], coord_A16[1])
  cache    = precompute_SOI(ds, rSOI=[rSOI], ejecta_model=ejecta_model)
  Tprimary = cache['thickness_primary'][0]
  ### //Compute vertical mixing//
  ds_profile = compute_ejecta_mixing_multi_basin(ds, coord_A16[0], coord_A16[1], ejecta_model=ejecta_model)
  elevation  = ds_profile['elevation'].values
  abundances = ds_profile['abundance'].values.squeeze()
  frac_excavatedlocal = get_coverage_fraction(
    depths      = np.maximum(-elevation, 0),
    layer_mode  = 'all',
    soi_cache   = cache,
    i_soi       = 0,
  )
  ### //Plot//
  with plt.rc_context({
      'font.family': 'Myriad Pro',
      'figure.dpi': 300,
      'lines.linewidth': 1.5,
  }):
    fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    # Coverage fraction vs depth
    ax[0].semilogx(
      -elevation, 
      frac_excavatedlocal * 100, 
      color = [0, 0, 0]
    )
    ax[0].set_xlabel("Depth from pre-impact surface (m)")
    ax[0].set_ylabel("Fraction of excavated local materials (%)")
    ax[0].set_xlim((1e0, 2e3))
    ax[0].set_ylim((0, 100))
    # Orientale vs. local abundance vs depth
    ax[1].semilogx(
      -elevation + Tprimary, 
      ds_profile.sel(basin='Orientale')['abundance'].values.squeeze() * 100, 
      color = [1, 0, 0],
      linewidth = 2,
      label = "Orientale ejecta"
    )
    ax[1].semilogx(
      -elevation + Tprimary, 
      ds_profile.sel(basin='preimpact')['abundance'].values.squeeze() * 100, 
      color = [0, 0, 0], 
      linewidth = 2, 
      label = "Pre-Orientale materials"
    )
    ax[1].vlines(Tprimary, 0, 100, color=[0, 0, 1], linestyle='--', linewidth=.7, label="Pre-impact surface")
    ax[1].set_xlabel("Depth from surface (m)")
    ax[1].set_ylabel("Abundance of materials in deposits (%)")
    ax[1].set_xlim((1e0, 2e3))
    ax[1].set_ylim((0, 100))
    ax[1].legend()
    # Formatting
    for a in ax:
      a.set_yticks(np.linspace(0, 100, 11))
      a.minorticks_on()
      a.tick_params(which='both', direction='in', right=True, top=True)
      if grid:
        a.grid()
    plt.show()
  return Tprimary, frac_excavatedlocal, abundances, elevation


def Xie_figure10c(grid=False) -> tuple:
  """
  Code to reproduce Figure 10c of Xie et al. (2020):
  Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples. Journal of Geophysical Research: Planets, 125, e2019JE006113. https://doi.org/10.1029/2019JE006113.

  Adapted to a semi-analytical approach by Matt J. Jones 2026.
  
  Originally created with MATLAB R2016b:
  Xie, M. Liu, T. and Xu, A. (2020), Ballistic sedimentaiton model [Code], Zenodo. https://doi.org/10.5281/zenodo.3692887.
  """
  ### //Basin definitions//
  basin_names = ["Nectaris", "Humorum", "Crisium", "Serenitatis", "Imbrium", "Orientale"]
  basin_lats  = [-16.15, -24.28, 18.03, 26.57,  34.71, -19.83]
  basin_lons  = [ 34.59, -39.35, 60.12, 18.05, -17.07, -94.58]
  basin_Dats  = [339, 300, 370, 350, 402, 418]
  ds_basin = xr.Dataset(
    {
      'order': (('basin'), range(len(basin_names))),
      'clat': (('basin'), basin_lats),
      'clon': (('basin'), basin_lons),
      'Dat':  (('basin'), basin_Dats)
    },
    coords = {
      'basin': basin_names
    }
  )
  ds_basin = ds_basin.assign(Rat = lambda ds: ds['Dat'] / 2)
  ds_basin = ds_basin.assign(R = lambda ds: ds['Rat'] * 1.3)
  coord_A16   = (-8.973, 15.5)
  ### //Initialize variables//
  # Elevation grid
  dz        = 0.05
  max_depth = 1e4
  nz        = int(max_depth / dz) + 1
  elevation = None
  # Ejecta model
  ejecta_model   = EjectaModel(material="sand")
  ejecta_model.Y = 10e6
  ### //Emplace basins chronologically//
  ds_profile = compute_ejecta_mixing_multi_basin(ds_basin, coord_A16[0], coord_A16[1], elevation, ejecta_model, preimpact_label="Pre-Nectarian")
  elevation = ds_profile['elevation'].values
  ### //Plot//
  depth = -elevation + ds_profile['total_thickness'].item()
  with plt.rc_context({
      'font.family': 'Myriad Pro',
      'figure.dpi': 300,
      'lines.linewidth': 1.5,
  }):
    fig, ax = plt.subplots(1, 1, figsize=(6.67, 4))
    C1 = ['#888888', '#ff0000', '#00ffff', '#0000ff', '#000000', '#ff00ff', '#00ff00']
    # Plot basins
    for i, b in enumerate(ds_profile['basin'].values):
      basin = ds_profile.sel(basin=b)
      if i == 0:
        ax.loglog(
          depth,
          (1 - ds_profile.sel(basin='Nectaris')['abundance_iflast'].values.squeeze()) * 100,
          color = C1[i],
          linestyle = '--'
        )
      else:
        ax.loglog(
          depth,
          basin['abundance_iflast'].values.squeeze() * 100,
          color = C1[i],
          linestyle = '--'
        )
      ax.loglog(
        depth,
        basin['abundance'].values.squeeze() * 100,
        color = C1[i]
      )
      ax.text(1800, 7 * 2**(0.5 * i), b, color=C1[i])
    # Formatting
    ax.set_ylim([0.1, 100])
    ax.set_xlim([0.1, 20000])
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], labels=['1', '2', '5', '10', '20', '50', '100'])
    ax.set_xlabel("Depth from surface (m)")
    ax.set_ylabel("Abundance of basin ejecta in deposits (%)")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    if grid:
      ax.grid()
    plt.show()
  return ds_profile


### Backend for mass-continuous ballistic sedimentation
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
  ejecta_model: dict | EjectaModel = None,
  verbose: bool | int = False
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
  verbose : bool or int
    If True or >0, print additional information.

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
  if (min_dist < min_dist_safe) and (verbose > 1):
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
      raise ValueError(f"{ds_basin.basin.item()} (R={R_km:.2f}km, Rat={Rat_km:.2f}km): All values in `rSOI` must be between {min_dist:.2f} km and {max_dist:.2f} km. Invalid values are {dist_km[(dist_km < min_dist) | (dist_km > max_dist)]}.")
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
    cache: dict,
    i_soi: int,
) -> float:
  """
  Maximum (i.e., central) excavation depth of the crater formed by a fragment of mass `m`.
  Eq. 17 of Xie et al. (2020).

  Parameters
  ----------
  m : float
    Mass of the fragment (g).
  cache : dict
    Dictionary containing pre-computed per-SOI parameters for ballistic sedimentation calculations, as 
    returned by `precompute_SOI()`.
  i_soi : int
    Index for arrays in `cache` corresponding to the SOI in which to do the calculation.
  
  Returns
  -------
  deff : float
    Central effective excavation depth for a fragment of mass `m` in the SOI with index `i_soi`.
  """
  pre_frag_radius          = cache['pre_frag_radius']
  exp_sec_transient_radius = cache['exp_sec_transient_radius']
  pre_sec1                 = cache['pre_sec_transient_radius1'][i_soi]
  pre_sec2                 = cache['pre_sec_transient_radius2'][i_soi]
  pre_deff                 = cache['pre_central_effective_depth'][i_soi]
  a = pre_frag_radius * m**(1/3)
  sec_transient_radius = a * (pre_sec1*a + pre_sec2)**(exp_sec_transient_radius)
  return pre_deff * sec_transient_radius


def compute_coverage_kernel(
    layer_mode: str,
    soi_cache: dict,
    i_soi: int,
    nmass: int = 2048,
    depth_in: np.ndarray = None,
    coverage_in: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Fast vectorized approximation of mass-continuous forms of Eq. 19 or 20 of Xie et al. (2020).
  Can be used to build a mass-space coverage kernel (if `depth_in` and `coverage_in` are both None) 
  or to compute (1) coverage fraction(s) at `depth_in` or (2) depth(s) at `coverage_in`.

  Parameters
  ----------
  layer_mode : {"all", "one"}
    Whether to compute coverage fraction assuming all layers of primary ejecta are emplaced 
    ("all", corresponding to Eq. 19 of Xie et al. (2020)) or assuming only one layer of primary 
    ejecta is emplaced ("one", corresponding to Eq. 20 of Xie et al. (2020)).
  soi_cache : dict
    Dictionary containing pre-computed per-SOI parameters for ejecta thickness calculations, as 
    returned by `precompute_SOI()`.
  i_soi : int
    Index of the SOI for which to compute the coverage kernel.
  nmass : int
    Number of mass bins to use for numerical integration. Higher values yield more accurate results 
    but increase computation time. Default is 2048, which provides a good balance of accuracy and 
    speed for typical use cases.
  depth_in : ndarray, optional
    If specified, `coverage_in` must be None and `coverage_fraction` will be computed exactly at 
    the input depths. If None (default), `depth` will be computed as a mass-space grid of effective 
    excavation depths, and `coverage_fraction` will be computed at those depths.
  coverage_in : ndarray, optional
    If specified, `depth_in` must be None and `depths` will be interpolated from a mass-space grid 
    of effective excavation depths. If None (default), `coverage_fraction` will be computed on `depth`.
  
  Returns
  -------
  depth_out : ndarray
    Array of grid point depths. If `depth_in` is None, this is a mass-space grid of effective 
    excavation depths. Otherwise, returns `depth_in`.
  coverage_out : ndarray
    Array of coverage fractions computed at the points in `depth_out`.
  """
  ### //Process inputs//
  # Normalize parameters
  if layer_mode not in ['all', 'one']:
    raise ValueError("`layer_mode` must be 'all' [for coverage after full primary ejecta deposit is emplaced] or 'one' [for coverage after a single layer is emplaced].")
  if (depth_in is not None) and (coverage_in is not None):
    raise ValueError("At least one of `depth_in` or `coverage_in` must be None.")
  if coverage_in:
    coverage_in = np.asarray(coverage_in)
    if np.any((coverage_in < 0) | (coverage_in > 1)):
      raise ValueError("All values in `coverage_in` must be between 0 and 1.")
  pre_frag_radius = soi_cache['pre_frag_radius']
  exp_sec_transient_radius = soi_cache['exp_sec_transient_radius']
  ml       = soi_cache['mass_lower'][i_soi]
  mh       = soi_cache['mass_upper'][i_soi]
  C        = soi_cache['mass_norm_constant'][i_soi]
  S        = soi_cache['soi_area'][i_soi]
  pthick   = soi_cache['thickness_primary'][i_soi]
  pre_sec1 = soi_cache['pre_sec_transient_radius1'][i_soi]
  pre_sec2 = soi_cache['pre_sec_transient_radius2'][i_soi]
  pre_deff = soi_cache['pre_central_effective_depth'][i_soi]
  if mh <= ml or C <= 0 or S <= 0:
    raise ValueError("Invalid mass bounds, normalization constant, or SOI area in `soi_cache`.")
  ### //Prepare for the vectorized integral//
  # Calculate mass-discretized excavation depths
  b      = soi_cache['ejecta_model'].b
  mspace = np.logspace(np.log10(ml), np.log10(mh), int(nmass))
  a      = pre_frag_radius * mspace**(1/3)
  R_at   = a * (pre_sec1*a + pre_sec2)**(exp_sec_transient_radius)
  deff   = pre_deff * R_at
  # Define depth grid based on input
  depths = deff if depth_in is None else depth_in
  # Calculate pre-integral mass-discretized exponent of Eq. 19/20 of Xie et al. (2020)
  area = np.pi * R_at**2
  DN   = C * b * mspace**(-b - 1)     #implicitly includes the sum over layers from Eq. 19 of Xie et al. (2020)
  if layer_mode == 'all':             #shielding effect for mass-continuous form of Eq. 19 of Xie et al. (2020)
    if pthick == 0:
      layer_correction = np.ones_like(deff)
    elif pthick < 0:
      raise ValueError("Primary ejecta thickness must be non-negative.")
    else:
      layer_correction = np.where(pthick <= deff, 1 - pthick/(2*deff), deff/(2*pthick))
  elif layer_mode == 'one':           #mass-continuous form of Eq. 20 of Xie et al. (2020)
    deff_max           = deff[-1]
    Tprimary_perlayer  = max(deff_max / 5000, pthick / 100)
    N_layers           = int(np.ceil(pthick / Tprimary_perlayer))
    layer_correction   = 1 / N_layers #this cancels the implicit sum over layers in `DN`
  base = area * layer_correction * DN / S
  ### //Decompose the (1 - T_LM/d_eff) depth correction into two integrals to enable vectorization//
  # Compute reverse cumulative integrals from mh down to each m (avoids "catastrophic cancellation" from forward integral)
  # `cumulative_simpson` needs monotonically increasing x, so to do the reverse integral we negate and flip
  neg_mspace = -np.flip(mspace)                                                        #strictly increasing: -mh, ..., -ml
  base_rev   = np.flip(base)
  deff_rev   = np.flip(deff)
  I0_rev = np.flip(cumulative_simpson(base_rev,            x=neg_mspace, initial=0.0)) #(reverse) integral of base       from m to mh
  I1_rev = np.flip(cumulative_simpson(base_rev / deff_rev, x=neg_mspace, initial=0.0)) #(reverse) integral of base/deff  from m to mh
  I0_tot = I0_rev[0]
  I1_tot = I1_rev[0]
  # Interpolate reverse cumulative integrals at m0 corresponding to each query depth
  deff_min = deff[0]
  deff_max = deff[-1]
  m0       = np.interp(depths, deff, mspace, left=ml, right=mh)
  I0_hi    = np.interp(m0, mspace, I0_rev)
  I1_hi    = np.interp(m0, mspace, I1_rev)
  # Calculate the exponent: integral from m0 to mh of base*(1 - T_LM/d_eff)
  exponent = I0_hi - depths * I1_hi
  exponent = np.where(depths >= deff_max, 0.0, exponent)              #zero coverage beyond deepest excavation
  shallow_exponent = I0_tot - depths * I1_tot                         #full integral for depths shallower than smallest fragment
  exponent = np.where(depths <= deff_min, shallow_exponent, exponent)
  exponent = np.maximum(exponent, 0.0)                                #clamp: exponent is analytically non-negative, but there are sometimes numerical artifacts
  ### //Return the coverage fraction at each depth//
  coverage_fraction = 1 - np.exp(-exponent)
  if any(coverage_fraction < 0) or any(coverage_fraction > 1):
    idx_out_of_bounds = (coverage_fraction < 0) | (coverage_fraction > 1)
    raise ValueError(f"At least one (n={idx_out_of_bounds.sum()}; depths: {depths[idx_out_of_bounds]}) coverage fraction is out of bounds (should be between 0 and 1).")
  if coverage_in is not None:
    coverage_out = coverage_in
    depth_out = np.interp(coverage_out, np.flip(coverage_fraction), np.flip(depths), left=deff_max, right=0.0)
  else:
    depth_out = depths
    coverage_out = coverage_fraction
  return depth_out, coverage_out


def get_coverage_fraction(
    depths: np.ndarray,
    layer_mode: str,
    soi_cache: dict,
    i_soi: int,
    nmass: int = 2048,
) -> np.ndarray:
  """
  Returns coverage fractions at specified depths by interpolating the output of `compute_coverage_kernel`.

  Parameters
  ----------
  depths : ndarray
    Depths at which to compute coverage fractions. Specified with respect to the pre-impact surface, so must be non-negative.
  layer_mode : {"all", "one"}
    Whether to compute coverage fraction assuming all layers of primary ejecta are emplaced ("all", corresponding to Eq. 19 of Xie et al. (2020)) or assuming only one layer of primary ejecta is emplaced ("one", corresponding to Eq. 20 of Xie et al. (2020)).
  soi_cache : dict
    Dictionary containing pre-computed per-SOI parameters for ejecta thickness calculations, as returned by `precompute_SOI()`.
  i_soi : int
    Index of the SOI for which to compute the coverage fraction.
  nmass : int
    Number of mass bins to use for numerical integration in `compute_coverage_kernel_fast`. Higher values yield more accurate results but increase computation time. Default is 2048, which provides a good balance of accuracy and speed for typical use cases.

  Returns
  -------
  coverage_fraction : ndarray
    Array of coverage fractions corresponding to each input depth.
  """
  if any(depths < 0):
    raise ValueError("Depths must be non-negative.")
  _, W = compute_coverage_kernel(layer_mode, soi_cache, i_soi, nmass, depth_in=depths)
  return W


def get_coverage_depth(
    coverage_fractions: float | np.ndarray,
    layer_mode: str,
    soi_cache: dict,
    i_soi: int,
    nmass: int = 2048,
) -> np.ndarray:
  """
  Returns coverage depths corresponding to specified coverage fractions by interpolating the output of 
  `compute_coverage_kernel`.

  Parameters
  ----------
  coverage_fractions : float | np.ndarray
    Coverage fractions at which to compute corresponding depths. Must be between 0 and 1.
  layer_mode : {"all", "one"}
    Whether to compute coverage fraction assuming all layers of primary ejecta are emplaced ("all", 
    corresponding to Eq. 19 of Xie et al. (2020)) or assuming only one layer of primary ejecta is emplaced 
    ("one", corresponding to Eq. 20 of Xie et al. (2020)).
  soi_cache : dict
    Dictionary containing pre-computed per-SOI parameters for ejecta thickness calculations, as returned 
    by `precompute_SOI()`.
  i_soi : int
    Index of the SOI for which to compute the coverage depth.
  nmass : int
    Number of mass bins to use for numerical integration in `compute_coverage_kernel`. Higher values 
    yield more accurate results but increase computation time. Default is 2048, which provides a good 
    balance of accuracy and speed for typical use cases.

  Returns
  -------
  depths : float | np.ndarray
    Depths corresponding to each input coverage fraction.
  """
  if any(coverage_fractions < 0) or any(coverage_fractions > 1):
    raise ValueError("Coverage fractions must be between 0 and 1.")
  depths, _ = compute_coverage_kernel(layer_mode, soi_cache, i_soi, nmass, coverage_in=coverage_fractions)
  return depths


### Vertical mixing from ballistic sedimentation
def build_mixing_kernel(
    cache: dict,
    i_soi: int,
) -> dict | None:
  """
  Build the per-layer depth-excavation coverage kernel for a single SOI and basin.
  The kernel can be referenced when computing layer-by-layer vertical ejecta mixing.

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
  pthick          = cache['thickness_primary'][i_soi] #[m]
  # No mixing if no primary ejecta is deposited
  if pthick == 0:
    return None
  elif pthick < 0:
    raise ValueError(f"Primary ejecta thickness is <0 ({pthick} m).")
  ### //Predefine d_eff function for this SOI's parameters//
  deff_max = compute_central_effective_depth(
    m     = cache['mass_upper'][i_soi], #[g]
    cache = cache,
    i_soi = i_soi,
  )
  ### //Calculate primary ejecta deposit layer parameters//
  Tprimary_onelayer = max(deff_max / 5000, pthick / 100)           #[m]
  N_layers = int(np.ceil(pthick / Tprimary_onelayer))
  Tprimary_onelayer = pthick / N_layers                            #[m]
  elevation_layerbottoms = np.arange(N_layers) * Tprimary_onelayer #[m]
  ### // Define depth grid//
  zmax        = deff_max                                                     #[m]
  mixing_grid = np.arange(-Tprimary_onelayer / 2, -zmax, -Tprimary_onelayer) #[m]
  dz          = Tprimary_onelayer                                            #[m]
  ### //Compute mixing kernel//
  T_LM_values  = -mixing_grid               #[m] -- make values positive
  Wmz_onelayer = np.zeros_like(T_LM_values) #[area fraction]
  Wmz_onelayer[:] = get_coverage_fraction(
    depths = T_LM_values,
    layer_mode = 'one',
    soi_cache = cache,
    i_soi = i_soi,
    nmass = 2048,
  )
  # Ensure no invalid values
  if np.any(Wmz_onelayer < 0) or np.any(Wmz_onelayer > 1):
    raise ValueError("Computed coverage fractions are out of bounds [0, 1]. Check the integration and input parameters.")
  ### //Return the kernel//
  Texcavated_onelayer = np.sum(dz * Wmz_onelayer) #[m]
  return {
    'primary_thickness':      pthick,
    'Wmz_onelayer':           Wmz_onelayer,
    'mixing_grid':            mixing_grid,
    'dz':                     dz,
    'Texcavated_onelayer':    Texcavated_onelayer,
    'Tprimary_onelayer':      Tprimary_onelayer,
    'N_layers':               N_layers,
    'elevation_layerbottoms': elevation_layerbottoms,
    'zmax':                   zmax,
  }


def compute_ejecta_mixing(
    kernel: dict,
    elevation: np.ndarray,
    abundances: np.ndarray = None,
    surface_elevation: float = 0.0,
) -> np.ndarray:
  """
  Compute vertical mixing of primary ejecta with local material for a single SOI using a pre-built 
  kernel from `build_mixing_kernel`. Accounts for multiple components if `abundances` is 2-D.

  This is a semi-analytical pipeline for the layer-by-layer mixing algorithm of Xie et al. (2020).

  Parameters
  ----------
  kernel : dict
    Output of `build_mixing_kernel`.
  elevation : ndarray, shape (m_elev,)
    Grid of elevation w.r.t. pre-impact surface.
  abundances : ndarray, shape (m_elev, n_existing) or None
    Abundance of each pre-existing ejecta component versus `elevation`.
    Passing `None` (or an empty array) computes only this basin's mixing profile.
  surface_elevation : float
    Elevation of the surface before deposition of this basin's ejecta. Used to adjust for a non-zero
    elevation of the pre-impact surface. Default is 0 m. 

  Returns
  -------
  new_abundances : ndarray, shape (m_elev, n_existing + 1)
    Updated abundances after mixing.  The last column is the newly emplaced
    primary ejecta component.
  """
  ### //Handle case of no primary ejecta, or compute mixing//
  if kernel is None:
    if abundances is None:
      return np.zeros((len(elevation), 1))
    else:
      return np.column_stack([abundances, np.zeros(len(elevation))])
  ### //Fetch parameters from kernel//
  Tprimary               = kernel['primary_thickness']      #[m]
  Wmz_onelayer           = kernel['Wmz_onelayer']           #[area fraction]
  mixing_grid            = kernel['mixing_grid']            #[m] -- w.r.t. pre-impact surface
  dz                     = kernel['dz']                     #[m]
  Texcavated_onelayer    = kernel['Texcavated_onelayer']    #[m]
  Tprimary_onelayer      = kernel['Tprimary_onelayer']      #[m]
  N_layers               = kernel['N_layers']
  zmax                   = kernel['zmax']                   #[m]
  ### //Define interpolation helper function to prevent invalid values//
  def _safe_interp(x, xp, fp, left=None, right=None):
    result = np.interp(x, xp, fp, left=left, right=right)
    if np.any(result > 1) or np.any(result < 0):
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
  elevation_mixinggrid  = np.concatenate((elevation_deposit, mixing_grid))                   #[m]
  # Correct grid for surface elevation
  elevation_mixinggrid    += surface_elevation                                                        #[m]
  elevation_initialsurface = surface_elevation                                                        #[m] -- elevation of the surface before any of this basin's ejecta is deposited
  ### //Initialize preexisting abundances on the mixing grid//
  idx_mixingzone                             = np.where((elevation_mixinggrid > (elevation_initialsurface - zmax)) & (elevation_mixinggrid < elevation_initialsurface))[0] #starting indices of the moving mixing-zone window for `elevation_mixinggrid`
  abundances_mixinggrid                      = np.zeros((len(elevation_mixinggrid), n_component)) #[area fraction]
  for k in range(idx_newcomponent): #initialize the first layer's mixing zone with preexisting abundances
    abundances_mixinggrid[idx_mixingzone, k] = _safe_interp(elevation_mixinggrid[idx_mixingzone], np.flip(elevation), np.flip(abundances_2d[:, k]), right=0) #[area fraction] -- abundances of preexisting components within mixing zone for the first layer
  ### //Compute vertical mixing//
  # Convert fancy index to slice for fast view-based access (mixing zone is always contiguous)
  mz_start = int(idx_mixingzone[0])
  mz_stop  = int(idx_mixingzone[-1]) + 1
  n_mix    = mz_stop - mz_start
  Wmz_col  = Wmz_onelayer[:n_mix, None]                                              #(n_mix, 1) for broadcasting
  dz_Wmz   = dz * Wmz_col                                                            #pre-multiply constant factors
  one_m_W  = 1.0 - Wmz_col                                                           #pre-compute (1 - W)
  Ttotal_onelayer = Tprimary_onelayer + Texcavated_onelayer                           #[m] -- constant across layers
  inv_Ttotal      = 1.0 / Ttotal_onelayer                                             #pre-compute reciprocal
  newfrac_k       = np.empty(n_component)                                             #pre-allocate once
  for i in range(N_layers): #emplace primary ejecta layer-by-layer
    # Compute excavated thicknesses for ALL components at once (slice = view, no copy)
    window = abundances_mixinggrid[mz_start:mz_stop]                                  #(n_mix, n_component) view
    Texcavated_k = (dz_Wmz * window).sum(axis=0)                                     #(n_component,)
    # Compute deposited fractions
    newfrac_k[:idx_newcomponent] = Texcavated_k[:idx_newcomponent] * inv_Ttotal
    newfrac_k[idx_newcomponent]  = (Tprimary_onelayer + Texcavated_k[idx_newcomponent]) * inv_Ttotal
    # Update mixing zone in-place — all components at once
    window[:] = window * one_m_W + newfrac_k * Wmz_col
    # Deposit layer
    abundances_mixinggrid[mz_start - 1, :] = newfrac_k
    # Shift the mixing zone up for the next layer
    mz_start -= 1
    mz_stop  -= 1
  ### //Stack the new/mixed deposit on top of the original elevation grid and interpolate results onto that grid//
  # Identify elevations outside the mixing zone where extrapolation would be invalid
  deep_mask      = elevation < np.min(elevation_mixinggrid) # below mixing zone on the original elevation grid
  new_abundances = np.zeros((len(elevation), n_component))  #[area fraction]
  # Interpolate onto the original grid
  deposit_top   = surface_elevation + Tprimary
  above_deposit = elevation > deposit_top
  for k in range(n_component):
    new_abundances[:, k] = _safe_interp(elevation, np.flip(elevation_mixinggrid), np.flip(abundances_mixinggrid[:, k])) #[area fraction]
    # Zero above the deposit top (where no material has been placed yet)
    new_abundances[above_deposit, k] = 0.0
    # Below the mixing zone, restore original abundances for preexisting components
    if k < idx_newcomponent:
      new_abundances[:, k] = np.where(deep_mask, abundances_2d[:, k], new_abundances[:, k])
  return new_abundances


def compute_ejecta_mixing_multi_basin(
    ds_basin: xr.Dataset,
    profile_lat: float,
    profile_lon: float,
    elevation: np.ndarray = None,
    ejecta_model: dict | EjectaModel = None,
    preimpact_label: str = None,
    verbose: bool = False,
) -> xr.Dataset:
  """
  Run a vertical mixing simulation at a coordinate given a chronological sequence of basin-forming impacts.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain dimensions:
    - basin : name of each basin
    and variables:
    - order : stratigraphic order of each basin (ascending)
    - clat  : latitude of each basin center (degrees)
    - clon  : longitude of each basin center (degrees)
    - R     : rim radius of each basin (km)
    - Rat   : apparent transient radius of each basin (km)
  profile_lat, profile_lon : float
    Coordinates of the point at which to compute the vertical mixing profile.
  elevation : optional, ndarray or None
    Elevation grid in meters for the vertical mixing profile. If `None`, elevation grid will be handled automatically.
  ejecta_model : dict or EjectaModel
    If dict, can contain keys as described in the EjectaModel class. If `None`, default parameters from the EjectaModel class will be used.
  return_caches : bool
    Whether to return a copy of `ds_basin` with pre-computed SOI caches for each basin. Default is False.
  preimpact_label : str, optional
    Name for the mixing component corresponding to local materials that predate the first impact. If `None`, defaults to 'preimpact'.
  verbose : bool
    Whether to print progress bars for the pre-computation and mixing steps. Default is False.
  
  Returns
  -------
  ds_profile : xr.Dataset
    Dataset containing the vertical mixing profile at the specified location, with dimensions:
    - lat       : latitude of the profile point (degrees)
    - lon       : longitude of the profile point (degrees)
    - elevation : elevation of grid points w.r.t. pre-impact surface (m)
    - basin     : name of each basin involved in mixing, plus "preimpact"
    and variables:
    - total_thickness   : total thickness of primary ejecta from all basins at this location
    - primary_thickness : thickness of primary ejecta from each basin at this location
    - primary_thickness_cumulative : cumulative thickness of primary ejecta from this and all previous basins at this location
    - abundance         : area or volume fraction of each basin's primary ejecta at each elevation
    - abundance_iflast  : area or volume fraction of each basin's primary ejecta at each elevation if there were no subsequent impacts
  """
  ### //Initialize the computation//
  ds_caches = ds_basin.sortby('order', ascending=False) #sort from youngest -> oldest to efficiently break at the youngest overlapping basin, if necessary
  # Initialize basin caches and thicknesses, breaking at the youngest basin that overlaps the SOI 
  caches                 = []
  cutoff_order           = None
  primary_thicknesses    = []
  for b in tqdm(ds_caches['basin'].values, desc="Pre-computing basin SOI caches", disable=verbose<1):
    basin                = ds_caches.sel(basin=b)
    rSOI                 = great_circle_distance(profile_lat, profile_lon, basin.clat, basin.clon) #[km]
    if rSOI < basin['R'].item(): ## TODO: should this use R or Rat, or something else? I'm assuming all basins would reset stratigraphy but that wouldn't necessarily be true near the rim
      # This point is inside the rim of this basin, so all previous ejecta is erased; break, because all necessary caches have been computed
      cutoff_order       = ds_caches.sel(basin=b)['order'].item()
      nreset             = len(ds_caches['basin']) - len(caches)
      empty_caches       = [None] * nreset
      reset_thicknesses  = [0.0] * nreset
      caches.extend(empty_caches)                   #fill the rest of the caches with None since they won't be used
      primary_thicknesses.extend(reset_thicknesses) #stratigraphy is reset, so all older basins have zero thickness
      break
    try:
      cache              = precompute_SOI(ds_basin=basin, rSOI=[rSOI], ejecta_model=ejecta_model)
    except ValueError: ##TODO: investigate remaining ValueErrors in precompute_SOI to ensure they're handled appropriately here
      cache              = {'thickness_primary': np.array([0.0])} #[m] -- point is outside valid range; treat as zero ejecta
    caches.append(cache)
    primary_thicknesses.append(cache['thickness_primary'][0])     #[m]
  ds_caches['soi_cache']         = (('basin',), np.array(caches, dtype=object))
  ds_caches['primary_thickness'] = (('basin',), np.array(primary_thicknesses)) #[m]
  Tprimary_total                 = ds_caches['primary_thickness'].sum()        #[m]
  # Set up the elevation grid
  if elevation is None:
    dz              = .5                                                          #[m]
    max_depth       = 1e4                                                         #[m]
    elevation_below = np.arange(-dz/2, -max_depth, -dz)                           #[m]
    elevation_above = np.arange(dz/2, Tprimary_total, dz)                         #[m]
    elevation       = np.concatenate((np.flip(elevation_above), elevation_below)) #[m]
  else:
    elevation       = np.asarray(elevation)                                       #[m]
  # Fetch the ordered list of basins
  basins_ordered = ds_caches.sortby('order', ascending=True)
  # Define the output dataset
  pre_lbl    = preimpact_label or 'preimpact'
  ds_profile = xr.Dataset(
    {
      'total_thickness'  : (('lat', 'lon'), np.atleast_2d(Tprimary_total)), #[m]
      'primary_thickness': (('lat', 'lon', 'basin'), np.reshape(np.concatenate(([0], basins_ordered['primary_thickness'].values)), (1, 1, -1))), #[m]
    },
    coords = {
      'lat'      : np.asarray([profile_lat]),
      'lon'      : np.asarray([profile_lon]),
      'basin'    : np.concatenate(([pre_lbl], basins_ordered['basin'].values)),
      'elevation': elevation, #[m]
    },
  )
  ds_profile = ds_profile.assign(primary_thickness_cumulative = lambda ds: ds['primary_thickness'].cumsum(dim='basin'))
  ### //Run the chronological vertical mixing//
  abundances          = np.where(elevation[:, None] < 0, 1.0, 0.0)
  abundances_iflast   = abundances.copy()
  for b in tqdm(basins_ordered['basin'].values, desc="Computing each basin's vertical mixing", disable=verbose<1):
    if (cutoff_order is not None) and (ds_caches.sel(basin=b)['order'].item() <= cutoff_order):
      # This basin's ejecta is erased at this point by a later impact; add a zero column and skip the mixing computation
      abundances        = np.column_stack([abundances, np.zeros(len(elevation))])
      abundances_iflast = np.column_stack([abundances_iflast, np.full(len(elevation), np.nan)])
      continue
    cache             = ds_caches.sel(basin=b)['soi_cache'].item() #fetch the pre-computed SOI cache for this basin
    kernel            = build_mixing_kernel(cache, 0)
    basin             = ds_profile.sel(basin=b)
    abundances        = compute_ejecta_mixing(kernel, elevation, abundances, surface_elevation=basin['primary_thickness_cumulative'].item()-basin['primary_thickness'].item())
    abundances_iflast_thisbasin = abundances[:, -1].copy()[:, None]
    abundances_iflast_thisbasin[abundances_iflast_thisbasin == 0] = np.nan
    abundances_iflast = np.concatenate((abundances_iflast, abundances_iflast_thisbasin), axis=1)
  for var_name, var in zip(['abundance', 'abundance_iflast'], [abundances, abundances_iflast]):
    ds_profile[var_name] = (('lat', 'lon', 'basin', 'elevation'), np.reshape(var.T, (1, 1, var.shape[1], var.shape[0])))
  return ds_profile


def build_global_mixing_dataset(
  ds_basin: xr.Dataset,
  grid_lats: list | np.ndarray,
  grid_lons: list | np.ndarray,
  elevation: np.ndarray = None,
  ejecta_model: dict | EjectaModel = None,
  preimpact_label: str = None,
  verbose: int = 0
) -> xr.Dataset:
  """
  Run a vertical mixing simulation at all points on a coordinate grid given a chronological sequence of basin-forming impacts.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain dimensions 'basin' and data_vars 'order', 'clat', 'clon', 'R', 'Rat'.
  grid_lats, grid_lons : list or np.ndarray
    Coordinates of the points at which to compute the vertical mixing profiles.
  elevation : optional, ndarray or None
    Elevation grid in meters for the vertical mixing profile. If `None`, elevation grid will be handled automatically.
  ejecta_model : dict or EjectaModel
    If dict, can contain keys as described in the EjectaModel class. If `None`, default parameters from the EjectaModel class will be used.
  preimpact_label : str, optional
    Name for the mixing component corresponding to local materials that predate the first impact. If `None`, defaults to 'preimpact'.
  verbose : int
    Verbosity level for progress bars. 0 = no progress bars, 1 = progress bar for each basin, 2 = additional progress bars for sub-steps. Default is 0.
  
  Returns
  -------
  ds_profiles : xr.Dataset
    Dataset containing the vertical mixing profiles at the specified locations, with dimensions:
    - lat       : latitude of each profile point
    - lon       : longitude of each profile point
    - elevation : elevation of grid points w.r.t. pre-impact surface (m)
    - basin     : name of each basin involved in mixing, plus "preimpact"
    and variables:
    - primary_thickness : thickness of primary ejecta from each basin at this location
    - total_thickness   : total thickness of primary ejecta + local excavation at each location
    - abundance         : area or volume fraction of each basin's primary ejecta at each elevation
  """
  ds_profiles = xr.Dataset(
    coords = {
      'lat': grid_lats,
      'lon': grid_lons,
    },
  )
  all_profiles = []
  for lat, lon in tqdm([(lat, lon) for lat in grid_lats for lon in grid_lons], desc="Computing profiles at each grid point", disable=verbose < 1):
    ds_profile = compute_ejecta_mixing_multi_basin(
      ds_basin = ds_basin,
      profile_lat = lat,
      profile_lon = lon,
      elevation = elevation,
      ejecta_model = ejecta_model,
      preimpact_label = preimpact_label
    )
    ds_profile = ds_profile.drop(['abundance_iflast', 'primary_thickness_cumulative'])
    all_profiles.append(ds_profile)
  ds_profiles = xr.merge((ds_profiles, *all_profiles), join='outer')
  return ds_profiles


def _worker_mixing_profile(args):
  """Helper function for parallel processing of mixing profiles. Not intended for external use."""
  i_lat, i_lon, lat, lon, ds_basin, elevation, ejecta_model, preimpact_label, verbose = args
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="`rSOI` is set")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    ds = compute_ejecta_mixing_multi_basin(
      ds_basin = ds_basin,
      profile_lat = lat,
      profile_lon = lon,
      elevation = elevation,
      ejecta_model = ejecta_model,
      preimpact_label = preimpact_label,
      verbose = verbose,
    ).drop(['abundance_iflast', 'primary_thickness_cumulative'])
  data = (
    os.getpid(),
    i_lat, i_lon,
    ds['total_thickness'].item(),
    ds['primary_thickness'].values[0, 0, :], #shape (nbasin,)
    ds['abundance'].values[0, 0, :, :]       #shape (nbasin, nelevation)
  )
  return data


def build_global_mixing_dataset_parallel(
  ds_basin: xr.Dataset,
  elevation: np.ndarray,
  grid_lats: list | np.ndarray,
  grid_lons: list | np.ndarray,
  ejecta_model: dict | EjectaModel = None,
  preimpact_label: str = None,
  verbose: int = 1
) -> xr.Dataset:
  """
  Run a vertical mixing simulation at all points on a coordinate grid given a chronological sequence of basin-forming impacts.

  Parameters
  ----------
  ds_basin : xr.Dataset
    Must contain dimensions 'basin' and data_vars 'order', 'clat', 'clon', 'R', 'Rat'.
  elevation : ndarray
    Elevation grid in meters for the vertical mixing profile.
  grid_lats, grid_lons : list or np.ndarray
    Coordinates of the points at which to compute the vertical mixing profiles.
  ejecta_model : dict or EjectaModel
    If dict, can contain keys as described in the EjectaModel class. If `None`, default parameters from the EjectaModel class will be used.
  preimpact_label : str, optional
    Name for the mixing component corresponding to local materials that predate the first impact. If `None`, defaults to 'preimpact'.
  verbose : int
    Verbosity level for progress bars. 0 = no progress bars, 1 = progress bar for each basin, 2 = additional progress bars for sub-steps. Default is 0.
  
  Returns
  -------
  ds_profiles : xr.Dataset
    Dataset containing the vertical mixing profiles at the specified locations, with dimensions:
    - lat       : latitude of each profile point
    - lon       : longitude of each profile point
    - elevation : elevation of grid points w.r.t. pre-impact surface (m)
    - basin     : name of each basin involved in mixing, plus "preimpact"
    and variables:
    - primary_thickness : thickness of primary ejecta from each basin at this location
    - total_thickness   : total thickness of primary ejecta + local excavation at each location
    - abundance         : area or volume fraction of each basin's primary ejecta at each elevation
  """
  from concurrent.futures import ProcessPoolExecutor, as_completed
  ds_sorted = ds_basin.sortby('order')
  tasks = [
    (i_lat, i_lon, lat, lon, ds_sorted, elevation, ejecta_model, preimpact_label, verbose-1) 
    for i_lat, lat in enumerate(grid_lats) 
    for i_lon, lon in enumerate(grid_lons)
  ]
  nbasin     = len(ds_sorted['basin']) + 1 #+1 for pre-impact component
  nlat       = len(grid_lats)
  nlon       = len(grid_lons)
  nelevation = len(elevation)
  total_thickness   = np.zeros((nlat, nlon))
  primary_thickness = np.zeros((nlat, nlon, nbasin))
  abundance         = np.zeros((nlat, nlon, nbasin, nelevation))
  with ProcessPoolExecutor() as executor:
    if verbose > 0:
      print(f"{executor._max_workers} workers available for parallel vertical mixing.")
    futures = [executor.submit(_worker_mixing_profile, t) for t in tasks]
    used_pids = set()
    for f in tqdm(as_completed(futures), total=len(futures), desc="Computing profiles at each grid point (parallel)", disable=verbose<1):
      pid, i_lat, i_lon, total_thickness_val, primary_thickness_val, abundance_val = f.result()
      used_pids.add(pid)
      total_thickness[i_lat, i_lon]      = total_thickness_val
      primary_thickness[i_lat, i_lon, :] = primary_thickness_val
      abundance[i_lat, i_lon, :, :]      = abundance_val
    if verbose > 0:
      print(f"All profiles computed. Maximum concurrent workers used: {len(used_pids)}.")
  ds_profiles = xr.Dataset(
    {
      'total_thickness':   (('lat', 'lon'), total_thickness),
      'primary_thickness': (('lat', 'lon', 'basin'), primary_thickness),
      'abundance':         (('lat', 'lon', 'basin', 'elevation'), abundance),
    },
    coords = {
      'lat': grid_lats,
      'lon': grid_lons,
      'basin': np.concatenate(([preimpact_label or 'preimpact'], ds_sorted['basin'].values)),
      'elevation': elevation,
    }
  )
  return ds_profiles


### Compute 1D radial profiles of ejecta thickness
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
    thickness_local[i] = get_coverage_depth(np.asarray([ejecta_model.cov]), 'all', cache, i)[0]
  thickness_total = thickness_primary + thickness_local #T_ED_med [*median only if `cov=0.5] -- see text below Eq. 19
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
  ejecta_model: EjectaModel = None,
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
  ejecta_model : optional, EjectaModel
    EjectaModel object that specifies base parameters for ballistic sedimentation model. If `None`, defaults will be used.
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
  d0, p0, l0, t0 = compute_thickness_1D(example, nSOI=nSOI, radius_cutoff=radius_cutoff, ejecta_model=ejecta_model)
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
    d1, p1, l1, t1 = compute_thickness_1D(b, nSOI=nSOI, radius_cutoff=radius_cutoff, ejecta_model=ejecta_model)
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
    dists2d    = (("basin","lat","lon"),  dist2d),
    primary2d  = (("basin","lat","lon"),  primary2d),
    local2d    = (("basin","lat","lon"),  local2d),
    total2d    = (("basin","lat","lon"),  total2d),
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
    zero_basins: bool = False,
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
  
  if zero_basins:
    for b in ordered_basins:
      basin_dists  = ds.dists2d.sel(basin=b).values #km (lat, lon) -- distance of each grid point from basin center
      basin_radius = ds.R.sel(basin=b).item()       #km -- final basin rim radius
      mask = (basin_dists <= basin_radius)
      total_compacted[mask] = 0.0

  # write the new global_total back into a copy of ds
  return ds.assign(global_total=(('lat', 'lon'), total_compacted))
