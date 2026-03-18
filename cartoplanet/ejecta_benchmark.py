import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
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
    T_LM_med[i], T_PE[i], abundance_PEinNewED, secondary_craters[i], Fraction_ExcavatedLM[:, i, :] = compute_ejecta_thickness_and_mixing(
        Rat=Rat[i],
        r_SOI_center=r_SOI[:, i],
        Elevation_from_Local_Surface=elevation,
        abundance_local_PE=abundance_PEinNewED,
        **kw_args
    )
    abundance_PEinED_withoutMixingbyLaterEjecta[:, i, :] = abundance_PEinNewED[:, -1, :]
  return T_LM_med, T_PE, abundance_PEinNewED, secondary_craters, Fraction_ExcavatedLM, elevation, abundance_PEinED_withoutMixingbyLaterEjecta


def Xie_figure10c_benchmark(grid=False):
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
    ax.set_ylabel("Abundance of basin ejecta in deposits (#)")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    if grid:
      ax.grid()
    plt.show()
  return


def Xie_figure5_benchmark(grid=False):
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
    ax[1].semilogx(-elevation, abundance_PEinED[:, 0, 0]*100, color=[1, 0, 0], linewidth=2, label="Orientale ejecta")
    ax[1].semilogx(-elevation, (1-abundance_PEinED[:, 0, 0])*100, color=[0, 0, 0], linewidth=2, label="Pre-Orientale materials")
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
      if grid:
        a.grid()
    plt.show()
  return