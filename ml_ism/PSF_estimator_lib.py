import matplotlib.pyplot as plt
import numpy as np

from scipy.special import kl_div as kl
from scipy.stats import pearsonr as pearson
from scipy.signal import find_peaks

import brighteyes_ism.simulation.PSF_sim as psf
import brighteyes_ism.analysis.Tools_lib as tool
from brighteyes_ism.analysis.APR_lib import ShiftVectors

from . import ShiftVectorMinimizer as svm
from . import MagnificationFinder as mag


class GridFinder(psf.GridParameters):
    def __init__(self, grid_par: psf.GridParameters = None):
        psf.GridParameters.__init__(self)
        if grid_par is not None:
            vars(self).update(vars(grid_par))
        self.shift = None  # um

    def estimate(self, dset, wl_ex, wl_em, na):
        ref = dset.shape[-1] // 2
        usf = 50
        shift, _ = ShiftVectors(dset, usf, ref)
        par = svm.FindParam(shift)
        self.shift = par[0] * self.pxsizex # um
        self.rotation = par[1] # rad
        self.mirroring = par[2] # +/- 1
        self.M = np.round(
            mag.find_mag(self.shift, wl_ex=wl_ex, wl_em=wl_em, pxpitch=self.pxpitch, pxdim=self.pxdim, NA=na))


def PSFwidth(pxsizex, pxsizez, Nz, simPar, spad_size):

    '''
    Function calculating the beam waist along the z-axis and the number of pixel required to have the whole simulated PSF in the FOV, minimizing simulation time complexitu

    Parameters
    ----------
    pxsizex : float
        dimension in [nm] of every pixel
    pxsizez : float
        discretization step along the z-axis.
    Nz : int
        number of axial planes on which discretize the object
    simPar: object 
        object describing the features of excitation light 
    spadsize: float
        dimension of the SPAD array detector

    Returns
    -------
    Nx : int
        number of pixel required to have the whole simulated PSF in the FOV 
    '''

    z = pxsizez * (Nz // 2)

    M2 = 3

    w0 = simPar.airy_unit/2
    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt( 1 + (M2 * z / z_r )**2)

    Nx = int(np.round((2 * (w_z + spad_size) / pxsizex)))

    if Nx % 2 == 0:
        Nx += 1

    return Nx

def psf_estimator_from_data(data: np.ndarray, exPar, emPar, pxsizex : float,  Nz : int = 2, ups: int = 10, downsample: bool = True, stedPar=None, z_out_of_focus = 'ToFind'):
    '''
    Function generating rotated PSFs according to the pinholes distribution of the SPAD

    Parameters
    ----------
    pxpitch : float
        distance between two SPAD's channels centroid with at least one shared side
    pxdim : float
        side dimension of one SPAD's channel
    Nx : int
        number of pixel of the desidered returned PSFs
    pxsizex : float
        dimension in [nm] of every pixel
    Nz : int
        number of axial plane on which generate the set of PSFs
    pxsizez : float
        distance between every couple of axial planes 
    M : int
        magnification factor
    exPar : object
        excitation parameters
    emPar : object
        emission parameters
    ups : int
        upsampling factor used in the PSFs simulation process
    data : np.ndarray
        DESCRIPTION.
    stedPar : TYPE, optional
        set of parameters describing the STED beam. The default is None.

    Returns
    -------
    Psf3_f : np.ndarray ( Nz x Nx x Ny x Nch )
        complete PSF for every element of the array detector
    detPsf3_f : np.ndarray ( Nz x Nx x Ny x Nch )
        detection PSF for every element of the array detector
    exPsf3_f : np.ndarray ( Nz x Nx x Ny )
        excitation PSFs

    '''
    szd = data.shape

    Nx = szd[0]
    Ny = szd[1]

    dset = data.copy()

    if Nx % 2 == 0:
        dset = tool.CropEdge(dset, npx=1, edges='l', order='xyc')
        Nx -= 1
    if Ny % 2 == 0:
        dset = tool.CropEdge(dset, npx=1, edges='u', order='xyc')
        Ny -= 1

    N = int(np.sqrt(data.shape[-1]))

    #if the depth of the background plane is not set, those lines calculates it by minimizing the correlation between focal PSF and the PSFs along the z-axis.
    if isinstance(z_out_of_focus, str) and z_out_of_focus == 'ToFind':
        pxsizez = FindOutOfFocus(pxsizex ,exPar , emPar, mode='KL', stack='positive')[0]  #generating a stack of PSFs along the z-axis and calculating the correlation curve to minimize

    else:
        pxsizez = float(z_out_of_focus)

    # find rotation, mirroring, and magnification parameters from the data

    grid_simul = GridFinder() #TODO: why this object is not an input? How to set pxpitch and pxdim?
    grid_simul.pxsizex = pxsizex
    grid_simul.estimate(data, exPar.wl, emPar.wl, emPar.na)

    # calculate optimal simulation range
    Nx_simul = PSFwidth(pxsizex, pxsizez, Nz, emPar, grid_simul.spad_size())

    # calculate upsampled pixel size and number of pixels to have a more precise simulation of the PSFs
    pxsize_simul = pxsizex / ups
    Nx_up = Nx_simul * ups

    grid_simul.Nx = Nx_up
    grid_simul.pxsizex = pxsize_simul
    grid_simul.Nz = Nz
    grid_simul.pxsizez = pxsizez
    grid_simul.N = N

    Psf, detPsf, exPsf = psf.SPAD_PSF_3D(grid_simul, exPar, emPar, stedPar=stedPar, spad=None, stack='symmetrical') # up sampled PSFs generation

    #downsampling the PSFs to the original pixel size
    if downsample == True:
        Psf_ds = tool.DownSample(Psf, ups, order='zxyc')
        detPsf_ds = tool.DownSample(detPsf, ups, order='zxyc')
        exPsf_ds = tool.DownSample(exPsf, ups, order='zxy')
    else:
        Psf_ds = Psf
        detPsf_ds = detPsf
        exPsf_ds = exPsf

    #padding the PSFs to have the same size of the original data
    topad = int((Nx - Nx_simul) / 2) # TODO: Is the PSF padding really necessary?

    #generating the array to perform padding exclusively on the proper axis of the array
    padarray_big = [ [0,0] ] * 4
    padarray_sm = [ [0,0] ] * 3
    padarray_big[1:3] = [[topad, topad]] * 2
    padarray_sm[1:3] = [[topad, topad]] * 2

    #PSFs padding
    Psf_f = np.squeeze( np.pad(Psf_ds, padarray_big, mode='constant') )
    detPsf_f = np.squeeze( np.pad(detPsf_ds, padarray_big, mode='constant') )
    exPsf_f = np.squeeze( np.pad(exPsf_ds, padarray_sm, mode='constant') )

    return Psf_f, detPsf_f, exPsf_f


def FindUps(pxsize_exp, pxsize_sim=4):
    '''
    Function to find the up sampling factor to use in the PSFs simulation process, if not passed by the user as input

    Parameters
    ----------
    pxsize_exp : TYPE
        DESCRIPTION.
    pxsize_sim : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    ups_opt : TYPE
        DESCRIPTION.

    '''
    ups = np.arange(1, np.floor(pxsize_exp)).astype(int) #generating an array containing a set of valid values as up sampling factors

    l = int(len(ups))

    res = np.empty(l)

    for i in range(l):
        res[i] = (pxsize_exp / ups[i] - pxsize_sim) ** 2 #norm function to find the upsampling value that minimizes the difference between the experimental and simulated pixel size (passed as default in this function)

    index = np.argmin(res)

    ups_opt = ups[index] #optimal upsampling value retrieved

    return ups_opt


def FindOutOfFocus(pxsizex : float = None , exPar = None, emPar = None, grid = None, mode: str ='KL', stack: str ='positive', graph: bool = False, input_psf: list = None):
    '''
    Function retrieving the optimal out-of-focus depth of the background plane by minimizing the Pearson correlation/Kullback-Leibler divergence between the focal PSF and the PSFs along the z-axis

    Parameters
    ----------
    pxsizex : float, optional
        lateral dimension in [nm] of one image voxel
    exPar : object
        object descibing the excitation step features
    emPar : object
        object describing the emission parameters features
    grid : object
        parameters describing features of the final image (e.g. number of pixel, lateral and axial pixel size...)
    mode : str, optional
        Choose how to perform the correlation measure.It can be performed evaluating the Kullback-Leibler divergence or the Pearson correlation as metric. The default is 'KL'.
    stack : str, optional
        Here one can choose how to generate the PSF stack ( simmetrically or not with respect to the focal plane). The default is 'positive'.
    graph : bool, optional
        One can choose if visualize the PSFs correlation graph along the axial dimension. The default is False.
    input_psf : list, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    optimal_depth : TYPE
        DESCRIPTION.
    PSF : TYPE
        DESCRIPTION.

    '''

    #raising exception if some parameters needed for the simulation are not passed as input

    if input_psf is None and exPar is None:
        raise Exception("PSF is not an input. PSF parameters are needed.")

    if input_psf is None and emPar is None:
        raise Exception("PSF is not an input. PSF parameters are needed.")

    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")

    if grid is None:
        gridPar = psf.GridParameters()
        gridPar.Nz = 40
        gridPar.pxsizez = 25
        gridPar.pxsizex = pxsizex
    else:
        gridPar = grid.copy()

    Nx_temp = PSFwidth(pxsizex, gridPar.pxsizez, gridPar.Nz ,exPar, gridPar.spad_size())  #finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV (in order to minimize the time complexity of the simulation process)
    gridPar.Nx= Nx_temp

    correlation, PSF = Conditioning(gridPar=gridPar, emPar=emPar,
                                    exPar=exPar, mode=mode,
                                    stack=stack, input_psf=input_psf) #function calculating the correlation curve with Pearson correlation or Kullback-Leibler divergence, as the user choose

    if mode == 'KL':                        #finding minimum of the correlation curve in the KL divergence scenario
        idx, _ = find_peaks(correlation)
    elif mode == 'Pearson':
        idx, _ = find_peaks(-correlation)  #finding the minimum of the correlation curve in the Pearson scenario
    else:
        raise Exception("Similarity mode unknown.")

    optimal_depth = idx * gridPar.pxsizez #optimal out-of-focus depth retrieved

    if graph == True:
        z = np.arange(0, gridPar.Nz * gridPar.pxsizez, gridPar.pxsizez)
        plt.figure()
        plt.plot(z, correlation)
        plt.plot(optimal_depth, correlation[idx], 'ro')
        plt.vlines(optimal_depth , 0 , correlation[idx], linestyles='dashed', label ='optimal_depth')  #plotting the correlation/divergence curve and the optimal out-of-focus depth
    return optimal_depth, PSF


def Conditioning(gridPar, exPar =  None, emPar = None, mode='KL', stack='positive', input_psf = None):
    '''
    Function calculating the correlation/divergence along the z-axis between the in focus PSFs and the PSFs at different planes

    Parameters
    ----------
    N : int
       square root of the amount SPAD's channels
    pxpitch : float
        distance between two SPAD's channels centroid with at least one shared side
    pxdim : float
        side dimension of one SPAD's channel
    pxsizex : float
        dimension in [nm] of every pixel
    M : int
        magnification of the system
    exPar : TYPE
        DESCRIPTION.
    emPar : TYPE
        DESCRIPTION.
    spad : np.ndarray
        distribution of the SPAD's pinhole
    depth : float
        maximum depth at which generate the PSFs
    density_step : float
        the simulator will generate one PSFs set at every density step, starting from 0 up to the maximum depth
    mode : TYPE, optional
        parameter describing which pseudo-metric use to calculate the correlations along depth. The default is 'KL'.
    stack : TYPE, optional
        how to manage the PSFs simulation along z-axis. The default is 'positive'.[--> that means firts PSF generated on the focal plane and then up to the sky]
    ups : int, optional
        upsampling factor. The default is 10.
    psf : TYPE, optional
         The default is 'None', in this case the PSf stack will be automatically generated inside the function.
         The user can pass the stack as input.

    Returns
    -------
    corr : np.ndarray
        correlation value between the in-focus PSFs and the PSFs varying along the z-axis 

    '''

    if input_psf is None and exPar is None:
        raise Exception("PSF is not an input. PSF parameters are needed.")

    if input_psf is None and emPar is None:
        raise Exception("PSF is not an input. PSF parameters are needed.")

    depth = gridPar.Nz * gridPar.pxsizez
    Nx_t = PSFwidth(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, emPar , gridPar.spad_size())  #finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV (in order to minimize the time complexity of the simulation process)
    gridPar.Nx = Nx_t

    if input_psf is None:
        # Simulating PSFs stack
        PSF, detPSF, exPSF = psf.SPAD_PSF_3D(gridPar, exPar, emPar, spad=None, stack=stack)
        # calculating crop value taking into accout that PSFs inerehntly contain a shift as we move apart from the central pixel of the detector array
        npx = int(np.round(((gridPar.N // 2) * gridPar.pxpitch + gridPar.pxdim / 2) / gridPar.M / gridPar.pxsizex))

        PSF = tool.CropEdge(PSF, npx, edges='l,r,u,d', order='zxyc')
        detPSF = tool.CropEdge(detPSF, npx, edges='l,r,u,d', order='zxyc')
        exPSF = tool.CropEdge(exPSF, npx, edges='l,r,u,d', order='zxy')
    else:
        PSF, detPSF, exPSF = input_psf

    # Normalizing PSF of each axial plane with respect to the total flux of each axial plane
    for i in range(gridPar.Nz):
        PSF[i] = PSF[i] / np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    #calculating the correlation/divergence between the in-focus PSF and the PSFs at different planes
    if mode == 'KL':
        for i in range(gridPar.Nz):
            corr[i] = kl(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()

    elif mode == 'Pearson':
        for i in range(gridPar.Nz):
            corr[i] = pearson(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]


def UpsamplingError(groundtruth , reconstruction):
    #TODO: not very well written + this is the wrong library
    #function calculating the upsampling error between upsampled reconstruction and reference image (ground truth when user has it)

    sz = groundtruth.shape
    szt = reconstruction.shape

    if sz!=szt:
        raise Exception('Input dimensions mismatch.')

    Nx = sz[0]
    Ny= sz[1]

    flux_gt = np.sum(groundtruth)  #calculating photon flux of the ground truth
    flux_rec = np.sum(reconstruction) #calculating photon flux of the reconstruction

    gt_res = np.empty(sz)
    rec_res = np.empty(sz)

    up_error = 0

    for i in range(Nx):
        for j in range(Ny):
            gt_res[i,j] = groundtruth[i,j]/flux_gt
            rec_res[i,j] = reconstruction[i,j]/flux_rec
            up_error += np.abs(gt_res[i,j] - rec_res[i,j])  #upsampling error function is a sum of the absolute value of the difference between the normalized ground truth and the normalized reconstruction
                                                            #normalization is done dividing each pixel by the total photon flux of the image

    return up_error







