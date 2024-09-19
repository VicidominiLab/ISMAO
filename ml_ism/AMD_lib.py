import numpy as np
from scipy.signal import convolve
import os
from collections.abc import Iterable
from . import PSF_estimator_lib as svr
from scipy.special import kl_div as kl
import brighteyes_ism.analysis.Tools_lib as tool
from time import sleep
from tqdm import tqdm
import scipy


def AMDupdate_2(img: np.ndarray, obj: np.ndarray, psf: np.ndarray, psf_m: np.ndarray, eps: float) -> np.ndarray:
    '''
    It performs an iteration of the AMD algorithm.

    Parameters
    ----------
    img : np.ndarray
        Input image ( Nx x Ny x Nch ).
    obj : np.ndarray
        Oject estimated from the previous iteration ( Nz x Nx x Ny ).
    psf : np.ndarray
        Point spread function ( Nz x Nx x Ny x Nch ).
    psf_m : np.ndarray
        Point spread function with flipped X and Y axis ( Nz x Nx x Ny x Nch ).
    eps : float
        Division threshold.

    Returns
    -------
    obj_new : np.ndarray ( Nz x Nx x Ny )
        New estimate of the object.

    '''

    # Variables initialization

    sz_o = obj.shape
    Nz = sz_o[0]

    sz_i = img.shape
    Nx = sz_i[0]
    Ny = sz_i[1]
    Nch = sz_i[2]

    den = np.empty([Nz, Nx, Ny, Nch])
    up = np.empty([Nz, Nx, Ny, Nch])

    # Update

    for z in range(Nz):
        for c in range(Nch):
            den[z, :, :, c] = convolve(obj[z, :, :], psf[z, :, :, c], mode='same')
    img_estimate = den.sum(axis=0)

    fraction = np.where(img_estimate < eps, 0, img / img_estimate)

    for z in range(Nz):
        for c in range(Nch):
            up[z, :, :, c] = convolve(psf_m[z, :, :, c], fraction[:, :, c], mode='same')
    update = up.sum(axis=-1)

    obj_new = obj * update

    return obj_new


def AMDstop(O_old: np.ndarray, O_new: np.ndarray, pre_flag: bool, flag: bool, stop, max_iter: int, threshold: float,
            tot: float, Nz: int, k: int):
    '''
    function dealing with the iteration stop of the algorithm

    Parameters
    ----------
    O_old : np.ndarray
        Object obtained at the latter iteration ( Nz x Nx x Ny ).
    O_new : np.ndarray
        Object obtained at the current iteration ( Nz x Nx x Ny ).   
    pre_flag : bool
        first alert that the derivative of the photon counts has reached the threshold. To stop the algorithm both flags must turn into False.
    flag : bool
        second alert that the derivative of the photon counts has reached the threshold.
    stop : np.string
        DESCRIPTION.
    max_iter : int
        maximum number of iterations for the algorithm.
    threshold : float
        when the derivative of the photon counter function reaches this value the algorithm halt .
    tot : float
        total number of photons in the ISM dataset.
    Nz : int
        number of axial planes of interest.
    k : int
        indexing the current algorithm iteraton.

    Returns
    -------
    pre_flag : boolean
        first alert that the derivative of the photon counts has reached the threshold. To stop the algorithm both flags must turn into False.
    flag : boolean
        second alert that the derivative of the photon counts has reached the threshold.
    list
        [total number of photons in the focal plane, total number of photons in the out-of-focus planes]
    list
        [derivative of the photons count at the current iteration in the focal plane, derivative of the photons count at the current iteration in the out-of-focus planes]

    '''

    int_f_old = np.sum(O_old[Nz // 2, :, :]) #calculating photon flux in the focal plane reconstruction at the previous iteration

    int_f_new = np.sum(O_new[Nz // 2, :, :]) #calculating the photon flux in the focal plane reconstruction at the current iteration

    d_int_f = (int_f_new - int_f_old) / tot #calculating the derivative of the photon count function in the focal plane

    int_bkg_old = np.sum(O_old) - int_f_old #calculating the photon flux in the out-of-focus planes reconstruction at the previous iteration

    int_bkg_new = np.sum(O_new) - int_f_new #calculating the photon flux in the out-of-focus planes reconstruction at the current iteration

    d_int_bkg = (int_bkg_new - int_bkg_old) / tot #calculating the derivative of the photon count function in the out-of-focus planes

    # controlling if the derivative value is under the threshold. The algorithm derivative has to lye under the threshold for two consecutive iterations to stop.
    if isinstance(stop, str) and stop == 'auto':
        if np.abs(d_int_f) < threshold:
            if pre_flag == False:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print('Reached maximum number of iterations.')
    #if the iteration stop rule il claimed to be fixed, the algorithm stop when the maximum number of iterations is reached, default value is 100.
    elif isinstance(stop, str) and stop == 'fixed':
        if k == max_iter:
            flag = False

    return pre_flag, flag, [int_f_new, int_bkg_new], [d_int_f, d_int_bkg]


def AMDcore(I: np.ndarray, PSF: np.ndarray, stop='auto', max_iter: int = 100, threshold: float = 1e-3,
            rep_to_save: bool = False, initialization: str = 'flat'):
    '''
    Core function of the algorithm 

    Parameters
    ----------
    I : np.ndarray
        Input image ( Nx x Ny x Nch ).
    PSF : np.ndarray
        Point spread function ( Nz x Nx x Ny x Nch ).
    stop : np.string, optional
        String describing how to stop the algorithm. The default is 'auto'.
    max_iter : int, optional
        maximum number of iteration. The default is 100.
    threshold : float, optional
        DESCRIPTION. The default is 1e-3.
    rep_to_save : iter, optional
        object containing the iteration at which one wants to save the algorithm reconstruction. The default is None.

    Returns
    -------
    O : np.ndarray ( Nz x Nx x Nx )
        reconstructed object.
    k : int
        iteration.

    '''

    # Variables initialization taking into account if the data is spread along the axial dimension or not

    Nx, Ny, Nch = I.shape
    if PSF.ndim > 3:
        Nz = PSF.shape[0]
        H = PSF.copy()
    else:
        Nz = 1
        H = np.expand_dims(PSF.copy(), axis=0)

    b = np.finfo(float).eps  #assigning the error machine value

    # initialization of the object to empty array
    O = np.empty((Nz, Nx, Ny))


    #user can decide how to initialize the object, either with the photon flux of the input image or with a flat initialization
    if initialization == 'sum':
        S = I.sum(axis=-1) / Nz
        for z in range(Nz):
            O[z, ...] = S
    elif initialization == 'flat':
        O = np.ones((Nz, Nx, Nx), dtype=np.float64) * I.sum() / Nz / Nx / Ny
    else:
        raise Exception('Initialization mode unknown.')
        # O_old = np.ones((Nz,Nx, Nx),dtype=np.float64)
    # O_new = np.ones((Nz,Nx, Nx),dtype=np.float64)
    k = 0

    counts = np.zeros([2, max_iter + 1])
    diff = np.zeros([2, max_iter + 1])
    O_all = np.empty((max_iter+1, Nz, Nx, Ny))
    tot = np.sum(I)

    pre_flag = True
    flag = True

    # PSF normalization axial plane wise, with respect to the flux of each plane

    for j in range(Nz):
        H[j, :, :, :] = H[j, :, :, :] / (np.sum(H[j, :, :, :]))

    #calculating the flip of the PSF
    Ht = np.flip(H, axis=(1, 2))

    # Iterative reconstruction process
    if stop != 'auto':
        total = max_iter
    else:
        total = None

    pbar = tqdm(total=total, desc='Progress', position=0)
    for i in range(10):
        while flag:

            O_new = AMDupdate_2(I, O, H, Ht, b)

            pre_flag, flag, counts[:, k], diff[:, k] = AMDstop(O, O_new, pre_flag, flag, stop, max_iter, threshold, tot, Nz,
                                                               k)
            O = O_new.copy()
            O_all[k, ...] = O_new.copy()

            k += 1
            pbar.update(1)
    pbar.close()

    if rep_to_save == False:
        return O, counts[:, :k], diff[:, :k], k
    else:
        return  O_all[:k, ...], counts[:, :k], diff[:, :k], k



def AMD(I : np.ndarray, PSF : np.ndarray, gridPar, exPar, emPar, rep_to_save: bool = False, initialization: str = 'flat',
        max_iter: int = 100, stop='auto', threshold: float = 1e-3, z_out_of_focus='ToFind'):
    '''
     Parameters
     ----------
     I : np.ndarray ( Nx x Ny x Nch)
         ISM dataset to reconstruct
     PSF : np.ndarray (Nz x Nx x Ny x Nch)
         set of PSFs to perform deconvolution
    gridPar : object
        parameters describing features of the acquired image (e.g. number of pixel, lateral and axial pixel size...)
    exPar : object
        excitation parameters
    emPar : object
        emission parameters
     rep_to_save : bool, optional
        if False, the algorithm returns just the final reconstruction. If True it returns the reconstruction at every iteration
     initialization: string
         if flat the first instance of the iterative algorithm is initialized to the total flux of the ISM dataset,
         if sum the first instance of the iterative algorithm is initialized such that the total flux of the ISM dataset is equally divided on every object's pixel
     max_iter : int, optional
         maximum number of iterations that the algorithm can perform both in fixed and auto stop mode. The default is 100.
     stop : str, optional
         string describing how one wants to stop the algorithm . in auto mode the algorithm halt when the derivative of the photon counts reach the setted threshold.
         In fixed mode the algorithm halt when the iterations reaches the maximum number of iterations passed as max_iter. The default is 'auto'.
     threshold : float, optional
         If the stop rule is setted as auto, the algorithm halt when the photon counts derivative reaches this value. The default is 1e-3.
     z_out_of_focus :
         if equal to 'ToFind' the algorithm will find the optimal depth of reconstruciton through a correlative minimization procedure
         if passed as a float the algorithm will place at that depth the background plane of reconstruction


     Returns
     -------
     O : np.ndarray  ( Nz x Nx x Ny)
         reconstructed object.
     counts : np.ndarray ( k x 2 )
         number of the photons on the axial planes and on the out-of-focus planes for every iteration of the algorithm.
     diff : np.ndarray ( k x 2 )
         photon counts derivative on the axial planes and on the out-of-focus planes for every iteration of the algorithm.
     k : int
         total number of iterations performed by the algorithm.

     '''

    sz = I.shape

    gridPar.Nx = sz[0]
    gridPar.Ny = sz[1]
    gridPar.N = int(np.sqrt(sz[2]))

    dset = I.copy()

    #if the datasset to analize is even in one of the two lateral dimensions, the algorithm crop the dataset to the nearest odd number. That is to have a 'centroid' of the image well-defined
    if gridPar.Nx % 2 == 0:
        dset = tool.CropEdge(dset, npx = 1, edges = 'l', order = 'xyc')
    if gridPar.Ny % 2 == 0:
        dset = tool.CropEdge(dset, npx = 1, edges = 'u', order = 'xyc')

    # if the PSF is not passed as input the algorithm estimate it by extracting parameters from the ISM dataset
    if isinstance(PSF, str) and PSF == 'blind':
        PSF, _, _ = svr.psf_estimator_from_data(dset, exPar, emPar, gridPar.pxsizex, gridPar.Nz, z_out_of_focus=z_out_of_focus)

    O, counts, diff, k = AMDcore(dset, PSF, stop=stop, max_iter=max_iter, threshold=threshold, rep_to_save=rep_to_save,
                                 initialization=initialization)

    #re-padding the object to the original size

    if gridPar.Nz>1:
        if gridPar.Ny % 2 == 0:
            O = np.pad(O, ((0, 0), (0, 0), (1, 0)), 'constant')
            PSF = np.pad(PSF, ((0, 0), (0, 0), (1, 0), (0, 0)), 'constant')
        if gridPar.Nx % 2 == 0:
            O = np.pad(O, ((0, 0), (1, 0), (0, 0)), 'constant')
            PSF = np.pad(PSF, ((0, 0), (1, 0), (0, 0), (0, 0)), 'constant')

    if gridPar.Nz == 1:
        if gridPar.Nx % 2 == 0 :
            O = np.pad(O, ((0, 0), (1, 0), (0, 0)), 'constant')
            PSF = np.pad(PSF, ( (0, 0), (1, 0), (0, 0)), 'constant')
        if gridPar.Ny % 2 == 0 :
            O = np.pad(O, ((0, 0), (0, 0), (1, 0)), 'constant')
            PSF = np.pad(PSF, ( (0, 0), (0, 0), (1, 0)), 'constant')

    return O, counts, diff, k, PSF


def KL_divergence(ground_truth, recon):

    num_rec = recon.shape[0]
    kl = np.zeros( [2, num_rec] )

    for i in range(num_rec):
        kl[0,i] = np.sum(scipy.special.kl_div(ground_truth[0,...], recon[i,0,...]))
        kl[1,i] =  np.sum(scipy.special.kl_div( ground_truth[1,...],  recon[i,1,...]))

    return kl


def Wasserstein(ground_truth, recon):

    num_rec = recon.shape[0]
    wass = np.zeros( [2, num_rec] )

    for i in range(num_rec):
        wass[0,i] = scipy.stats.wasserstein_distance(ground_truth[0,...].flatten(), recon[i,0,...].flatten())
        wass[1,i] = scipy.stats.wasserstein_distance(ground_truth[1,...].flatten(), recon[i,1,...].flatten())

    return wass

