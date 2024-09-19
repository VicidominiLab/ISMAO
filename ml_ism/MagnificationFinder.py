import numpy as np
from scipy.special import jv
from scipy.optimize import minimize
from scipy.signal import convolve
def scalar_psf(r, wl, NA):

    k = 2 * np.pi / wl  # 1/nm
    x = k * r * NA

    psf = np.ones_like(x) * 0.5
    np.divide(jv(1, x), x, out=psf, where=(x != 0))
    psf = np.abs(psf) ** 2

    return psf


def scalar_psf_det(r, wl, NA, pxdim, pxpitch, M):

    psf = scalar_psf(r, wl, NA)
    pinhole = rect(r - pxpitch / M, pxdim / M)
    psf_det = convolve(psf, pinhole, mode='same')

    return psf_det


def rect(r, D):
    r = np.where(abs(r) <= D / 2, 1, 0)
    return r / D


def shift_value(M, wl_ex, wl_em, pxpitch, pxdim, NA):
    airy_unit = 1.22 * wl_em / NA  # nm
    pxsize = 0.1  # nm
    rangex = int(airy_unit / pxsize)

    r = np.arange(-rangex, rangex + 1) * pxsize

    psf_det = scalar_psf_det(r, wl_em, NA, pxdim, pxpitch, M)

    psf_ex = scalar_psf(r, wl_ex, NA)

    psf_clsm = psf_det * psf_ex

    shift = r[np.argmax(psf_clsm)]

    return shift


def loss_shift(x, shift_exp, wl_ex, wl_em, pxpitch, pxdim, NA):
    shift_t = shift_value(x, wl_ex, wl_em, pxpitch, pxdim, NA)

    loss = np.linalg.norm(shift_t - shift_exp) ** 2

    return loss


def loss_minimizer(shift_t, wl_ex, wl_em, pxpitch, pxdim, NA, M_0, tol, opt):
    result = minimize(loss_shift, x0=M_0, args=(shift_t, wl_ex, wl_em, pxpitch, pxdim, NA), options=opt, tol=tol,
                      method='Nelder-Mead')

    if result.success == True:
        M = result.x[0]
    else:
        print('Minimization did not succed.')
        print(result.message)

    return M


def find_mag(shift, wl_ex, wl_em, pxpitch, pxdim, NA):
    tol = 1e-6
    opt = {'maxiter': 10000}
    M_0 = 500

    M = loss_minimizer(shift, wl_ex, wl_em, pxpitch, pxdim, NA, M_0, tol, opt)

    return M