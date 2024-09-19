import numpy as np
from scipy.optimize import minimize

def ShiftMatrix() -> np.ndarray:
    '''
    Function calculating the 3x3 squared shift-vectors according to the input shift-vectors

    Returns
    -------
   np.ndarray (3 x 2)
        squared theoretical shift vectors oriented as the experimental ones given as input

    '''

    shift_t = np.zeros((3, 3, 2))

    for i, n in enumerate(np.arange(-1, 2)):
        for j, m in enumerate(np.arange(-1, 2)):
            shift_t[i, j, :] = [-n, -m]

    return shift_t.reshape(9, 2)


def RotationMatrix2D(theta: float) -> np.ndarray:
    '''
    function calculating the 2x2 rotation matrix of the desidered angle passed as input

    Parameters
    ----------
    theta : float
        angle of rotation.

    Returns
    -------
    rot : TYPE
        2 x 2 linear operato performing rotation of the input angle.

    '''

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    rot = np.squeeze(rotMatrix)

    return rot

def MirrorMatrix(alpha):
    mirroring = np.array([[1, 0],
                          [0, alpha]])

    return mirroring

def CropShift(shift_exp: np.ndarray) -> np.ndarray:
    '''
    Function cropping shift on the central 3x3 ring

    Parameters
    ----------
    shift_exp : np.ndarray
        experimental shift vectors to crop .

    Returns
    -------
    np.ndarray
        cropped shift vectors on the central 3 x 3 ring.

    '''

    N = int(np.sqrt(shift_exp.shape[0]))

    shift_exp = shift_exp.reshape(N, N, -1)

    shift_cropped = np.zeros((3, 3, 2))

    for i, n in enumerate(np.arange(-1, 2)):
        for j, m in enumerate(np.arange(-1, 2)):
            shift_cropped[i, j, :] = shift_exp[N // 2 + n, N // 2 + m, :]

    return shift_cropped.reshape(9, 2)

def TransformShiftVectors(param, shift_t):

    A = param[0]
    R = RotationMatrix2D(param[1])
    M = MirrorMatrix(param[2])

    TransformMatrix = A * R @ M

    shift_tt = np.einsum('ij,kj -> ki', TransformMatrix, shift_t)

    return shift_tt

def loss_shifts(x0, shift_exp: np.ndarray, shift_t: np.ndarray, mirror: float) -> float:
    '''
    Norm-2 loss function associated to the rotation and dilatation minimization with respect to the experimental shifts

    Parameters
    ----------
    param : TYPE
        DESCRIPTION.
    shift_exp : np.ndarray
        experimental shift vectors to fit with theoretical ones
    shift_t : np.ndarray
        theoretical shift vectors fitting experimental shift vectors

    Returns
    -------
    float
        residual of the fitting between experimental shift vectors and dilatated and rotated simulated shift vectors.

    '''

    param = [*x0, mirror]
    shift_tt = TransformShiftVectors(param, shift_t)

    loss = np.linalg.norm(shift_exp - shift_tt) ** 2

    return loss

def LossMinimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror):

    result = minimize(loss_shifts, x0=(alpha_0, theta_0), args=(shift_m, shift_t, mirror), options=opt, tol=tol,
                      method='Nelder-Mead')

    if result.success == True:
        alpha = result.x[0]
        theta = result.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror

    else:
        print('Minimization did not succeded.')
        print(result.message)

        alpha = result.x[0]
        theta = result.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror

def FindParam(shift_exp: np.ndarray, alpha_0: float = 2, theta_0: float = 0.5):
    '''
    Function minimizing the loss on angle and dilation parameters

    Parameters
    ----------
    shift_exp : np.ndarray
        experimental shift vectors to fit with theoretical ones through dilatation and rotation operators
    alpha_0 : float
        starting point for the dilation parameter
    theta_0 : float
        starting point for the rotation parameter
    Returns
    -------
    alpha : float
        dilatation factor describing the dilatation operator
    theta : float
        rotation angle describing the rotation operator

    '''

    shift_m = CropShift(shift_exp)

    shift_t = ShiftMatrix()

    tol = 1e-6
    opt = {'maxiter': 10000}

    params = LossMinimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror=1)

    params_mirror = LossMinimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror=-1)

    Loss_0 = loss_shifts(params, shift_m, shift_t, 1)
    Loss_1 = loss_shifts(params_mirror, shift_m, shift_t, -1)

    if Loss_0 < Loss_1:
        alpha = params[0]
        theta = params[1]
        mirror = 1
    else:
        alpha = params_mirror[0]
        theta = params_mirror[1]
        mirror = -1

    return alpha, theta, mirror

def FittedShiftVectors(param):

    shift_t = ShiftMatrix()

    shift_vectors_array = TransformShiftVectors(param, shift_t)

    return shift_vectors_array