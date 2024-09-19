import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


def PhaseAberrationGibsonLanni(rho, phi, w0, f, k, NA, h, z, zp, ns, ng, ng0, ni, ni0, ti0, tg, tg0):
    
    a = min([NA, ns, ni, ni0, ng, ng0]) / NA
    
    r = rho/h
	
    if r > a:
    
        return 1
    
    else:

        OPDs = zp * np.sqrt(ns * ns - NA * NA * r * r) # OPD in the sample
        OPDi = (z + ti0) * np.sqrt(ni * ni - NA * NA * r * r) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * r * r) # OPD in the immersion medium
        OPDg = tg * np.sqrt(ng * ng - NA * NA * r * r) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * r * r) # OPD in the coverslip
        W    = k * 1e-6 * (OPDs + OPDi + OPDg) # FIX to a Pyfocus bug. k is not given in rad/nm. Check here: https://github.com/fcaprile/PyFocus/blob/a3c13b209f4d558322027e0f579fd6175ca4a426/PyFocus/custom_mask_functions.py#L10
    
        phase = np.exp(1j*W)
        
        return phase





def Plot(O , s=1 , scale=False , cmap='inferno'):
   
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = ax.imshow(O , cmap = cmap)
    plt.axis('off')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(data, cax=cax)
    ax.text(1.45, 0.1, r'Intensity [photons] ', rotation=90, transform=ax.transAxes)
    if scale==True:
        scalebar = ScaleBar(
            s, "nm",  # default, extent is calibrated in meters
            box_alpha=0,
            color='w',
            length_fraction=0.25)
        ax.add_artist(scalebar)
    
    return fig




def ChPlot(O , s=1, scale=False, cmap='inferno'):

    N=int(np.sqrt(O.shape[-1]))
    
    fig = plt.figure()
    plt.axis('off')
    for i in range(N*N):
        ax = fig.add_subplot(N, N, i+1)
        ax.imshow(O[:,:,i],cmap=cmap)
        plt.axis('off')
        if scale==True:
            if i==N**2-1:
                scalebar = ScaleBar(
                  s, "nm", # default, extent is calibrated in meters
                  box_alpha=0,
                  color='w',
                  length_fraction=0.25)
                ax.add_artist(scalebar)
    
    return fig





def PatchPlot(img , pxsize ,  bottom_left_square_x , bottom_left_square_y, height , width , edgecolor ='g' , facecolor = 'none' , cmap='inferno'):
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1,1)
    data = ax.imshow(img ,cmap=cmap)
    rect = patches.Rectangle((bottom_left_square_x, bottom_left_square_y), height, width, linewidth=1, edgecolor = edgecolor, facecolor = facecolor)
    ax.add_patch(rect)
    plt.axis('off')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(data, cax=cax)
    ax.text(1.45, 0.1, r'Intensity [photons] ', rotation=90, transform=ax.transAxes)
    scalebar = ScaleBar(
        pxsize, "nm",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)
    ax.add_artist(scalebar)

    return fig

def ChPad(data, topad, ):
    
    if data.ndim == 3:
        Nx = data.shape[0]
        N = data.shape[-1]
        
        data = data.reshape(1,Nx,Nx,N)
    
    N = data.shape[-1]
    
    Nx = data.shape[1]
    
    Nz = data.shape[0]
    
    Nn = int( Nx + 2*topad )
    
    data_n = np.empty(( Nz , Nn , Nn , N ))
    
    for j in range(N):
        for i in range(Nz):
            data_n[i,:,:,j] = np.pad (data[i,:,:,j],topad)
            
            
    data_n = np.squeeze(data_n)
            
    return data_n
        
    
    
    