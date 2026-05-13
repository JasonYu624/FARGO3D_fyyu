import numpy as np
import scipy as sp
import h5py
from pathlib import Path


class Fargo3D_Cyl_Data:
    """
    Handles **2D** cylindrical data (r, phi) without dummy dimensions; gas data only
    
    usage: 
    >>> output_dir = "fargo/outputs/run1/"
    >>> P = 2*np.pi
    >>> ds = Fargo3D_Cyl_Data(output_dir, 1234, t=1234*P)
    
    """
    
    def __init__(self, sim_dir, outid, t=0.0, interp_method='manual_linear', h5=False, kflag=False, **kwargs):
        self.sim_dir, self.outid = sim_dir, outid
        postfix = "_0" if kflag else ""
        ext = '.h5' if h5 else '.dat'
        self.Ndim = 2  # Force 2D cylindrical mode
        
        # Load domain files - cylindrical coordinates (r, phi)
        self.ciphi = np.loadtxt(f"{sim_dir}/domain_x.dat")  # azimuthal interfaces
        self.cir = np.loadtxt(f"{sim_dir}/domain_y.dat")    # radial interfaces
        
        # Build cell center/left coordinates - radial direction
        if kflag:  # including ghost zones
            self.Nr, self.Nphi = self.cir.size - 1, self.ciphi.size - 1
            self.ccr, self.clr = 0.5 * (self.cir[1:] + self.cir[:-1]), self.cir[:-1]
        else:  # exclude ghost zones (3 per radial side)
            self.Nr, self.Nphi = self.cir.size - 7, self.ciphi.size - 1
            self.ccr, self.clr = 0.5 * (self.cir[3:-4] + self.cir[4:-3]), self.cir[3:-4]
        
        # Phi coordinates
        self.ccphi, self.clphi = 0.5 * (self.ciphi[1:] + self.ciphi[:-1]), self.ciphi[:-1]
        
        # Data shape: pure 2D (Nr, Nphi)
        self.shape = (self.Nphi, self.Nr)
        self.time = t
        
        # Attempt to get time from summary file
        self.summary_time = -1
        summary_file = Path(f"{sim_dir}/summary{outid}.dat")
        if summary_file.is_file():
            with open(summary_file, 'r') as f:
                for line in f:
                    if "at simulation time " in line:
                        self.summary_time = float(line.split("at simulation time ")[1].split()[0])
                        break
        
        # Read data as pure 2D arrays
        if h5:
            h5_path = kwargs.get("h5_path", f"{sim_dir}/gas{outid}{ext}")
            with h5py.File(h5_path, 'r') as h5f:
                self.dtype = h5f['dens'].dtype
                print(f"Detected data type: {self.dtype}")
                self.rho_gas = h5f['dens'][:].reshape(self.Nr, self.Nphi)
                self.ux = h5f['vx'][:].reshape(self.Nr, self.Nphi)
                self.uy = h5f['vy'][:].reshape(self.Nr, self.Nphi)
                self.energy = h5f['energy'][:].reshape(self.Nr, self.Nphi)
        else:
            self.dtype = np.float64
            self.rho_gas = np.fromfile(f"{sim_dir}/gasdens{outid}{postfix}{ext}").reshape(self.Nr, self.Nphi)
            self.ux = np.fromfile(f"{sim_dir}/gasvx{outid}{postfix}{ext}").reshape(self.Nr, self.Nphi)
            self.uy = np.fromfile(f"{sim_dir}/gasvy{outid}{postfix}{ext}").reshape(self.Nr, self.Nphi)
            self.energy = np.fromfile(f"{sim_dir}/gasenergy{outid}{postfix}{ext}").reshape(self.Nr, self.Nphi)
        
        # Interpolate velocities to cell centers (pure 2D operations)
        self.uphi = np.zeros_like(self.ux)
        self.uphi[:, :-1] = 0.5 * (self.ux[:, :-1] + self.ux[:, 1:])  # Phi-direction
        self.uphi[:, -1] = self.ux[:, -1] + 0.5 * (self.ux[:, -1] - self.ux[:, -2])
        
        self.ur = np.zeros_like(self.uy)
        self.ur[:-1, :] = 0.5 * (self.uy[:-1, :] + self.uy[1:, :])  # Radial-direction
        self.ur[-1, :] = self.uy[-1, :] + 0.5 * (self.uy[-1, :] - self.uy[-2, :])


class Fargo3D_Sph_Data:
    """ 
    Handles **3D** spherical data (r, phi, theta); gas data only
    
    usage: 
    >>> ds = Fargo3D_Sph_Data("./fargo3d/output/p3d/", 1234, t=1234*2*np.pi)
    """
    
    def __init__(self, sim_dir, outid, t=0.0, interp_method='manual_linear', h5=False, kflag=False, **kwargs):
        self.sim_dir, self.outid = sim_dir, outid
        postfix = "_0" if kflag else ""  # useful when debugging with "-k" flag
        ext = '.h5' if h5 else '.dat'  # make it compatible with both output format
        
        # ci = cell interface, including left and right edge, and including ghost zone for radius and z
        # this is now hard-coded for 2D cylindrical
        self.ciphi = np.loadtxt(f"{sim_dir}/domain_x.dat")    # ciphi = cell interface for phi
        self.cir = np.loadtxt(f"{sim_dir}/domain_y.dat")      # cir = cell interface for radius        
        self.citheta = np.loadtxt(f"{sim_dir}/domain_z.dat")  # citheta = cell interface for theta
        if self.citheta.size <= 2:
            self.Ndim = 2
            raise NotImplementedError("This function is now hard-coded for 3D spherical only...")
        else:
            self.Ndim = 3
        # build cell center/left coordinates
        if kflag:  # if output ghost zone
            self.Nr, self.Nphi, self.Ntheta =  self.cir.size-1, self.ciphi.size-1, self.citheta.size-7
            self.ccr, self.clr = 0.5 * (self.cir[1:] + self.cir[:-1]), self.cir[:-1]
            self.cctheta, self.cltheta = 0.5 * (self.citheta[1:] + self.citheta[:-1]), self.citheta[:-1]
        else:
            self.Nr, self.Nphi, self.Ntheta =  self.cir.size-7, self.ciphi.size-1, self.citheta.size-7
            self.ccr, self.clr = 0.5 * (self.cir[3:-4] + self.cir[4:-3]), self.cir[3:-4]
            self.cctheta, self.cltheta = 0.5 * (self.citheta[3:-4] + self.citheta[4:-3]), self.citheta[3:-4]
        self.ccphi, self.clphi = 0.5 * (self.ciphi[1:] + self.ciphi[:-1]), self.ciphi[:-1]
        self.Nzyx = np.array([self.Ntheta, self.Nr, self.Nphi])
        self.shape = np.array([self.Nphi, self.Ntheta, self.Nr])
        self.time = t
        # also try to get a rough time from summary file
        # example line: OUTPUT 500 at simulation time 31415.9 (2023-6-11 22:55:18)
        summary_file = Path(f"{sim_dir}/summary{outid}.dat")
        self.summary_time = -1
        if summary_file.is_file():  # if file exist
            with open(summary_file, 'r') as f:
                for line in f:  # Read lines using for loop
                    if "at simulation time " in line:
                        self.summary_time = float(line.split("at simulation time ")[1].split()[0])
                        break        
        # now read
        if h5:
            h5_path = kwargs.get("h5_path", f"{sim_dir}/gas{outid}{ext}")
            with h5py.File(h5_path, 'r') as h5f:
                # Detect the data type of the 'dens' dataset
                # If it's fp32 or fp64, load directly and ensure float32
                self.dtype = h5f['dens'].dtype
                #print(f"Detected data type: {self.dtype}")                    
                # HDF5 files support lazy loading, meaning data is not actually read into memory until you explicitly request it. 
                # h5f['dens'] gives a proxy object that refers to the dataset. Using [:] loads the data into memory.
                self.rho_gas = h5f['dens'][:].reshape(self.Nzyx)
                self.ux = h5f['vx'][:].reshape(self.Nzyx)
                self.uy = h5f['vy'][:].reshape(self.Nzyx)
                if self.Ndim == 3:
                    self.uz = h5f['vz'][:].reshape(self.Nzyx)
                self.energy = h5f['energy'][:].reshape(self.Nzyx)                

        else:
            self.dtype = np.float64
            q = "gasdens"
            self.rho_gas = np.fromfile(f"{sim_dir}/{q}{outid}{postfix}{ext}").reshape(self.Nzyx)
            q = "gasvx"  # in Fargo3D+Spherical, this is azimuthal velocity, staggered at phi
            self.ux = np.fromfile(f"{sim_dir}/{q}{outid}{postfix}{ext}").reshape(self.Nzyx)
            q = "gasvy"  # in Fargo3D+Spherical, this is radial velocity, staggered at r
            self.uy = np.fromfile(f"{sim_dir}/{q}{outid}{postfix}{ext}").reshape(self.Nzyx)
            if self.Ndim == 3:
                q = "gasvz"  # in Fargo3D+Spherical, this is colatitude velocity, staggered at theta
                self.uz = np.fromfile(f"{sim_dir}/{q}{outid}{postfix}{ext}").reshape(self.Nzyx)
            q = "gasenergy"
            self.energy = np.fromfile(f"{sim_dir}/{q}{outid}{postfix}{ext}").reshape(self.Nzyx)
        
        # interpolate to cell centers
        if interp_method == "manual_linear":  # should be much faster
            self.uphi = np.zeros_like(self.ux)
            self.uphi[:, :, :-1] = 0.5 * (self.ux[:, :, :-1] + self.ux[:, :, 1:])
            self.uphi[:, :, -1] = self.ux[:, :, -1] + 0.5 * (self.ux[:, :, -1] - self.ux[:, :, -2])
            self.ur = np.zeros_like(self.uy)
            self.ur[:, :-1, :] = 0.5 * (self.uy[:, :-1, :] + self.uy[:, 1:, :])
            self.ur[:, -1, :] = self.uy[:, -1, :] + 0.5 * (self.uy[:, -1, :] - self.uy[:, -2, :])
            if self.Ndim == 3:
                self.utheta = np.zeros_like(self.uz)
                self.utheta[:-1, :, :] = 0.5 * (self.uz[:-1, :, :] + self.uz[1:, :, :])
                self.utheta[-1, :, :] = self.uz[-1, :, :] + 0.5 * (self.uz[-1, :, :] - self.uz[-2, :, :])
        else:
            mesh_cctheta, mesh_ccr, mesh_ccphi = np.meshgrid(self.cctheta, self.ccr, self.ccphi, indexing='ij')
            interp_cc_points = np.array([mesh_cctheta.ravel(), mesh_ccr.ravel(), mesh_ccphi.ravel()]).T

            # It seems RegularGridInterpolator by default returns float64 when calling, and there is no API to change that
            cc_ux_generator = sp.interpolate.RegularGridInterpolator((self.cctheta, self.ccr, self.clphi), self.ux, bounds_error=False, fill_value=None)
            self.uphi = cc_ux_generator(interp_cc_points, method=interp_method).astype(self.dtype).reshape(self.Nzyx)
            cc_uy_generator = sp.interpolate.RegularGridInterpolator((self.cctheta, self.clr, self.ccphi), self.uy, bounds_error=False, fill_value=None)            
            self.ur = cc_uy_generator(interp_cc_points, method=interp_method).astype(self.dtype).reshape(self.Nzyx)
            if self.Ndim == 3:
                cc_uz_generator = sp.interpolate.RegularGridInterpolator((self.cltheta, self.ccr, self.ccphi), self.uz, bounds_error=False, fill_value=None)            
                self.utheta = cc_uz_generator(interp_cc_points, method=interp_method).astype(self.dtype).reshape(self.Nzyx)