import math, pickle
import numpy as np
import cupy as cp
from cupyx.scipy.special import sph_harm as cp_sph_harm
import mrcfile
from joblib import Parallel, delayed
from scipy.ndimage import map_coordinates
import scipy
import scipy.ndimage
import time
from numba import jit
from tqdm import tqdm
import random
import ray
import json
import sys
import nvidia_smi
import zarr
import psutil
from numcodecs import Blosc
import os, textwrap, argparse
from indexing import *
import numpy.lib.recfunctions as rf
np.seterr(invalid='ignore')

class Moments:
    def __init__(self):
        print("Created `Moments` instance")
        self.order = None
        self.voxdim = 64
        self.dimensions = None
        self.apix = 1
        self.moments = {}
        self.invariants = {}
        self.ndmoments = {}
        self.ndinvariants = {}
        self.scaled_moments = {}
        self.latents = None

    def _divide_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def _chunk_list(self, seq, size):
        return (list[i::size] for i in range(size))

    def _ndmoment_to_ndarray(self):
        arr = np.zeros((self.dimensions,self.order+1, self.order+1, self.order+1), dtype=np.complex64)
        mask = np.zeros((self.order+1, self.order+1, self.order+1), dtype=bool)
        for dim, moments in self.ndmoments.items():
            for nlm, mom in moments.items():
                n,l,m = nlm
                arr[dim,n,l,m] = mom
                mask[n,l,m] = True
        return arr, mask

    def add_dimension(self):
        dim = len(self.ndmoments)
        self.ndmoments[dim] = self.moments
        self.moments = {}

    def calculate_invariants(self):
        for dimension, moments in self.ndmoments.items():
            partial = {}
            for nlm, mom in moments.items():
                n, l, m = nlm
                if m == 0:
                    partial.setdefault((n, l), []).append(np.abs(mom))
                else:
                    partial.setdefault((n, l), []).append(np.abs(mom))
                    partial.setdefault((n, l), []).append(np.abs(mom))

            for nl, invar in partial.items():
                self.invariants[nl] = np.linalg.norm(invar)
            self.ndinvariants[dimension] = self.invariants
            self.invariants = {}

    def apply_scaling(self, z_ind=None, scales=None):
        momentsarray, momentsmask = self._ndmoment_to_ndarray()
        self.scales = np.hstack([np.ones_like(self.latents[:, :1]), self.latents])
        if z_ind is None and scales is None:
            print(f"No latent coordinates were provided `apply_scaling(scales=(3.2,4.5,-0.9,...,3.1,0.2))` and "
                  f"no Z-index was provided `apply_scaling(z_ind=5537)`, so returning a random latent coordinate.")
            z_ind = np.random.choice(len(self.latents))
            print(f"Chose {self.scales[z_ind]}")
            scaled_moments = np.tensordot(self.scales[z_ind], momentsarray, axes=([-1], [0]))
        elif z_ind is not None:
            scaled_moments = np.tensordot(self.scales[z_ind], momentsarray, axes=([-1], [0]))
        elif scales is not None:
            scales = np.array([1,*scales]).astype(np.float32)
            scaled_moments = np.tensordot(scales, momentsarray, axes=([-1], [0]))
        else:
            scaled_moments = None

        self.scaled_moments = {(n, l, m): moment for (n, l, m), moment in np.ndenumerate(scaled_moments) if momentsmask[n,l,m]}

    def get_moment_size(self):
        if self.ndmoments:
            return len(self.ndmoments[0])
        else:
            return len(self.moments)

    def get_order(self):
        if self.ndmoments:
            return list(self.ndmoments[0].keys())[-1][0]
        else:
            return list(self.moments.keys())[-1][0]

    def get_size(self):
        if self.ndmoments:
            return sum([self._size(v) for v in self.ndmoments.values()])
        else:
            return self._size(self.moments)

    def stats(self, quiet=False):
        self.dimensions = len(self.ndmoments)
        if len(self.ndinvariants) == 0: initialised = False
        else: initialised = True

        moment_size = self.get_moment_size()
        self.order = self.get_order()
        size = self.get_size()
        if not quiet:
            if bool(self.ndmoments):
                type = self.ndmoments[0][(0, 0, 0)].dtype
            else:
                type = self.moments[(0,0,0)].dtype
            print("+-------------------------------------------------------+")
            print("|{:<55}|".format(f"Moment statistics"))
            print("|{:<55}|".format(f"   Dimensions: {self.dimensions}"))
            print("|{:<55}|".format(f"   Moments: {moment_size}"))
            print("|{:<55}|".format(f"   Order: {self.order}"))
            print("|{:<55}|".format(f"   nbytes: {size}"))
            print("|{:<55}|".format(f"   type: {type}"))
            print("|{:<55}|".format(f""))
            print("|{:<55}|".format(f"   Invariants: {initialised}"))
            if initialised:
                print("|{:<55}|".format(f"   Total: {len(self.ndinvariants[0])}"))
                print("|{:<55}|".format(f"   nbytes: {sum([self._size(v) for v in self.ndinvariants.values()])}"))
                print("|{:<55}|".format(f"   type: {self.ndinvariants[0][(0, 0)].dtype}"))
            if self.latents is not None:
                print("|{:<55}|".format(f"   particles: {self.latents.shape}"))
            print("+-------------------------------------------------------+")

    def _size(self, obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([v.nbytes for v in obj.values()])
            size += sum([self._size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += self._size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self._size(i, seen) for i in obj])
        return size

    def check_allowed(self, n, l, m):
        if n <= self.order:
            if l <= n:
                if (n - l) % 2 == 0:
                    if -l <= m <= l:
                        return True
                    else:
                        print(f"m = {m} not allowed, required: -l <= m <= l")
                        return False
                else:
                    print(f"l = {l} not allowed, n - l must be even")
                    return False
            else:
                print(f"l = {l} not allowed, required: l < n or l = n")
                return False
        else:
            print(f"n = {n} not allowed, required: n = {self.order}")
            return False

    def get_moment(self, n, l, m, latent_dim=0):
        if self.check_allowed(n,l,m):
            if self.ndmoments:
                return self.ndmoments[latent_dim][(n, l, m)]
            else:
                return self.moments[(n, l, m)]

    def get_invariant(self, n, l, latent_dim=0):
        if self.check_allowed(n,l,m=0):
            if self.ndinvariants:
                return self.ndinvariants[latent_dim][(n, l)]
            else:
                return self.invariants[(n, l)]

    def set_moment(self, n, l, m, moment):
        self.moments[(n, l, m)] = moment

    def set_latents(self, particles, components):
        keys = [''.join(('components_mode_', str(component), '/value')) for component, _ in
                enumerate(range(components))]
        n = particles[keys]
        n = rf.repack_fields(n)
        self.latents = n.copy().view('<f4').reshape((n.shape[0], components))

    def display_moments(self, latent_dim=0, nlm=None, n_max=None):
        if nlm is not None:
            n,l,m = nlm
            if m < 0:
                mom = self.get_moment(n=n, l=l, m=abs(m), latent_dim=latent_dim)
                print(((-1) ** (m)) * np.conj(mom))
            else:
                print(self.get_moment(n=n, l=l, m=m, latent_dim=latent_dim))
            return

        if self.ndmoments:
            try:
                self.ndmoments[latent_dim]
            except:
                print(f"Key error: There are only {self.dimensions} dimensions, you gave dim={latent_dim}")
            else:
                for key, val in self.ndmoments[latent_dim].items():
                    if n_max is None:
                        print(key, val)
                    elif key[0] == n_max+1:
                        break
                    else:
                        print(key, val)
        else:
            for key, val in self.moments.items():
                if n_max is None:
                    print(key, val)
                elif key[0] == n_max+1:
                    break
                else:
                    print(key, val)

    def display_invariants(self, latent_dim=0, nl=None, n_max=None):
        if nl is not None:
            n, l = nl
            print(self.get_invariant(n=n, l=l, latent_dim=latent_dim))
            return

        if self.ndinvariants:
            try:
                self.ndinvariants[latent_dim]
            except:
                print(f"Key error: There are only {self.dimensions} dimensions, you gave dim={latent_dim}")
            else:
                for key, val in self.ndinvariants[latent_dim].items():
                    if n_max is None:
                        print(key, val)
                    elif key[0] == n_max + 1:
                        break
                    else:
                        print(key, val)
        else:
            for key, val in self.invariants.items():
                if n_max is None:
                    print(key, val)
                elif key[0] == n_max + 1:
                    break
                else:
                    print(key, val)

    def load(self, input):
        print(f"Loading {input} into Moments object")
        with np.load(input) as data:
            if ('arr_0' in data.files) and (len(data.files)==3):
                print(f"Detected full precision moments (single).")
                moments = data['arr_0']
                self.voxdim = data['dim']
                self.apix = data['apix']

                for mu, mom in enumerate(moments):
                    n, l, m = mu_to_nlm(mu)
                    self.set_moment(n, l, m, mom)
                self.add_dimension()
            elif ('arr_0' in data.files) and (len(data.files) > 3):
                print(f"Detected full precision moments (multiple).")
                self.voxdim = data['dim']
                self.apix = data['apix']
                self.latents = data['latents']
                files = data.files
                files.remove('dim')
                files.remove('apix')
                files.remove('latents')

                for file in files:
                    moments = data[file]
                    for mu, mom in enumerate(moments):
                        n, l, m = mu_to_nlm(mu)
                        self.set_moment(n, l, m, mom)
                    self.add_dimension()
            elif ('args_0' in data.files) and (len(data.files)==5):
                print(f"Detected 8 bit precision moments (single).")
                self.voxdim = data['dim']
                self.apix = data['apix']
                dr_min, dr_max = data['deltas_0'][0], data['deltas_0'][1]
                r_vals = np.geomspace(dr_min, dr_max, num=256)
                r = r_vals[data['args_0']]
                theta = data['phases_0'] * data['deltas_0'][2]
                z = r * (np.cos(theta) + np.sin(theta) * 1j)
                for mu, mom in enumerate(z):
                    n, l, m = mu_to_nlm(mu)
                    self.set_moment(n, l, m, mom)
                self.add_dimension()
            elif ('args_0' in data.files) and (len(data.files) > 5):
                print(f"Detected 8 bit precision moments (multiple).")
                self.voxdim = data['dim']
                self.apix = data['apix']
                self.latents = data['latents']
                arrays = data.files
                arrays.remove('dim')
                arrays.remove('apix')
                files.remove('latents')

                for group in self._divide_chunks(arrays, 3):
                    args, phases, deltas = group
                    dr_min, dr_max = data[deltas][0], data[deltas][1]
                    r_vals = np.geomspace(dr_min, dr_max, num=256)
                    r = r_vals[data[args]]
                    theta = data[phases] * data[deltas][2]
                    z = r * (np.cos(theta) + np.sin(theta) * 1j)
                    for mu, mom in enumerate(z):
                        n, l, m = mu_to_nlm(mu)
                        self.set_moment(n, l, m, mom)
                    self.add_dimension()
            self.stats(quiet=True)
        return len(self.ndmoments)

    def save(self, output='moments', compress=False, dim=64, apix=1):
        if self.ndmoments:
            i = 0
            _temp = {}
            for key, moments in self.ndmoments.items():
                if compress:
                    args_8bit, phases_8bit, deltas = self._save_compressed_moments(moments)
                    _temp['args_'+str(i)] = args_8bit
                    _temp['phases_' + str(i)] = phases_8bit
                    _temp['deltas_' + str(i)] = deltas
                else:
                    _out = self._save_moments(moments)
                    _temp['arr_'+str(i)] = _out
                i += 1
            if compress:
                file, ext = output.split('.')
                output = str(file)+'.8bit'
            np.savez_compressed(output, latents=self.latents, dim=dim, apix=apix, **_temp)
        else:
            if compress:
                file, ext = output.split('.')
                output = str(file) + '.8bit'
                args_8bit, phases_8bit, deltas = self._save_compressed_moments(self.moments)
                np.savez_compressed(output, dim=dim, apix=apix, args_0=args_8bit, phases_0=phases_8bit, deltas_0=deltas)
            else:
                _out = self._save_moments(self.moments)
                np.savez_compressed(output, dim=dim, apix=apix, arr_0=_out)

    def _save_compressed_moments(self, moments):
        temp = []
        for key, val in sorted(moments.items()):
            temp.append(val)
        z = np.array(temp).astype(np.complex64)
        args, phases = np.abs(z), np.angle(z)
        dr_min, dr_max = min(args), max(args)
        dr_min = 1e-100 if dr_min==0 else dr_min
        rbins = np.geomspace(dr_min, dr_max, num=256)
        dtheta = 2 * np.pi / 256
        tbins = np.arange(-np.pi, np.pi, dtheta)
        args_8bit = np.digitize(args, rbins, right=False).astype(np.uint8)
        phases_8bit = np.digitize(phases, tbins, right=False).astype(np.uint8)
        deltas = np.array([dr_min, dr_max, dtheta]).astype(np.float64)
        return args_8bit, phases_8bit, deltas

    def _save_moments(self, moments):
        temp = []
        for key, val in sorted(moments.items()):
            temp.append(val)
        return np.array(temp).astype(np.complex64)

class ComplexArray:
    """
    A class to convert np.array or cp.array of type complex64 to a reduced 8bit format and vice-versa

    Class ComplexArray is a utility class designed to convert NumPy or CuPy arrays of complex numbers
    with 64-bit precision into a reduced 8-bit format and vice versa. The purpose of this class is to
    reduce memory usage while retaining the essential information required for many signal processing
    tasks. The class provides two methods for conversion: reduce_to_8bit() method to convert from
    complex64 to 8-bit format, and read_as_complex64() method to convert back from 8-bit format to
    complex64. The class also includes helper methods _polar2rect() and _rect2polar() to convert
    between rectangular and polar representations of complex numbers.
    """
    def __init__(self):
        # self.tbins = xp.arange(-xp.pi, xp.pi, 2 * xp.pi / 256)
        print('8bit precision mode (low memory)')

    def _polar2rect(self, r, theta, gpu=False):
        xp = cp if gpu else np
        return r * (xp.cos(theta) + xp.sin(theta) * 1j)

    def _rect2polar(self, z, gpu=False):
        xp = cp if gpu else np
        return xp.abs(z), xp.angle(z)

    def reduce_to_8bit(self, complex_array, in_gpu=False, out_gpu=False):
        """
        Converts an array of complex numbers of type complex64 to a reduced 8-bit format.

        Args:
            complex_array (ndarray or cupy.ndarray): An array of complex numbers of type complex64 to be converted.
            in_gpu (bool, optional): A boolean flag indicating whether the input complex array is stored in CuPy format.
                Defaults to False.
            out_gpu (bool, optional): A boolean flag indicating whether the output reduced 8-bit format complex array
                should be stored in CuPy format. Defaults to False.

        Returns:
            tuple: A tuple containing the reduced 8-bit format complex array and its associated parameters.
                The tuple has the form (args, phases, dr, dtheta), where:
                - args (ndarray or cupy.ndarray): An array of shape (N,) containing the quantized magnitudes of the complex numbers.
                - phases (ndarray or cupy.ndarray): An array of shape (N,) containing the quantized phases of the complex numbers.
                - dr (float): The magnitude resolution of the quantized values.
                - dtheta (float): The phase resolution of the quantized values.
                The output arrays are of type uint8.
                If in_gpu is True and out_gpu is False, the output arrays are in NumPy format.
                If in_gpu is False and out_gpu is True, the output arrays are in CuPy format.
                If both in_gpu and out_gpu are False, the output arrays are in NumPy format.
                If both in_gpu and out_gpu are True, the output arrays are in CuPy format.
        """
        xp = cp if in_gpu else np
        args, phases = self._rect2polar(complex_array, in_gpu)
        max_args = float(xp.amax(args))
        dr = max_args/256
        if dr == 0:
            print('got zero')
            return (xp.zeros(complex_array.shape, dtype=np.uint8), xp.zeros_like(complex_array, dtype=np.uint8), 0, 0)

        rbins = xp.arange(0, max_args, dr)
        _args = xp.digitize(args, rbins, right=False).astype(np.uint8)

        dtheta = 2*np.pi/256
        tbins = xp.arange(-np.pi, np.pi, dtheta)
        _phases = xp.digitize(phases, tbins, right=False).astype(np.uint8)

        del args, phases, tbins, rbins
        if in_gpu and not out_gpu:
            return (cp.asnumpy(_args), cp.asnumpy(_phases), dr, dtheta)
        elif not in_gpu and out_gpu:
            return (cp.asarray(_args), cp.asarray(_phases), dr, dtheta)
        else:
            return (_args, _phases, dr, dtheta)

    def read_as_complex64(self, complex_array_8bit, in_gpu=False, out_gpu=False):
        """
        Converts a complex array represented by a reduced 8-bit format and associated parameters back to an array of type complex64.

        Args:
            complex_array_8bit (tuple): A tuple containing the reduced 8-bit format complex array and its associated parameters.
                The tuple should have the form (args, phases, dr, dtheta), where:
                - args (ndarray): An array of shape (N,) containing the quantized magnitudes of the complex numbers.
                - phases (ndarray): An array of shape (N,) containing the quantized phases of the complex numbers.
                - dr (float): The magnitude resolution of the quantized values.
                - dtheta (float): The phase resolution of the quantized values.
            in_gpu (bool, optional): A boolean flag indicating whether the input complex array is stored in CuPy format.
                Defaults to False.
            out_gpu (bool, optional): A boolean flag indicating whether the output complex array should be stored in CuPy format.
                Defaults to False.

        Returns:
            ndarray or cupy.ndarray: An array of type complex64 with the original precision and shape.
                If in_gpu is True and out_gpu is False, the output array is in NumPy format.
                If in_gpu is False and out_gpu is True, the output array is in CuPy format.
                If both in_gpu and out_gpu are False, the output array is in NumPy format.
                If both in_gpu and out_gpu are True, the output array is in CuPy format.
        """
        args, phases, dr, dtheta = complex_array_8bit
        complex64 = self._polar2rect(args * dr, phases * dtheta, in_gpu)

        if in_gpu and not out_gpu:
            return cp.asnumpy(complex64).astype(np.complex64)
        elif not in_gpu and out_gpu:
            return cp.asarray(complex64).astype(np.complex64)
        else:
            return complex64.astype(np.complex64)

    def __size__(self, input):
        return input[0].nbytes + input[1].nbytes + input[2].__sizeof__() + input[3].__sizeof__() + input.__sizeof__()

class ZernikeMonomial:
    # Modified from Scott Grandison, Richard Morris

    ########################
    # This code takes a voxelised object and represents the shape of the object by computing a set
    # of Zernike moments.  It is also able to take a set of moments and from them reconstruct a
    # voxelised object.  Code is present that will load a pdb (protein databank) file and creat
    # a voxelised object from the protein structure.
    #
    # The code was used in scientific research and as far as I know is correct.  However it is
    # mostly uncommented and can be improved in many ways.  For further details please see the
    # article:
    #
    # Grandison S., Roberts C., Morris R. J. (2009)
    # The application of 3D zernike moments for the description of "model-free" molecular
    # structure, functional motion, and structural reliability
    # Journal of Computational Biology 16 (3) 487-500
    # DOI:10.1089/cmb.2008.0083
    #
    # Or contact the author direct:  s.grandison@uea.ac.uk
    #
    # Copyright (C) 2009  Scott Grandison, Richard Morris
    #
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>.
    #######################

    "A class for calculating Zernike moments from voxels"
    factdict = {}
    bindict = {}
    clmdict = {}
    Qklvdict = {}
    voxels = 0
    axis = 0
    axes = 0
    tabSpaceSum = []
    reconmoments = []
    chidict = {}
    H_nlm = {}
    P_lm_dict = {}

    #RAM STORAGE
    R_nl_dict = {}
    Y_lm_dict = {}

    Y_lm = {'vram': {}, 'ram': {}, 'zarr': None}
    R_nl = {'vram': {}, 'ram': {}, 'zarr': None}
    Y_lm_memory_map = {}
    R_nl_memory_map = {}

    def __init__(self):
        print("Created a Zernike calculator instance")

    def crop_array(self, map, crop):
        if map.shape[0] != map.shape[1] != map.shape[2]:
            print("Map is not cubic... taking the first dimension. This may fail.")
        if map.shape[0] % 2 == 0:
            m = int((map.shape[0] / 2) - 1)
            l = m - crop//2
            h = m + crop//2
            return map[l:h, l:h, l:h]
        else:
            m = int((map.shape[0] / 2) - 1)
            l = m - crop//2
            h = m + crop//2
            return map[l:h, l:h, l:h]

    def cartesian_2_polar(self, x, y, z):
        r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        theta[y < 0] = np.pi * 2 - theta[y < 0]
        return r, theta, phi

    def interpolate_polar2cartesian(self, vol, o=None, r=None, output=None, order=1):  # Extend to 3D
        if r is None: r = vol.shape[0]
        if output is None:
            output = np.zeros((r, r, r), dtype=vol.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=vol.dtype)
        if o is None: o = np.array(output.shape) / 2 - 0.5
        x, y, z = output.shape
        xs, ys, zs = np.mgrid[:x, :y, :z] - o[:, None, None, None]
        rs = (ys ** 2 + xs ** 2 + zs ** 2) ** 0.5
        ts = np.arccos(zs / rs)
        ps = np.arctan2(ys, xs)

        ts[ys < 0] = np.pi * 2 - ts[ys < 0]
        # ts *= (vol.shape[1] - 1) / (np.pi * 2)
        # ps[ps < 0] = np.pi - ps[ps < 0]
        map_coordinates(vol, (ps, ts, rs), order=order, output=output)
        return output

    def precompute_binominals(self, N):
        @np.vectorize
        def binomial(n, r):
            return math.comb(n, r)
        q = np.linspace(0, N, N+1, dtype=int)
        n, r = np.meshgrid(q,q,indexing='ij')
        return binomial(n, r)

    def precompute_clm(self, order):
        dict = {(0, 0): 1}
        for l in range(1, order + 1):
            for m in range(0, l + 1):
                if m == 0:
                    C = math.sqrt(2*l+1)
                else:
                    C = math.sqrt((l + m) / (l - m + 1)) * dict[(l, m-1)]
                dict[(l, m)] = C
        return dict

    def precompute_Tklv(self, order):
        dict = {}
        for n in range(0, order + 1):
            for l in range(0, n + 1):
                if ((n - l) % 2) == 0:
                    k_dict = {}
                    for k in range(0, int(((n - l) / 2)) + 1):
                        v_list = []
                        for v in range(0, k + 1):
                            T = (math.comb(2 * k, k) * math.comb(k, v) * math.comb(2 * (k + l + v) + 1, 2 * k)) / math.comb(k + l + v, k)
                            v_list.append(T)
                        k_dict[k] = v_list
                    dict[l] = k_dict
        return dict

    def R_nl_recurrence(self, n, l, r):
        if l == n:
            R = r**n
            self.R_nl_dict[(n, l)] = R
            return R
        elif l == (n - 2):
            R = (n + 0.5)*(r**n) - (n - 0.5)*(r**(n-2))
            self.R_nl_dict[(n, l)] = R
            return R
        elif l <= (n - 4):
            k_0 = (n - l) * (n + l + 1) * (2 * n - 3)
            k_1 = (2 * n - 1) * (2 * n + 1) * (2 * n - 3)
            k_2 = 0.5 * (-2 * n + 1) * ((2 * l + 1)**2) - (k_1/2)
            k_3 = -1 * (n - l - 2) * (n + l - 1) * (2 * n + 1)
            K_1 = k_1 / k_0
            K_2 = k_2 / k_0
            K_3 = k_3 / k_0

            R = (K_1*(r**2)+K_2)*self.R_nl_dict.get((n - 2, l), self.R_nl(n - 2, l, r)) + K_3*self.R_nl_dict.get((n - 4, l), self.R_nl(n-4, l, r))
            self.R_nl_dict[(n,l)] = R
            return R
        else:
            print(f"Error, got invalid n or l")

    def Y_lm_recurrence(self, l, m, theta, phi):
        if l==m==0:
            Y = 0.5 * np.sqrt(1 / np.pi)
            self.Y_lm_dict[(0,0)] = Y
            return Y
        elif l > 1 and 0 <= m < (l-1):
            Y = np.sqrt(((2*l+1)*(2*l-1))/((l+m)*(l-m)))*np.cos(theta)*self.Y_lm_dict[(l-1,m)] - \
                np.sqrt(((2*l+1)*(l+m-1)*(l-m-1))/((2*l+3)*(l+m)*(l-m)))*self.Y_lm_dict[(l-2,m)]
            self.Y_lm_dict[(l,m)] = Y
            return Y
        elif l > 0 and m == (l-1):
            Y = np.sqrt(2*l+1)*np.cos(theta)*self.Y_lm_dict[(l-1,l-1)]
            self.Y_lm_dict[(l,m)] = Y
            return Y
        elif l > 0 and m == l:
            Y = -1*np.sin(theta)*np.sqrt((2*l+1)/(2*l))*np.exp(1j*phi)*self.Y_lm_dict[(l-1,l-1)]
            self.Y_lm_dict[(l,m)] = Y
            return Y
        else:
            print(f"Error, got invalid l or m")

    def Y_lm_scipy(self, l, m, theta, phi, save=False):
        try:
            Y = self.Y_lm_dict[(l, m)]
        except:
            Y = np.nan_to_num(scipy.special.sph_harm(m, l, phi, theta))
            if save:
                self.Y_lm_dict[(l, m)] = Y
        return Y

    def R_nl_scipy(self, n, l, r, save=False):
        try:
            R = self.R_nl_dict[(n, l)]
        except:
            R = np.nan_to_num((r**l) * scipy.special.jacobi(int(n-l)/2, 0, l + 0.5)(2*(r**2)-1))
            if save:
                self.R_nl_dict[(n, l)] = R
        return R

    def _get_Ylm(self, l, m):
        loc = Y_lm_memory_map[(l,m)]
        if loc == 'zarr':
            return Y_lm_zarr[l, m, :]
        else:
            return Y_lm[loc][(l, m)]

    def _set_Ylm(self, l, m, array, mem):
        Y_lm_memory_map[(l, m)] = mem
        Y_lm[mem][(l, m)] = array

    def _get_Rnl(self, n, l):
        loc = R_nl_memory_map[(n,l)]
        if loc == 'zarr':
            return R_nl_zarr[n, l, :]
        else:
            return R_nl[loc][(n,l)]

    def _set_Rnl(self, n, l, array, mem):
        R_nl_memory_map[(n, l)] = mem
        R_nl[mem][(n, l)] = array

    @jit(nopython=True)
    def P_lm(self, l, m, theta):
        if l==m==0:
            P = np.ones_like(theta)
            self.P_lm_dict[(0,0)] = P
            return P
        elif l==m:
            C3 = np.sqrt((2 * l + 1) / 2 * l)
            P = -C3 * np.sin(theta)*self.P_lm_dict[(l-1, m-1)]
            self.P_lm_dict[(l,m)] = P
            return P
        elif m < l-1:
            C = np.sqrt((2 * l + 1) / ((l + m) * (l - m)))
            C1 = C * np.sqrt(2 * l + 1)
            C2 = C * np.sqrt(((l + m - 1) * (l - m - 1)) / (2 * l - 3))
            P = C1 * np.cos(theta)*self.P_lm_dict[(l-1, m)]-C2*self.P_lm_dict[(l-2, m)]
            self.P_lm_dict[(l, m)] = P
            return P
        elif m == l-1:
            P = np.zeros_like(theta)
            self.P_lm_dict[(l, m)] = P
            return P

    @jit(nopython=True)
    def Qklv(self, k, l, v):
        if k == 0:
            return math.sqrt((2*l+3)/3)
        else:
            return (((-1)**(k + v)) / (2**(2 * k))) * math.sqrt((2 * l + 4 * k + 3) / 3) * self.Tklv[l][k][v]

    @jit(nopython=True)
    def sum6(self, n, l, m, nu, alpha, beta, u, mu, x, y, z):
        sum6 = 0 + 0j
        for v in range(mu + 1):
            r = 2 * (v + alpha) + u
            s = 2 * (mu - v + beta) + m - u
            t = 2 * (nu - alpha - beta - mu) + l - m
            if x is None:
                sum6 += self.nCr[mu][v] * self.tabSpaceSum[r][s][t]
            else:
                sum6 += self.nCr[mu][v] * (x**r) * (y**s) * (z**t)
        return sum6

    @jit(nopython=True)
    def sum5(self, n, l, m, nu, alpha, beta, u, x, y, z):
        sum5 = 0 + 0j
        for mu in range(int((l - m) // 2) + 1):
            sum5 += ((-1)**mu) * (2**(-2*mu)) * self.nCr[l][mu] * self.nCr[l-mu][m+mu] * self.sum6(n, l, m, nu, alpha, beta, u, mu, x, y, z)
        return sum5

    @jit(nopython=True)
    def sum4(self, n, l, m, k, nu, alpha, beta, x, y, z):
        sum4 = 0 + 0j
        for u in range(m + 1):
            sum4 += ((-1)**(m - u)) * self.nCr[m][u] * (1j**u) * self.sum5(n, l, m, nu, alpha, beta, u, x, y, z)
        return sum4

    @jit(nopython=True)
    def sum3(self, n, l, m, k, nu, alpha, x, y, z):
        sum3 = 0 + 0j
        for beta in range(nu - alpha + 1):
            sum3 += self.nCr[nu-alpha][beta] * self.sum4(n, l, m, k, nu, alpha, beta, x, y, z)
        return sum3

    @jit(nopython=True)
    def sum2(self, n, l, m, k, nu, x, y, z):
        sum2 = 0 + 0j
        for alpha in range(nu + 1):
            sum2 += self.nCr[nu][alpha] * self.sum3(n, l, m, k, nu, alpha, x, y, z)
        return sum2

    @jit(nopython=True)
    def sum1(self, n, l, m, k, x, y, z):
        sum1 = 0 + 0j
        for nu in range(int(k+1)):
            sum1 += self.Qklv(k, l, nu) * self.sum2(n, l, m, k, nu, x, y, z)
        return sum1

    @jit(nopython=True)
    def Chinlmrst(self, n, l, m, x=None, y=None, z=None, do_reconstruct=False):
        if do_reconstruct:
            try:
                val = self.chidict[(n, l, m)]
            except:
                k = (n - l) / 2
                coeff = self.clm[(l, abs(m))] * (2.0**(-m))
                coeff = coeff * 0.75 / 3.1415926
                val = coeff * self.sum1(n, l, m, k, x, y, z)
                self.chidict[(n, l, m)] = val
        else:
            k = (n - l) / 2
            coeff = self.clm[(l, abs(m))] * (2.0**(-m))
            coeff = coeff * 0.75 / 3.1415926
            val = coeff * self.sum1(n, l, m, k, x, y, z)
        return val

    def SpaceSum(self, a, b, g, x, y, z, mask, step):
        temp2 = (x+step)**(a + 1) - x**(a + 1)
        temp2 *= (y+step)**(b + 1) - y**(b + 1)
        temp2 *= (z+step)**(g + 1) - z**(g + 1)
        temp2 /= (a + 1) * (b + 1) * (g + 1)
        temp2 = cp.multiply(temp2, cp.asarray(self.voxels.voxels_array))
        return cp.asnumpy(cp.sum(temp2*mask))

    def ConstructSpaceSumTable(self, order):
        # side = np.linspace(-1, 1 - 2/self.voxels.GetResolution(), self.voxels.GetResolution())
        side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 96)
        x, y, z = cp.asarray(np.meshgrid(side, side, side, indexing='ij'))
        mask = cp.asarray(np.where(((x ** 2 + y ** 2 + z ** 2) ** 0.5) <= 1.0, 1, 0))
        diff = side[1]-side[0]
        for r in range(order + 1):
            print("Constructing order", r, "integral")
            page = []
            for s in range(order + 1):
                line = []
                for t in range(order + 1):
                    val = 0.0
                    if r + s + t <= order:
                        val = self.SpaceSum(r, s, t, x, y, z, mask, diff)
                    line.append(val)
                page.append(line)
            self.tabSpaceSum.append(page)
        self.tabSpaceSum = np.array(self.tabSpaceSum)

    def CalculateMoments_Monomial_sysmem(self, passedvoxels, order):
        self.voxels = passedvoxels
        self.order = order
        self.clm = self.precompute_clm(self.order)
        self.Tklv = self.precompute_Tklv(self.order)
        self.nCr = self.precompute_binominals(self.order)
        self.ConstructSpaceSumTable(order)

        calcedmoments = Moments()
        for n in range(order + 1):
            print("Doing:", n)
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(l + 1):
                        chi = self.Chinlmrst(n, l, m)
                        calcedmoments.SetMoment(n, l, m, chi)
                        if m > 0:
                            negchi = (-1)**m * np.conj(chi)
                            calcedmoments.SetMoment(n, l, -m, negchi)
        return calcedmoments

    def CalculateMoments_Spherical_sysmem(self, passedvoxels, order):
        self.voxels = passedvoxels
        self.order = order
        side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 128)
        x, y, z = np.asarray(np.meshgrid(side, side, side))
        r, theta, phi = self.cartesian_2_polar(x, y, z)
        volume = cp.asarray(self.voxels.voxels_array)
        N = self.voxels.voxels_array.shape[0]

        calcedmoments = Moments()
        for n in range(order + 1):
            print("Doing:", n)
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    R = cp.asarray(self.R_nl_scipy(n, l, r, save=True))
                    if np.isnan(np.sum(R)):
                        print(f"Detected NaN in {n, l, m} of R")
                    for m in range(l + 1):
                        Y = cp.asarray(self.Y_lm_scipy(l, m, theta, phi, save=True))
                        if np.isnan(np.sum(R)):
                            print(f"Detected NaN in {n, l, m} of Y")
                        Z = cp.conj(R * Y)
                        coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * cp.pi * (N**3))
                        inner_product = cp.tensordot(volume, Z, axes=3)
                        if np.isnan(np.sum(inner_product)):
                            print(f"Detected NaN in {n, l, m} of inner_product")
                        chi = coeff * cp.asnumpy(inner_product)
                        calcedmoments.SetMoment(n, l, m, chi)
                        if m > 0:
                            negchi = (-1)**m * np.conj(chi)
                            calcedmoments.SetMoment(n, l, -m, negchi)
        return calcedmoments

    def CalculateMoments_Spherical_sysmem_ray(self, passedvoxels, order, mode='serial'):
        num_cpus = 16
        ray.init(num_cpus=num_cpus,
                 num_gpus=4,
                 _system_config={
                     "object_spilling_config": json.dumps({"type": "filesystem",
                                                           "params": {"directory_path": "/mnt/SSD/spill"}
                                                           })
                 }
                 )
        self.voxels = passedvoxels
        volume = self.voxels.voxels_array
        dim = volume.shape[0]
        side = np.linspace(-1 / np.sqrt(3), 1 / np.sqrt(3), dim)
        x, y, z = np.asarray(np.meshgrid(side, side, side))
        r, theta, phi = self.cartesian_2_polar(x, y, z)

        N = self.voxels.voxels_array.shape[0]
        self.moments = Moments()

        R_nl_dict = {}
        Y_lm_dict = {}

        N_of_R = ((order + 2) / 2) ** 2
        N_of_Y_approx = 2 * N_of_R
        Y_size_p = N_of_Y_approx * (dim**3) * 8
        R_size_p = N_of_R * (dim**3) * 4
        print(f"Predicted size of Y(l, m) array : {(Y_size_p/1e9):.3f} Gb")
        print(f"Predicted size of R(n, l) array : {(R_size_p/1e9):.3f} Gb")

        nvidia_smi.nvmlInit()

        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"Device {i}: {nvidia_smi.nvmlDeviceGetName(handle)}, "
                  f"Memory: {(100 * info.free / info.total):.2f}% free. "
                  f"[{(info.total/1e9):.4f} (total), {(info.free/1e9):.4f} (free), {(info.used/1e9):.4f} (used)]")
        if info.free < Y_size_p+R_size_p:
            print("Insufficent GPU VRAM, fallback to mode 2")
            # mode = 'serial'
            mode = 'GPU_Ylm'
        else:
            print("Looks like this can all be done in GPU VRAM!")
            mode = 'GPU_Ylm'

        nvidia_smi.nvmlShutdown()

        def numpy_math_polar2rect(r, theta, gpu=False):
            xp = cp if gpu else np
            return r * (xp.cos(theta) + xp.sin(theta) * 1j)

        def numpy_math_rect2polar(z, gpu=False):
            xp = cp if gpu else np
            return xp.abs(z), xp.angle(z)

        def reduce_to_8bit(complex_array, gpu=False):
            xp = cp if gpu else np
            args, phases = numpy_math_rect2polar(complex_array, gpu)
            max_args = xp.max(args)
            rcounts, rbins = xp.histogram(args, bins=255, range=(0, max_args))
            args = xp.digitize(args, rbins, right=False).astype(xp.uint8)
            dr = rbins[1] - rbins[0]

            tcounts, tbins = xp.histogram(phases, bins=255, range=(-xp.pi, xp.pi))
            phases = xp.digitize(phases, tbins, right=False).astype(xp.uint8)
            dtheta = tbins[1] - tbins[0]
            return (args, phases, dr, dtheta)

        def read_as_complex64(complex_array_8bit, gpu=False):
            args, phases, dr, dtheta = complex_array_8bit
            complex64 = numpy_math_polar2rect(args * dr, phases * dtheta, gpu)
            return complex64

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        def chunker_list(seq, size):
            return (seq[i::size] for i in range(size))

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        @ray.remote(num_gpus=1)
        def _calculate_Y_GPU(jobs, theta, phi):
            dict = {}
            for job in jobs:
                l, m = job
                _arr = cp.nan_to_num(cp_sph_harm(m, l, phi, theta)).astype(cp.complex64)
                # pc = 100*((_arr.shape[0]**3) - np.count_nonzero(_arr)) / (_arr.shape[0]**3)
                # print(f"Y {pc}% zeros. Also dtype {_arr.dtype}")
                # print(f"Y - Sparse array {sparse.GCXS(_arr, compressed_axes=[0,1,2]).nbytes/1e6} and non-sparse {_arr.nbytes/1e6}")
                # return l, m, sparse.COO(_arr)
                dict[(l,m)] = _arr
            return dict

        @ray.remote
        def _calculate_Y(job, theta, phi):
            l, m = job
            _arr = np.nan_to_num(scipy.special.sph_harm(m, l, phi, theta)).astype(np.complex64)
            # pc = 100*((_arr.shape[0]**3) - np.count_nonzero(_arr)) / (_arr.shape[0]**3)
            # print(f"Y {pc}% zeros. Also dtype {_arr.dtype}")
            # print(f"Y - Sparse array {sparse.GCXS(_arr, compressed_axes=[0,1,2]).nbytes/1e6} and non-sparse {_arr.nbytes/1e6}")
            # return l, m, sparse.COO(_arr)
            return l, m, _arr

        @ray.remote
        def _calculate_R(job, r):
            n, l = job
            _arr = np.nan_to_num((r**l) * scipy.special.jacobi(int(n-l)/2, 0, l + 0.5)(2*(r**2)-1)).astype(np.float32)
            # pc = 100*((_arr.shape[0]**3) - np.count_nonzero(_arr)) / (_arr.shape[0]**3)
            # print(f"R {pc}% zeros. Also dtype {_arr.dtype}")
            # return n, l, sparse.COO(_arr)
            # print(f"R - Sparse array {sparse.GCXS(_arr, compressed_axes=[0,1,2]).nbytes/1e6} and non-sparse {_arr.nbytes/1e6}")
            return n, l, _arr

        @ray.remote
        def _calculate_M_serial(job, volume, Ydict, Rdict):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            R = Rdict[(n, l)]
            Y = Ydict[(l, m)]
            Z = R*Y.conj()

            inner_product = np.tensordot(volume, Z, axes=3)
            chi = coeff * inner_product
            if m > 0:
                negchi = ((-1)**(m)) * np.conj(chi)
                return [(n,l,m,chi),(n,l,-m,negchi)]
            return [(n,l,m,chi)]

        def _calculate_M_single(job, volume):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            R = R_nl_dict[(n, l)]
            Y = Y_lm_dict[(l, m)]
            Z = R*np.conj(Y)

            inner_product = np.tensordot(volume, Z, axes=3)
            chi = coeff * inner_product
            if m > 0:
                negchi = ((-1)**(m)) * np.conj(chi)
                return [(n,l,m,chi),(n,l,-m,negchi)]
            return [(n,l,m,chi)]

        def _calculate_M_single_GPU(jobs, volume, Y_array, Y_lookup, R_array, R_lookup): #phi, theta):
            temp = []
            for job in tqdm(jobs, total=len(jobs)):
                n, l, m = job
                coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
                R_ind = R_lookup[(n, l)]
                Y_ind = Y_lookup[(l, m)]

                R = R_array[R_ind,:]
                Y = Y_array[Y_ind,:]
                # Y = cp.nan_to_num(cp_sph_harm(m, l, phi, theta)).astype(np.complex64)
                Z = R * cp.conj(Y)

                inner_product = cp.tensordot(volume, Z, axes=3)
                chi = coeff * cp.asnumpy(inner_product)
                if m > 0:
                    negchi = ((-1) ** (m)) * np.conj(chi)
                    temp.append((n, l, m, chi))
                    temp.append((n, l, -m, negchi))
                temp.append((n, l, m, chi))
            Y_array = None
            R_array = None
            Z = None
            R = None
            Y = None
            inner_product = None
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            return temp

        def _calculate_M_single_GPU_ylm(jobs, volume, R_array, R_lookup, Y_lm_dict):
            temp = []
            for job in tqdm(jobs, total=len(jobs)):
                n, l, m = job
                coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
                R_ind = R_lookup[(n, l)]
                Y = read_as_complex64(Y_lm_dict[(l, m)], gpu=True) ## CHANGED HERE

                R = R_array[R_ind,:]
                # Y = cp.nan_to_num(cp_sph_harm(m, l, phi, theta)).astype(np.complex64)
                Z = R * cp.conj(Y)

                inner_product = cp.tensordot(volume, Z, axes=3)
                chi = coeff * cp.asnumpy(inner_product)
                n = np.array(n).astype(np.uint8)
                l = np.array(l).astype(np.uint8)
                m = np.array(m).astype(np.uint8)
                if m > 0:
                    negchi = ((-1) ** (m)) * np.conj(chi)
                    temp.append((n, l, m, chi))
                    temp.append((n, l, -m, negchi.astype(np.complex64)))
                temp.append((n, l, m, chi.astype(np.complex64)))
            print("Got to end of M calc")
            return temp

        def get_size(obj, seen=None):
            """Recursively finds size of objects"""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            # Important mark as seen *before* entering recursion to gracefully handle
            # self-referential objects
            seen.add(obj_id)
            if isinstance(obj, dict):
                size += sum([v.nbytes for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            return size

        # def check_for_existing(jobs, type=None):
        #     new_jobs = []
        #     for job in jobs:
        #         if type == 'R':
        #             n, l = job
        #             if f'R/{n}/{l}' not in f1:
        #                 new_jobs.append(job)
        #         if type == 'Y':
        #             l, m = job
        #             with h5py.File('Y_lm.h5', 'r') as f2:
        #                 if f'Y/{l}/{m}' not in f2:
        #                     new_jobs.append(job)
        #         if type == None:
        #             new_jobs = jobs
        #     return new_jobs

        jobs_for_R = []
        jobs_for_Y = []
        for n in range(0, order + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    jobs_for_R.append((n, l))
                    for m in range(l + 1):
                        jobs_for_Y.append((l, m))

        jobs_for_Y = list(set(jobs_for_Y))
        jobs_for_R = list(set(jobs_for_R))

        # _jobs_for_R = check_for_existing(_jobs_for_R, type='R')
        # _jobs_for_Y = check_for_existing(_jobs_for_Y, type='Y')

        print(f"Calculating all radial functions, R(n,l)")
        _r = ray.put(r)
        obj_ids = [_calculate_R.remote(job, _r) for job in jobs_for_R]
        for i,x in enumerate(tqdm(to_iterator(obj_ids), total=len(obj_ids))):
            n, l, R = x
            R_nl_dict[(n, l)] = R
            # pass

        _jobs_for_M = []
        for n in range(0, order + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(l + 1):
                        _jobs_for_M.append((n,l,m))

        if mode=='serial':
            # if info.free < Y_size_p:
            #     print(f"Calculating all spherical harmonics (CPU), Y(l,m)")
            #     _theta = ray.put(theta)
            #     _phi = ray.put(phi)
            #     obj_ids = [_calculate_Y.remote(job, _theta, _phi) for job in jobs_for_Y]
            #     for i,x in enumerate(tqdm(to_iterator(obj_ids), total=len(obj_ids))):
            #         l, m, Y = x
            #         Y_lm_dict[(l,m)] = Y
            # else:
            print(f"Calculating all spherical harmonics (GPU), Y(l,m)")
            _theta = cp.asarray(theta)
            _phi = cp.asarray(phi)
            for job in tqdm(jobs_for_Y, total=len(jobs_for_Y)):
                l, m = job
                Y_lm_dict[(l, m)] = cp.asnumpy(cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64))

            # print(f"Calculating all spherical harmonics (multiGPU), Y(l,m)")
            # _theta = ray.put(cp.asarray(theta))
            # _phi = ray.put(cp.asarray(phi))
            # obj_ids = [_calculate_Y_GPU.remote(jobs, _theta, _phi) for jobs in chunker_list(jobs_for_Y, 4)]
            # for x in ray.get(obj_ids):
            #     Y_lm_dict.update(x)
            # Y_lm_dict = {k: cp.asnumpy(v) for k, v in Y_lm_dict.items()}

            print(f"Calculating the Zernike moments, Z_nlm")
            volume = self.voxels.voxels_array
            random.shuffle(_jobs_for_M)
            for job in tqdm(_jobs_for_M, total=len(_jobs_for_M)):
                x = _calculate_M_single(job, volume)
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.SetMoment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.SetMoment(n, l, m, mom)
        elif mode in ['GPU', 'gpu']:
            print(f"Calculating all spherical harmonics, Y(l,m)")
            _theta = ray.put(theta)
            _phi = ray.put(phi)
            obj_ids = [_calculate_Y.remote(job, _theta, _phi) for job in jobs_for_Y]
            for i,x in enumerate(tqdm(to_iterator(obj_ids), total=len(obj_ids))):
                l, m, Y = x
                Y_lm_dict[(l,m)] = Y

            print(f"Calculating the Zernike moments, Z_nlm")
            volume = cp.asarray(volume)
            R_lookup = {}
            poolArr_R = []
            for i, (key, value) in enumerate(R_nl_dict.items()):
                R_lookup[key] = i
                poolArr_R.append(value)
            R_nl_dict = {}
            _arr2 = np.array(poolArr_R)
            poolArr_R = cp.asarray(_arr2)

            Y_lookup = {}
            poolArr_Y = []
            for i, (key, value) in enumerate(Y_lm_dict.items()):
                Y_lookup[key] = i
                poolArr_Y.append(value)
            Y_lm_dict = {}
            poolArr_Y = cp.asarray(np.array(poolArr_Y))

            moms = _calculate_M_single_GPU(_jobs_for_M, volume, poolArr_Y, Y_lookup, poolArr_R, R_lookup)
            for mom in moms:
                n,l,m,val = mom
                self.moments.SetMoment(n,l,m,val)
        elif mode in ['GPU_batch', 'gpu_batch']:
            print(f"Calculating the Zernike moments, Z_nlm")
            volume = cp.asarray(volume)
            phi = cp.asarray(phi)
            theta = cp.asarray(theta)
            batch_size = 2500
            for batch in tqdm(chunks(_jobs_for_M, batch_size), total=int(len(_jobs_for_M)//batch_size)+1):
                c = 0
                R_lookup_slice = {}
                R = []
                g = 0
                Y_lookup_slice = {}
                Y = []
                for job in batch:
                    n,l,m = job
                    if not (n,l) in R_lookup_slice:
                        R.append(R_nl_dict[(n, l)])
                        R_lookup_slice[(n,l)] = c
                        c += 1
                    # if not (l,m) in Y_lookup_slice:
                    #     Y.append(Y_lm_dict[(l,m)])
                    #     Y_lookup_slice[(l, m)] = g
                    #     g += 1

                cp_poolArr_R = None
                # cp_poolArr_Y = None
                cp._default_memory_pool.free_all_blocks()
                cp_poolArr_R = cp.asarray(np.array(R))
                # cp_poolArr_Y = cp.asarray(np.array(Y))
                #cp_poolArr_Y, Y_lookup_slice,

                moms = _calculate_M_single_GPU(batch, volume, cp_poolArr_R, R_lookup_slice, phi, theta)
                for mom in moms:
                    n, l, m, val = mom
                    self.moments.SetMoment(n, l, m, val)
        elif mode in ['GPU_Ylm']:
            volume = cp.asarray(volume)
            R_lookup = {}
            poolArr_R = []
            for i, (key, value) in enumerate(R_nl_dict.items()):
                R_lookup[key] = i
                poolArr_R.append(value)
            R_nl_dict = {}
            _arr2 = np.array(poolArr_R)
            cp_poolArr_R = cp.asarray(_arr2)

            print(f"Calculating all spherical harmonics on GPU, Y(l,m)")
            _theta = cp.asarray(theta)
            _phi = cp.asarray(phi)
            for job in tqdm(jobs_for_Y, total=len(jobs_for_Y)):
                l, m = job
                Y_lm_dict[(l, m)] = reduce_to_8bit(cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64), gpu=True) ### CHANGED HERE

            print(f"Calculating the Zernike moments, Z_nlm")
            moms = _calculate_M_single_GPU_ylm(_jobs_for_M, volume, cp_poolArr_R, R_lookup, Y_lm_dict)
            print("Skipping moments")
            # for mom in moms:
            #     n, l, m, val = mom
            #     self.moments.SetMoment(n, l, m, val)
            print("Got to end of block")

        else:
            volume = ray.put(volume)
            _Ydict = ray.put(Y_lm_dict)
            _Rdict = ray.put(R_nl_dict)
            random.shuffle(_jobs_for_M)
            obj_ids = [_calculate_M_serial.remote(job, volume, _Ydict, _Rdict) for job in _jobs_for_M]
            for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.SetMoment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.SetMoment(n, l, m, mom)

        # with open('R_nl_dict.pkl', 'wb') as f:
        #     pickle.dump(R_nl_dict, f)
        # with open('Y_lm_dict.pkl', 'wb') as f:
        #     pickle.dump(Y_lm_dict, f)

        return self.moments

    def CalculateMoments_Spherical_zarr_ray(self, passedvoxels, order, mode='serial'):
        import zarr
        from numcodecs import Blosc
        num_cpus = 20
        ray.init(num_cpus=num_cpus)
        self.voxels = passedvoxels
        side = np.linspace(-1 / np.sqrt(3), 1 / np.sqrt(3), 64)
        x, y, z = np.asarray(np.meshgrid(side, side, side))
        r, theta, phi = self.cartesian_2_polar(x, y, z)
        volume = self.voxels.voxels_array
        N = self.voxels.voxels_array.shape[0]
        self.moments = Moments()
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

        Y_lm_zarr = zarr.open('Y_lm.test.zarr',
                              mode='a',
                              shape=(order+1,order+1,*volume.shape),
                              chunks=(1,1,64,64,64),
                              dtype=np.complex64,
                              compressor=compressor)

        R_nl_zarr = zarr.open('R_nl.test.zarr',
                              mode='a',
                              shape=(order+1,order+1,*volume.shape),
                              chunks=(1,1,64,64,64),
                              dtype=np.float32,
                              compressor=compressor)

        # Y_lm_zarr = zarr.create(
        #     shape=(order+1,order+1,*volume.shape),
        #     chunks=(1,1,128,128,128),
        #     dtype=np.complex64,
        #     compressor=compressor,
        # )
        #
        # R_nl_zarr = zarr.create(
        #     shape=(order+1,order+1,*volume.shape),
        #     chunks=(1,1,128,128,128),
        #     dtype=np.float32,
        #     compressor=compressor,
        # )

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        @ray.remote
        def _calculate_Y(job, theta, phi):
            l, m = job
            # self.Y_lm_scipy(l, m, theta, phi, save=True)
            return l, m, np.nan_to_num(scipy.special.sph_harm(m, l, phi, theta))
            # Y_lm_zarr[l,m,:] = np.nan_to_num(scipy.special.sph_harm(m, l, phi, theta))

        @ray.remote
        def _calculate_R(job, r):
            n, l = job
            # self.R_nl_scipy(n, l, r, save=True)
            return n, l, np.nan_to_num((r**l) * scipy.special.jacobi(int(n-l)/2, 0, l + 0.5)(2*(r**2)-1))
            # R_nl_zarr[n,l,:] = np.nan_to_num((r**l) * scipy.special.jacobi(int(n-l)/2, 0, l + 0.5)(2*(r**2)-1))

        @ray.remote
        def _calculate_M_chunk(job, volume, Y_dict, R_dict):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            R = R_dict[(n, l)]
            Y = Y_dict[(l, m)]
            Z = np.conj(R * Y)

            inner_product = np.tensordot(volume, Z, axes=3)
            chi = coeff * inner_product
            if m > 0:
                negchi = (-1) ** m * np.conj(chi)
                return [(n,l,m,chi),(n,l,-m,negchi)]
            return [(n,l,m,chi)]

        @ray.remote
        def _calculate_M_serial(job, volume):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            R = R_nl_zarr[n, l, :]
            Y = Y_lm_zarr[l, m, :]
            # R = self.R_nl_dict[(n,l)]
            # Y = self.Y_lm_dict[(l,m)]
            Z = R*np.conj(Y)

            inner_product = np.tensordot(volume, Z, axes=3)
            chi = coeff * inner_product
            if m > 0:
                negchi = ((-1)**(m)) * np.conj(chi)
                return [(n,l,m,chi),(n,l,-m,negchi)]
            return [(n,l,m,chi)]

        def _calculate_M_single(job, volume):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            R = R_nl_zarr[n, l, :]
            Y = Y_lm_zarr[l, m, :]
            Z = R*np.conj(Y)

            inner_product = np.tensordot(volume, Z, axes=3)
            chi = coeff * inner_product
            if m > 0:
                negchi = ((-1)**(m)) * np.conj(chi)
                return [(n,l,m,chi),(n,l,-m,negchi)]
            return [(n,l,m,chi)]

        # def check_for_existing(jobs, type=None):
        #     new_jobs = []
        #     for job in jobs:
        #         if type == 'R':
        #             n, l = job
        #             if f'R/{n}/{l}' not in f1:
        #                 new_jobs.append(job)
        #         if type == 'Y':
        #             l, m = job
        #             with h5py.File('Y_lm.h5', 'r') as f2:
        #                 if f'Y/{l}/{m}' not in f2:
        #                     new_jobs.append(job)
        #         if type == None:
        #             new_jobs = jobs
        #     return new_jobs

        jobs_for_R = []
        jobs_for_Y = []
        for n in range(0, order + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    jobs_for_R.append((n, l))
                    for m in range(l + 1):
                        jobs_for_Y.append((l, m))

        jobs_for_Y = list(set(jobs_for_Y))
        jobs_for_R = list(set(jobs_for_R))

        # _jobs_for_R = check_for_existing(_jobs_for_R, type='R')
        # _jobs_for_Y = check_for_existing(_jobs_for_Y, type='Y')

        print(f"Calculating all spherical harmonics, Y(l,m)")
        theta = ray.put(theta)
        phi = ray.put(phi)
        obj_ids = [_calculate_Y.remote(job, theta, phi) for job in jobs_for_Y]
        for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
            l, m, Y = x
            Y_lm_zarr[l, m, ...] = Y
            # pass

        print(f"Calculating all radial functions, R(n,l)")
        r = ray.put(r)
        obj_ids = [_calculate_R.remote(job, r) for job in jobs_for_R]
        for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
            n, l, R = x
            R_nl_zarr[n, l, ...] = R
            # pass
        ray.shutdown()

        print("Library statistics")
        print(Y_lm_zarr.info)
        print(R_nl_zarr.info)
        N_of_R = ((order+2)/2)**2
        N_of_Y_approx = 2*N_of_R
        print(f"Predicted size {N_of_Y_approx*64*64*64*8 /1e9}")
        print(f"Predicted size {N_of_R*64*64*64*4 /1e9}")


        _jobs_for_M = []
        for n in range(0, order + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(l + 1):
                        _jobs_for_M.append((n,l,m))

        print(f"Calculating the Zernike moments, Z_nlm")
        if mode=='serial':
            # batch_size = 1000
            # for batch in tqdm(chunks(_jobs_for_M, batch_size), total=int(len(_jobs_for_M)/batch_size)):
            #     nl = []
            #     lm = []
            #     for nlm in batch:
            #         n,l,m = nlm
            #         nl.append((n,l))
            #         lm.append((l,m))
            #     nl = list(sorted(set(nl)))
            #     lm = list(sorted(set(lm)))
            #
            #     R_dict = {el: R_nl_zarr[n, l, :] for el in nl}
            #     Y_dict = {el: Y_lm_zarr[l, m, :] for el in lm}
            #
            #     R_dict = ray.put(R_dict)
            #     Y_dict = ray.put(Y_dict)
            #     obj_ids = [_calculate_M_chunk.remote(job, volume, Y_dict, R_dict) for job in batch]
            volume = self.voxels.voxels_array
            random.shuffle(_jobs_for_M)
            for job in tqdm(_jobs_for_M, total=len(_jobs_for_M)):
                x = _calculate_M_single(job, volume)
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.SetMoment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.SetMoment(n, l, m, mom)
        else:
            ray.init(num_cpus=4)
            volume = ray.put(volume)
            # Yzarr = ray.put(Y_lm_zarr)
            # Rzarr = ray.put(R_nl_zarr)
            random.shuffle(_jobs_for_M)
            obj_ids = [_calculate_M_serial.remote(job, volume) for job in _jobs_for_M]
            for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.SetMoment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.SetMoment(n, l, m, mom)
            ray.shutdown()

        # zarr.save('Y_lm.test.zarr', Y_lm_zarr)
        # zarr.save('R_nl.test.zarr', R_nl_zarr)
        return self.moments

    def CalculateMoments_Spherical_Multi_sysmem(self, input_volume, order):
        def _calculate_m(job):
            n, l, m, R, theta, phi, volume = job
            N = volume.shape[0]
            Y = cp.asarray(self.Y_lm_scipy(l, m, theta, phi))
            Z = cp.conj(R * Y)
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (N ** 3))
            inner_product = cp.tensordot(volume, Z, axes=3)
            val = coeff * cp.asnumpy(inner_product)
            del Z, inner_product, Y
            return (n, l, m, val)

        def _calculate_l(job):
            n, l, R, theta, phi, volume = job
            jobs_of_m = []
            for m in range(l + 1):
                jobs_of_m.append((n, l, m, R, theta, phi, volume))
            moments_m = Parallel(n_jobs=8)(delayed(_calculate_m)(job_m) for job_m in jobs_of_m)
            return moments_m

        def _calculate_n(job):
            n, r, theta, phi, volume = job
            print(f"Calculating moments of the {n}th order")
            jobs_of_l = []
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    R = cp.asarray(self.R_nl_scipy(n, l, r))
                    if np.isnan(np.sum(R)):
                        print(f"Detected NaN in {n, l, m} of R")
                    jobs_of_l.append((n, l, R, theta, phi, volume))
            moments_l = Parallel(n_jobs=1)(delayed(_calculate_l)(job_l) for job_l in jobs_of_l)
            return moments_l
        #
        # def _calculate_moment(job):
        #     n, l, m, gid = job
        #     with cp.cuda.Device(gid):
        #         volume = cp.asarray(self.voxels.voxels_array)
        #         R = cp.asarray(self.R_nl_scipy(n, l, r))
        #         Y = cp.asarray(self.Y_lm_scipy(l, m, theta, phi))
        #         Z = cp.conj(R * Y)
        #         coeff = (2 * (2 * n + 3)) / (3 * cp.sqrt(3) * cp.pi * (N**3))
        #         inner_product = cp.tensordot(volume, Z, axes=3)
        #         val = coeff * cp.asnumpy(inner_product)
        #         return (n, l, m, val)

        side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 96)
        x, y, z = np.meshgrid(side, side, side)
        r, theta, phi = self.cartesian_2_polar(x, y, z)
        volume = cp.asarray(input_volume.voxels_array)

        jobs_of_n = []
        # if order > 20:
        #     for n in range(0, 20):
        #         #Calculate in sequence without parallel / overhead is too high
        #         self.CalculateMoments_Spherical()
        def flatten(l):
            return [item for sublist in l for item in sublist]

        for n in range(0, order + 1):
            jobs_of_n.append((n,r,theta,phi,volume))
        moments = Parallel(n_jobs=1)(delayed(_calculate_n)(job_n) for job_n in jobs_of_n)
        moments = flatten(flatten(moments))

        calcedmoments = Moments()
        for mom in moments:
            n, l, m, val = mom
            calcedmoments.SetMoment(n, l, m, val)
            if m > 0:
                negchi = (-1) ** m * np.conj(val)
                calcedmoments.SetMoment(n, l, -m, negchi)
        return calcedmoments

    def InitialiseReconstruction(self, passedmoments, order):
        print("Reconstructing voxel")
        self.order = order
        self.voxels = Voxels()
        # self.clm = self.precompute_clm(self.order)
        # self.Tklv = self.precompute_Tklv(self.order)
        # self.nCr = self.precompute_binominals(self.order)
        self.voxels.SetResolution(128)
        self.reconmoments = passedmoments
        self.reconvoxels = Voxels()
        self.reconvoxels.SetResolution(128)
        print("Done initialisation")

    def ReconstructGridMulti_msum(self, job):
        n, l, m, x, y, z, mask = job
        mom = self.reconmoments.GetMoment(n, l, m)
        grid = 0 + 1j
        if mom != 0.0 and mom != "VOID":
            grid = self.Chinlmrst(n, l, m, x, y, z, do_reconstruct=True) * mom
        mom = self.reconmoments.GetMoment(n, l, -m)
        if mom != 0.0 and mom != "VOID":
            grid += self.Chinlmrst(n, l, -m, x, y, z, do_reconstruct=True) * mom
        return grid*mask

    def ReconstructGridMulti_lsum(self, job):
        n, l, x, y, z, mask = job
        job_list = [(n, l, m, x, y, z, mask) for m in range(l + 1)]
        grid_list = Parallel(n_jobs=4)(delayed(self.ReconstructGridMulti_msum)(job) for job in job_list)
        out = 0 + 1j
        for grid in grid_list:
            out += grid
        return out

    def ReconstructGridMulti_nsum(self, job):
        n, x, y, z, mask = job
        print(f"Doing n: {n}")
        job_list = [(n, l, x, y, z, mask) for l in range(n + 1)]
        grid_list = Parallel(n_jobs=8)(delayed(self.ReconstructGridMulti_lsum)(job) for job in job_list)
        out = 0 + 1j
        for grid in grid_list:
            out += grid
        return out

    def ReconstructAllGridMultiDynamic(self, apix, outname):
        side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), self.voxels.GetResolution())
        x, y, z = np.asarray(np.meshgrid(side, side, side, indexing='ij'))
        num_cores = multiprocessing.cpu_count()
        momorder = self.reconmoments.GetMomentSize()
        mask = np.asarray(np.where(((x ** 2 + y ** 2 + z ** 2) ** 0.5) <= 1.0, 1, 0))
        job_list = [(n, x, y, z, mask) for n in range(min(self.order, momorder)+1)]
        grid_list = Parallel(n_jobs=4)(delayed(self.ReconstructGridMulti_nsum)(job) for job in job_list)
        out = 0 + 1j
        for grid in grid_list:
            out += grid
        out = np.real(out)
        with mrcfile.open(''.join([str(outname), '.mrc']), mode='w+') as mrc:
            mrc.set_data(np.array(np.real(out), np.float32))
            mrc.voxel_size = 2*1/self.voxels.GetResolution()

    def ReconstructAllGridMulti(self, apix, outname):
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        momorder = self.reconmoments.GetMomentSize()
        def _get_grid(n):
            side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), self.voxels.GetResolution())
            x, y, z = np.asarray(np.meshgrid(side, side, side, indexing='ij'))
            grid = 0.0
            print("Reconstructing w/:", n)
            for l in range(n + 1):
                for m in range(l + 1):
                    mom = self.reconmoments.GetMoment(n, l, m)
                    if mom != 0.0 and mom != "VOID":
                        grid += self.Chinlmrst(n, l, m, x, y, z, do_reconstruct=True) * mom
                    mom = self.reconmoments.GetMoment(n, l, -m)
                    if mom != 0.0 and mom != "VOID":
                        grid += self.Chinlmrst(n, l, -m, x, y, z, do_reconstruct=True) * mom
            mask = np.asarray(np.where(((x ** 2 + y ** 2 + z ** 2) ** 0.5) <= 1.0, 1, 0))
            return grid*mask

        inputs = [n for n in range(min(self.order, momorder)+1)]
        grid_list = Parallel(n_jobs=num_cores)(delayed(_get_grid)(i) for i in inputs)
        out = 0
        for grid in grid_list:
            out += grid
        out = np.real(out)
        with mrcfile.open(''.join([str(outname), '.mrc']), mode='w+') as mrc:
            mrc.set_data(np.array(np.real(out), np.float32))
            mrc.voxel_size = 2*1/self.voxels.GetResolution()

    def Reconstruct_PolarCoords_Ray(self, start, apix, outname):
        import zarr
        from numcodecs import Blosc
        num_cpus = 20
        # ray.init(num_cpus=num_cpus, num_gpus=2, logging_level=0)
        ray.init(num_cpus=num_cpus)
        order = min(self.reconmoments.GetMomentSize(),self.order)
        side = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 64)
        x, y, z = np.asarray(np.meshgrid(side, side, side))
        r, theta, phi = self.cartesian_2_polar(x, y, z)
        mask = np.asarray(np.where(((x ** 2 + y ** 2 + z ** 2) ** 0.5) < 0.99, 1, 0))
        # Y_lm_zarr = zarr.load('Y_lm.test.zarr')
        # R_nl_zarr = zarr.load('R_nl.test.zarr')

        Y_lm_zarr = zarr.open('Y_lm.test.zarr',
                              mode='r+')

        R_nl_zarr = zarr.open('R_nl.test.zarr',
                              mode='r+')

        print(f"Min: {np.min(theta * 180 / np.pi)}, max: {np.max(theta * 180 / np.pi)}")
        print(f"Min: {np.min(phi * 180 / np.pi)}, max: {np.max(phi * 180 / np.pi)}")

        @ray.remote
        def _calculate_Z_component(job):
            grid = 0.0 + 1j
            n, l, m, mom = job
            R = R_nl_zarr[n, l, :]
            Y = Y_lm_zarr[l, m, :]
            # R = Rzarr[n, l, :]
            # Y = Yzarr[l, m, :]
            grid += np.nan_to_num(mom * np.multiply(R,Y))
            if m != 0:
                grid += np.nan_to_num(np.conj(mom) * np.multiply(R,np.conj(Y)))
            return grid

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        out = 0
        # Yzarr = ray.put(Y_lm_zarr)
        # Rzarr = ray.put(R_nl_zarr)
        print(f"Calculating the Zernike components up to order, n: {order}")
        for n in tqdm(range(order+1)):
            _component_jobs = []
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(l + 1):
                        mom = self.reconmoments.GetMoment(n, l, m)
                        _component_jobs.append((n,l,m,mom))
            #Do parallel code here for batches of l
            obj_ids = [_calculate_Z_component.remote(job) for job in _component_jobs]
            cumulative = []
            for grid in to_iterator(obj_ids):
            # for grid in tqdm(to_iterator(obj_ids), total=len(obj_ids), leave=False):
                out += grid
                cumulative.append(out)

            if n % 5 == 0:
                with mrcfile.open(''.join(['intermediate_ray_component_', str(n).zfill(3), '.mrc']), mode='w+') as mrc_out:
                    mrc_out.set_data(np.array(np.real(out*mask), np.float32))
                    mrc_out.voxel_size = apix

        out *= mask
        with mrcfile.open(''.join([str(outname), '.mrc']), mode='w+') as mrc:
            mrc.set_data(np.array(np.real(out), np.float32))
            mrc.voxel_size = apix
        ray.shutdown()
        return cumulative

    def FourierShellCorrelation(self, m1, m2):
        '''
        Compute the fourier shell correlation between the 3D maps m1 and m2,
        which must be n x n x n in size with the same n, assumed to be even.

        The FSC is defined as
                    sum{F1 .* conj(F2)}
        c(i) = -----------------------------
               sqrt(sum|F1|^2 * sum|F2|^2)
        where F1 and F2 are the Fourier components at the given spatial
        frequency i. i ranges from 1 to n/2-1, times the unit frequency 1/(n*res)
        where res is the pixel size.

        Examples:
        --------
        c=FSCorr(m1,m2)

        Parameters
        ----------
        m1,m2 : ndarray
            n x n x n array. n assumed to be even

        Returns
        -------
        c : 1-D vector with Fourier shell correlation coefficients computed as
        described above.

        '''

        # First, construct the radius values for defining the shells.
        n, n1, n2 = m1.shape
        ctr = (n + 1) / 2
        origin = np.transpose(np.array([ctr, ctr, ctr]))

        x, y, z = np.meshgrid(np.arange(1 - origin[0], n - origin[0] + 1), np.arange(1 - origin[1], n - origin[1] + 1),
                         np.arange(1 - origin[2], n - origin[2] + 1), indexing='ij')
        R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        eps = 0.0001

        # Fourier-transform the maps
        f1 = np.fft.fftshift(np.fft.fftn(m1))
        f2 = np.fft.fftshift(np.fft.fftn(m2))

        # Perform the sums
        d0 = R < 1 + eps
        c = np.zeros((int(np.floor(2*n / 2)), 1))

        for i in np.arange(1, int(np.floor(2*n / 2) + 1)).reshape(-1):
            d1 = R < 1 + (i/2) + eps
            ring = np.logical_xor(d1, d0)

            r1 = f1[ring]
            r2 = f2[ring]
            num = np.real(sum(np.multiply(r1, np.conjugate(r2))))
            den = np.sqrt(sum(np.abs(r1) ** 2) * sum(np.abs(r2) ** 2))

            c[i - 1] = num / den
            d0 = d1

        return np.nan_to_num(c, nan=1)

class ZernikeSpherical:
    "A class for calculating Zernike moments from voxels"
    def __init__(self, args):
        self.args = args
        if self.args.bit8:
            self.C = ComplexArray()
        self.moments = Moments()

        self.Y_lm = {'vram': {}, 'ram': {}, 'zarr': {}}
        self.P_lm = {'vram': {}, 'ram': {}, 'zarr': {}}
        self.R_nl = {'vram': {}, 'ram': {}, 'zarr': {}}

        self.Y_lm_zarr = None
        self.R_nl_zarr = None

        self.Y_lm_memory_map = {}
        self.R_nl_memory_map = {}

        num_cpus = 16 if self.args.cpu_tasks is None else self.args.cpu_tasks
        ray.init(num_cpus=num_cpus)

        print("Created `Zernike` instance")

    def calc_COM(self, volume):
        _mass = volume.sum()
        x_mass = (volume * np.indices(volume.shape)[0]).sum()
        y_mass = (volume * np.indices(volume.shape)[1]).sum()
        z_mass = (volume * np.indices(volume.shape)[2]).sum()
        return (x_mass / _mass, y_mass / _mass, z_mass / _mass)

    def center_volume(self, volume):
        com = self.calc_COM(volume)
        print(f"Center of mass calculated: {com} (pixels)")
        s = volume.shape[0]
        if (s % 2) == 0:
            c = s / 2
        else:
            c = (s - 1) / 2

        offset = (-1 * (com[0] - c), -1 * (com[1] - c), -1 * (com[2] - c))
        centered = shift(volume, offset)
        return centered

    def cartesian_2_polar_WORKING(self, x, y, z):
        r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        theta[y < 0] = np.pi * 2 - theta[y < 0]
        return r, theta, phi

    def cartesian_2_polar(self, x, y, z):
        r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def interpolate_polar2cartesian(self, vol, o=None, r=None, output=None, order=1):  # Extend to 3D
        if r is None: r = vol.shape[0]
        if output is None:
            output = np.zeros((r, r, r), dtype=vol.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=vol.dtype)
        if o is None: o = np.array(output.shape) / 2 - 0.5
        x, y, z = output.shape
        xs, ys, zs = np.mgrid[:x, :y, :z] - o[:, None, None, None]
        rs = (ys ** 2 + xs ** 2 + zs ** 2) ** 0.5
        ts = np.arccos(zs / rs)
        ps = np.arctan2(ys, xs)

        ts[ys < 0] = np.pi * 2 - ts[ys < 0]
        # ts *= (vol.shape[1] - 1) / (np.pi * 2)
        # ps[ps < 0] = np.pi - ps[ps < 0]
        map_coordinates(vol, (ps, ts, rs), order=order, output=output)
        return output

    def _parity_Ylm(self, arr, m, gpu=False):
        """
        Compute the parity-transformed array of the input array `arr` up to order `l` and return the result.

        Parameters
        ----------
        arr : np.ndarray
            The input 3D array.
        m : int
            The azimuthal number.
        gpu : bool, optional
            Whether to use GPU acceleration (default is False).

        Returns
        -------
        np.ndarray
            The parity-transformed array. If `arr` has equal dimensions, returns the upper half of `arr`.
            Otherwise, returns the result of concatenating `arr` with the parity-transformed array (with
            the upper half of the array multiplied by (-1)^l) along the "z" dimension.
        """
        xp = cp if gpu else np
        z_dim, y_dim, x_dim = arr.shape
        if z_dim == y_dim == x_dim:
            return arr[z_dim // 2:, :, :].astype(np.complex64)
        else:
            if (m%2) == 0:
                return xp.concatenate((arr[::-1], arr), axis=0, dtype=np.complex64)[::-1]
            else:
                return xp.concatenate((-arr[::-1], arr), axis=0, dtype=np.complex64)[::-1]

    def _get_Ylm(self, l, m, gpu=False):
        loc = self.Y_lm_memory_map[(l,m)]

        if loc == 'zarr':
            Y = self.Y_lm_zarr[l, m, :]
        else:
            Y = self.Y_lm[loc][(l, m)]

        if gpu and loc in ['zarr', 'ram']:
            return cp.asarray(Y)
        else:
            return Y

    def _set_Ylm(self, l, m, array, mem):
        self.Y_lm_memory_map[(l, m)] = mem
        if mem == 'zarr':
            self.Y_lm_zarr[l, m, :] = array
        else:
            self.Y_lm[mem][(l, m)] = array

    def _get_Rnl(self, n, l, gpu=False):
        loc = self.R_nl_memory_map[(n,l)]
        if loc == 'zarr':
            R = self.R_nl_zarr[n, l, :]
        else:
            R = self.R_nl[loc][(n,l)]

        if gpu and loc in ['zarr', 'ram']:
            return cp.asarray(R)
        else:
            return R

    def _set_Rnl(self, n, l, array, mem):
        self.R_nl_memory_map[(n, l)] = mem
        if mem == 'zarr':
            self.R_nl_zarr[n, l, :] = array
        else:
            self.R_nl[mem][(n, l)] = array

    def R_nl_recurrence(self, n, l, r, mem):
        xp = cp if mem == 'vram' else np

        if l == n:
            R = r**n
            self._set_Rnl(n, l, R.astype(xp.float32), mem)
            return R.astype(xp.float32)
        elif l == (n - 2):
            R = (n + 0.5)*(r**n) - (n - 0.5)*(r**(n-2))
            self._set_Rnl(n, l, R.astype(xp.float32), mem)
            return R.astype(xp.float32)
        elif l <= (n - 4):
            k_0 = (n - l) * (n + l + 1) * (2 * n - 3)
            k_1 = (2 * n - 1) * (2 * n + 1) * (2 * n - 3)
            k_2 = 0.5 * (-2 * n + 1) * ((2 * l + 1)**2) - (k_1/2)
            k_3 = -1 * (n - l - 2) * (n + l - 1) * (2 * n + 1)
            K_1 = k_1 / k_0
            K_2 = k_2 / k_0
            K_3 = k_3 / k_0

            R = (K_1*(r**2)+K_2) * self.R_nl[mem][(n - 2, l)] + K_3*self.R_nl[mem][(n - 4, l)]
            self._set_Rnl(n, l, R.astype(xp.float32), mem)
            return R.astype(xp.float32)
        else:
            print(f"Error, got invalid n or l")

    def Y_lm_recurrence(self, l, m, theta, phi, mem):
        xp = cp if mem=='vram' else np

        if l == m == 0:
            Y = xp.ones_like(theta)
            return Y*xp.sqrt(1/xp.pi)*0.5
        elif l == m:
            C_3 = xp.sqrt((2*l + 1) / (2*l))
            Y = -C_3 * xp.sin(theta) * self.P_lm[mem][(l-1, m-1)]
            return Y
        elif m == (l-1):
            C_0 = xp.sqrt((2*l + 1) / ((l + m) * (l - m)))
            C_1 = C_0 * xp.sqrt(2*l - 1)
            Y = C_1 * xp.cos(theta) * self.P_lm[mem][(l - 1, m)]
            return Y
        else:
            C_0 = xp.sqrt((2*l + 1) / ((l + m) * (l - m)))
            C_1 = C_0 * xp.sqrt(2*l - 1)
            C_2 = C_0 * xp.sqrt(((l + m - 1) * (l - m - 1)) / (2*l - 3))
            Y = C_1 * xp.cos(theta) * self.P_lm[mem][(l - 1, m)] - C_2 * self.P_lm[mem][(l - 2, m)]
            return Y

    def _check_resources_mode(self, Y_size_predict, R_size_predict):
        GPUs = {}
        max_RAM = psutil.virtual_memory().total
        curr_RAM = psutil.virtual_memory()[3]

        nvidia_smi.nvmlInit()

        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            GPUs[i] = info.free
            if self.args.verbose:
                print(f"Device {i}: {nvidia_smi.nvmlDeviceGetName(handle)}, "
                      f"Memory: {(100 * info.free / info.total):.2f}% free. "
                      f"[{(info.total/1e9):.4f} (total), {(info.free/1e9):.4f} (free), {(info.used/1e9):.4f} (used)]")

        if self.args.gpu_id is None:
            max_VRAM_device = max(GPUs, key=GPUs.get)
            max_VRAM = GPUs[max_VRAM_device]
            cp.cuda.runtime.setDevice(max_VRAM_device)
        elif self.args.gpu_id > deviceCount-1:
            raise ValueError(f"Invalid GPU id. Got id: {self.args.gpu_id}, found device ids: {list(range(deviceCount))}")
        else:
            max_VRAM_device = self.args.gpu_id
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(max_VRAM_device))
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            max_VRAM = info.free
            cp.cuda.runtime.setDevice(max_VRAM_device)

        requirement = Y_size_predict+R_size_predict
        self.budgetGPU = 0.9*max_VRAM
        self.budgetCPU = 0.9*(max_RAM - curr_RAM)
        print(f"Total RAM {max_RAM/1e9}, used {curr_RAM/1e9}. Larged GPU VRAM {max_VRAM/1e9} on device {int(max_VRAM_device)}")

        if self.args.compute_mode is not None:
            self.mode = self.args.compute_mode
            print(f'Forcing mode {self.mode}')
            return

        if requirement < 0.9*max_VRAM:
            if self.args.verbose:
                print("Mode 3 : Sufficient GPU VRAM")
            self.mode = '3'
        elif requirement > 0.9*max_VRAM and requirement < 0.9*(max_RAM - curr_RAM):
            if self.args.mixed:
                if self.args.verbose:
                    print(f"Fallback: insufficient GPU VRAM for size {requirement/1E9} Gb")
                    print("Mode 3 (mixed) : Use GPU VRAM for most frequent (but not all).")
                self.mode = '3m'
            else:
                if self.args.verbose:
                    print("Fallback: insufficient GPU VRAM")
                    print("Mode 2 : Calculate Y_lm on GPU and offload to CPU.")
                self.mode = '2'
        elif requirement < 0.9*(max_RAM - curr_RAM):
            if self.args.verbose:
                print("Fallback: GPU disabled or no GPU found.")
                print("Mode 1: Sufficient CPU RAM")
            self.mode = '1'
        else:
            if self.args.mixed:
                if self.args.verbose:
                    print("Fallback: GPU disabled or no GPU found. Insufficient CPU RAM.")
                    print("Mode 1 (mixed) : Mixed enabled. Use CPU RAM for most frequent (but not all).")
                self.mode = '1m'
            if self.args.verbose:
                print("Fallback: insufficient CPU RAM")
                print("Mode 0 : Using disk I/O (this will be slow)")
            self.mode = '0'

        nvidia_smi.nvmlShutdown()

    def _get_jobs(self, order):
        jobs_for_R = []
        jobs_for_Y = []
        _jobs_for_M = []
        for n in range(0, order + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    jobs_for_R.append((n, l))
                    for m in range(l + 1):
                        jobs_for_Y.append((l, m))
                        _jobs_for_M.append((n, l, m))
        jobs_for_Y = list(set(jobs_for_Y))
        jobs_for_R = list(set(jobs_for_R))
        return jobs_for_R, jobs_for_Y, _jobs_for_M

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    @staticmethod
    def chunker_list(seq, size):
        return (seq[i::size] for i in range(size))

    @staticmethod
    def to_iterator(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    @staticmethod
    @ray.remote
    def _calculate_Y_np(job, theta, phi):
        l, m = job
        _arr = np.nan_to_num(scipy.special.sph_harm(m, l, phi, theta)).astype(np.complex64)
        return l, m, _arr

    @staticmethod
    @ray.remote
    def _calculate_R(job, r):
        n, l = job
        _arr = np.nan_to_num((r**l) * scipy.special.jacobi(int(n-l)/2, 0, l + 0.5)(2*(r**2)-1)).astype(np.float32)
        return n, l, _arr

    def _calculate_Z_CPU(self, job, volume):
        n, l, m = job
        coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (self.dim ** 3))
        R = self._get_Rnl(n, l)
        Y = self._get_Ylm(l, m)

        if self.args.bit8:
            Y = self.C.read_as_complex64(Y, in_gpu=False, out_gpu=False)

        Z = R*np.conj(Y)

        inner_product = np.tensordot(volume, Z, axes=3)
        chi = coeff * inner_product
        # if m > 0:
        #     negchi = ((-1)**(m)) * np.conj(chi)
        #     return [(n,l,m,chi),(n,l,-m,negchi)]
        return [(n,l,m,chi)]

    @staticmethod
    @ray.remote
    def _calculate_Z_CPU_multi(job, vol, dim, Y_lm_memory_map, Y_lm_zarr, Y_lm, R_nl_memory_map, R_nl_zarr, R_nl):
        def _get_Ylm(l, m):
            loc = Y_lm_memory_map[(l, m)]
            if loc == 'zarr':
                return Y_lm_zarr[l, m, :]
            else:
                return Y_lm[loc][(l, m)]

        def _get_Rnl(n, l):
            loc = R_nl_memory_map[(n, l)]
            if loc == 'zarr':
                return R_nl_zarr[n, l, ...]
            else:
                return R_nl[loc][(n, l)]

        n, l, m = job
        coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (dim ** 3))
        R = _get_Rnl(n, l)
        Y = _get_Ylm(l, m)
        Z = R*Y.conj()

        inner_product = np.tensordot(vol, Z, axes=3)
        chi = coeff * inner_product
        # if m > 0:
        #     negchi = ((-1)**(m)) * np.conj(chi)
        #     return [(n,l,m,chi),(n,l,-m,negchi)]
        return [(n,l,m,chi)]

    def _calculate_Z_GPU(self, jobs, volume):
        temp = []
        for job in tqdm(jobs, total=len(jobs)):
            n, l, m = job
            coeff = (2 * (2 * n + 3)) / (3 * np.sqrt(3) * np.pi * (self.dim ** 3))
            R = self._get_Rnl(n, l, gpu=True)
            Y = self._get_Ylm(l, m, gpu=True)

            if self.args.bit8:
                Y = self.C.read_as_complex64(Y, in_gpu=True, out_gpu=True)

            if self.args.parity:
                Y = self._parity_Ylm(Y, m, gpu=True)

            Z = R * cp.conj(Y)

            inner_product = cp.tensordot(volume, Z, axes=3)
            chi = coeff * cp.asnumpy(inner_product)
            # if m > 0:
            #     negchi = ((-1) ** (m)) * np.conj(chi)
            #     temp.append((n, l, m, chi))
            #     temp.append((n, l, -m, negchi.astype(np.complex64)))
            temp.append((n, l, m, chi.astype(np.complex64)))
        return temp

    def get_size(self, obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([v.nbytes for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size

    def CalculatePolynomials(self):
        order = self.args.order

        side = np.linspace(-1 / np.sqrt(3), 1 / np.sqrt(3), self.dim)
        x, y, z = np.asarray(np.meshgrid(side, side, side))
        r, theta, phi = self.cartesian_2_polar(x, y, z)

        quantize_factor = 0.25 if self.args.bit8 else 1
        parity_factor = 0.5 if self.args.parity else 1

        Y_size_p = mu_Y(order) * (self.dim ** 3) * 8 * quantize_factor * parity_factor
        R_size_p = mu_R(order) * (self.dim ** 3) * 4
        if self.args.verbose:
            print(f"Predicted size of Y(l, m) array : {(Y_size_p/1e9):.3f} Gb")
            print(f"Predicted size of R(n, l) array : {(R_size_p/1e9):.3f} Gb")

        # Check system resources
        self._check_resources_mode(Y_size_p, R_size_p)

        #Initialise zarr in case of mode `1m` or `0`
        if self.mode in ['1m', '0']:
            data_type = np.complex64 if self.args.bit8 is None else self.args.bit8
            compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
            self.Y_lm_zarr = zarr.open('Y_lm.zarr',
                                  mode='a',
                                  shape=(order+1,order+1,*volume.shape),
                                  chunks=(1,1,*volume.shape),
                                  dtype=np.complex64,
                                  compressor=compressor)

            self.R_nl_zarr = zarr.open('R_nl.zarr',
                                  mode='a',
                                  shape=(order+1,order+1,*volume.shape),
                                  chunks=(1,1,*volume.shape),
                                  dtype=np.float32,
                                  compressor=compressor)

        # Maybe do a check for pre-caluclated results to avoid having to re-calculate
        # or allow user to provide the pre-calculated moments...

        # Generate a list of Rnl and Ylm components that need to be calculated
        jobs_for_R, jobs_for_Y, _jobs_for_M = self._get_jobs(order)

        # BEGIN EXECUTION OF JOBS
        if self.mode == '3':
            # CALCULATE THE SPHERICAL HARMONICS
            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m), on GPU")
            _theta = cp.asarray(theta)
            _phi = cp.asarray(phi)
            for job in tqdm(sorted(jobs_for_Y), total=len(jobs_for_Y)):
                l, m = job
                # _arr = cp_sph_harm(m, l, _phi, _theta)
                Plm = self.Y_lm_recurrence(l, m, _theta, _phi, 'vram').astype(np.float32)
                self.P_lm['vram'][(l, m)] = Plm

            if self.args.verbose:
                print(f"Compressing, and memory clean up")
            for count,keys in tqdm(enumerate(list(self.P_lm['vram'].keys())), total=len(self.P_lm['vram'])):
                l, m = keys
                _arr = self.P_lm['vram'][keys] * cp.exp(1j * m * _phi)
                Y = cp.nan_to_num(_arr).astype(np.complex64)

                if self.args.parity:
                    Y = self._parity_Ylm(Y, m, gpu=True)

                if self.args.bit8:
                    Y = self.C.reduce_to_8bit(Y, in_gpu=True, out_gpu=True)

                self._set_Ylm(l, m, Y, 'vram')
                del self.P_lm['vram'][keys]
                if (count%1000 == 0):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

            # CALCULATE THE RADIAL COMPONENT
            if self.args.verbose:
                print(f"Calculating radial functions, R(n,l), on GPU")
            _r = cp.asarray(r)
            for job in tqdm(sorted(jobs_for_R), total=len(jobs_for_R)):
                n, l = job
                R = cp.nan_to_num(self.R_nl_recurrence(n, l, _r, 'vram'))
                self._set_Rnl(n, l, R, 'vram')

            del _r, _theta, _phi
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        elif self.mode == '3m':
            mem = self.budgetGPU - R_size_p
            mem_chunk = (self.dim ** 3) * 2 * m_factor
            if self.args.verbose:
                print(f'GPU memory budget set to: {mem / 1E9} Gb. Each component uses: {mem_chunk / 1E9} Gb.')
            Y_jobs_on_GPU = []
            Y_jobs_on_CPU = []
            for job in sorted(jobs_for_Y):
                if mem > 0:
                    Y_jobs_on_GPU.append(job)
                else:
                    Y_jobs_on_CPU.append(job)
                mem -= mem_chunk

            # CALCULATE THE RADIAL COMPONENT
            if self.args.verbose:
                print(f"Calculating radial functions, R(n,l), on CPU (ray parallel)")
            _r = ray.put(r)
            obj_ids = [self._calculate_R.remote(job, _r) for job in jobs_for_R]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                n, l, R = x
                R = cp.asarray(R)
                self._set_Rnl(n, l, R, 'vram')

            # CALCULATE THE SPHERICAL HARMONICS
            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m) | CPU batch")
            _theta = cp.asarray(theta)
            _phi = cp.asarray(phi)

            # CALCULATE THE EXCESS ON GPU THEN PASS TO CPU
            for job in tqdm(Y_jobs_on_CPU, total=len(Y_jobs_on_CPU)):
                l, m = job
                if self.args.bit8:
                    Y = cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64)
                    Y = self.C.reduce_to_8bit(Y, in_gpu=True, out_gpu=False)
                else:
                    Y = cp.asnumpy(cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64))
                self._set_Ylm(l, m, Y, 'ram')

            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m) | GPU batch")
            #STORE THE REMAINDER IN GPU RAM
            for job in tqdm(Y_jobs_on_GPU, total=len(Y_jobs_on_GPU)):
                l, m = job
                if self.args.bit8:
                    Y = cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64)
                    Y = self.C.reduce_to_8bit(Y, in_gpu=True, out_gpu=True)
                else:
                    Y = cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64)
                self._set_Ylm(l, m, Y, 'vram')

        elif self.mode == '2':
            # CALCULATE THE RADIAL COMPONENT
            if self.args.verbose:
                print(f"Calculating radial functions, R(n,l), on CPU (ray parallel)")
            _r = ray.put(r)
            obj_ids = [self._calculate_R.remote(job, _r) for job in jobs_for_R]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                n, l, R = x
                self._set_Rnl(n, l, R, 'ram')

            # CALCULATE THE SPHERICAL HARMONICS
            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m)")
            _theta = cp.asarray(theta)
            _phi = cp.asarray(phi)
            for job in tqdm(jobs_for_Y, total=len(jobs_for_Y)):
                l, m = job
                if self.args.bit8:
                    Y = cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64)
                    Y = self.C.reduce_to_8bit(Y, in_gpu=True, out_gpu=False)
                else:
                    Y = cp.asnumpy(cp.nan_to_num(cp_sph_harm(m, l, _phi, _theta)).astype(cp.complex64))
                self._set_Ylm(l, m, Y, 'ram')

        elif self.mode == '1':
            # CALCULATE THE RADIAL COMPONENT
            if self.args.verbose:
                print(f"Calculating radial functions, R(n,l), on CPU (ray parallel)")
            _r = ray.put(r)
            obj_ids = [self._calculate_R.remote(job, _r) for job in jobs_for_R]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                n, l, R = x
                self._set_Rnl(n, l, R, 'ram')

            # CALCULATE THE SPHERICAL HARMONICS
            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m)")
            _phi = ray.put(phi)
            _theta = ray.put(theta)
            obj_ids = [self._calculate_Y_np.remote(job, _phi, _theta) for job in jobs_for_Y]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                l, m, Y = x
                self._set_Ylm(l, m, Y, 'ram')

        elif self.mode == '1m':
            mem = self.budgetCPU - R_size_p
            mem_chunk = (self.dim ** 3) * 2 * m_factor
            if self.args.verbose:
                print(f'CPU memory budget set to: {mem/1E9} Gb. Each component uses: {mem_chunk/1E9} Gb.')
            Y_jobs_on_CPU = []
            Y_jobs_on_zarr = []
            for job in sorted(jobs_for_Y):
                if mem > 0:
                    Y_jobs_on_CPU.append(job)
                else:
                    Y_jobs_on_zarr.append(job)
                mem -= mem_chunk

            # CALCULATE THE RADIAL COMPONENT
            if self.args.verbose:
                print(f"Calculating radial functions, R(n,l), on CPU (ray parallel)")
            _r = ray.put(r)
            obj_ids = [self._calculate_R.remote(job, _r) for job in jobs_for_R]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                n, l, R = x
                self._set_Rnl(n, l, R, 'ram')

            # CALCULATE THE SPHERICAL HARMONICS
            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m) | CPU batch")
            _phi = ray.put(phi)
            _theta = ray.put(theta)
            obj_ids = [self._calculate_Y_np.remote(job, _phi, _theta) for job in Y_jobs_on_CPU]
            for i, x in enumerate(tqdm(self.to_iterator(obj_ids), total=len(obj_ids))):
                l, m, Y = x
                self._set_Ylm(l, m, Y, 'ram')

            if self.args.verbose:
                print(f"Calculating spherical harmonic functions, Y(l,m) | zarr batch")
            obj_ids = [self._calculate_Y_np.remote(job, _theta, _phi) for job in Y_jobs_on_zarr]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                l, m, Y = x
                self._set_Ylm(l, m, Y, 'zarr')

        elif self.mode == '0':
            if self.args.verbose:
                print(f"Calculating all radial functions, R(n,l)")
            r = ray.put(r)
            obj_ids = [self._calculate_R.remote(job, r) for job in jobs_for_R]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                n, l, R = x
                self._set_Rnl(n, l, R, 'zarr')

            if self.args.verbose:
                print(f"Calculating all spherical harmonics, Y(l,m)")
            _theta = ray.put(theta)
            _phi = ray.put(phi)
            obj_ids = [self._calculate_Y_np.remote(job, _theta, _phi) for job in jobs_for_Y]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                l, m, Y = x
                self._set_Ylm(l, m, Y, 'zarr')

            if self.args.verbose:
                print("Library statistics")
                print(self.Y_lm_zarr.info)
                print(self.R_nl_zarr.info)

        if self.args.save:
            with open('R_nl_dict.pkl', 'wb') as f:
                pickle.dump(R_nl_dict, f)
            with open('Y_lm_dict.pkl', 'wb') as f:
                pickle.dump(Y_lm_dict, f)

        return _jobs_for_M

    def Decompose(self, volume, dim, jobs):
        _jobs_for_M = jobs

        # BEGIN EXECUTION OF JOBS
        if self.mode == '3':
            _volume = cp.asarray(volume)

            if self.args.verbose:
                print(f"Calculating the Zernike moments, Z_nlm")
            moms = self._calculate_Z_GPU(_jobs_for_M, _volume)
            for moment in moms:
                n, l, m, mom = moment
                self.moments.set_moment(n, l, m, mom)

        elif self.mode == '3m':
            _volume = cp.asarray(volume)

            if self.args.verbose:
                print(f"Calculating the Zernike moments, Z_nlm")
            moms = self._calculate_Z_GPU(_jobs_for_M, _volume)
            for moment in moms:
                n, l, m, mom = moment
                self.moments.set_moment(n, l, m, mom)

        elif self.mode == '2':
            if self.args.verbose:
                print(f"Calculating the Zernike moments, Z_nlm")

            random.shuffle(_jobs_for_M)
            for job in tqdm(_jobs_for_M, total=len(_jobs_for_M)):
                x = self._calculate_Z_CPU(job, volume)
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.set_moment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.set_moment(n, l, m, mom)

        elif self.mode == '1':

            _vol = ray.put(volume)
            Y_lm_memory_map = ray.put(self.Y_lm_memory_map)
            Y_lm = ray.put(self.Y_lm)
            R_nl_memory_map = ray.put(self.R_nl_memory_map)
            R_nl = ray.put(self.R_nl)
            random.shuffle(_jobs_for_M)
            obj_ids = [self._calculate_Z_CPU_multi.remote(job, _vol, self.dim,
                                                              Y_lm_memory_map, self.Y_lm_zarr, Y_lm,
                                                              R_nl_memory_map, self.R_nl_zarr, R_nl) for job in _jobs_for_M]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.set_moment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.set_moment(n, l, m, mom)

        elif self.mode == '1m':
            if self.args.verbose:
                print(f"Calculating the Zernike moments, Z_nlm")
            _vol = ray.put(volume)
            Y_lm_memory_map = ray.put(self.Y_lm_memory_map)
            Y_lm = ray.put(self.Y_lm)
            R_nl_memory_map = ray.put(self.R_nl_memory_map)
            R_nl = ray.put(self.R_nl)
            random.shuffle(_jobs_for_M)
            obj_ids = [self._calculate_Z_CPU_multi.remote(job, _vol, self.dim,
                                                          Y_lm_memory_map, self.Y_lm_zarr, Y_lm,
                                                          R_nl_memory_map, self.R_nl_zarr, R_nl) for job in _jobs_for_M]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.set_moment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.set_moment(n, l, m, mom)

        elif self.mode == '0':
            if self.args.verbose:
                print(f"Calculating the Zernike moments, Z_nlm")
            _vol = ray.put(volume)
            Y_lm_memory_map = ray.put(self.Y_lm_memory_map)
            Y_lm = ray.put(self.Y_lm)
            R_nl_memory_map = ray.put(self.R_nl_memory_map)
            R_nl = ray.put(self.R_nl)
            random.shuffle(_jobs_for_M)
            obj_ids = [self._calculate_Z_CPU_multi.remote(job, _vol, self.dim,
                                                          Y_lm_memory_map, self.Y_lm_zarr, Y_lm,
                                                          R_nl_memory_map, self.R_nl_zarr, R_nl) for job in _jobs_for_M]
            for x in tqdm(self.to_iterator(obj_ids), total=len(obj_ids)):
                if len(x) > 1:
                    for out in x:
                        n, l, m, mom = out
                        self.moments.set_moment(n, l, m, mom)
                else:
                    n, l, m, mom = x[0]
                    self.moments.set_moment(n, l, m, mom)

        # if self.args.verbose:
        #     self.moments.display_moments()

    def _calculate_cum_GPU(self, jobs):
        grid = 0.0 + 1j
        # for job in tqdm(jobs, total=len(jobs)):
        for job in jobs:
            n, l, m, mom = job

            R = self._get_Rnl(n, l, gpu=True)
            Y = self._get_Ylm(l, m, gpu=True)

            if self.args.bit8:
                Y = self.C.read_as_complex64(Y, in_gpu=True, out_gpu=True)

            if self.args.parity:
                Y = self._parity_Ylm(Y, l, gpu=True)

            grid += cp.nan_to_num(mom * cp.multiply(R, Y))
            if m != 0:
                grid += cp.nan_to_num(cp.conj(mom) * cp.multiply(R, cp.conj(Y)))
        return grid

    @staticmethod
    @ray.remote
    def _calculate_cum_CPU(job, Y_lm_memory_map, Y_lm_zarr, Y_lm, R_nl_memory_map, R_nl_zarr, R_nl):
        def _get_Ylm(l, m):
            loc = Y_lm_memory_map[(l, m)]
            if loc == 'zarr':
                return Y_lm_zarr[l, m, :]
            else:
                return Y_lm[loc][(l, m)]

        def _get_Rnl(n, l):
            loc = R_nl_memory_map[(n, l)]
            if loc == 'zarr':
                return R_nl_zarr[n, l, ...]
            else:
                return R_nl[loc][(n, l)]

        grid = 0.0 + 1j
        n, l, m, mom = job
        R = _get_Rnl(n, l)
        Y = _get_Ylm(l, m)
        grid += np.nan_to_num(mom * np.multiply(R,Y))
        if m != 0:
            grid += np.nan_to_num(np.conj(mom) * np.multiply(R,np.conj(Y)))
        return grid

    def Reconstruct(self, jobs, apply_latent_scaling=False):
        # Check for minimum number of moments or user defined limit
        # order = min(self.moments,self.order) # to do
        # self.moments.load(moments)

        _rec_jobs = range(1) if apply_latent_scaling else range(self.moments.dimensions)
        _jobs_for_M = jobs
        order = self.args.order
        for _rec in _rec_jobs:
            print(f"Summing the Zernike components up to order, n: {order}")
            if self.mode == 2:
                Y_lm_memory_map = ray.put(self.Y_lm_memory_map)
                Y_lm = ray.put(self.Y_lm)
                R_nl_memory_map = ray.put(self.R_nl_memory_map)
                R_nl = ray.put(self.R_nl)
                out = 0
                for n in tqdm(range(self.order+1)):
                    _component_jobs = []
                    for l in range(n + 1):
                        if (n - l) % 2 == 0:
                            for m in range(l + 1):
                                mom = self.moments.get_moment(n, l, m, latent_dim=_rec)
                                _component_jobs.append((n,l,m,mom))

                    # RAY PARALLEL
                    obj_ids = [self._calculate_cum_CPU.remote(job, Y_lm_memory_map, self.Y_lm_zarr, Y_lm,
                                                                  R_nl_memory_map, self.R_nl_zarr, R_nl) for job in _component_jobs]
                    cumulative = []
                    for grid in self.to_iterator(obj_ids):
                        out += grid
                        cumulative.append(out)

                    if n % 5 == 0:
                        name = ''.join(['intermediate_vol_component_', str(_rec), 'n_', str(n).zfill(3), '.mrc'])
                        self.write_mrc_file(volume=np.array(np.real(out*mask), np.float32), name=name)

                if apply_latent_scaling:
                    name = str(self.args.output).split('.')[0] + '_scaled.mrc'
                else:
                    name = str(self.args.output).split('.')[0] + '_' + str(_rec) + '.mrc'
                self.write_mrc_file(volume=np.array(np.real(out), np.float32), name=name)
                ray.shutdown()
                return cumulative

            else:
                cumulative = 0 + 1j
                for n in tqdm(range(order + 1)):
                    _component_jobs = []
                    for l in range(n + 1):
                        if (n - l) % 2 == 0:
                            for m in range(l + 1):
                                if apply_latent_scaling:
                                    mom = self.moments.scaled_moments[n,l,m]
                                else:
                                    mom = self.moments.get_moment(n, l, m, latent_dim=_rec)
                                _component_jobs.append((n, l, m, mom))

                    #GPU Parallel
                    component = self._calculate_cum_GPU(_component_jobs)
                    cumulative += component

                    if (n % 5 == 0) and self.args.verbose:
                        name = ''.join(['intermediate_vol_component_', str(_rec), 'n_', str(n).zfill(3), '.mrc'])
                        self.write_mrc_file(volume=cp.asnumpy(cp.real(cumulative)), name=name)

                if apply_latent_scaling:
                    if self.args.z_ind:
                        name = f"{str(self.args.output).split('.')[0]}_scaled_{self.args.z_ind}.mrc"
                    elif self.args.scales:
                        _s = '_'.join([s for s in self.args.scales])
                        name = f"{str(self.args.output).split('.')[0]}_scaled_{_s}.mrc"
                else:
                    name = str(self.args.output).split('.')[0] + '_' + str(_rec) + '.mrc'
                self.write_mrc_file(volume=cp.asnumpy(cp.real(cumulative)), name=name)

    def write_mrc_file(self, volume, name):
        with mrcfile.open(name, mode='w+') as mrc_out:
            data = volume
            mrc_out.set_data(np.array(data, np.float32))
            mrc_out.voxel_size = 1 if self.args.apix is None else self.args.apix

    def crop_array(self, array, crop):
        def crop_it(_array, _crop):
            if _array.shape[0] != _array.shape[1] != _array.shape[2]:
                print("Map is not cubic... taking the first dimension. This may fail.")
            if _array.shape[0] % 2 == 0:
                m = int((_array.shape[0] / 2) - 1)
                l = m + 1 - crop
                h = m + crop
                return _array[l:h, l:h, l:h]
            else:
                m = int((_array.shape[0] / 2) - 1)
                l = m - crop
                h = m + crop + 1
                return _array[l:h, l:h, l:h]
        if isinstance(array, list):
            return [crop_it(vol, crop) for vol in array]
        else:
            return crop_it(array, crop)

    def crop_array_center(self, array, crop, center=(0, 0, 0)):
        if (crop%2)==0:
            crop = int(crop/2)
        else:
            crop -= 1
            crop = int(crop/2)

        def crop_it(_array, _crop, _center):
            if _array.shape[0] != _array.shape[1] != _array.shape[2]:
                print("Map is not cubic... taking the first dimension. This may fail.")
            if _array.shape[0] % 2 == 0:
                cx, cy, cz = _center
                l = cx - crop
                h = cx + crop + 1
                _array = _array[l:h, :, :]
                l = cy - crop
                h = cy + crop + 1
                _array = _array[:, l:h, :]
                l = cz - crop
                h = cz + crop + 1
                _array = _array[:, :, l:h]
            else:
                cx, cy, cz = _center
                l = cx - crop
                h = cx + crop
                _array = _array[l:h + 1, :, :]
                l = cy - crop
                h = cy + crop
                _array = _array[:, l:h + 1, :]
                l = cz - crop
                h = cz + crop
                _array = _array[:, :, l:h + 1]
            return _array

        if isinstance(array, list):
            return [crop_it(vol, crop, center) for vol in array]
        else:
            return crop_it(array, crop, center)

    def bin_crop_array_center(self, array, crop, bin=None, center=(0, 0, 0)):
        if (crop%2)==0:
            crop = int(crop/2)
        else:
            crop -= 1
            crop = int(crop/2)

        def bin_it(_array, scale):
            return scipy.ndimage.zoom(_array, scale, order=0)

        def crop_it(_array, _crop, _center):
            if _array.shape[0] != _array.shape[1] != _array.shape[2]:
                print("Map is not cubic... taking the first dimension. This may fail.")
            cx, cy, cz = _center
            l = cx - crop
            h = cx + crop
            _array = _array[l:h, :, :]
            l = cy - crop
            h = cy + crop
            _array = _array[:, l:h, :]
            l = cz - crop
            h = cz + crop
            _array = _array[:, :, l:h]
            return _array

        if isinstance(array, list):
            if bin is None:
                return [crop_it(vol, crop, center) for vol in array]
            else:
                return [bin_it(crop_it(vol, crop, center), 1/bin) for vol in array]
        else:
            if bin is None:
                return crop_it(array, crop, center)
            else:
                return bin_it(crop_it(array, crop, center), 1/bin) #crop_it(bin_it(array, 1/bin), crop, tuple(int(ti/bin) for ti in center))

    def parse_marker_set(self, xml_string):
        # find the start and end indices of the x, y, and z attributes
        x_start = xml_string.find('x="') + 3
        x_end = xml_string.find('"', x_start)
        y_start = xml_string.find('y="') + 3
        y_end = xml_string.find('"', y_start)
        z_start = xml_string.find('z="') + 3
        z_end = xml_string.find('"', z_start)
        pix_start = xml_string.find('radius="') + 8
        pix_end = xml_string.find('"', pix_start)

        # extract the x, y, and z coordinates of the marker and store them in a tuple
        apix = float(xml_string[pix_start:pix_end])
        x = float(xml_string[x_start:x_end])/apix
        y = float(xml_string[y_start:y_end])/apix
        z = float(xml_string[z_start:z_end])/apix
        coordinates = (int(z), int(y), int(x))

        # return the tuple of coordinates
        return coordinates

    def _read_wiggle(self, input):
        _load = np.load(input, allow_pickle=True)
        labels = ['particles', 'consensus_map', 'components']
        if all(label in _load.files for label in labels):
            return _load['particles'], _load['consensus_map'], _load['components']

    def run(self, decompose=False, reconstruct=False, flip=False):
        #Initiate decompose or reconstruct
        volume = []
        if decompose:
            # Single volume or multiple volumes (in or out)
            base, ext = os.path.splitext(self.args.input)
            if ext in ('.mrc', '.MRC'):
                with mrcfile.open(self.args.input) as mrc:
                    volume = mrc.data
                    self.apix = mrc.voxel_size['x'] if self.args.apix is None else self.args.apix
                single = True
            elif ext in ('.npz', '.npy'):
                particles, consensus, components = self._read_wiggle(self.args.input)
                volume.append(consensus)
                for c in components:
                    volume.append(c)

                single = False
                self.apix = 1 if self.args.apix is None else self.args.apix
            else:
                print("Got invalid input")
                sys.exit(2)

            if self.args.com is not None:
                with open(self.args.com, 'r') as f:
                    marker_set = f.read()
                    center = self.parse_marker_set(marker_set)
                volume = self.bin_crop_array_center(volume, self.args.crop_to, bin=self.args.bin, center=center)
            elif self.args.bin is not None:
                if isinstance(volume, list):
                    volume = [scipy.ndimage.zoom(vol, 1/self.args.bin, order=0) for vol in volume]
                else:
                    volume = scipy.ndimage.zoom(volume, 1/self.args.bin, order=0)

            self.dim = volume.shape[0] if single else volume[0].shape[0]
            print(f'Detected dimensions: {self.dim}')
            jobs = self.CalculatePolynomials()

            if single:
                self.Decompose(volume, self.dim, jobs)
                self.moments.save(output=self.args.output, compress=self.args.save_bit8, dim=self.dim, apix=self.apix)
            else:
                for _,vol in enumerate(volume):
                    print(f"Calculating Zernike moments for component {_}")
                    self.Decompose(vol, self.dim, jobs)
                    self.moments.add_dimension()
                self.moments.set_latents(particles=particles, components=len(volume)-1)
                self.moments.save(output=self.args.output, compress=self.args.save_bit8, dim=self.dim, apix=self.apix)

        if reconstruct:
            # Single volume or multiple volumes (in or out)
            base, ext = os.path.splitext(self.args.moments)
            if ext not in ('.npz', '.npy'):
                print("Got invalid input")
                sys.exit(2)

            self.dim = self.args.dimensions if not None else np.load(self.args.moments)['dim']
            print(f'Detected dimensions: {self.dim}')
            jobs = self.CalculatePolynomials()

            self.moments.load(self.args.moments)
            if self.args.z_ind:
                print(f"Got z_ind {self.args.z_ind}, scaling moments and reconstructing.")
                self.moments.apply_scaling(z_ind=self.args.z_ind)
                self.Reconstruct(jobs, apply_latent_scaling=True)
            elif self.args.scales:
                print(f"Got scales {self.args.scales}, scaling moments and reconstructing.")
                self.moments.apply_scaling(scales=self.args.scales)
                self.Reconstruct(jobs, apply_latent_scaling=True)
            else:
                print(f"No scales or z_ind, will reconstruct base volume and component maps.")
                self.Reconstruct(jobs)

if __name__ == "__main__":
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def valid_input(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.npz', '.mrc', '.MRC'):
            raise argparse.ArgumentTypeError('File must have a .npz or .mrc extension')
        return param

    def valid_mrc(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.mrc','.MRC'):
            raise argparse.ArgumentTypeError('File must have a .mrc extension. See README.')
        return param

    def valid_com(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.com','.cmm'):
            raise argparse.ArgumentTypeError('File must have a .com or .cmm extension. See README.')
        return param

    def valid_output(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.npz',):
            return param + '.npz'
        return param

    # [print(bcolors.__dict__[key]+'TEST'+bcolors.ENDC) for key, value in bcolors.__dict__.items() if not bcolors.__dict__[key].startswith('__')]
    # create the top-level parser
    my_parser = argparse.ArgumentParser(
        prog=bcolors.OKBLUE+bcolors.BOLD+'zernike.py'+bcolors.ENDC,
        description=textwrap.dedent(bcolors.BOLD+bcolors.HEADER+'''    ###########################################################################################################

        *** ZERNIKE *** v0.1 - Zernike decomposition and reconstruction of cryo-EM volumes (mrc format)

        Charles Bayly-Jones (2022-2023) - Monash University, Melbourne, Australia

    ###########################################################################################################'''+bcolors.ENDC),
        formatter_class=argparse.RawTextHelpFormatter #ArgumentDefaultsHelpFormatter
    )

    # create sub-parser
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode: `decomposition` or `reconstruct`.",
        dest="operating_mode",
        required=True,
    )

    # create the parser for the "DECOMPOSE" sub-command
    parser_agent = sub_parsers.add_parser("decomposition", help="Decompose an input map to Zernike moments")
    parser_agent.add_argument(
        "--mode",
        choices=['spharm', 'monomial'],
        type=str,
        help="Algorithm for moment calculation. Spharm is stable up to high `n` (>100), "
             "the monomial approach diverges at high `n` (therefore not recommended, available for purposes of comparison)",
        required=True
    )
    parser_agent.add_argument(
        "--input",
        type=valid_input,
        help="Input map (.mrc format) or latent space (WIGGLE .npz) to be decomposed.",
        required=True
    )
    # parser_agent.add_argument(
    #     "--volume",
    #     type=valid_mrc,
    #     help="Input map (.mrc format) to be decomposed.",
    #     required=True
    # )
    parser_agent.add_argument(
        "--order",
        type=int,
        help="Order `n` of Zernike components to calculate.",
        required=True
    )
    parser_agent.add_argument(
        "--apix",
        type=float,
        help="Override the input pixel size (Angstrom per pixel).",
        required=False
    )
    parser_agent.add_argument(
        "--com",
        type=valid_com,
        help="Define a center point within the volume around which to extract a smaller box.",
        required=False
    )
    parser_agent.add_argument(
        "--crop_to",
        type=int,
        help="Crop box to this size (pixels).",
        required=False
    )
    parser_agent.add_argument(
        "--bin",
        type=float,
        help="Bin the input volume by this factor.",
        default=None,
        required=False
    )
    parser_agent.add_argument(
        "--compute_mode",
        help="Force the use of a particular computational mode ('3', '3m', '2', '1', '1m', '0'). See README for details. [Default: dynamic]",
        type=str
    )
    parser_agent.add_argument(
        "--gpu_id",
        help="Force GPU device ID. [Default: Device with most vRAM]",
        type=int
    )
    parser_agent.add_argument(
        "--cpu_tasks",
        help="Number of parallel tasks. [Default: all]",
        type=int
    )
    parser_agent.add_argument(
        "--output",
        type=valid_output,
        help="Name of the output file. If no extension given, string will be appended (npz).",
        required=True
    )
    #This mode is a compromise between precision and memory requirements.
    # `bit8` mode will give significant speed up by allowing components to
    # be simultaneously stored in memory. If components aren't stored then
    # 8bit incurs a computational cost (and precision loss) and is not recommended.
    parser_agent.add_argument(
        "--bit8",
        action='store_true',
        help="Reduce complex arrays from complex64 to 8bit encoding.",
    )
    parser_agent.add_argument(
        "--save_bit8",
        action='store_true',
        help="Also save the moments in 8bit encoding.",
    )
    parser_agent.add_argument(
        "--parity",
        action='store_true',
        help="Use parity relation to reduce array size.",
    )
    parser_agent.add_argument(
        "--mixed",
        action='store_true',
        help="Enable use of mixed memory types (e.g. use both GPU/CPU or CPU/Zarr, see README)",
    )
    parser_agent.add_argument(
        "--onthefly",
        help="Don't pre-calculate spherical harmonics, instead calculate these on the fly. Overrides `mixed` and `compute_mode`.",
        action='store_true'
    )
    parser_agent.add_argument(
        "--save",
        action='store_true',
        help="Save Zernike components to disk for use in later experiments.",
    )
    parser_agent.add_argument(
        "--verbose",
        help="Make verbose.",
        action='store_true'
    )
    parser_agent.add_argument(
        "--preallocate_memory_off",
        action='store_true',
        help="Preallocate GPU memory is much faster. Turn off pre-allocation using --preallocate_memory_off to use less memory in case of out-of-memory errors (slower).",
    )

    # create the parse for the "RECONSTRUCT" sub-command
    parser_learner = sub_parsers.add_parser("reconstruct", help="Reconstruct the object from Zernike moments")
    parser_learner.add_argument(
        "--moments",
        type=valid_input,
        help="Complex coefficients ('moments.npz')",
        required=True
    )
    parser_learner.add_argument(
        "--order",
        type=int,
        help="Order `n` of Zernike components to calculate",
        required=True
    )
    parser_learner.add_argument(
        "--mode",
        choices=['spharm', 'monomial'],
        type=str,
        help="Algorithm for moment calculation. Spharm is stable up to high `n` (>100), "
             "the monomial approach diverges at high `n` (therefore not recommended, available for purposes of comparison)",
        required=True
    )
    parser_learner.add_argument(
        "--mixed",
        action='store_true',
        help="Enable use of mixed memory types (e.g. use both GPU/CPU or CPU/Zarr, see README)",
    )
    parser_learner.add_argument(
        "--dimensions",
        type=int,
        help="Reconstruction box size.",
        default=None
    )
    parser_learner.add_argument(
        "--z_ind",
        type=int,
        help="Index of the scaling latent coodinates to use.",
        default=None
    )
    parser_learner.add_argument(
        "--scales",
        nargs='+',
        help="List of scaling factors e.g. 4.5 78 2 ... Must match the dimensions of the space!",
        default=None
    )
    parser_learner.add_argument(
        "--bit8",
        action='store_true',
        help="Reduce complex arrays from complex64 to 8bit encoding.",
    )
    parser_learner.add_argument(
        "--parity",
        action='store_true',
        help="Use parity relation to reduce array size.",
    )
    parser_learner.add_argument(
        "--apix",
        type=float,
        help="Override the output pixel size (Angstrom per pixel).",
        required=False
    )
    parser_learner.add_argument(
        "--output",
        type=valid_mrc,
        help="Name of the output volume.",
        required=True
    )
    parser_learner.add_argument(
        "--cpu_tasks",
        help="Number of parallel tasks. [Default: all]",
        type=int
    )
    parser_learner.add_argument(
        "--save",
        action='store_true',
        help="Save Zernike components to disk for use in later experiments. "
             "Useful if generating very large order moments and falling back to mode X (using zarr only).",
    )
    parser_learner.add_argument(
        "--gpu_id",
        help="Force GPU device ID. [Default: Device with most vRAM]",
        type=int
    )
    parser_learner.add_argument(
        "--compute_mode",
        help="Force the use of a particular computational mode ('3', '3m', '2', '1', '1m', '0'). See README for details. [Default: dynamic]",
        type=str
    )
    parser_learner.add_argument(
        "--verbose",
        action='store_true',
        help="Make verbose.",
    )
    parser_learner.add_argument(
        "--preallocate_memory_off",
        action='store_true',
        help="Preallocate GPU memory is much faster. Turn off pre-allocation using --preallocate_memory_off to use less memory in case of out-of-memory errors (slower).",
    )

    args = my_parser.parse_args()
    if args.preallocate_memory_off:
        cp.cuda.set_allocator(None)
        cp.cuda.set_pinned_memory_allocator(None)

    if args.operating_mode == 'decomposition':
        if args.com and (args.crop_to is None):
            my_parser.error(
                bcolors.WARNING + "`--com` flag requires `--crop_to BOX_SIZE` e.g. --com --crop_to 32" + bcolors.ENDC)
        if (args.compute_mode in ['1m', '0']) and (args.bit8):
            print("ERROR: `bit8` low memory mode is not compatible with compute modes `0` or `1m`.")
            sys.exit(2)

        if args.mode == 'spharm':
            Z = ZernikeSpherical(args)
        else:
            Z = ZernikeMonomial(args)
        Z.run(decompose=True)
    elif args.operating_mode == 'reconstruct':
        if args.mode == 'spharm':
            Z = ZernikeSpherical(args)
        else:
            Z = ZernikeMonomial(args)
        Z.run(reconstruct=True)