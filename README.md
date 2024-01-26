# Zernike3d for cryo-EM analysis
Computes the Zernike moments (complex coefficients of the Zernike expansion) and Zernike invariants (rotationally invariant norms) from voxel-based MRC density maps (cryo-EM).

![example](https://github.com/charbj/zernike3d/blob/main/reconstruction.gif)

## Requirements
#### Hardware
The fastest and most luxurious option would be to have a GPU with a lot of VRAM, e.g. an NVIDIA RTX A6000 with 48 Gb of VRAM, however this is an extreme. Most people will have 12-24 Gb of VRAM. Alternatively, a lot of system RAM is also a good option. Many HPC nodes have 500 Gb - 1 Tb of RAM nowadays. Computing everything on the GPU and in GPU memory is the fastest. Unfortunately due to the large number of arrays and limited bandwidth, computing on the GPU and passing these to the CPU memory is quite slow and in this case it's usually just faster to compute directly with the CPU. The optimal approach is somewhat system dependent... 

Zernike3D *should* check your system and determine the fastest computation strategy depending on your system hardware. 
It prioritises:
  - GPU VRAM > GPU/CPU VRAM/RAM > CPU RAM > CPU SSD/HD I/O. 

#### Memory
Computing moments to order n=50 represents an expansion of ~12,051 Zernike polynomials. Zernike3D will minimise this by exploiting redundancy and recurrence. However, this still requires 675 radial and 1326 spherical harmonics. Consequently, memory usage scales poorly with volume size. It is strongly recommended to crop the volume to remove as much unnecessary solvent as possible without cropping the signal of the macromolecule. Alternatively, a soft mask can be applied and the masked subvolume can be decomposed (once again, after cropping this region). 

![memory_usage](https://github.com/charbj/zernike3d/blob/main/memory_usage_Figure_1.png)

#### Resolution
To further minimise memory requirements, consider whether the full pixel size sampling is necessary (e.g. by binning). A factor of 2, e.g. 200 -> 100 px, requires 90 Gb less memory! Here is an example of resolutions obtained for a 64-pixel volume with 1.4 Å pix<sup>-1</sup>. Currently, this relationship depends on the pixel size and the box size (aiming to remove/resolve this).

![resolution_vs_order](https://github.com/charbj/zernike3d/blob/main/resolution.png))

## Dependencies
- python==3.10
- numpy
- numba
- cupy    
- rcfile
- joblib
- scipy
- tqdm
- zarr
- ray
- nvidia-ml-py3

##### Cupy install can be tricky, check your CUDA version and follow the [documentation](https://docs.cupy.dev/en/stable/install.html). I recommend the `pip install cupy-cuda12x` approach. Note `nvidia-smi` doesn't always report the correct cuda version that is on your system path. You should check this. 

## Install
~~~
mamba create -n zernike3d python==3.10
mamba activate zernike3d
mamba install -c conda-forge numpy numba mrcfile joblib scipy tqdm zarr cupy
python -m pip install ray nvidia-ml-py3 cupy-cuda12x
~~~

## Examples 

#### Decompose a single volume in MRC format into the Zernike moments (saved as a .npz file).
~~~
python zernike_master.py decomposition \
--mode spharm --input meprin_64pix.mrc --order 50 --compute_mode 3 --output test.npz --gpu_id 0 --verbose
~~~

#### Reconstruct a single volume from a set of Zernike moments
~~~
python zernike_master.py reconstruct \
--mode spharm --moments test.npz --order 50 --compute_mode 3 --output reconstructed.mrc --gpu_id 0 --verbose --dimensions 64
~~~

#### Decompose a 3DVA job from cryoSPARC in WIGGLE (.wgl/.npz) format into Zernike moments (saved as a .npz file).
##### In order to handle the 3DVA base volume and component maps, these must be bundled together along with the scaling factors from cryoSPARC. [WIGGLE](https://github.com/charbj/wiggle/tree/main) provides a utility to take the `map.mrc`, `component_0.mrc`, `component_1.mrc`, etc, and `particles.cs` files and output a single `bundled_3dva.npz` file.

~~~
python zernike_master.py decomposition \
--mode spharm --input bundle.npz --order 50 --compute_mode 3 --output test_3dva.npz --gpu_id 0 --verbose
~~~

#### Reconstruct a single volume from Zernike moments for a given latent coordinate in the 3DVA space
~~~
python zernike_master.py reconstruct \
--mode spharm --moments test.npz --z_ind 3450 --order 50 --compute_mode 3 --output reconstructed.mrc --gpu_id 0 --verbose --dimensions 64
~~~

#### The `Moments` class has utilities for loading, handling, and viewing these moments.
```pycon
>>> from zernike_master import Moments

>>> m = Moments()

>>> m.load('test.npz')
Loading test.npz into Moments object
Detected full precision moments (single).

>>> m.
m.add_dimension()         m.display_invariants()    m.get_order()             m.load(                   m.order                   m.stats()
m.apix                    m.display_moments()       m.get_size()              m.moments                 m.save(                   m.voxdim
m.calculate_invariants()  m.get_moment(             m.invariants              m.ndinvariants            m.set_latents(            
m.dimensions              m.get_moment_size()       m.latents                 m.ndmoments               m.set_moment(

>>> m.stats()
+-------------------------------------------------------+
|Moment statistics                                      |
|   Dimensions: 2                                       |
|   Moments: 12051                                      |
|   Order: 50                                           |
|   nbytes: 2918032                                     |
|   type: complex64                                     |
|                                                       |
|   Invariants: False                                   |
+-------------------------------------------------------+

>>> m.display_moments(n_max=5)
(0, 0, 0) (0.001322321+0j)
(1, 1, 0) (9.049198e-06+0j)
(1, 1, 1) (9.950537e-06+1.2573148e-05j)
(2, 0, 0) (-0.0038312061+0j)
(2, 2, 0) (6.18744e-05+0j)
(2, 2, 1) (-8.5120555e-05+0.000100565776j)
(2, 2, 2) (2.1655213e-05-3.1847885e-05j)
...

>>> m.display_moments(nlm=(3,3,2))
(3.8573184e-05+1.8685685e-06j)

>>> m.display_invariants(n_max=10)
(0, 0) 0.001322321
(1, 1) 2.44148e-05
(2, 0) 0.0038312061
(2, 2) 0.00020374708
(3, 1) 9.3017414e-05
(3, 3) 9.50426e-05
...

>>> m.display_invariants(nl=(0,0))
0.001322321
```

#### Options and help
~~~
(zernike3d) charles $ python zernike_master.py decomposition --help
usage: zernike.py decomposition [-h] --mode {spharm,monomial} --input INPUT --order ORDER [--apix APIX] [--com COM] [--crop_to CROP_TO] [--bin BIN]
                                             [--compute_mode COMPUTE_MODE] [--gpu_id GPU_ID] [--cpu_tasks CPU_TASKS] --output OUTPUT [--bit8] [--save_bit8]
                                             [--parity] [--mixed] [--onthefly] [--save] [--verbose]

options:
  -h, --help            show this help message and exit
  --mode {spharm,monomial}
                        Algorithm for moment calculation. Spharm is stable up to high `n` (>100), the monomial approach diverges at high `n` (therefore not
                        recommended, available for purposes of comparison)
  --input INPUT         Input map (.mrc format) or latent space (WIGGLE .npz) to be decomposed.
  --order ORDER         Order `n` of Zernike components to calculate.
  --apix APIX           Override the input pixel size (Angstrom per pixel).
  --com COM             Define a center point within the volume around which to extract a smaller box.
  --crop_to CROP_TO     Crop box to this size (pixels).
  --bin BIN             Bin the input volume by this factor.
  --compute_mode COMPUTE_MODE
                        Force the use of a particular computational mode ('3', '3m', '2', '1', '1m', '0'). See README for details. [Default: dynamic]
  --gpu_id GPU_ID       Force GPU device ID. [Default: Device with most vRAM]
  --cpu_tasks CPU_TASKS
                        Number of parallel tasks. [Default: all]
  --output OUTPUT       Name of the output file. If no extension given, string will be appended (npz).
  --bit8                Reduce complex arrays from complex64 to 8bit encoding.
  --save_bit8           Also save the moments in 8bit encoding.
  --parity              Use parity relation to reduce array size.
  --mixed               Enable use of mixed memory types (e.g. use both GPU/CPU or CPU/Zarr, see README)
  --onthefly            Don't pre-calculate spherical harmonics, instead calculate these on the fly. Overrides `mixed` and `compute_mode`.
  --save                Save Zernike components to disk for use in later experiments.
  --verbose             Make verbose.


(zernike3d) charles $ python zernike_master.py reconstruct --help
usage: zernike.py reconstruct [-h] --moments MOMENTS --order ORDER --mode {spharm,monomial} [--mixed] [--dimensions DIMENSIONS] [--bit8] [--parity]
                                           [--apix APIX] --output OUTPUT [--cpu_tasks CPU_TASKS] [--save] [--gpu_id GPU_ID] [--compute_mode COMPUTE_MODE]
                                           [--verbose]

options:
  -h, --help            show this help message and exit
  --moments MOMENTS     Complex coefficients ('moments.npz')
  --order ORDER         Order `n` of Zernike components to calculate
  --mode {spharm,monomial}
                        Algorithm for moment calculation. Spharm is stable up to high `n` (>100), the monomial approach diverges at high `n` (therefore not
                        recommended, available for purposes of comparison)
  --mixed               Enable use of mixed memory types (e.g. use both GPU/CPU or CPU/Zarr, see README)
  --dimensions DIMENSIONS
                        Reconstruction box size.
  --bit8                Reduce complex arrays from complex64 to 8bit encoding.
  --parity              Use parity relation to reduce array size.
  --apix APIX           Override the output pixel size (Angstrom per pixel).
  --output OUTPUT       Name of the output volume.
  --cpu_tasks CPU_TASKS
                        Number of parallel tasks. [Default: all]
  --save                Save Zernike components to disk for use in later experiments. Useful if generating very large order moments and falling back to mode X
                        (using zarr only).
  --gpu_id GPU_ID       Force GPU device ID. [Default: Device with most vRAM]
  --compute_mode COMPUTE_MODE
                        Force the use of a particular computational mode ('3', '3m', '2', '1', '1m', '0'). See README for details. [Default: dynamic]
  --verbose             Make verbose.
~~~
