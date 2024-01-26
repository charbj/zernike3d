import os, textwrap, argparse, sys
import numpy as np
from zernike_master import ZernikeSpherical, Moments
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from sklearn.preprocessing import normalize
from scipy import stats
import pandas as pd
import joypy as jp
import itertools
import mrcfile
np.random.seed(42)

class Z_3DVA:
    def __init__(self, args):
        self.args = args
        self.query = self.args.query_space
        self.target = self.args.target_spaces
        self.tgs = len(self.target)

    def read_3dva_npz(self, input):
        f = np.load(input, allow_pickle=True)
        m = f['consensus_map']
        m = m[None, :]
        c = f['components']
        return np.vstack([m, c])

    def save_vol(self, name, vol):
        vol = self.crop_center_outwards(np.nan_to_num(vol), 196)
        with mrcfile.new(name) as mrc:
            mrc.set_data(vol.astype(np.float32))

    def scaled_invariant_old(self, start, stop):
        return cp.dot(self._particles[start:stop], self._invariants), self._particles[start:stop]

    def thr(self):
        tgs = range(self.tgs+1)
        while True:
            for j in tgs:
                yield j

    def scaled_invariants_m_concatination(self, start, stop, thr):
        scaled = cp.abs(cp.tensordot(self.particles[thr][start:stop], self.moments[thr], axes=([-1],[0])))
        norm = cp.linalg.norm(scaled, axis=-1)
        out = norm[:,self.masks[thr]]
        out_n = out[..., 1:] / np.linalg.norm(out[..., 1:], ord=2, axis=0)
        return cp.asnumpy(out_n), self.particles[thr][start:stop]

    def scaled_invariants(self, start, stop, thr):
        # print(self.particles[thr][start:stop])
        # print(self.moments[thr])
        # print(cp.tensordot(self.particles[thr][start:stop], self.moments[thr], axes=([-1],[0])))
        #
        # print(self.particles[thr][start:stop].shape)
        # print(self.moments[thr].shape)
        # print(cp.tensordot(self.particles[thr][start:stop], self.moments[thr], axes=([-1], [0])).shape)

        scaled = cp.abs(cp.tensordot(self.particles[thr][start:stop], self.moments[thr], axes=([-1],[0])))
        # print(scaled[0,0:10])
        # print(scaled.shape)
        # norm = cp.linalg.norm(scaled, axis=-1)
        # print(norm[0,0:10])
        # print(norm.shape)
        # out = norm[:, self.masks[thr]]
        # print(self.masks[thr])
        # print(out.shape)
        # print(out[...,1:].shape)
        # print(out[...,1:])
        # print(scaled.shape)
        if scaled.shape[0] != 0:
            flat = scaled.reshape(scaled.shape[0], -1)
            # print(flat.shape)
            out = flat[~np.isnan(flat) & (flat != 0)].reshape(scaled.shape[0],-1)
            # print(out.shape)
            out_n = out / np.linalg.norm(out, ord=2, axis=0)

            # out_n = out[...,1:] / np.linalg.norm(out[...,1:], ord=2, axis=0)
            # print(out_n, out_n.shape)
            # print(np.sum(out_n[0]))
            return cp.asnumpy(out_n), self.particles[thr][start:stop]
        else:
            return np.array([]), self.particles[thr][start:stop]

    def fetch_data_chunk(self):
        s = 0
        step = 200
        thr = self.thr()
        while True:
            t = next(thr)
            invariants, scalars = self.scaled_invariants_m_concatination(s, s + step - 1, t)
            yield invariants, scalars, t
            if t == self.tgs:
                s += step

    def unison_shuffled_copies(self, arr, brr):
        assert len(arr) == len(brr)
        p = np.random.permutation(len(arr))
        return arr[p], brr[p]

    def crop_center_outwards(self, arr, crop_shape):
        center = np.array(arr.shape) // 2
        start = center - (crop_shape // 2)
        end = start + crop_shape
        return arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def incrementalPCA(self):
        from sklearn.decomposition import IncrementalPCA
        from sklearn.preprocessing import StandardScaler

        self.scales = [np.hstack([np.ones_like(m.latents[:, :1]), m.latents]) for m in self.m]

        # print(self.scales_1.shape, self.scales_2.shape)

        # if self.scales_1.shape[0] != self.scales_2.shape[0]:
        #     print("Dataset sizes are uneven. Truncating the larger of the two.")
        r = min([scale.shape[0] for scale in self.scales])

        self.particles = {key: cp.asarray(scale) for key, scale in enumerate(self.scales)}
        self.moments = {key: cp.asarray(moms) for key, moms in enumerate(self.moms)}
        self.masks = {key: cp.asarray(mask) for key, mask in enumerate(self.mask)}

        n_components = 8
        ipca = IncrementalPCA(n_components=n_components)
        sc = StandardScaler()
        steps = (r // 200) + 1
        early_stop = 0
        print("Performing principal component analysis (FIT) on Zernike invariant/scalars")
        for chunk, scales, thr in tqdm(self.fetch_data_chunk(), total=(self.tgs+1)*steps):
            early_stop += 1
            # print(chunk.shape)
            if chunk.shape[0] != 0:
                # sc.fit(chunk)
                # chunk = sc.transform(chunk)
                ipca.partial_fit(chunk[20:])
            else:
                print(chunk.shape)
                break
            # sc.fit(chunk)
            # chunk = sc.transform(chunk)

            if early_stop == -1:
                break

        # sys.exit()

        early_stop = 0
        transform = {key: np.ndarray(shape=(0, n_components)) for key,_ in enumerate(self.m)}
        scales_test = {key: np.ndarray(shape=(0, 4)) for key, _ in enumerate(self.m)}
        print("Performing principal component analysis (TRANSFORM) on Zernike invariant/scalars")
        for chunk, scales, thr in tqdm(self.fetch_data_chunk(), total=(self.tgs+1)*steps):
            early_stop += 1
            if chunk.shape[0] != 0:
                # sc.fit(chunk)
                # chunk = sc.transform(chunk)
                partial = ipca.transform(chunk[20:])
                transform[thr] = np.vstack((transform[thr], partial))
                scales_test[thr] = np.vstack((scales_test[thr], cp.asnumpy(scales)))
            else:
                print(chunk.shape)
                break

            if early_stop == -1:
                break

        # One-dimensional plot (aggregate)
        label = self.args.labels  # List of labels for categories
        x = np.linspace(0.0, 1.0, self.tgs+1)

        cl = mpl.cm.viridis(x)
        fig, axs = plt.subplots(n_components)
        fig.set_figheight(14)
        fig.set_figwidth(6)
        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.92, top=0.945, wspace=None, hspace=0.42)
        for dim, ax in enumerate(axs):
            # One-dimensional plot (aggregate)
            for key, obj in transform.items():
                data = transform[key][:, dim]
                kde = stats.gaussian_kde(data)
                xx = np.linspace(data.min(), data.max(), 100)
                ax.hist(data, density=True, bins=100, alpha=0.5, color=cl[key], label=label[key], log=True)
                # ax.plot(xx, kde(xx), color=cl[0])
                # ax.yaxis.offsetText.set_visible(False)
                # ax.xaxis.offsetText.set_visible(False)

        # Put a legend below current axis
        axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0.08, ncol=3)

        # plt.show()
        plt.savefig(f'{self.args.outname}')
        self.sample_volumes(transform, scales_test)


    def sample_volumes(self, transform, scales):
        maps = self.read_3dva_npz(self.args.target_3dva)
        setA = pd.DataFrame(transform[0])
        setB = pd.DataFrame(transform[1])

        scalesA = pd.DataFrame(scales[0])
        scalesB = pd.DataFrame(scales[1])

        # print(self.particles[0][setA.iloc[0:50].index.to_numpy()])
        # print(scalesA.head(50))

        sortA = setA.sort_values(by=0)
        sortB = setB.sort_values(by=0)

        # print(sortA)
        print(sortA)

        # print(sortA.iloc[0:10])
        # print(sortA.iloc[0:10].index.to_numpy())
        # print(self.particles[0][sortA.iloc[0:10].index.to_numpy()])
        # print(sortA[0].min(), sortA[0].max())
        pc0_range = np.linspace(sortA[0].min(), sortA[0].max(), 20)
        for i,r0 in enumerate(pc0_range[:-1]):
            print(f'Generating volume {i} in series for PC {str(0)}')
            chunk = sortA[(sortA[0] >= r0) & (sortA[0] <= pc0_range[i+1])]
            print(chunk[:5])
            print(chunk.shape)
            if chunk.shape[0] == 0:
                continue

            chunk = self.particles[1][chunk.index.to_numpy()]
            scales = np.mean(cp.asnumpy(chunk), axis=0)
            print(scales)
            s = scales[:, None, None, None]

            vol = np.sum(np.multiply(s, maps), axis=0)
            self.save_vol(f'pc_0_lin_vol_{str(i).zfill(2)}.mrc', vol)

        chunk_size = 10000
        for i,stp in enumerate(range(0, len(sortA), chunk_size)):
            chunk = sortA.iloc[stp:stp+chunk_size]
            if chunk.shape[0] == 0:
                continue

            chunk = self.particles[1][chunk.index.to_numpy()]
            scales = np.mean(cp.asnumpy(chunk), axis=0)
            s = scales[:, None, None, None]

            vol = np.sum(np.multiply(s, maps), axis=0)
            self.save_vol(f'pc_0_grp_vol_{str(i).zfill(2)}.mrc', vol)


    def incrementalPCA_sampleVolumes(self):
        from sklearn.decomposition import IncrementalPCA
        from sklearn.preprocessing import StandardScaler
        from column_transformer_partial import ColumnTransformerPartial


        self.scales = [np.hstack([np.ones_like(m.latents[:, :1]), m.latents]) for m in self.m]

        # print(self.scales_1.shape, self.scales_2.shape)

        # if self.scales_1.shape[0] != self.scales_2.shape[0]:
        #     print("Dataset sizes are uneven. Truncating the larger of the two.")
        r = min([scale.shape[0] for scale in self.scales])

        self.particles = {key: cp.asarray(scale) for key, scale in enumerate(self.scales)}
        self.moments = {key: cp.asarray(moms) for key, moms in enumerate(self.moms)}
        self.masks = {key: cp.asarray(mask) for key, mask in enumerate(self.mask)}

        n_components = 8
        ipca = IncrementalPCA(n_components=n_components)
        sc = StandardScaler()
        featurizer = ColumnTransformerPartial([('pca', ipca, ['c1', 'c2']), ('preserve', 'passthrough', ['index'])])


        #PERFORM THE FIT
        steps = (r // 200) + 1
        early_stop = 0
        for chunk, thr in tqdm(self.fetch_data_chunk(), total=(self.tgs+1)*steps):
            early_stop += 1
            if chunk.shape[0] != 0:
                # sc.fit(chunk)
                # chunk = sc.transform(chunk)
                featurizer.partial_fit(chunk)
            else:
                break
            # sc.fit(chunk)
            # chunk = sc.transform(chunk)

            if early_stop == -1:
                break

        early_stop = 0
        transform = {key: np.ndarray(shape=(0, n_components)) for key,_ in enumerate(self.m)}
        for chunk, thr in tqdm(self.fetch_data_chunk(), total=(self.tgs+1)*steps):
            early_stop += 1
            if chunk.shape[0] != 0:
                # sc.fit(chunk)
                # chunk = sc.transform(chunk)
                partial = ipca.transform(chunk)
                transform[thr] = np.vstack((transform[thr], partial))
            else:
                break

            if early_stop == -1:
                break

        # One-dimensional plot (aggregate)
        label = self.args.labels  # List of labels for categories
        x = np.linspace(0.0, 1.0, self.tgs+1)

        cl = mpl.cm.viridis(x)
        fig, axs = plt.subplots(8)
        fig.set_figheight(14)
        fig.set_figwidth(6)
        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.92, top=0.945, wspace=None, hspace=0.42)
        for dim, ax in enumerate(axs):
            # One-dimensional plot (aggregate)
            for key, obj in transform.items():
                data = transform[key][:, dim]
                kde = stats.gaussian_kde(data)
                xx = np.linspace(data.min(), data.max(), 100)
                ax.hist(data, density=True, bins=100, alpha=0.5, color=cl[key], label=label[key], log=True)
                # ax.plot(xx, kde(xx), color=cl[0])
                # ax.yaxis.offsetText.set_visible(False)
                # ax.xaxis.offsetText.set_visible(False)

        # Put a legend below current axis
        axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0.08, ncol=3)

        # plt.show()
        plt.savefig(f'{self.args.outname}')

    def ndmoment_to_ndarray(self, obj):
        arr = np.zeros((obj.dimensions,obj.order+1, obj.order+1, obj.order+1), dtype=np.complex64)
        mask = np.zeros((obj.order+1, obj.order+1), dtype=bool)
        for dim, moments in obj.ndmoments.items():
            for nlm, mom in moments.items():
                n,l,m = nlm
                arr[dim,n,l,m] = mom
                mask[n,l] = True
        return arr, mask

    def all_vs_all(self):
        print("Running all_vs_all")

        m1 = Moments()
        m1.load(self.args.query_space)

        # Fudge
        # m1.latents = np.random.normal(0, 0.1, 10000).reshape(10000,1)
        # m1.add_dimension()
        # m1.ndmoments[1] = m1.ndmoments[0]

        m1.calculate_invariants()
        m1.stats()
        m1_moms, m1_mask = self.ndmoment_to_ndarray(m1)

        self.m = [m1]
        self.moms = [m1_moms]
        self.mask = [m1_mask]

        for t in self.args.target_spaces:
            m = Moments()
            m.load(t)

            # Fudge
            # m.latents = m1.latents
            # m.add_dimension()
            # m.ndmoments[1] = m.ndmoments[0]

            m.calculate_invariants()
            m.stats()
            m_moms, m_mask = self.ndmoment_to_ndarray(m)
            self.m.append(m)
            self.mask.append(m_mask)
            self.moms.append(m_moms)


        # cum = 0
        # coefficients = 0
        # diff = []
        #
        # for key, val in self.m1.elements2[0].items():
        #     A = self.m1.elements[0][key]
        #     B = self.m2.elements[0][key]
        #     mu = np.mean([A, B])
        #     diff.append(np.abs(A - B))
        #     sigma = np.std([A, B])
        #     err = (sigma / mu) * 100
        #     if sigma != 0:
        #         cum += err
        #     coefficients += 1
        #
        # print(f"Average error: (\u03C3/\u03BC)%: {cum / coefficients}")
        # print(f"Minkowski distance: {np.linalg.norm(diff)}")

        self.incrementalPCA()

    def search_trajectory(self):
        self.coords = self.args.trajectory if not None else []
        pass

    def calc_ZCC(self, mapA, mapB, name='_output.png'):
        c = []
        order = 50
        for n in range(order + 1):
            A = []
            B = []
            _sum = []
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(l + 1):
                        momA = mapA[(n,l,m)]
                        momB = np.conjugate(mapB[(n,l,m)])
                        _sum.append(np.abs(momA*momB))
                        A.append(momA)
                        B.append(momB)
            nom = np.sum(_sum)
            dom_1 = np.sum(np.abs(A)**2)
            dom_2 = np.sum(np.abs(B)**2)
            dom = np.sqrt(dom_1 * dom_2)
            c.append(nom/dom)
        fig, ax = plt.subplots(constrained_layout=True)
        ax.grid(visible=True, axis='both')
        ax.plot(np.arange(1, len(c) + 1), c, '-g', linewidth=2.0)
        plt.xlim(np.array([1, len(c)]))
        plt.ylim(np.array([- 0.1, 1.05]))
        plt.savefig(''.join(('./ZCC_analysis_component_maps', name)))
        # plt.show()

    def ZCC(self):
        #Calculate a zernike component correlation ("FSC") curve.
        m1 = Moments()
        m1.load(self.args.query_space)
        m1.calculate_invariants()
        m1.stats()
        m1_moms, m1_mask = self.ndmoment_to_ndarray(m1)

        self.m = [m1]
        self.moms = [m1_moms]
        self.mask = [m1_mask]

        m2 = Moments()
        m2.load(self.args.target_spaces[0])
        m2.calculate_invariants()
        m2.stats()
        m2_moms, m2_mask = self.ndmoment_to_ndarray(m2)

        self.m.append(m2)
        self.moms.append(m2_moms)
        self.mask.append(m2_mask)

        mapA = self.m[0].ndmoments
        mapB = self.m[1].ndmoments

        for i,maps in enumerate(itertools.product(mapA, mapB)):
            mapA = self.m[0].ndmoments[maps[0]]
            mapB = self.m[1].ndmoments[maps[1]]
            self.calc_ZCC(mapA, mapB, name=str(i))


if __name__ == "__main__":
    class bcolors:
        HEADER = '\033[91m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def valid_npz(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.npz'):
            raise argparse.ArgumentTypeError('File must have a .npz extension')
        return param

    def valid_mrc(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.mrc','.MRC'):
            raise argparse.ArgumentTypeError('File must have a .mrc extension')
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

        *** ZERNIKE *** v0.1 - Zernike moment analysis of 3DVA cryo-EM latent space (WIGGLE npz format)

                Charles Bayly-Jones (2022-2023) - Monash University, Melbourne, Australia

    ###########################################################################################################'''+bcolors.ENDC),
        formatter_class=argparse.RawTextHelpFormatter #ArgumentDefaultsHelpFormatter
    )

    # create sub-parser
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode: `trajectory`, `all-vs-all`, or `merge`.",
        dest="operating_mode",
        required=True,
    )

    # create the parser for the "trajectory" sub-command
    parser_1 = sub_parsers.add_parser("trajectory", help="Search an input list of latent coordinates (query space) for similar paths in target spaces")
    parser_1.add_argument(
        "--query_space",
        type=valid_npz,
        help="Zernike moments of query latent space (in .npz format) from which search queries are generated.",
        required=True
    )
    parser_1.add_argument(
        "--target_spaces",
        type=valid_npz,
        help="Zernike moments of target latent space(s) (in .npz format) against which queries are searched.",
        required=True
    )
    parser_1.add_argument(
        "--trajectory",
        type=valid_mrc,
        help="A single coordinate or list of coordinates in index format to search.",
        required=True
    )

    # create the parse for the "all-vs-all" sub-command
    parser_2 = sub_parsers.add_parser("all-vs-all", help="Construct a KD-tree to query all latent coordinates of query space against all other spaces")
    parser_2.add_argument(
        "--query_space",
        type=valid_npz,
        help="Zernike moments of query latent space (in .npz format) from which search queries are generated.",
        required=True
    )
    parser_2.add_argument(
        "--do_reconstruct",
        action='store_true',
        help="Whether to reconstruct a volume series along the principal components given in --PC_travers list."
    )
    parser_2.add_argument(
        "--pc_travers",
        help="List of PC components from which to generate volume series e.g. --pc_travers 0 1 2",
        nargs='+'
    )
    parser_2.add_argument(
        "--query_3dva",
        type=valid_npz,
        help="Original 3DVA bundle file for the query set (in .npz format)."
    )
    parser_2.add_argument(
        "--target_3dva",
        type=valid_npz,
        help="Original 3DVA bundle file for the target set (in .npz format)."
    )
    parser_2.add_argument(
        "--target_spaces",
        type=valid_npz,
        help="Zernike moments of target latent space(s) (in .npz format) against which queries are searched.",
        required=True,
        nargs='+'
    )
    parser_2.add_argument(
        "--labels",
        help="Graph labels.",
        required=True,
        nargs='+'
    )
    parser_2.add_argument(
        "--outname",
        help="Output figure name.",
        required=True
    )

    # create the parse for the "all-vs-all" sub-command
    parser_3 = sub_parsers.add_parser("ZCC", help="Construct a KD-tree to query all latent coordinates of query space against all other spaces")
    parser_3.add_argument(
        "--query_space",
        type=valid_npz,
        help="Zernike moments of query latent space (in .npz format) from which search queries are generated.",
        required=True
    )
    parser_3.add_argument(
        "--target_spaces",
        type=valid_npz,
        help="Zernike moments of target latent space(s) (in .npz format) against which queries are searched.",
        required=True,
        nargs='+'
    )


    args = my_parser.parse_args()
    Z = Z_3DVA(args)
    if args.operating_mode == 'trajectory':
        Z.search_trajectory()
    elif args.operating_mode == 'all-vs-all':
        Z.all_vs_all()
    elif args.operating_mode == 'ZCC':
        Z.ZCC()