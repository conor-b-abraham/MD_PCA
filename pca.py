import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
import sys
import argparse
import multiprocessing
import warnings
from tqdm import tqdm

# ------------------------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------------------------
W = "\033[0m"
R = "\033[1;31m"
E = "\033[0m"
print(f"""
{R}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
{R}░ {W}╔════════════════════════════════════════════════════════════════╗ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}██████╗{R}░░░{W}█████╗{R}░░░{W}█████╗{R}░░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}██╔══██╗{R}░{W}██╔══██╗{R}░{W}██╔══██╗{R}░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}██████╔╝{R}░{W}██║{R}░░{W}╚═╝{R}░{W}███████║{R}░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}██╔═══╝{R}░░{W}██║{R}░░{W}██╗{R}░{W}██╔══██║{R}░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}██║{R}░░░░░░{W}╚█████╔╝{R}░{W}██║{R}░░{W}██║{R}░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░{W}╚═╝{R}░░░░░░░{W}╚════╝{R}░░{W}╚═╝{R}░░{W}╚═╝{R}░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}║ {R}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ {W}║ {R}░
{R}░ {W}╚════════════════════════════════════════════════════════════════╝ {R}░
{R}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░{E}
""")


parser = argparse.ArgumentParser(description='PERFORM PRINCIPAL COMPONENT ANALYSIS ON MD TRAJECTORY',
                                 epilog='Will always save principal component analysis results to pca_results.npz containing "principal_coordinates", "cumulative_variances", "eigenvectors", and "eigenvalues". If KDE is performed (it is by default), PCA_KDE.npz will also be saved containing the PC1 datapoints ("xdata"), PC2 datapoints ("ydata"), the datapoints probability densities ("zdata"), the datapoints energies ("edata"), the grid x values ("xgrid"), the grid y values ("ygrid"), the grid points probability densities ("zgrid"), the grid points energies ("egrid"), and the bandwidth used ("bandwidth"). Energies are reported in kcal/mol and are dependent upon the temperature specified with -t/--temperature. If no temperature is given or if temperature is set to -1, energies are reported in kT.')
INPUT_FILES = parser.add_argument_group('INPUT')
INPUT_FILES.add_argument('-p', '--protein', required=True, help='(REQUIRED) Name of protein (i.e. BACE1 or ADAM10)')
INPUT_FILES.add_argument('-m','--membrane', required=True, help='(REQUIRED) Name of membrane (i.e. DUPC90-CHOL10 or DPPC70-CHOL30)')
OPTIONS = parser.add_argument_group('OPTIONS')
OPTIONS.add_argument('-k', '--nokde', action='store_false', help='(OPTIONAL) Do NOT perform kernel density estimation (KDE) on first and second principal components. By default, KDE is performed. ')
OPTIONS.add_argument('-r', '--kderoundedge', default=-1, type=int, help='(OPTIONAL; Default=-1) Integer to round KDE boundaries to. Be aware that this will result in the KDE being extrapolated beyond the bounds of the data. For example, if principal component 1 extends from -117.5 to 140.3 and the 25 is used here the KDE grid along principal component 1 will be extended to -125 to 150. Use option -1 to simply round grid boundaries to the nearest integer. If -k/--nokde is used this option will be ignored.' )
OPTIONS.add_argument('-t', '--temperature', default=-1, type=float, help="(OPTIONAL; Default=-1) The temperature (in Kelvin) for converting KDE probability density to energy. If -k/--nokde is used this option will be ignored. If -1 is used, energies will be reported in units of kT.")
OPTIONS.add_argument('-n', '--nprocs', default=-2, type=int, help='(OPTIONAL; Default=-2) Number of processors to use for kernel density estimation. Use -1 to use all processors. Use -2 (default) to use half of the available processors.')
args = parser.parse_args()

p = args.protein
m = args.membrane
DO_KDE = args.nokde
KDE_ROUND = args.kderoundedge
TEMP = args.temperature
NP = args.nprocs
dir = f"{p}/{m}/analysis"
TOP = f"{dir}/3_PCA.pdb"
TRAJ = f"{dir}/3_PCA.xtc"
print("\033[1;36m"+"JOB PARAMETERS"+'\033[0m'+f"""
\033[1m         Protein: \033[0m {p}
\033[1m        Membrane: \033[0m {m}
\033[1m   Topology File: \033[0m {TOP}
\033[1m Trajectory File: \033[0m {TRAJ}
\033[1m    Perform KDE?: \033[0m {DO_KDE}""")

if DO_KDE:
    if TEMP == -1:
        print("\033[1m     Temperature: \033[0m None, Energy will be reported in kT")
    else:
        print(f"\033[1m    Temperature: \033[0m {TEMP}K")
    if KDE_ROUND != -1:
        print(f"\033[1m  Round KDE Grid: \033[0m to {KDE_ROUND}")
    else:
        print(f"\033[1m  Round KDE Grid: \033[0m to nearest integer")
    if NP == -1:
        NP = multiprocessing.cpu_count()
    elif NP == -2:
        NP = int(multiprocessing.cpu_count()/2)
    print(f"\033[1m  No. Processors: \033[0m {NP}")
elif KDE_ROUND != -1:
    print(f"\033[1m Option -r/--kderoundedge ignored because KDE is not being performed. \033[0m")
print("\n")

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def _rmsd(ref, coords):
    """
    Compute the rmsd between ref and coords
    """
    return np.std(ref-coords)

def kabsch(coordinates, masses):
    """
    RMSD fit coordinates to first set of coordinates using the Kabsch algorithm.

    Arguments
    ---------
    coordinates : Numpy.ndarray (n_frames, n_atoms, 3)
        Coordinates to fit
    masses : Numpy.ndarray (n_atoms)
        Masses of atoms

    Returns
    -------
    coordinates : Numpy.ndarray (n_frames, n_atoms, 3)
        Updated coordinates with all sets of coordinates fit to the first set of
        coordinates.
    """
    # Center reference coordinates
    coordinates[0] -= np.average(coordinates[0], axis=0, weights=masses)

    rmsd_i, rmsd_f = 0, 0
    for ts, coords in enumerate(coordinates[1:]):
        # Center mobile coordinates
        coords -= np.average(coords, axis=0, weights=masses)

        # Calculate initial RMSD
        rmsd_i += _rmsd(coordinates[0], coords)

        # Kabsch Algorithm
        h = coords.T @ coordinates[0] # Cross-Covariance Matrix
        (u, s, vt) = np.linalg.svd(h) # Singular Value Decomposition
        v = vt.T # Numpy's svd function returns the transpose of v, so transpose it back
        d = np.sign(np.linalg.det(v @ u.T)) # Correct rotation for right-handed coordinate system
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d]])
        rot = v @ mat @ u.T # find rotation matrix
        coordinates[ts+1] = coords @ rot.T # apply rotation matrix

        # Calculate final RMSD
        rmsd_f += _rmsd(coordinates[0], coordinates[ts+1])

    print("\033[1;36m"+"TRAJECTORY FITTING"+'\033[0m')
    print(f"\033[1m Average Initial RMSD: \033[0m {rmsd_i/(coordinates.shape[0]-1)}")
    print(f"\033[1m   Average Final RMSD: \033[0m {rmsd_f/(coordinates.shape[0]-1)}")
    print()
    return coordinates

def _cumvar(e):
    '''
    Compute the cumulative variance given 1D array of eigenvalues
    '''
    cumvar = np.cumsum(e/np.sum(e))
    print("\033[1;36m"+"PRINCIPAL COMPONENT ANALYSIS"+'\033[0m')
    print("\033[1m"+f"{'Principal Component':>25}{'Cumulative Variance':>25} \033[0m")
    for i, c in enumerate(cumvar):
        if c <= 0.95:
            cr = f"{np.round(c, 3):.3f}"
            print(f"{i+1:>25}{cr:>25}")
    return cumvar

def find_mean_coords(coords):
    '''
    Find set of coordinates that has the smallest rmsd from the mean
    '''
    ref = np.mean(coords, axis=0)
    RMSD = np.zeros(coords.shape[0])
    for i, f in tqdm(enumerate(coords)):
        RMSD[i] = np.mean((f-ref)**2) # Not taking square root because we just need to know the relative magnitudes
    medcoords = coords[np.argmin(RMSD)]
    return medcoords

def write_pca_files(ag, dir, coords, pcs, evals):
    # First, Write PDB File with mean coordinates
    ag.positions = coords
    warnings.filterwarnings(action='ignore', category=UserWarning) # Prevent Warning from missing information from PDB
    ag.write(f"{dir}/PCA_mean.pdb")
    print("\033[1m"+f"Mean Coordinates saved to {dir}/PCA_mean.pdb\033[0m")

    # Now, write normal mode wizard file
    title = "PCA"
    atomnames = " ".join(ag.names.tolist())
    resnames = " ".join(ag.resnames.tolist())
    chids = " ".join(["X"]*ag.n_atoms)
    resnums = " ".join(map(str, ag.resids.tolist()))
    betas = " ".join(["0.0"]*ag.n_atoms)
    coordinates = " ".join(map(str, ag.positions.flatten().tolist()))

    # Write the NMD file
    with open(f'{dir}/PCA_results.nmd', 'w') as w:
        w.write(f'title {title}\n')
        w.write(f'names {atomnames}\n')
        w.write(f'resnames {resnames}\n')
        w.write(f'chids {chids}\n')
        w.write(f'resnums {resnums}\n')
        w.write(f'betas {betas}\n')
        w.write(f'coordinates {coordinates}\n')
        for i in range(0, pcs.shape[1]):
            modes = " ".join([f"{m:.3f}" for m in np.round(pcs[:,i], 3)])
            w.write(f'mode {i+1} {np.sqrt(evals[i])} {modes}\n') # Sqrt of eval/variance might not be correct
    print("\033[1m"+f"PCA NMD file saved to {dir}/PCA_results.nmd \033[0m")

def round_up(x, base=1):
    '''
    Round up to nearest value divisible by base
    '''
    rounded = base * np.ceil(x/base)
    if base % 1 == 0:
        rounded = int(rounded)
    return rounded

def round_down(x, base=1):
    '''
    Round down to nearest value divisible by base
    '''
    rounded = base * np.floor(x/base)
    if base % 1 == 0:
        rounded = int(rounded)
    return rounded

def calculate_energies(logprob, TEMP):
    '''
    Calculate energies (in kcal/mol if TEMP != -1, else in kT) given probability densities
    '''
    if TEMP == -1:
        energies = -logprob
    else:
        energies = -1.380649e-23*6.02214076e23*TEMP*0.001*logprob/4.184
    energies -= np.min(energies)
    return energies

def KDE(x, y, KDE_ROUND, TEMP, NP):
    '''
    Perform gaussian kernel density estimation on two principal components.
    i.e. generate probability densities.

    Arguments
    ---------
    x : numpy.ndarray (n_frames,)
        First principal component
    y : numpy.ndarray (n_frames,)
        Second principal component

    Returns
    -------
    kde_results : Dict{xdata, ydata, zdata, xgrid, ygrid, zgrid, bandwidth}
        xdata, ydata : numpy.ndarrays (n_frames,)
            The input data points (xdata is x, ydata is y). Included for convenience.
        zdata : numpy.ndarray (n_frames, )
            The probability of each snapshot
        edata : numpy.ndarray (n_frames, )
            The energies of each snapshot (in kcal/mol)
        xgrid, ygrid : numpy.ndarrays (300,300)
            Equally spaced gridpoints from min(xdata) to max(xdata) and min(ydata)
            to max(ydata)
        zgrid : numpy.ndarray (300,300)
            The probability of each gridpoint
        egrid : numpy.ndarray (300,300)
            The energies of each gridpoint (in kcal/mol)
        bandwidth : numpy.ndarray (1,)
            The optimized bandwidth
    '''
    print("\033[1;36m"+f"\nKERNEL DENSITY ESTIMATION \033[0m")
    print("\033[1m (This might take a while) \033[0m")
    start_time = time.time()
    # Sampling Grid
    if KDE_ROUND != -1:
        xmin = round_down(x.min(), KDE_ROUND)
        xmax = round_up(x.max(), KDE_ROUND)
        ymin = round_down(y.min(), KDE_ROUND)
        ymax = round_up(y.max(), KDE_ROUND)

    else:
        xmin = int(np.round(x.min()))
        xmax = int(np.round(x.max()))
        ymin = int(np.round(y.min()))
        ymax = int(np.round(y.max()))

    nbinsx = int(xmax-xmin)+1

    nbinsy = int(ymax-ymin)+1
    # while nbinsy < 300:
    #     nbinsy += int(ymax-ymin)

    # xx, yy = np.mgrid[xmin:xmax:int(xmax-xmin)j,ymin:ymax:int(ymax-ymin)j]
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, int(xmax-xmin)+1),
                         np.linspace(ymin, ymax, int(ymax-ymin)+1))
    xx = xx.T
    yy = yy.T
    gridpoints = np.vstack([xx.ravel(), yy.ravel()]).T

    # Randomly shuffled datapoints
    random_order = np.arange(x.size)
    np.random.shuffle(random_order)
    datapoints = np.vstack([x[random_order], y[random_order]]).T

    # KERNEL DENSITY ESTIMATION
    # Optimize Bandwidth
    bandwidths = np.round(10 ** np.linspace(-1, 1, 50), 3)
    CV = GridSearchCV(KernelDensity(), {'bandwidth':bandwidths}, cv=10, n_jobs=NP)
    CV.fit(datapoints)
    bandwidth = CV.best_params_['bandwidth']
    print('\033[1A', end='\x1b[2K') # Clear might take a while line
    print(f"\033[1m Optimized Bandwidth: \033[0m {bandwidth}")
    print(f"\033[1m               Score: \033[0m {CV.best_score_}")
    print(f"\033[1m        Grid X-Range:  \033[0m ({xx.min()}, {xx.max()})")
    print(f"\033[1m        Grid Y-Range:  \033[0m ({yy.min()}, {yy.max()})")
    print(f"\033[1m     Grid X-N_points:  \033[0m {xx.shape[0]}")
    print(f"\033[1m     Grid Y-N_points:  \033[0m {yy.shape[1]}")
    print(f"\033[1m Collecting Results \033[0m")
    # Choose Best Estimator
    kde = CV.best_estimator_
    # Collect Data and Compute Energies
    datalogprob = kde.score_samples(datapoints)
    dataprob = np.exp(datalogprob)
    dataenergies = calculate_energies(datalogprob, TEMP)
    gridlogprob = kde.score_samples(gridpoints)
    gridprob = np.exp(gridlogprob)
    gridenergies = calculate_energies(gridlogprob, TEMP)
    elapsed_time = time.time()-start_time
    print('\033[1A', end='\x1b[2K') # Clear collecting results line
    print(f"\033[1m          Time Taken:  \033[0m {elapsed_time/60:.2f} minutes")
    results = {'xdata':x,
               'ydata':y,
               'zdata':dataprob,
               'edata':dataenergies,
               "xgrid":xx,
               "ygrid":yy,
               "zgrid":gridprob.reshape(xx.shape),
               "egrid":gridenergies.reshape(xx.shape),
               "bandwidth":np.array(bandwidth)}

    return results

def PCA(u, sel, dir, DO_KDE, KDE_ROUND, TEMP, NP):
    '''
    Perform principal component analysis for atomgroup (selected with sel) given
    trajectory defined by universe, u.

    Arguments
    ---------
    u : MDAnalysis.universe
        Universe containing trajectory and masses
    sel : String
        Selection string to create atomgroup, ag, for which to compute the covariance
        matrix
    dir : String
        directory to write results
    DO_KDE : Bool
        if True, perform KDE on first and second principal components.
    '''

    # Get coordinates of trajectory and masses of atoms
    ag = u.select_atoms(sel)
    print("\033[1;36m"+'Collecting Coordinates:'+'\033[0m')
    coordinates = AnalysisFromFunction(lambda atoms: atoms.positions.copy(),
                                       ag).run(verbose=True).results['timeseries']
    masses = ag.masses
    print('\033[1A', end='\x1b[2K') # Clear progress bar
    print('\033[1A', end='\x1b[2K') # Clear collecting coordinates line

    # RMSD fit coordinates to first frame
    coordinates = kabsch(coordinates, masses)

    # # Find Median Structure
    # print("\033[1;36m"+'Finding Median Structure:'+'\033[0m')
    near_mean_coordinates = find_mean_coords(coordinates)
    # print('\033[1A', end='\x1b[2K') # Clear progress bar
    # print('\033[1A', end='\x1b[2K') # Clear finding median structure line

    # Mass weight and reshape coordinates array
    mass_weight_array = (np.sqrt(ag.masses)[np.newaxis, :, np.newaxis]) # ?
    coordinates = coordinates*mass_weight_array # Mass weight the coordinates
    coordinates = coordinates.reshape(coordinates.shape[0], 3*coordinates.shape[1]) # Reshape coordinate array

    # Construct Covariance Matrix : covmat = <(r_i-<r_i>)(r_j-<r_j>)>
    mean_coordinates = np.mean(coordinates, axis=0)
    coordinates = (coordinates - mean_coordinates) # values of (r_i-<r_i>) and (r_j-<r_j>)
    covmat = np.cov(coordinates, rowvar=False)

    # Compute Eigenvalues and Eigenvectors & Sort them
    e_values, e_vectors = np.linalg.eigh(covmat)
    order = np.argsort(e_values)[::-1]
    e_values = e_values[order]
    e_vectors = e_vectors[:, order]

    # Calculate Cumulative Variances
    cumulative_variances = _cumvar(e_values)

    # Transform Original Data
    principal_coordinates = np.dot(coordinates, e_vectors) # Weights
    meancoords_pcs = np.dot(near_mean_coordinates.flatten(), e_vectors)
    # Save PCA Results
    np.savez(f"{dir}/pca_results.npz", principal_coordinates=principal_coordinates,
             cumulative_variances=cumulative_variances, eigenvectors=e_vectors,
             eigenvalues=e_values, meancoords_pcs=meancoords_pcs)
    print("\033[1m"+f"\nPCA results saved to {dir}/pca_results.npz \033[0m")

    # Write VMD Visualization Files (mean coord PDB and NMD)
    print("\033[1;36m"+'\nWriting PCA Visualization Files:'+'\033[0m')
    write_pca_files(ag, dir, mean_coordinates.reshape((-1, 3)), e_vectors, e_values)
    # for pc in range(2):
    #     proj_coordinates = (np.outer(principal_coordinates[:, pc], e_vectors[:, pc]) + mean_coordinates).reshape(len(principal_coordinates[:, pc]), -1, 3)/mass_weight_array
    #     pu = mda.Merge(ag)
    #     pu.load_new(proj_coordinates, order="fac")
    #     pu.atoms.write(f"{dir}/pc_top.gro")
    #     print(f"Wrote topology for PC trajectories to {dir}/pc_top.gro")
    #     with mda.Writer(f"{dir}/pc{pc+1}_traj.xtc", pu.atoms.n_atoms) as w:
    #         for ts in tqdm(pu.trajectory):
    #             w.write(pu.atoms)
    #     print(f"Wrote Vis for PC{pc} to {dir}/pc{pc+1}_traj.xtc")


    # Perform Kernel Density Estimation (if on)
    if DO_KDE:
        kde_results = KDE(principal_coordinates[:, 0], principal_coordinates[:, 1], KDE_ROUND, TEMP, NP)
        np.savez(f"{dir}/PCA_KDE.npz", **kde_results)
        print(f"\033[1m"+f"\nKDE results saved to {dir}/PCA_KDE.npz \033[0m")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
u = mda.Universe(TOP, TRAJ)
PCA(u, "protein and name CA", f"{p}/{m}/analysis", DO_KDE, KDE_ROUND, TEMP, NP)

# with open("3_PCA_pca.nmd") as file:
#     modes = []
#     for l in file:
#         ll = l.split()
#         if ll[0] == "mode":
#             modes.append(list(map(float, ll[3:])))
#     modes = np.array(modes)

# with open("PCA_results.nmd") as file:
#     vectors = []
#     for l in file:
#         ll = l.split()
#         if ll[0] == "mode":
#             vectors.append(list(map(float, ll[3:])))
#     vectors = np.array(vectors)

# def compare(modes, vectors, pcnum):
#     pcmode = modes[pcnum, :]
#     pcmode = pcmode.reshape(int(np.round(pcmode.size/3)),3)
#     pcvec = vectors[pcnum, :]
#     pcvec = pcvec.reshape(int(np.round(pcvec.size/3)),3)
#     for i, (m, v) in enumerate(zip(pcmode, pcvec)):
#         mn = np.linalg.norm(m)
#         vn = np.linalg.norm(v)
#         cc = np.dot(m, v)/(mn*vn)
#         print(f"{i}{mn:>10.3f}{vn:>10.3f}{cc:>10.3f}")