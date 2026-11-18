import argparse
import time
from pathlib import Path
import h5py
import numpy as np
import MAS_library as MASL

BASE_PATH = Path("../../../../PCS/")
SNAP_TEMPLATE = BASE_PATH / "snapshots/{}/snapdir_004/snap_004.{}.hdf5"
HR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_128_z=0.npy"
LR_FOLDER_TEMPLATE = BASE_PATH / "df_m_{}_64_r_z=0.npy"

NUM_SIMULATIONS = 2000
NUM_FILES = 8 # Sub-files of each snapshot
BOX_SIZE = 1e6 # kpc/h
GRID_SIZE_HR = 128
GRID_SIZE_LR = 64

def read_positions(sim):
    pos = []
    for j in range(NUM_FILES):
        with h5py.File(str(SNAP_TEMPLATE).format(sim, j), "r") as f:
            pos.append(f["PartType1/Coordinates"][:])
    return np.vstack(pos)

def density_contrast(pos, grid_size):
    field = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    MASL.MA(pos, field, BOX_SIZE, "PCS", verbose=False)
    return field / np.mean(field) - 1.0

def mass_assign(sim, hr_path: Path, lr_path: Path):
    if hr_path.exists() and lr_path.exists(): return
    try:
        start = time.time()
        pos = read_positions(sim)
        hr_path.parent.mkdir(parents=True, exist_ok=True)
        lr_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hr_path, density_contrast(pos, GRID_SIZE_HR))
        np.save(lr_path, density_contrast(pos, GRID_SIZE_LR))
        print(f"Simulation {sim}: {len(pos)} particles, {time.time() - start:.0f} s")
    except FileNotFoundError: print(f"Warning: snapshot of simulation {sim} not found")
    except Exception as e: print(f"Error processing simulation {sim}: {e}")

def run_mass_assignment(sims):
    print("Starting mass assignment...")
    for i in sims:
        mass_assign(i, Path(str(HR_FOLDER_TEMPLATE).format(i)), Path(str(LR_FOLDER_TEMPLATE).format(i)))
    print("Mass assignment complete.")

# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", nargs=2, type=int, default=[0, NUM_SIMULATIONS], metavar=("START", "STOP"),
                    help="simulations from START to STOP excluded (default: all)")
    a = ap.parse_args()
    run_mass_assignment(range(*a.sims))
