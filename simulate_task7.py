#!/usr/bin/env python3
import argparse
from os.path import join
import time
from multiprocessing import Pool
import numpy as np
from numba import njit

# LOAD_DIR = '../modified_swiss_dwellings/'
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@njit
def jacobi_numba(u_init, interior_mask, max_iter, atol=1e-6):
    u = np.ascontiguousarray(u_init)
    m, n = interior_mask.shape

    u_new = np.empty_like(u)

    for it in range(max_iter):

        u_new[:] = u
        diff = 0.0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if interior_mask[i - 1, j - 1]:
                    val = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    u_new[i, j] = val

                    d = abs(val - u[i, j])
                    if d > diff:
                        diff = d

        u, u_new = u_new, u

        if diff < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

def load_data_for_building(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

def process_building(bid):
    u0, interior_mask = load_data_for_building(bid)
    u_final = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u_final, interior_mask)

    return bid, stats

def main():
    parser = argparse.ArgumentParser(description="Parallel static scheduling simulation")
    parser.add_argument('--n', type=int, default=100, help="Number of floorplans to process")
    parser.add_argument('--workers', type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    building_ids = building_ids[:args.n]
    print(f"Processing {len(building_ids)} floorplans with {args.workers} workers")

    u0, mask0 = load_data_for_building(building_ids[0])
    jacobi_numba(np.ascontiguousarray(u0), np.ascontiguousarray(mask0), 1, ABS_TOL)

    start_time = time.time()
    with Pool(processes=args.workers) as pool:
        jobs = [
            (np.ascontiguousarray(*load_data_for_building(bid)),) + (bid,)
            for bid in building_ids
        ]
        results = pool.map(process_building, building_ids)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Time of using {args.workers} workers to process {args.n} floors: {total_time:.2f} seconds." )

    # Output the results
    print("building_id, mean_temp, std_temp, pct_above_18, pct_below_15")
    for bid, stats in results:
        print(f"{bid}, {stats['mean_temp']:.2f}, {stats['std_temp']:.2f}, {stats['pct_above_18']:.2f}, {stats['pct_below_15']:.2f}")

if __name__ == '__main__':
    main()