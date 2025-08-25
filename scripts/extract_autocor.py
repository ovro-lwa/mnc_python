#!/usr/bin/env python3
#
# This script extracts self-correlation spectra for all antennas for a
# specified day/hr/min/sec. It combines all 16 subbands to create the
# spectra and saves them as compressed NumPy '.npz' arrays for further processing.
#
# This version is OPTIMIZED FOR PARALLEL EXECUTION.
#

import argparse
import logging
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from casacore.tables import table
from tqdm import tqdm

# --- Setup ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# --- Core Functions (largely unchanged) ---

def _get_autocorr_indices(num_antennas: int) -> np.ndarray:
    """
    Calculate the indices of the autocorrelation data in a flattened visibility array.
    """
    steps = np.arange(num_antennas, 1, -1)
    return np.cumsum(np.insert(steps, 0, 0))


def read_ms_data(ms_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Extracts frequency, time, antenna names, and autocorrelation data from a single MS file.
    """
    if not ms_path.is_dir():
        logging.warning(f"Measurement Set not found or is not a directory: {ms_path}")
        return None
    try:
        with table(str(ms_path / 'SPECTRAL_WINDOW'), readonly=True, ack=False) as tb:
            freq = tb.getcol('CHAN_FREQ')
        with table(str(ms_path / 'ANTENNA'), readonly=True, ack=False) as tb:
            antname = tb.getcol('NAME')
            num_antennas = len(antname)
        with table(str(ms_path), readonly=True, ack=False) as tb:
            data = tb.getcol('DATA')
            time_val = tb.getcol('TIME')
        indices = _get_autocorr_indices(num_antennas)
        autocor = data[indices, :, :]
        return {"freq": freq, "time": time_val, "autocor": autocor, "antname": antname}
    except Exception as e:
        logging.error(f"Failed to read data from {ms_path}: {e}")
        return None


def process_timestamp(timestamp: str, ms_files_dict: Dict[str, List[Path]], output_dir: Path):
    """
    Processes all sub-bands for a single timestamp, combines them, and saves the result.
    
    Note: This function is now designed to be called by a parallel worker.
    """
    ms_files = ms_files_dict[timestamp]
    # The logging inside this function will appear from the worker processes.
    logging.info(f"Processing {len(ms_files)} sub-bands for timestamp: {timestamp}")

    freq_list, time_list, autocor_list = [], [], []
    antname = None

    for ms_path in sorted(ms_files):
        data = read_ms_data(ms_path)
        if data:
            freq_list.append(data["freq"])
            time_list.append(data["time"])
            autocor_list.append(data["autocor"])
            if antname is None:
                antname = data["antname"]

    if not autocor_list:
        logging.warning(f"No valid data found for timestamp {timestamp}. Skipping.")
        return

    combined_freq = np.concatenate(freq_list, axis=0).flatten()
    combined_time = np.concatenate(time_list, axis=0)
    combined_autocor = np.concatenate(autocor_list, axis=1)

    output_filename = output_dir / f"{timestamp}.npz"
    np.savez_compressed(
        output_filename,
        antname=antname,
        time=combined_time,
        freq=combined_freq,
        autocor=combined_autocor
    )
    # Return the path to indicate completion, useful for the progress bar.
    return output_filename


def find_and_group_ms_files(base_path: Path, date: str, time_filter: str) -> Dict[str, List[Path]]:
    """
    Finds all MS files matching the date and time criteria and groups them by timestamp.
    """
    date_path_str = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
    time_glob_pattern = time_filter + '*'

    #search_pattern = f"*/{date_path_str}/*/{date}_{time_glob_pattern}*.ms"
    search_pattern = f"*/{date_path_str}/*/{date}_{time_glob_pattern}.ms"
    
    logging.info(f"Searching for MS files in '{base_path}' with pattern: '{search_pattern}'")

    files_by_timestamp = defaultdict(list)
    ms_paths = list(base_path.glob(search_pattern))

    if not ms_paths:
        logging.warning("No Measurement Set files found for the specified criteria.")
        return {}

    logging.info(f"Found {len(ms_paths)} total MS files. Grouping by timestamp...")

    for path in ms_paths:
        try:
            parts = path.name.split('_')
            timestamp = f"{parts[0]}_{parts[1]}"
            files_by_timestamp[timestamp].append(path)
        except IndexError:
            logging.warning(f"Could not parse timestamp from filename: {path.name}")
            continue
            
    return files_by_timestamp


def extract_autocorrelations(path: str, date: str, time: str, step: int, workingdir: str, workers: int):
    """
    Main function to find, process, and save self-correlation spectra in parallel.
    """
    base_data_path = Path(path)
    output_dir = Path(workingdir) / date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_by_timestamp = find_and_group_ms_files(base_data_path, date, time)
    
    if not files_by_timestamp:
        return

    sorted_timestamps = sorted(files_by_timestamp.keys())
    timestamps_to_process = sorted_timestamps[::step]
    
    num_jobs = len(timestamps_to_process)
    logging.info(f"Found {len(sorted_timestamps)} unique timestamps. "
                 f"Will process {num_jobs} in parallel using {workers} workers.")
    
    # *** NEW: PARALLEL EXECUTION BLOCK ***
    # Use 'partial' to create a function that only needs the 'timestamp' argument
    task_function = partial(process_timestamp, ms_files_dict=files_by_timestamp, output_dir=output_dir)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs to the executor
        futures = {executor.submit(task_function, ts): ts for ts in timestamps_to_process}
        
        # Use tqdm to create a progress bar as jobs are completed
        for future in tqdm(as_completed(futures), total=num_jobs, desc="Processing Timestamps"):
            try:
                result = future.result()
                # You could add logging here, e.g., logging.info(f"Successfully wrote {result}")
            except Exception as exc:
                timestamp = futures[future]
                logging.error(f"Timestamp {timestamp} generated an exception: {exc}")


def main():
    """Argument parsing and script execution."""
    parser = argparse.ArgumentParser(
        description='Extract self-correlation spectra from Measurement Set (MS) files in parallel.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (all other arguments are the same)
    parser.add_argument(
        '-p', '--path', type=str, required=True,
        help='Absolute path to the parent directory containing the data (e.g., /lustre/pipeline/).'
    )
    parser.add_argument(
        '-d', '--date', type=str, required=True,
        help='Date of the observations in YYYYMMDD format (e.g., 20231214).'
    )
    parser.add_argument(
        '-t', '--time', type=str, required=False, default='',
        help='Time of observations. Can be HHMMSS, HHMM, or HH. \nIf not provided, all observations for the specified day will be processed.'
    )
    parser.add_argument(
        '-s', '--step', type=int, required=False, default=60,
        help='Step used to sub-sample the data. \n'
             'step=1 -> use every file (~10s cadence).\n'
             'step=6 -> use 1 of every 6 files (~1min cadence).\n'
             'step=60 -> use 1 of every 60 files (~10min cadence, default).'
    )
    parser.add_argument(
        '-w', '--workingdir', type=str, required=False, default='/lustre/ai/',
        help='Path to save the output autocorrelation files (default: /lustre/ai/).'
    )
    # *** NEW: WORKERS ARGUMENT ***
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help=f"Number of parallel worker processes to use (default: number of CPU cores, {os.cpu_count()})."
    )
    
    args = parser.parse_args()
    extract_autocorrelations(args.path, args.date, args.time, args.step, args.workingdir, args.workers)


if __name__ == '__main__':
    main()
