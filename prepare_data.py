# prepare_data.py
# This script downloads the MIT-BIH Arrhythmia database,
# extracts normal heartbeats, and saves them for training.

import wfdb
import numpy as np
import os

# --- Configuration ---
DATASET_DIR = 'mit-bih-data'
PROCESSED_DATA_FILE = 'processed_ecg_data.npy'
SAMPLE_RATE = 360  # The sample rate of the MIT-BIH database
WINDOW_SIZE = 180  # We'll take a window of 180 samples around each heartbeat peak

def download_and_prepare_data():
    """
    Downloads the MIT-BIH Arrhythmia database and processes the ECG signals.
    """
    print("--- Phase 1: Downloading Data ---")
    
    # Create a directory to store the dataset if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created directory: {DATASET_DIR}")

    # Download the dataset using the wfdb library
    # This will download all the records from the 'mitdb' database
    # into the specified directory. It might take a few minutes.
    try:
        # CORRECTED LINE: Changed 'dir_name' to 'dl_dir'
        wfdb.dl_database('mitdb', dl_dir=DATASET_DIR)
        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print("\n--- Phase 2: Processing Data ---")
    
    # Get a list of all the record names in the database
    record_names = wfdb.get_record_list('mitdb')
    all_normal_beats = []

    # Loop through each record in the dataset
    for record_name in record_names:
        print(f"Processing record: {record_name}...")
        
        try:
            # Construct the full path to the record file
            record_path = os.path.join(DATASET_DIR, record_name)
            
            # Read the ECG signal data and its metadata
            record = wfdb.rdrecord(record_path)
            # Read the annotations (which tell us where the heartbeats are)
            annotation = wfdb.rdann(record_path, 'atr')

            # We'll use the first channel of the ECG signal (MLII)
            ecg_signal = record.p_signal[:, 0]

            # Get the locations (indices) of the R-peaks and their corresponding symbols
            r_peak_locations = annotation.sample
            beat_symbols = annotation.symbol

            # Find the R-peaks that are annotated as 'N' (Normal beat)
            for i, symbol in enumerate(beat_symbols):
                if symbol == 'N':
                    peak_loc = r_peak_locations[i]
                    
                    # Define the start and end of the window around the R-peak
                    start = peak_loc - WINDOW_SIZE // 2
                    end = peak_loc + WINDOW_SIZE // 2
                    
                    # Ensure the window is within the signal boundaries
                    if start >= 0 and end < len(ecg_signal):
                        # Extract the heartbeat segment (the window)
                        beat_segment = ecg_signal[start:end]
                        all_normal_beats.append(beat_segment)
        
        except Exception as e:
            print(f"Could not process record {record_name}. Error: {e}")

    print(f"\nExtracted {len(all_normal_beats)} normal heartbeats.")

    # Convert the list of beats into a NumPy array
    data = np.array(all_normal_beats)

    # --- Phase 3: Normalizing and Saving Data ---
    print("\n--- Phase 3: Normalizing and Saving ---")

    # Normalize the data to a range of [-1, 1]
    # This is important for training GANs effectively
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    
    # Reshape the data to add a "channels" dimension for the model
    # The shape will be (num_beats, window_size, 1)
    normalized_data = np.expand_dims(normalized_data, axis=-1)

    # Save the processed data to a file
    np.save(PROCESSED_DATA_FILE, normalized_data)
    print(f"Processed data saved to '{PROCESSED_DATA_FILE}'")
    print(f"Data shape: {normalized_data.shape}")


if __name__ == '__main__':
    download_and_prepare_data()
