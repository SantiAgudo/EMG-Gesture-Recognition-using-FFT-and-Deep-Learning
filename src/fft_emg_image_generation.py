import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import gc

def calculate_fft_matrices(time_array, emg_signals, activity_intervals, restimulus, sampling_frequency, movements_of_interest):
    fft_matrices = []
    window_samples = 200
    overlap = 100

    for start, end in activity_intervals:
        start_idx = np.argmax(time_array >= start)
        end_idx = np.argmax(time_array >= end)

        if start_idx >= end_idx:
            continue

        emg_segments = emg_signals[start_idx:end_idx, :]
        current_mov = restimulus[start_idx:end_idx]
        active_movements = current_mov[np.isin(current_mov, movements_of_interest)]

        if len(active_movements) == 0:
            continue

        num_samples = len(emg_segments)
        for i in range(0, num_samples - window_samples + 1, overlap):
            emg_window = emg_segments[i:i + window_samples, :]
            window_mov = current_mov[i:i + window_samples]

            if len(np.unique(window_mov)) > 1:
                continue

            if len(emg_window) == window_samples:
                fft_emg = np.abs(np.fft.fft(emg_window, axis=0))[:window_samples // 2, :]
                mov = int(np.unique(window_mov)[0])
                fft_matrices.append([mov, fft_emg])

    return fft_matrices


def process_and_save_images(base_path, output_base_path, num_subjects, experiment, channels, movements_of_interest):
    for subject in range(1, num_subjects + 1):
        folder_name = f's{subject}'
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            print(f'Path {folder_path} does not exist.')
            continue

        exp_file = f'S{subject}_{experiment}_A1.mat'
        mat_path = os.path.join(folder_path, exp_file)

        if not os.path.exists(mat_path):
            print(f'File {exp_file} does not exist in folder {folder_name}.')
            continue

        mat_file = loadmat(mat_path)
        emg_signals = mat_file['emg']
        restimulus = mat_file['restimulus'].flatten()
        sampling_frequency = mat_file['frequency'][0, 0]
        num_samples = emg_signals.shape[0]
        time_array = np.arange(0, num_samples) / sampling_frequency

        activity = restimulus > 0
        changes = np.diff(activity.astype(int))

        start_indices = np.where(changes == 1)[0] + 1
        end_indices = np.where(changes == -1)[0] + 1

        if activity[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if activity[-1]:
            end_indices = np.append(end_indices, len(activity))

        if len(start_indices) == 0 or len(end_indices) == 0:
            print(f'No activity intervals found in file {exp_file}.')
            continue

        activity_intervals = list(zip(start_indices / sampling_frequency, end_indices / sampling_frequency))

        desired_emg_signals = emg_signals[:, [channel - 1 for channel in channels]]
        fft_matrices = calculate_fft_matrices(time_array, desired_emg_signals, activity_intervals, restimulus, sampling_frequency, movements_of_interest)

        for idx, (mov, matrix) in enumerate(fft_matrices):
            output_folder = os.path.join(output_base_path, f'Movement_{mov}')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            filename = f'subject_{subject}_window_{idx+1}.png'
            filepath = os.path.join(output_folder, filename)

            # Normalize and save image
            fig, ax = plt.subplots()
            ax.imshow(matrix.T, cmap='gray', aspect='auto')
            ax.axis('off')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            gc.collect()

        print(f'Subject {subject} processed.')


# Example usage
base_path = '/content/drive/My Drive/'  # Make sure this path is correct
output_base_path = '/content/drive/My Drive/PAPER/'  # Path to save images
num_subjects = 10
experiment = 'E1'
channels = range(1, 9)
movements_of_interest = range(1, 13)

process_and_save_images(base_path, output_base_path, num_subjects, experiment, channels, movements_of_interest)
