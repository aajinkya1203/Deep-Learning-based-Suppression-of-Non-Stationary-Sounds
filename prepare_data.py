import os
import librosa
from data_tools import audioToNP
from data_tools import mixRandomNoise, npToSpectrogram
import numpy as np
import soundfile as sf

# creating randomly blended noisy inputs


def create_data(noise_dir, voice_dir, path_save_spectrogram, sample_rate,
                min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):

    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)

    # Extracting noise and voice from folder and convert to numpy
    noise = audioToNP(noise_dir, list_noise_files, sample_rate,
                      frame_length, hop_length_frame_noise, min_duration)
    print("NOISE DONE")
    voice = audioToNP(voice_dir, list_voice_files,
                      sample_rate, frame_length, hop_length_frame, min_duration)
    print("VOICE DONE")

    # Blend some clean voices with random selected noises (and a random level of noise)
    prod_voice, prod_noise, prod_noisy_voice = mixRandomNoise(
        voice, noise, nb_samples, frame_length)

    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = npToSpectrogram(
        prod_voice, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noisy_voice,  m_pha_noisy_voice = npToSpectrogram(
        prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

    # Save timeseries, spectrogram useful for training process
    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)
