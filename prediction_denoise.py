import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audioToNP, npToSpectrogram, spectrogramToNP
import soundfile as sf
import os
import math

# for prediction purposes


def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):

    # load json and create model
    json_file = open(weights_path+'/'+'model_unet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path+'/'+name_model+'.h5')
    print("Loaded model from disk")
    durations = []

    for file in audio_input_prediction:
        # open the audio file
        y, sr = librosa.load(os.path.join(
            audio_dir_prediction, file), sr=sample_rate)
        durations.append(math.floor(librosa.get_duration(y=y, sr=sr)))

    # Extracting noise and voice from folder and convert to numpy
    audio = audioToNP(audio_dir_prediction, audio_input_prediction, sample_rate,
                      frame_length, hop_length_frame, min_duration)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print("DIMENSION SPEC", dim_square_spec)
    prev = 0
    counter = 1
    for audio_inp_file, audio_dur in zip(audio_input_prediction, durations):
        # Create Amplitude and phase of the sounds
        m_amp_db_audio,  m_pha_audio = npToSpectrogram(
            audio[prev: prev + audio_dur], dim_square_spec, n_fft, hop_length_fft)
        prev += audio_dur
        # global scaling to have distribution -1/1
        X_in = scaled_in(m_amp_db_audio)
        # Reshape for prediction
        X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
        # Prediction using loaded network
        X_pred = loaded_model.predict(X_in)
        # Rescale back the noise model
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        # Remove noise model from noisy speech
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]
        # Reconstruct audio from denoised spectrogram and phase
        print("===> Processing Audio Sample #", counter)
        print("===> Denoised Shape:", X_denoise.shape)
        print("===> Phase Shape:", m_pha_audio.shape)
        print("===> Frame Length:", frame_length)
        print("===> Hop Length for FFT:", hop_length_fft)
        print("\n\n")
        counter += 1
        audio_denoise_recons = spectrogramToNP(
            X_denoise, m_pha_audio, frame_length, hop_length_fft)
        # Number of frames
        nb_samples = audio_denoise_recons.shape[0]
        # Save all frames in one file
        denoise_long = audio_denoise_recons.reshape(
            1, nb_samples * frame_length)*10
        sf.write(dir_save_prediction + audio_inp_file.split(".")[0] + "-" + audio_output_prediction,
                 denoise_long[0, :], sample_rate)
