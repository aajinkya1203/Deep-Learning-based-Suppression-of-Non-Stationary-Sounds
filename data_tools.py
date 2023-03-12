import librosa
import numpy as np
import os

# splits audio to several frames


def splitAudioToStack(sound_data, frame_length, hop_length_frame):
    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]

    if len(sound_data_list) > 0:
        return np.vstack(sound_data_list)
    return []


# merging audio files in a directory for a sliding window of size hop_length_frame
def audioToNP(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    list_sound_array = []

    for file in list_audio_files:
        # open the audio file
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            temp = splitAudioToStack(
                y, frame_length, hop_length_frame)
            if len(temp) != 0:
                list_sound_array.append(temp)
        else:
            print(
                f"The following file {os.path.join(audio_dir,file)} is below the min duration")
    return np.vstack(list_sound_array)

# blends noise and clean input data randomly


def mixRandomNoise(voice, noise, nb_samples, frame_length):
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice

# converts audio to spectrogram
# returns db and phase of spectrogram


def audio_to_magnitude_phase(n_fft, hop_length_fft, audio):
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude_db, stftaudio_phase


# same as above but with numpy data
def npToSpectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros(
        (nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase

# inverts a spectrogram to audio


def magnitude_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    stftaudio_magnitude_rev = librosa.db_to_amplitude(
        stftaudio_magnitude_db, ref=1.0)

    # taking magnitude and phase of audio
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(
        audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct


# same as above but to numpy audio
def spectrogramToNP(m_mag_db, m_phase, frame_length, hop_length_fft):
    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_phase_to_audio(
            frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)


# === helper functions below ===
def scaled_in(matrix_spec):
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


def scaled_ou(matrix_spec):
    matrix_spec = (matrix_spec - 6)/82
    return matrix_spec


def inv_scaled_in(matrix_spec):
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec


def inv_scaled_ou(matrix_spec):
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec
