from prepare_data import create_data
from train_model import training
from prediction_denoise import prediction
from args import parser

if __name__ == '__main__':

    args = parser.parse_args()

    mode = args.mode

    # Initialize all modes to zero
    data_mode = False
    training_mode = False
    prediction_mode = False

    # Update with the mode the user is asking
    if mode == 'prediction':
        prediction_mode = True
    elif mode == 'training':
        training_mode = True
    elif mode == 'data_creation':
        data_mode = True

    if data_mode:
        # Example: python main.py --mode="data_creation" --voice_dir="./datasets/cv-corpus/training_clips" --noise_dir="./datasets/UrbanSound8K/audio/fold1"

        # folder containing noises
        noise_dir = args.noise_dir
        # folder containing clean voices
        voice_dir = args.voice_dir
        # path to save spectrograms to be used for training purposes
        path_save_spectrogram = './output/train/spectrogram/'
        # Sample rate to read audio
        sample_rate = 50
        # Minimum duration of audio files to consider
        min_duration = 1.0
        # Frame length for training data
        frame_length = 8064
        # hop length for clean voice files
        hop_length_frame = 8064
        # hop length for noise files
        hop_length_frame_noise = 5000
        # How much frame to create for training
        nb_samples = 50
        # nb of points for fft(for spectrogram computation)
        n_fft = 255
        # hop length for fft
        hop_length_fft = 63

        create_data(noise_dir, voice_dir, path_save_spectrogram, sample_rate,
                    min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft)

    elif training_mode:
        # Example: python main.py --mode="training" --training_from_scratch=False

        # Path were to read spectrograms of noisy voice and clean voice
        path_save_spectrogram = './output/train/spectrogram/'
        # path to find pre-trained weights / save models
        weights_path = './weights'
        # pre trained model
        name_model = args.name_model
        # Training from scratch vs training from pre-trained weights
        training_from_scratch = args.training_from_scratch
        # epochs for training
        epochs = args.epochs
        # batch size for training
        batch_size = args.batch_size

        training(path_save_spectrogram, weights_path, name_model,
                 training_from_scratch, epochs, batch_size)

    elif prediction_mode:
        # Example: python main.py --mode="prediction" --name_model="model_best"

        # path to find pre-trained weights / save models
        weights_path = './weights'
        # pre trained model
        name_model = args.name_model
        # directory where read noisy sound to denoise
        audio_dir_prediction = './testing_batch/input_samples'
        # directory to save the denoise sound
        dir_save_prediction = './testing_batch/predicted_samples/'
        # Name noisy sound file to denoise
        audio_input_prediction = [
            'testing1.wav', 'testing2.wav', 'testing3.wav', 'testing4.wav', 'testing5.wav']
        # Name of denoised sound file to save
        audio_output_prediction = 'output.wav'
        # Sample rate to read audio
        sample_rate = 50
        # Minimum duration of audio files to consider
        min_duration = 1.0
        # Frame length for training data
        frame_length = 8064
        # hop length for sound files
        hop_length_frame = 8064
        # nb of points for fft(for spectrogram computation)
        n_fft = 255
        # hop length for fft
        hop_length_fft = 63

        prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
                   audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)
