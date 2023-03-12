import argparse

parser = argparse.ArgumentParser()

# mode to run the program (options: data creation, training or prediction)
parser.add_argument('--mode', default='prediction', type=str,
                    choices=['data_creation', 'training', 'prediction'])

# folders for noise and clean data
parser.add_argument(
    '--noise_dir', default='./datasets/UrbanSound8K/audio/fold1', type=str)

parser.add_argument(
    '--voice_dir', default='./datasets/cv-corpus/training_clips', type=str)

# no. of samples
parser.add_argument('--nb_samples', default=50, type=int)

# Training from scratch or pre-trained weights to be used
parser.add_argument('--training_from_scratch', default=True, type=bool)

# no. of epochs - hyperparameter
parser.add_argument('--epochs', default=10, type=int)

# batch size - hyperparameter
parser.add_argument('--batch_size', default=20, type=int)

# model name - can be changed in case you wish to use your own saved model
parser.add_argument('--name_model', default='model_unet', type=str)
