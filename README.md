# Deep Learning-based Suppression of Non-Stationary Sounds for Enhanced Audio Quality

### Introduction

---

The presence of ambient noise in hearing audio severely impairs the intelligibility of human speech. Most of the natural audio recorded, unless recorded in a closed studio type environment, is always susceptible to background noise. Removing this background noise which varies in type and intensity and produce a clean audio is a difficult task. Here, we propose a solution which uses a fully Convolutional Neural Network trained on a data mapping between noisy speech and clean speech which suppresses the background noise in audio samples producing a cleaner audio.

### Noise

---

There are two types of fundamental noise types that exist: Stationary and Non-Stationary.

![Stationary vs Non-Stationary Signal’s Spectrogram](https://user-images.githubusercontent.com/43881544/229710268-6dde8f91-179f-41bd-9a0c-83843d4e4c10.png)


Stationary vs Non-Stationary Signal’s Spectrogram

**Stationary background** noise is a type of noise that has a constant statistical property over time, and it persists at a relatively constant level. Examples of stationary background noise include the hum of a refrigerator or the hiss of white noise.

**Non-stationary background** noise, on the other hand, is a type of noise that changes over time and can have varying levels of intensity. Examples of non-stationary background noise include sounds from traffic, voices in a crowded room, and rustling leaves in the wind.

Both types of background noise can interfere with the clarity of speech, music, or other audio signals, making it difficult to hear or understand them. Non-stationary background noise can be more challenging to remove or filter out than stationary background noise since the noise's characteristics change over time.

### Data Acquisition

---

For the problem of speech denoising, we used two popular publicly available audio datasets.

- [The Mozilla Common Voice (MCV)](https://voice.mozilla.org/) — Contains clean voice
- [The UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) — Contains noisy samples

From the MCV Dataset, we selected the Common Voice Delta Segment 12.0 which consisted of 63 recorded hours and 64 validated hours of data in mp3 format and it’s roughly around 1.22 GB in size. One very good characteristic of this dataset is the vast variability of speakers. It contains recordings of men and women from a large variety of ages and accents, roughly around 1,152 different voices.

![MCV Dataset](https://user-images.githubusercontent.com/43881544/229710348-f2dc7775-54da-42e1-9e8b-9d17da49f6ea.png)


The UrbanSound8K dataset also contains small snippets (<=4s) of sounds. However, there are 8732 labeled examples of ten different commonly found urban sounds. The complete list includes:

- *0 = air_conditioner*
- *1 = car_horn*
- *2 = children_playing*
- *3 = dog_bark*
- *4 = drilling*
- *5 = engine_idling*
- *6 = gun_shot*
- *7 = jackhammer*
- *8 = siren*
- *9 = street_music*

With this approach, we’ll be using urban sounds as noise signals to the speech examples. 

Moreover, for the problem of testing the model, we used a different dataset (unseen data) which consisted of the audio files and it’s verified transcription.

- **[Hugging Face — joefox/LibriSpeech_test_noise](https://huggingface.co/datasets/joefox/LibriSpeech_test_noise)**


![Testing Dataset for Predicting the output of the model](https://user-images.githubusercontent.com/43881544/229710452-2c89621b-0cbb-41d3-a811-acecc976793f.png)


### Dataset Preprocessing

---

Since our primary focus is on a clear distinction of non-stationary noise from a signal, therefore we gathered clean speech dataset (from [The Mozilla Common Voice (MCV)](https://voice.mozilla.org/)) , and along with that we gathered only noisy data with no speech in it (like dog barking, music playing, alarms, construction work etc). 

This allowed us to mix and map various types of noisy data with a variety of clean speech from various speakers having different accents.

- This mix and match was coupled with different Data Augmentation techniques like:
    
    **→ 1. Combining various sorts of noises with a particular clean data:**
    
    - This way we’ll be able to achieve:
        
        $$
        m^n
        $$
        
        where $m$ = # of noise samples and $n$ = # of clean samples
        
        ![Data Augmentation Technique #1](https://user-images.githubusercontent.com/43881544/229710516-5896e5a1-c5c9-4b26-8054-5ffa4a03a0fa.png)
        
    
    **→ 2. Combining various snippets of a particular noise sample with any random snippet of clean sample**
    
    - This way we’ll be able to achieve:
        
        $$
        {}_{t_m}C_{1} * {}_{t_k}C_{1}
        $$
        
        where $t_m$ is the time of a particular noise sample segmented with sampling rate of 1sec
        
        where $t_m$ is the time of a particular clean sample segmented with sampling rate of 1sec
        
        ![Data Augmentation Technique #2
        This was used for environmental noises, which involved creating different noise windows by taking them at different times.](https://user-images.githubusercontent.com/43881544/229710583-d4193a08-9d96-48e0-b1ac-cf7da3028dd1.png)
        

In other words, we first take a small speech signal — this can be someone speaking a random sentence from the MCV dataset.

Then, we add noise to it — such as a woman speaking or a dog barking in the background. We then merge the two sounds. Finally, we use this artificially noisy signal as the input to our deep learning model. The Deep Neural Network, in turn, receives this noisy signal and tries to output a clean representation of it.

The image below displays a visual representation of a clean input signal from the MCV (top), a noise signal from the UrbanSound8K dataset (middle), and the resulting noisy input (bottom) — the input speech after adding the noise signal.

Also, note that the noise power is set so that the signal-to-noise ratio (SNR) is zero dB (decibel).

> A ratio higher than 1:1 (greater than 0 dB) indicates more signal than noise, hence better quality of communication
> 

![Clean Audio (*from MCV*) + Noisy Signal (*from UrbanSound8K*) ⇒ Noisy Input](https://user-images.githubusercontent.com/43881544/229710714-009809c4-cffb-40e6-bc65-d7319fd756e3.png)
Clean Audio (*from MCV*) + Noisy Signal (*from UrbanSound8K*) ⇒ Noisy Input


### Process Overview

---

- Let’s get a quick rundown as to what all steps are included to achieve our goal:
    1. `**Generating Spectrograms**`
    2. `**Training the Model**`
    3. `**Prediction**`
    4. `**Code Structure & Analysis**`

### Generating Spectrograms

---

- First, we down sampled the audio signals (from both datasets) to 8k Hz and removed the silent frames from it. This is done to simplify computational processing and time as well as data size.
    - *Audio data, in its raw form, is a one-dimensional time-series data.*
    - *Images, on the other hand, are two-dimensional representations of an instant moment in time.*
    - *For these reasons, audio signals are often transformed into (time/frequency) time-frequency decomposition*
- Then we convert these audio signals to time-frequency decomposition — Spectrograms have been proved to be a useful representation for audio processing.
    - Spectrograms are 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame.
    - Since, these are images hence they’re naturally perfect for a CNN Model to be applied to them.
    - Now, there are 2 sorts of spectrogram:
        - `Magnitude Spectrogram`: contain most the structure / information / strength of the signal
        - `Phase Spectrogram`: shows only little temporal and spectral regularities.
    - Hence, `Magnitude Spectrogram` is more important for us.

        ![Magnitude Spectrogram](https://user-images.githubusercontent.com/43881544/229710806-e5c54584-e6fb-4a2e-8d23-c77b7d64fee8.png)
        
- On this Magnitude Spectrogram, we then apply our CNN Model: (U-Net, a Deep Convolutional Autoencoder with symmetric skip connections) which has been adapted to denoise these spectrograms


### Training the Model

---

We trained our model using a U-Net architecture, a Deep Convolutional Autoencoder with symmetric skip connections, which is slightly adapted to denoise / figure out noise spectrums in spectrograms.

The U-Net architecture is designed to learn a mapping between the input spectrogram and the output noise model using a set of learned convolutional filters. 

During the training process, the network learns to identify the patterns in the input spectrogram of the noisy input voice sample (generated from the previous step) and to generate an output noise model that minimizes the difference between the noisy input spectrogram and the clean target spectrogram.

The network uses a combination of convolutional layers, activation functions, pooling layers, and skip connections to extract and propagate information at different scales.

- The convolutional layers apply a set of learned filters to the input spectrogram to extract local features, while the activation functions (in this case, LeakyReLU) introduce non-linearity to the model and help to capture more complex patterns.
- The pooling layers (in this case, maxpooling) downsample the feature maps, reducing the spatial dimensions of the input and allowing the model to capture more global information. The skip connections allow the network to propagate information from the encoder to the decoder and help to recover spatial details that may be lost during downsampling.
- The output layer of the network uses a hyperbolic tangent (tanh) activation function to scale the output noise model to the range of -1 to 1.
- By minimizing the difference between the noisy input spectrogram and the clean target spectrogram, the model learns to identify the noise components in the input and generate an output noise model that can be subtracted from the noisy input spectrogram to obtain a clean version of the signal.

![Training Process Simplified](https://user-images.githubusercontent.com/43881544/229711022-194e5409-6188-4622-a73d-966cb415d406.png)

### Predicting

---

During prediction, the noisy voice audio is divided (or sampled) into overlapping windows of slightly over 1 second each, and each window is converted into a magnitude spectrogram and a phase spectrogram using the Short-Time Fourier Transform (STFT).

These spectrograms are used for prediction. The magnitude spectrograms are then fed into the U-Net network, which predicts the noise model for each window / time serie.

The predicted noise model is then subtracted from the noisy voice spectrogram using a direct subtraction method. The output is a denoised, “cleaned” spectrogram is without the noise.

We then add this denoised spectrogram with the initial phase spectrogram to get the combined output which is then converted to cleaned output audio using Inverse Short Time Fourier Transform (ISTFT) 

![Prediction process shown diagrammatically](https://user-images.githubusercontent.com/43881544/229711077-ba353f27-f877-4cf0-b1f3-595203cb6fe8.png)


### Code Structure and Architecture

---

```jsx
├───datasets
│   ├───cv-corpus
│   │   ├───training_clips
│   │   └───training_clips2
│   └───UrbanSound8K
│       └───audio
│           ├───fold1
│           └───fold2
├───output
│   └───train
│       ├───sound
│       ├───spectrogram
│       └───time_serie
├───testing_batch
│   ├───input_samples
│   └───predicted_samples
└───weights
```

- Above is the code structure and architecture maintained throughout this project. Let’s understand it one by one.
    - `datasets`
        - As the name suggests, it consists of the data needed to train our model.
        - As we can see, we have 2 sorts of datasets. One for pure clean samples, other for pure noisy samples. We mix and match and blend as per our use case, involving different custom Data Augmentation Techniques.
        - If you notice, each of the 2 datasets has 2 separate folders — one containing more data, another containing less data samples. This way we can experiment the effect of data samples on the accuracy and generalization loss of our model. (It’ll be covered in detail in the Results and Outputs section of our report)
        - This folder is more useful for data-generation process.
    - `output`
        - This folder consists of the output during the data-handling step. It creates a random blend of noise samples, splits the time signal into 1sec numpy time series and finally a place to save the spectrogram which will be used for our training process
        - This folder is more relevant during data-generation and then training processes.
    - `testing_batch`
        - This contains the files or rather the audio samples that we wish to clean and accordingly it’s corresponding cleaned output
        - This folder is more relevant for prediction process.
    - `weights`
        - Used to save the weights and biases of our model in a `.h5` file to make sure all data and progress is saved. Next time, the accuracy will be improved by using the previous weights as a starting point unless a `--training_from_scratch = True` flag is passed during training process.

### Results and Outputs

- `Output 1: Less Training Data and Less Epochs (i.e. 5)`
    - `**Images**`
        
        ![image](https://user-images.githubusercontent.com/43881544/229711666-a30e4b8d-6201-4168-be07-e3fb89669677.png)

---

- `Output 2: More training data and More Epochs (i.e. 5 -> 10)`
    - `**Images**`
        
        ![image](https://user-images.githubusercontent.com/43881544/229711894-7efc36c1-c89e-47fa-97f1-b9e97d9d4f53.png)        

---

- `Output 3: More Training Data but same Epochs (i.e. 10)`
    - `**Images**`
        
        ![image](https://user-images.githubusercontent.com/43881544/229712012-fcfd33e8-9e49-4c80-b888-9ab6a31a2950.png)
        

### Future Scope

---

This project aims to enhance audio quality by reducing non-stationary sounds in noisy audio signals. They use the Mean Squared Error (MSE) technique to create a model that can smooth out noisy audio and provide a clean signal.

To improve the performance of the model, the authors suggest using Generative Adversarial Networks (GANs) to learn a more specific loss function for the task of source separation.

The GAN consists of a Generator that estimates a clean signal from noisy audio and a Discriminator that distinguishes between the generated clean signal and the actual clean signal.

The GAN is trained to map noisy audio to their respective clean signals, thus enhancing the audio quality.

### References

---

> Park, S.R. and Lee, J., 2016. A fully convolutional neural network for speech enhancement. *arXiv preprint arXiv:1609.07132*
> 
> 
> [[https://core.ac.uk/services/recommender](https://core.ac.uk/services/recommender)]
> 

> Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.Singing Voice Separation with Deep U-Net Convolutional Networks. ISMIR (2017).
> 
> 
> [[https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf)]
> 

> Grais, Emad M. and Plumbley, Mark D., Single Channel Audio Source Separation using Convolutional Denoising Autoencoders (2017).
> 
> 
> [[https://arxiv.org/abs/1703.08019](https://arxiv.org/abs/1703.08019)]
> 

> Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham
> 
> 
> [[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)]
> 

> K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.
> 
> 
> [DOI: [http://dx.doi.org/10.1145/2733373.2806390](http://dx.doi.org/10.1145/2733373.2806390)]
>
