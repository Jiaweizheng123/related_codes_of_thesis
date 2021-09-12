import tensorflow as tf
import os
from keras_radam import RAdam
import soundfile as sf
import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import models
from default_config import args
from data_processing.feature_extractor import FeatureExtractor
from tensorflow.python.client import device_lib
from utils import get_tf_feature, read_audio, write_audio, revert_features_to_audio, prepare_input_features, add_reverb_to_clean_audio, tf_record_parser

os.environ['TF_KERAS'] = '1'
device_lib.list_local_devices()
#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

tf.random.set_seed(999)
np.random.seed(999)

def l2_norm(vector):
    return np.square(vector)

def SDR(denoised, cleaned, eps=1e-7): # Signal to Distortion Ratio
    a = l2_norm(denoised)
    b = l2_norm(denoised - cleaned)
    a_b = a / b
    return np.mean(10 * np.log10(a_b + eps))




if __name__ == '__main__':

   home_dir = args.home_dir
   mozilla_basepath = args.mozilla_basepath
   reverb_basepath = args.reverb_basepath

   windowLength = args.windowLength
   overlap      = args.overlap
   ffTLength    = args.ffTLength
   inputFs      = args.inputFs
   fs           = args.fs
   numFeatures  = ffTLength//2 + 1
   numSegments  = 8

   model = models.build_model(l2_strength=0.0)
   model.summary()

   model.load_weights(os.path.join(mozilla_basepath, 'denoiser_cnn_log_mel_generator.h5'))

   cleanAudio, sr = read_audio(os.path.join(mozilla_basepath, 'clips', 'common_voice_en_16526.mp3'), sample_rate=fs)
   print("Min:", np.min(cleanAudio),"Max:",np.max(cleanAudio))
 
   reverbAudio, sr = read_audio(os.path.join(reverb_basepath, 'clips', 'common_voice_en_16526.mp3'), sample_rate=fs)
   print("Min:", np.min(reverbAudio),"Max:",np.max(reverbAudio))

   cleanAudioFeatureExtractor = FeatureExtractor(cleanAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   stft_features = cleanAudioFeatureExtractor.get_stft_spectrogram()
   stft_features = np.abs(stft_features)
   print("Min:", np.min(stft_features),"Max:",np.max(stft_features))

   reverbAudio = add_reverb_to_clean_audio(cleanAudio, reverbAudio)
   reverbAudioFeatureExtractor = FeatureExtractor(reverbAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
   reverb_stft_features = reverbAudioFeatureExtractor.get_stft_spectrogram()

   def revert_features_to_audio2(features, phase, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
       if cleanMean and cleanStd:
          features = cleanStd * features + cleanMean

       phase = np.transpose(phase, (1, 0))
       features = np.squeeze(features)
       features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

       features = np.transpose(features, (1, 0))
       return reverbAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)
       #return reverbAudioFeatureExtractor.get_audio_from_stft_spectrogram_GL(np.abs(features))

   reverbPhase = np.angle(reverb_stft_features)
   print(reverbPhase.shape)
   reverb_stft_features = np.abs(reverb_stft_features)

   mean = np.mean(reverb_stft_features)
   std = np.std(reverb_stft_features)
   reverb_stft_features = (reverb_stft_features - mean) / std


   predictors = prepare_input_features(reverb_stft_features, numSegments, numFeatures)

   predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
   predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
   print('predictors.shape:', predictors.shape)

   STFTFullyConvolutional = model.predict(predictors)
   print(STFTFullyConvolutional.shape)

   dereverbAudioFullyConvolutional = revert_features_to_audio2(STFTFullyConvolutional, reverbPhase,  mean, std)
   print("Min:", np.min(dereverbAudioFullyConvolutional),"Max:",np.max(dereverbAudioFullyConvolutional))
 #  ipd.Audio(data=dereverbAudioFullyConvolutional, rate=fs) # load a local WAV file


   f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

   ax1.plot(cleanAudio)
   ax1.set_title("Clean Audio")
   write_audio('./output/clean.wav', fs, cleanAudio)

   ax2.plot(reverbAudio)
   ax2.set_title("Reverb Audio")
   write_audio('./output/noisy.wav', fs, reverbAudio)

   ax3.plot(dereverbAudioFullyConvolutional)
   ax3.set_title("Dereverb Audio")
   write_audio('./output/denoised.wav', fs, dereverbAudioFullyConvolutional)
   plt.show()


