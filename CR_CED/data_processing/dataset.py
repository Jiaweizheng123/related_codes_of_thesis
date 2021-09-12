import librosa
import numpy as np
import math
from data_processing.feature_extractor import FeatureExtractor
from utils import prepare_input_features
import multiprocessing
import os
from utils import get_tf_feature, read_audio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


np.random.seed(999)
tf.random.set_seed(999)


class Dataset:
    def __init__(self, clean_filenames, reverb_filenames, **config):
        self.clean_filenames = clean_filenames
        self.reverb_filenames = reverb_filenames
        self.sample_rate = config['fs']
        self.overlap = config['overlap']
        self.window_length = config['windowLength']
        self.audio_max_duration = config['audio_max_duration']

    def _sample_reverb_filename(self):
        return np.random.choice(self.reverb_filenames)

    def _remove_silent_frames(self, audio):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, reverb_phase):
        assert clean_phase.shape == reverb_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - reverb_phase)

    def get_reverb_audio(self, *, filename):
        return read_audio(filename, self.sample_rate)

    def _audio_random_crop(self, audio, duration):
        audio_duration_secs = librosa.core.get_duration(audio, self.sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            # print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return audio

        audio_duration_ms = math.floor(audio_duration_secs * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx: idx + duration_ms]

    def _add_reverb_to_clean_audio(self, clean_audio, reverb_signal):
        if len(clean_audio) >= len(reverb_signal):
            # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
            while len(clean_audio) >= len(reverb_signal):
                reverb_signal = np.append(reverb_signal, reverb_signal)

        ## Extract a noise segment from a random location in the noise file
        ind = np.random.randint(0, reverb_signal.size - clean_audio.size)

        reverbSegment = reverb_signal[ind: ind + clean_audio.size]

        speech_power = np.sum(clean_audio ** 2)
        reverb_power = np.sum(reverbSegment ** 2)
        reverbAudio = clean_audio + np.sqrt(speech_power / reverb_power) * reverbSegment
        return reverbAudio

    def parallel_audio_processing(self, clean_filename):

        clean_audio, _ = read_audio(clean_filename, self.sample_rate)

        # remove silent frame from clean audio
        clean_audio = self._remove_silent_frames(clean_audio)

        reverb_filename = self._sample_reverb_filename()

        # read the noise filename
        reverb_audio, sr = read_audio(reverb_filename, self.sample_rate)

        # remove silent frame from noise audio
        reverb_audio = self._remove_silent_frames(reverb_audio)

        # sample random fixed-sized snippets of audio
        clean_audio = self._audio_random_crop(clean_audio, duration=self.audio_max_duration)

        # add noise to input image
        reverbInput = self._add_reverb_to_clean_audio(clean_audio, reverb_audio)

        # extract stft features from noisy audio
        reverb_input_fe = FeatureExtractor(reverbInput, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        reverb_spectrogram = reverb_input_fe.get_stft_spectrogram()

        # Or get the phase angle (in radians)
        # noisy_stft_magnitude, noisy_stft_phase = librosa.magphase(noisy_stft_features)
        reverb_phase = np.angle(reverb_spectrogram)

        # get the magnitude of the spectral
        reverb_magnitude = np.abs(reverb_spectrogram)

        # extract stft features from clean audio
        clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        clean_spectrogram = clean_audio_fe.get_stft_spectrogram()
        # clean_spectrogram = cleanAudioFE.get_mel_spectrogram()

        # get the clean phase
        clean_phase = np.angle(clean_spectrogram)

        # get the clean spectral magnitude
        clean_magnitude = np.abs(clean_spectrogram)
        # clean_magnitude = 2 * clean_magnitude / np.sum(scipy.signal.hamming(self.window_length, sym=False))

        clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, reverb_phase)

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        reverb_magnitude = scaler.fit_transform(reverb_magnitude)
        clean_magnitude = scaler.transform(clean_magnitude)

        return reverb_magnitude, clean_magnitude, reverb_phase

    def create_tf_record(self, *, prefix, subset_size, parallel=True):
        counter = 0
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        for i in range(0, len(self.clean_filenames), subset_size):

            tfrecord_filename = './records/' + prefix + '_' + str(counter) + '.tfrecords'

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                counter += 1
                continue
            print(os.getcwd())
            writer = tf.io.TFRecordWriter(tfrecord_filename)
            clean_filenames_sublist = self.clean_filenames[i:i + subset_size]

            print(f"Processing files from: {i} to {i + subset_size}")
            if parallel:
                out = p.map(self.parallel_audio_processing, clean_filenames_sublist)
            else:
                out = [self.parallel_audio_processing(filename) for filename in clean_filenames_sublist]

            for o in out:
                reverb_stft_magnitude = o[0]
                clean_stft_magnitude = o[1]
                reverb_stft_phase = o[2]

                reverb_stft_mag_features = prepare_input_features(reverb_stft_magnitude, numSegments=8, numFeatures=129)

                reverb_stft_mag_features = np.transpose(reverb_stft_mag_features, (2, 0, 1))
                clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                reverb_stft_phase = np.transpose(reverb_stft_phase, (1, 0))

                reverb_stft_mag_features = np.expand_dims(reverb_stft_mag_features, axis=3)
                clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=2)

                for x_, y_, p_ in zip(reverb_stft_mag_features, clean_stft_magnitude, reverb_stft_phase):
                    y_ = np.expand_dims(y_, 2)
                    example = get_tf_feature(x_, y_, p_)
                    writer.write(example.SerializeToString())

            counter += 1
            writer.close()
