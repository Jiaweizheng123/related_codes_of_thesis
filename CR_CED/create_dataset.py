from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from data_processing.urban_reverb import Reverb
from data_processing.dataset import Dataset
from default_config import args
import warnings

warnings.filterwarnings(action='ignore')
## read the clean and reverb audio
mcv = MozillaCommonVoiceDataset(args.mozilla_basepath, val_dataset_size=100)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

reverb_audio = Reverb(args.reverb_basepath, val_dataset_size=20)
reverb_train_filenames, reverb_val_filenames = reverb_audio.get_train_val_filenames()


config = {'windowLength': args.windowLength,
          'overlap': round(0.25 * args.windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, reverb_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=200)

train_dataset = Dataset(clean_train_filenames, reverb_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=400)

## Generate the dataset
clean_test_filenames = mcv.get_test_filenames()

reverb_filenames = reverb_audio.get_test_filenames()
reverb_filenames = reverb_filenames

test_dataset = Dataset(clean_test_filenames, reverb_filenames, **config)
#test_dataset.create_tf_record(prefix='test', subset_size=1, parallel=False)
test_dataset.create_tf_record(prefix='test', subset_size=100, parallel=False)

