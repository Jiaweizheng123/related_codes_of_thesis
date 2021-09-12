import pandas as pd
import numpy as np
import os


np.random.seed(999)


class ReverbDataset:

    def __init__(self, basepath, *, val_dataset_size):
        self.basepath = basepath
        self.val_dataset_size = val_dataset_size

    def _get_reverb_filenames(self, dataframe_name='train.tsv'):
        reverb_metadata = pd.read_csv(os.path.join(self.basepath, dataframe_name), sep='\t')
        reverb_files = reverb_metadata['path'].values
        np.random.shuffle(reverb_files)
        reverb_files = reverb_files[0:6000] #PRH reduce the number of files:
        print("Total number of training examples:", len(reverb_files))
        return reverb_files

    def get_train_val_filenames(self):
        reverb_files = self._get_reverb_filenames(dataframe_name='train.tsv')

        # resolve full path
        reverb_files = [os.path.join(self.basepath, 'clips',  filename) for filename in reverb_files]

        reverb_val_files = reverb_files[-self.val_dataset_size:]
        reverb_files = reverb_files[:-self.val_dataset_size]
        print("# of Training clean files:", len(reverb_files))
        print("# of  Validation clean files:", len(reverb_val_files))
        return reverb_files, reverb_val_files


    def get_test_filenames(self):
        reverb_files = self._get_reverb_filenames(dataframe_name='test.tsv')

        # resolve full path
        reverb_files = [os.path.join(self.basepath, 'clips',  filename) for filename in reverb_files]
        reverb_files = reverb_files[0:200] #PRH reduce the number of files
        print("# of Testing clean files:", len(reverb_files))
        return reverb_files
