
class args(object):
   home_dir            = '/mnt/storage/scratch/csprh/code/tmp/AudioDenoiser'
   mozilla_basepath    = '/mnt/storage/scratch/csprh/AUDIO/DATASETS/'
   reverb_basepath = '/mnt/storage/scratch/csprh/AUDIO/DATASETS/reverb'
   windowLength        = 256
   ffTLength           = windowLength
   inputFs             = 48e3
   fs                  = 16e3
   overlap             = round(0.25 * windowLength) # overlap of 75%
   numSegments         = 8
