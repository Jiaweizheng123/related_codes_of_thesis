#Author：Jiawei Zheng

#Email：hu20327@bristol.ac.uk

# Acknowledgement

My sincere thanks to my supervisor, Dr. Paul Hill, for his codes which has been uploaded in the GitHub(https://github.com/csprh/AudioDenoiser.git).
That codes ha been modified to processing reverb

Tensorflow 2.0 implementation of the paper [A Fully Convolutional Neural Network for Speech Enhancement](https://pdfs.semanticscholar.org/9ed8/e2f6c338f4e0d1ab0d8e6ab8b836ea66ae95.pdf)

This code is modified from[this code] (https://github.com/csprh/AudioDenoiser.git)
Which in turn is based on this blog post: [Practical Deep Learning Audio Denoising](https://medium.com/better-programming/practical-deep-learning-audio-denoising-79c1c1aea299)

## Dataset

The clean audio set is fromMozilla Common Voice and the reverb set is to use Adoble Audition to add in the training set.

- [Mozilla Common Voice](https://voice.mozilla.org/)
- [Adoble Audition](https://www.adobe.com/cn/products/audition.html)
Use ```create_dataset.sh``` script to create the TFRecord files. 

Update the paths of these datasets in ```default_config.py```.

## Training

Use train.sh to train an audio denoiser

## Testing

Use test.sh to test the audio denoiser with one track

## Requirements

tensorflow and keras and GPU

