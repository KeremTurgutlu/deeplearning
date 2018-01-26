**Tensorflow Speech Recognition**

Main link: https://www.tensorflow.org/versions/master/tutorials/audio_recognition

This is an ended Kaggle competition, for more information you may visit https://www.kaggle.com/c/tensorflow-speech-recognition-challenge. In this repo I've provided an end-to-end speech recognition pipeline for those who are familiar with CNNs. I've also used references from speech analysis blogs, kaggle winner solutions and tensorflow website. There are two ways of getting the full out of this repo:

- You may start going over the self-explanatory jupyter notebooks to better understand the end-to-end process.

- Then you may checkout the scripts folder which has all the necessary scripts for the processing and modeling that is already mentioned in jupyter notebooks. 

First you should prepare data for modeling:

`python prepare_data.py`

Later you may play around with train.py to set different hyperparameters, subsamples, etc...

At the final step running the following code will build the model for us.

`python main.py`

**PS.** All code is written in Pyton 3 and Pytorch is used as DL library. Notebooks may not be displayed in github so it's better to clone or fork if you wish.
