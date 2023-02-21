# A Fast and Stable GAN for  EEG data - pytorch
this project proposes a generative adversarial network model (FastGAN-NAM-SP) data generation method based on channel attention normalization and spatial pyramids for EEG data augmentation, where the EEG data are pre-processed EEG topographies.The classification performance of the EEG signal classifier is improved by augmenting the model-generated samples by adding them to the original dataset.The experiments were conducted using the BCI Competition IV-1 standard dataset. The quality of the generated data was assessed quantitatively using the FID ( Frecchet Inception Distance) method, and the usability of the generated data was assessed qualitatively using the Resnet-18 classification network. The improvement of generated sample quality and classification accuracy were verified, respectively.
## 0. Data
The datasets used in the paper can be found at [link](http://www.bbci.de/competition/iv/#dataset1)
classification of continuous EEG without trial structure (data sets 1).
Data sets 1: ‹motor imagery, uncued classifier application› (description)
provided by the Berlin BCI group: Technische Universität Berlin (Machine Learning Laboratory) and Fraunhofer FIRST (Intelligent Data Analysis Group) (Klaus-Robert Müller, Benjamin Blankertz, Carmen Vidaurre, Guido Nolte), and Campus Benjamin Franklin of the Charité - University Medicine Berlin, Department of Neurology, Neurophysics Group (Gabriel Curio)
EEG, motor imagery (2 classes of left hand, right hand, foot); evaluation data is continuous EEG which contains also periods of idle state
[64 EEG channels (0.05-200Hz), 1000Hz sampling rate, 2 classes (+ idle state), 7 subjects]
## 1. EEG transfor
The code is structured as follows:
* mat_poto01.py: The EEG signal was converted from .mat format to .png format.
  
* cut_file.py: Cut the EEG topographic map to size.
  
* images01.py: The clipped images are selected randomly as the input of the neural network.
## 2. pytorch-FastGAN-NAM
On the basis of FastGAN, NAM is added for optimization.
* models.py: all the models' structure definition.
  
* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model.

*  eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.
## 3. pytorch-FastGAN-SP
On the basis of FastGAN, SP is added for optimization.
* models.py: all the models' structure definition.
  
* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model.

*  eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.
## 4. pytorch-FastGAN-NAM-SP
On the basis of FastGAN, NAM and SP are added for optimization.
* models.py: all the models' structure definition.
  
* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model.

*  eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.
## 5. fid score
FID is a measure of similarity between two datasets of images. 
It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.
## 6. pytorch-resnet-ffhq-master
pytorch was used to train resnet18 model, and the generated EEG topographic map was classified.

