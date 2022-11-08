# Adversarial Attack on Image Data
> Create different types of adversarial examples to fool deep neural networks into misclassifying data.
<p align='center'>
<!-- PROJECT SHIELDS -->
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white">
<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252">
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white">
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white">
</p>

<!--ABOUT-->
## About the Project

Adversarial Attack is the methodology used to trick deep neural networks to misclassify data by slightly perturbing the original data. These perturbations are so small that they aren't even visible to human eye.

<img src="https://github.com/ACM40960/project-21200461/blob/main/images/adv_example.png">

In this project I implement three types of attacks on MNIST dataset which are implemented based on the following papers:

1. FGSM - [Link to paper](https://arxiv.org/abs/1412.6572)
2. Deepfool Attack - [Link to paper](https://arxiv.org/abs/1511.04599)
3. L-BFGS Attack - [Link to paper](https://arxiv.org/abs/1312.6199)

I have also built two simple defence techniques against a particular type of adversarial attack:

1. Adversarial training
2. APE-GAN - [Link to paper](https://arxiv.org/abs/1707.05474)

<!-- Software and Library details-->
## Built with

* <img src="https://img.shields.io/badge/python-v3.8.8-blue">
* <img src="https://img.shields.io/badge/tensorflow-v2.9.1-orange">
* <img src="https://img.shields.io/badge/numpy-v1.22.3-red">
* <img src="https://img.shields.io/badge/pandas-v1.4.2-brightgreen">
* <img src="https://img.shields.io/badge/seaborn-v0.11.2-yellowgreen">
* <img src="https://img.shields.io/badge/matplotlib-v3.5.1-yellow">
* <img src="https://img.shields.io/badge/xgboost-v1.6.1-lightgrey">
* <img src="https://img.shields.io/badge/sklearn-v1.1.1-blue">
* <img src="https://img.shields.io/badge/keras-v2.9.0-red">
* <img src="https://img.shields.io/badge/tensorflow__probability-v0.17.0-orange">


<!-- Installation details -->
## Getting Started

Since I have used only a notebook format to implement the project there are two ways you can run this project:

<details>
<summary> Google Collab notebook </summary>
<p>
The Hassle free approach

All the necessary libraries are downloaded as part of the notebook and uses its own compute power to run the project.
<br><br>
<a href="https://colab.research.google.com/drive/1PQHojePoskwSC3xi3BxVBrAjnngjWR6J?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" class="center" alt="Open In Colab"/>
</a>
</p>
</details>

<details>
<summary> Clone the repo and run Jupyter notebook on your system </summary>
<p>

#### Pre-requisite software Installation

- [Jupyter](https://jupyter.org/install)
- [Python](https://www.python.org/downloads/release/python-380/)
- [pip](https://pip.pypa.io/en/stable/installation/)

The tensorflow version I use only makes use of CPU so the following two software installations
arent mandatory. You may get a warning if your system does not have a NVIDIA GPU, while running the code 
but it doesnt impact the models that are built.

- [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN](https://developer.nvidia.com/cudnn)

#### Running the Jupyter notebook on your system

1. Clone the repo onto your system
```bash
git clone https://github.com/ACM40960/project-21200461.git
```

2. The repo you download should have this file structure on your system

```
.
├── images
├── src
│   └── adversarial_attack.ipynb
├── LICENSE.md
├── README.md
└── requirements.txt
```
3. Launch command prompt from the directory where the repo is installed and run the following command to install all the required libraries

```bash
pip install -r requirements.txt
```

4. Launch the **adversarial_attack.ipynb in the src folder** either by double clicking on it(if Jupyter is the default software for opening ipynb on your system) or run the following command from the **src folder** to launch jupyter.

``` bash
jupyter notebook
```

</p>
</details>

<!-- USAGE -->
## Usage

1. Ways to run the project on Colab/ Jupyter is illustrated below. Please note using Run All will take awhile for the entire notebook to finish running.
(Code section to build CNN ~ 10-15 minutes(depends on system), adversarial training on 5000 images ~ 10-15 minutes, APE-GAN training  60-120 minutes for 10 epochs)
<p float="left" >
<img src="https://github.com/ACM40960/project-21200461/blob/main/images/run.jpg"  width=300  title="Jupyter- Run options"/>
<img src="https://github.com/ACM40960/project-21200461/blob/main/images/colab_run.jpg"  width=300 title="Colab-Run options"/>
</p>

2. Sections of code that fits the images for predictions:

```python
cnn_model_fit = cnn_model.fit(x_train, 
                        y_train,
                        validation_data = (x_val, y_val),
                        batch_size=128,
                        epochs=10)
```
3. Various attacks and an example of results:
```python
perturbed_image = FGSM(cnn_model, image, label, eps=0.1)
```
<img src="https://github.com/ACM40960/project-21200461/blob/main/images/fgsm_example.png" title="FGSM">

```python
label_pert, pert_image = DeepFool(image, cnn_model)
```
<img src="https://github.com/ACM40960/project-21200461/blob/main/images/deepfool_example.png" title="DeepFool">

```python
 perturbed_image = LBFGS(cnn_model,image,actual_label)
```
<img src="https://github.com/ACM40960/project-21200461/blob/main/images/lbfgs_example.png" title="LBFGS">

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<!-- CONTACT -->
## Contact
[@vinuthags](https://github.com/vinuthags)
