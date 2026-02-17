<div align="center">
    <a href="https://krzjoa.github.io/awesome-python-data-science/"><img width="250" height="250" src="img/py-datascience.png" alt="pyds"></a>
    <br>
    <br>
    <br>
</div>

<h1 align="center">
    Awesome Python Data Science
</h1>
<div align="center"><a href="https://github.com/sindresorhus/awesome">
<img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Awesome" border="0">
</a>
</div>
</br>

> Probably the best curated list of data science software in Python

## Contents

* [Contents](#contents)
* [Machine Learning](#machine-learning)
  * [General Purpose Machine Learning](#general-purpose-machine-learning)
  * [Gradient Boosting](#gradient-boosting)
  * [Ensemble Methods](#ensemble-methods)
  * [Imbalanced Datasets](#imbalanced-datasets)
  * [Random Forests](#random-forests)
  * [Kernel Methods](#kernel-methods)
* [Deep Learning](#deep-learning)
  * [PyTorch](#pytorch)
  * [TensorFlow](#tensorflow)
  * [JAX](#jax)
  * [Others](#others)
* [Automated Machine Learning](#automated-machine-learning)
* [Natural Language Processing](#natural-language-processing)
* [Computer Audition](#computer-audition)
* [Computer Vision](#computer-vision)
* [Time Series](#time-series)
* [Reinforcement Learning](#reinforcement-learning)
* [Graph Machine Learning](#graph-machine-learning)
* [Graph Manipulation](#graph-manipulation)
* [Learning-to-Rank & Recommender Systems](#learning-to-rank-&-recommender-systems)
* [Probabilistic Graphical Models](#probabilistic-graphical-models)
* [Probabilistic Methods](#probabilistic-methods)
* [Model Explanation](#model-explanation)
* [Optimization](#optimization)
* [Genetic Programming](#genetic-programming)
* [Feature Engineering](#feature-engineering)
  * [General](#general)
  * [Feature Selection](#feature-selection)
* [Visualization](#visualization)
  * [General Purposes](#general-purposes)
  * [Interactive plots](#interactive-plots)
  * [Map](#map)
  * [Automatic Plotting](#automatic-plotting)
  * [NLP](#nlp)
* [Data Manipulation](#data-manipulation)
  * [Data Frames](#data-frames)
  * [Pipelines](#pipelines)
  * [Data-centric AI](#data-centric-ai)
  * [Synthetic Data](#synthetic-data)
* [Deployment](#deployment)
* [Statistics](#statistics)
* [Distributed Computing](#distributed-computing)
* [Experimentation](#experimentation)
* [Data Validation](#data-validation)
* [Evaluation](#evaluation)
* [Computations](#computations)
* [Web Scraping](#web-scraping)
* [Spatial Analysis](#spatial-analysis)
* [Quantum Computing](#quantum-computing)
* [Conversion](#conversion)
* [Contributing](#contributing)
* [License](#license)

## Machine Learning

### General Purpose Machine Learning

* [dlib](https://github.com/davisking/dlib) ⭐ 14,343 | 🐛 44 | 🌐 C++ | 📅 2026-02-15 - Toolkit for making real-world machine learning and data analysis applications in C++ (Python bindings).
* [PyCaret](https://github.com/pycaret/pycaret) ⭐ 9,698 | 🐛 423 | 🌐 Jupyter Notebook | 📅 2025-04-21 - An open-source, low-code machine learning library in Python.  <img height="20" src="img/R_big.png" alt="R inspired lib">
* [causalml](https://github.com/uber/causalml) ⭐ 5,734 | 🐛 50 | 🌐 Python | 📅 2026-02-15 - Uplift modeling and causal inference with machine learning algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [mlpack](https://github.com/mlpack/mlpack) ⭐ 5,599 | 🐛 24 | 🌐 C++ | 📅 2026-02-16 - A scalable C++ machine learning library (Python bindings).
* [cuML](https://github.com/rapidsai/cuml) ⭐ 5,118 | 🐛 893 | 🌐 C++ | 📅 2026-02-16 - RAPIDS Machine Learning Library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [MLxtend](https://github.com/rasbt/mlxtend) ⭐ 5,112 | 🐛 156 | 🌐 Python | 📅 2026-01-24 - Extension and helper modules for Python's data analysis and machine learning libraries. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [xLearn](https://github.com/aksnzhy/xlearn) ⭐ 3,097 | 🐛 194 | 🌐 C++ | 📅 2023-08-28 - High Performance, Easy-to-use, and Scalable Machine Learning Package.
* [Shogun](https://github.com/shogun-toolbox/shogun) ⭐ 3,068 | 🐛 423 | 🌐 C++ | 📅 2023-12-19 - Machine learning toolbox.
* [hyperlearn](https://github.com/danielhanchen/hyperlearn) ⭐ 2,397 | 🐛 2 | 🌐 Jupyter Notebook | 📅 2024-11-19 - 50%+ Faster, 50%+ less RAM usage, GPU support re-written Sklearn, Statsmodels. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [modAL](https://github.com/cosmic-cortex/modAL) ⭐ 2,339 | 🐛 107 | 🌐 Python | 📅 2024-02-26 - Modular active learning framework for Python3. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [metric-learn](https://github.com/all-umass/metric-learn) ⭐ 1,429 | 🐛 52 | 🌐 Python | 📅 2024-08-03 - Metric learning algorithms in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Sparkit-learn](https://github.com/lensacom/sparkit-learn) ⭐ 1,153 | 🐛 35 | 🌐 Python | 📅 2020-12-31 - PySpark + scikit-learn = Sparkit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) ⭐ 954 | 🐛 91 | 🌐 Python | 📅 2024-02-01 - Multi-label classification for python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyGAM](https://github.com/dswah/pyGAM) ⭐ 950 | 🐛 104 | 🌐 Python | 📅 2026-01-22 - Generalized Additive Models in Python.
* [seqlearn](https://github.com/larsmans/seqlearn) ⭐ 703 | 🐛 33 | 🌐 Python | 📅 2023-03-24 - Sequence classification toolkit for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Reproducible Experiment Platform (REP)](https://github.com/yandex/rep) ⭐ 700 | 🐛 31 | 🌐 Jupyter Notebook | 📅 2024-07-31 - Machine Learning toolbox for Humans. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pystruct](https://github.com/pystruct/pystruct) ⭐ 670 | 🐛 108 | 🌐 Python | 📅 2021-09-23 - Simple structured learning framework for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) ⭐ 489 | 🐛 5 | 🌐 Python | 📅 2017-08-11 - Highly interpretable classifiers for scikit learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [RuleFit](https://github.com/christophM/rulefit) ⭐ 440 | 🐛 28 | 🌐 Python | 📅 2023-10-08 - Implementation of the rulefit. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [SciPy](https://scipy.org/) - Fundamental algorithms for scientific computing in Python
* [scikit-learn](http://scikit-learn.org/stable/) - Machine learning in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Gradient Boosting

* [XGBoost](https://github.com/dmlc/xgboost) ⭐ 28,011 | 🐛 475 | 🌐 C++ | 📅 2026-02-14 - Scalable, Portable, and Distributed Gradient Boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [LightGBM](https://github.com/Microsoft/LightGBM) ⭐ 18,083 | 🐛 473 | 🌐 C++ | 📅 2026-02-13 - A fast, distributed, high-performance gradient boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [CatBoost](https://github.com/catboost/catboost) ⭐ 8,799 | 🐛 678 | 🌐 C++ | 📅 2026-02-17 - An open-source gradient boosting on decision trees library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [NGBoost](https://github.com/stanfordmlgroup/ngboost) ⭐ 1,829 | 🐛 54 | 🌐 Python | 📅 2025-11-21 - Natural Gradient Boosting for Probabilistic Prediction.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) ⭐ 710 | 🐛 39 | 🌐 C++ | 📅 2025-03-19 - Fast GBDTs and Random Forests on GPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests) ⭐ 694 | 🐛 49 | 🌐 Python | 📅 2026-02-05 - A collection of state-of-the-art algorithms for the training, serving and interpretation of Decision Forest models in Keras. <img height="20" src="img/keras_big.png" alt="keras"> <img height="20" src="img/tf_big2.png" alt="TensorFlow">

### Ensemble Methods

* [vecstack](https://github.com/vecxoz/vecstack) ⭐ 701 | 🐛 0 | 🌐 Python | 📅 2025-11-01 - Python package for stacking (machine learning technique). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Stacking](https://github.com/ikki407/stacking) ⭐ 230 | 🐛 20 | 🌐 Python | 📅 2017-12-21 - Simple and useful stacking library written in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [stacked\_generalization](https://github.com/fukatani/stacked_generalization) ⭐ 119 | 🐛 3 | 🌐 Python | 📅 2019-05-02 - Library for machine learning stacking generalization. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ML-Ensemble](http://ml-ensemble.com/) - High performance ensemble learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Imbalanced Datasets

* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) ⭐ 7,082 | 🐛 61 | 🌐 Python | 📅 2026-02-02 - Module to perform under-sampling and over-sampling with various techniques. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) ⭐ 241 | 🐛 1 | 🌐 Python | 📅 2022-01-29 - Python-based implementations of algorithms for learning on imbalanced data. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">

### Random Forests

* [rgf\_python](https://github.com/fukatani/rgf_python) ⭐ 383 | 🐛 9 | 🌐 C++ | 📅 2022-01-08 - Python Wrapper of Regularized Greedy Forest. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [rpforest](https://github.com/lyst/rpforest) ⭐ 225 | 🐛 0 | 🌐 Python | 📅 2020-02-08 - A forest of random projection trees. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-random-bits-forest](https://github.com/tmadl/sklearn-random-bits-forest) ⭐ 9 | 🐛 2 | 🌐 Python | 📅 2016-07-31 - Wrapper of the Random Bits Forest program written by (Wang et al., 2016).<img height="20" src="img/sklearn_big.png" alt="sklearn">

### Kernel Methods

* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) ⭐ 1,619 | 🐛 87 | 🌐 C++ | 📅 2024-04-01 - A fast SVM Library on GPUs and CPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [fastFM](https://github.com/ibayer/fastFM) ⭐ 1,090 | 🐛 51 | 🌐 Python | 📅 2022-07-17 - A library for Factorization Machines. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyFM](https://github.com/coreylynch/pyFM) ⭐ 926 | 🐛 44 | 🌐 Python | 📅 2020-10-01 - Factorization machines in python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tffm](https://github.com/geffy/tffm) ⭐ 781 | 🐛 19 | 🌐 Jupyter Notebook | 📅 2022-01-17 - TensorFlow implementation of an arbitrary order Factorization Machine. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">
* [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) ⭐ 237 | 🐛 13 | 🌐 Python | 📅 2025-08-14 - Relevance Vector Machine implementation using the scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [liquidSVM](https://github.com/liquidSVM/liquidSVM) ⭐ 71 | 🐛 17 | 🌐 C++ | 📅 2020-02-20 - An implementation of SVMs.

## Deep Learning

### PyTorch

* [PyTorch](https://github.com/pytorch/pytorch) ⭐ 97,448 | 🐛 18,015 | 🌐 Python | 📅 2026-02-17 - Tensors and Dynamic neural networks in Python with strong GPU acceleration. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pytorch-lightning](https://github.com/Lightning-AI/lightning) ⭐ 30,839 | 🐛 953 | 🌐 Python | 📅 2026-02-16 - PyTorch Lightning is just organized PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [skorch](https://github.com/dnouri/skorch) ⭐ 6,150 | 🐛 70 | 🌐 Jupyter Notebook | 📅 2026-02-16 - A scikit-learn compatible neural network library that wraps PyTorch. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ignite](https://github.com/pytorch/ignite) ⭐ 4,735 | 🐛 159 | 🌐 Python | 📅 2026-02-16 - High-level library to help with training neural networks in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Catalyst](https://github.com/catalyst-team/catalyst) ⭐ 3,371 | 🐛 3 | 🌐 Python | 📅 2025-06-27 - High-level utils for PyTorch DL & RL research. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ChemicalX](https://github.com/AstraZeneca/chemicalx) ⭐ 770 | 🐛 10 | 🌐 Python | 📅 2023-09-11 - A PyTorch-based deep learning library for drug pair scoring. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">

### TensorFlow

* [TensorFlow](https://github.com/tensorflow/tensorflow) ⭐ 193,745 | 🐛 3,606 | 🌐 C++ | 📅 2026-02-17 - Computation using data flow graphs for scalable machine learning by Google. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Ludwig](https://github.com/uber/ludwig) ⭐ 11,645 | 🐛 56 | 🌐 Python | 📅 2026-01-19 - A toolbox that allows one to train and test deep learning models without the need to write code. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Sonnet](https://github.com/deepmind/sonnet) ⭐ 9,905 | 🐛 39 | 🌐 Python | 📅 2026-02-10 - TensorFlow-based neural network library. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TFLearn](https://github.com/tflearn/tflearn) ⭐ 9,608 | 🐛 579 | 🌐 Python | 📅 2024-05-06 - Deep learning library featuring a higher-level API for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) ⭐ 7,389 | 🐛 36 | 🌐 Python | 📅 2023-02-18 - Deep Learning and Reinforcement Learning Library for Researcher and Engineer. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorpack](https://github.com/ppwwyyxx/tensorpack) ⭐ 6,295 | 🐛 14 | 🌐 Python | 📅 2023-08-06 - A Neural Net Training Interface on TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Polyaxon](https://github.com/polyaxon/polyaxon) ⭐ 3,697 | 🐛 123 | 📅 2026-02-16 - A platform that helps you build, manage and monitor deep learning models. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Hyperas](https://github.com/maxpumperla/hyperas) ⭐ 2,179 | 🐛 97 | 🌐 Python | 📅 2023-01-05 - Keras + Hyperopt: A straightforward wrapper for a convenient hyperparameter. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [TensorFlow Fold](https://github.com/tensorflow/fold) ⭐ 1,823 | 🐛 58 | 🌐 Python | 📅 2021-06-26 - Deep learning with dynamic computation graphs in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Mesh TensorFlow](https://github.com/tensorflow/mesh) ⚠️ Archived - Model Parallelism Made Easier. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [keras-contrib](https://github.com/keras-team/keras-contrib) ⚠️ Archived - Keras community contributions. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Elephas](https://github.com/maxpumperla/elephas) ⭐ 1,578 | 🐛 9 | 🌐 Python | 📅 2023-05-01 - Distributed Deep learning with Keras & Spark. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [tensorflow-upstream](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream) ⭐ 699 | 🐛 91 | 🌐 C++ | 📅 2026-02-12 - TensorFlow ROCm port. <img height="20" src="img/tf_big2.png" alt="sklearn"> <img height="20" src="img/amd_big.png" alt="Possible to run on AMD GPU">
* [qkeras](https://github.com/google/qkeras) ⭐ 577 | 🐛 49 | 🌐 Python | 📅 2025-06-13 - A quantization deep learning library. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [tfdeploy](https://github.com/riga/tfdeploy) ⭐ 355 | 🐛 12 | 🌐 Python | 📅 2025-01-04 - Deploy TensorFlow graphs for fast evaluation and export to TensorFlow-less environments running numpy. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLight](https://github.com/bsautermeister/tensorlight) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2022-10-06 - A high-level framework for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Keras](https://keras.io) - A high-level neural networks API running on top of TensorFlow.  <img height="20" src="img/keras_big.png" alt="Keras compatible">

### JAX

* [JAX](https://github.com/google/jax) ⭐ 34,881 | 🐛 2,469 | 🌐 Python | 📅 2026-02-17 - Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
* [FLAX](https://github.com/google/flax) ⭐ 7,078 | 🐛 464 | 🌐 Jupyter Notebook | 📅 2026-02-14 - A neural network library for JAX that is designed for flexibility.
* [Optax](https://github.com/google-deepmind/optax) ⭐ 2,182 | 🐛 65 | 🌐 Python | 📅 2026-02-09 - A gradient processing and optimization library for JAX.

### Others

* [transformers](https://github.com/huggingface/transformers) ⭐ 156,555 | 🐛 2,265 | 🌐 Python | 📅 2026-02-16 - State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Caffe](https://github.com/BVLC/caffe) ⭐ 34,834 | 🐛 1,175 | 🌐 C++ | 📅 2024-07-31 - A fast open framework for deep learning.
* [autograd](https://github.com/HIPS/autograd) ⭐ 7,456 | 🐛 190 | 🌐 Python | 📅 2026-02-16 - Efficiently computes derivatives of numpy code.
* [nnabla](https://github.com/sony/nnabla) ⭐ 2,774 | 🐛 35 | 🌐 Python | 📅 2025-08-29 - Neural Network Libraries by Sony.
* [Tangent](https://github.com/google/tangent) ⚠️ Archived - Source-to-Source Debuggable Derivatives in Pure Python.

## Automated Machine Learning

* [TPOT](https://github.com/rhiever/tpot) ⭐ 10,047 | 🐛 304 | 🌐 Jupyter Notebook | 📅 2025-09-11 - AutoML tool that optimizes machine learning pipelines using genetic programming. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AutoGluon](https://github.com/awslabs/autogluon) ⭐ 9,964 | 🐛 384 | 🌐 Python | 📅 2026-02-16 - AutoML for Image, Text, Tabular, Time-Series, and MultiModal Data.
* [AutoKeras](https://github.com/keras-team/autokeras) ⭐ 9,308 | 🐛 157 | 🌐 Python | 📅 2025-11-25 - AutoML library for deep learning. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [auto-sklearn](https://github.com/automl/auto-sklearn) ⭐ 8,053 | 🐛 207 | 🌐 Python | 📅 2026-01-20 - An AutoML toolkit and a drop-in replacement for a scikit-learn estimator. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) ⭐ 2,524 | 🐛 75 | 🌐 Python | 📅 2024-04-09 - Automatic architecture search and hyperparameter optimization for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [MLBox](https://github.com/AxeldeRomblay/MLBox) ⭐ 1,526 | 🐛 28 | 🌐 Python | 📅 2023-08-06 - A powerful Automated Machine Learning python library.

## Natural Language Processing

* [NLTK](https://github.com/nltk/nltk) ⭐ 14,505 | 🐛 290 | 🌐 Python | 📅 2026-01-10 -  Modules, data sets, and tutorials supporting research and development in Natural Language Processing.
* [flair](https://github.com/zalandoresearch/flair) ⭐ 14,356 | 🐛 28 | 🌐 Python | 📅 2025-10-27 - Very simple framework for state-of-the-art NLP.
* [torchtext](https://github.com/pytorch/text) ⚠️ Archived - Data loaders and abstractions for text and NLP. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Phonemizer](https://github.com/bootphon/phonemizer) ⭐ 1,512 | 🐛 46 | 🌐 Python | 📅 2024-09-26 - Simple text-to-phonemes converter for multiple languages.
* [KerasNLP](https://github.com/keras-team/keras-nlp) ⭐ 961 | 🐛 261 | 🌐 Python | 📅 2026-02-14 - Modular Natural Language Processing workflows with Keras. <img height="20" src="img/keras_big.png" alt="Keras based/compatible">
* [CLTK](https://github.com/cltk/cltk) ⭐ 893 | 🐛 5 | 🌐 Python | 📅 2026-02-12 - The Classical Language Toolkik.
* [skift](https://github.com/shaypal5/skift) ⭐ 233 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2022-06-07 - Scikit-learn wrappers for Python fastText. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyMorfologik](https://github.com/dmirecki/pyMorfologik) ⭐ 18 | 🐛 1 | 🌐 Python | 📅 2015-08-15 - Python binding for <a href="https://github.com/morfologik/morfologik-stemming">Morfologik</a>.
* [spaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing.
* [gensim](https://radimrehurek.com/gensim/) - Topic Modelling for Humans.

## Computer Audition

* [librosa](https://github.com/librosa/librosa) ⭐ 8,194 | 🐛 72 | 🌐 Python | 📅 2026-02-13 - Python library for audio and music analysis.
* [aubio](https://github.com/aubio/aubio) ⭐ 3,637 | 🐛 157 | 🌐 C | 📅 2025-11-20 - A library for audio and music analysis.
* [Essentia](https://github.com/MTG/essentia) ⭐ 3,405 | 🐛 408 | 🌐 C++ | 📅 2026-02-09 - Library for audio and music analysis, description, and synthesis.
* [torchaudio](https://github.com/pytorch/audio) ⭐ 2,829 | 🐛 319 | 🌐 Python | 📅 2026-02-16 - An audio library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [madmom](https://github.com/CPJKU/madmom) ⭐ 1,582 | 🐛 77 | 🌐 Python | 📅 2024-08-25 - Python audio and music signal processing library.
* [Marsyas](https://github.com/marsyas/marsyas) ⭐ 422 | 🐛 36 | 🌐 C++ | 📅 2023-04-19 - Music Analysis, Retrieval, and Synthesis for Audio Signals.
* [Yaafe](https://github.com/Yaafe/Yaafe) ⭐ 248 | 🐛 17 | 🌐 C++ | 📅 2021-06-21 - Audio features extraction.
* [muda](https://github.com/bmcfee/muda) ⭐ 236 | 🐛 9 | 🌐 Python | 📅 2021-05-03 - A library for augmenting annotated audio data.
* [LibXtract](https://github.com/jamiebullock/LibXtract) ⭐ 231 | 🐛 54 | 🌐 C++ | 📅 2020-04-03 - A simple, portable, lightweight library of audio feature extraction functions.

## Computer Vision

* [OpenCV](https://github.com/opencv/opencv) ⭐ 86,203 | 🐛 2,719 | 🌐 C++ | 📅 2026-02-15 - Open Source Computer Vision Library.
* [torchvision](https://github.com/pytorch/vision) ⭐ 17,513 | 🐛 1,188 | 🌐 Python | 📅 2026-02-16 - Datasets, Transforms, and Models specific to Computer Vision. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [albumentations](https://github.com/albu/albumentations) ⚠️ Archived - Fast image augmentation library and easy-to-use wrapper around other libraries.
* [imgaug](https://github.com/aleju/imgaug) ⭐ 14,730 | 🐛 309 | 🌐 Python | 📅 2024-07-30 - Image augmentation for machine learning experiments.
* [LAVIS](https://github.com/salesforce/LAVIS) ⭐ 11,166 | 🐛 500 | 🌐 Jupyter Notebook | 📅 2024-11-18 - A One-stop Library for Language-Vision Intelligence.
* [PyTorch3D](https://github.com/facebookresearch/pytorch3d) ⭐ 9,795 | 🐛 304 | 🌐 Python | 📅 2026-01-14 - PyTorch3D is FAIR's library of reusable components for deep learning with 3D data. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [scikit-image](https://github.com/scikit-image/scikit-image) ⭐ 6,450 | 🐛 867 | 🌐 Python | 📅 2026-02-13 - Image Processing SciKit (Toolbox for SciPy).
* [Augmentor](https://github.com/mdbloice/Augmentor) ⭐ 5,146 | 🐛 140 | 🌐 Python | 📅 2024-03-21 - Image augmentation library in Python for machine learning.
* [Decord](https://github.com/dmlc/decord) ⭐ 2,420 | 🐛 213 | 🌐 C++ | 📅 2024-07-17 - An efficient video loader for deep learning with smart shuffling that's super easy to digest.
* [MMEngine](https://github.com/open-mmlab/mmengine) ⭐ 1,452 | 🐛 257 | 🌐 Python | 📅 2025-12-23 - OpenMMLab Foundational Library for Training Deep Learning Models. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [KerasCV](https://github.com/keras-team/keras-cv) ⭐ 1,060 | 🐛 74 | 🌐 Python | 📅 2026-01-01 - Industry-strength Computer Vision workflows with Keras. <img height="20" src="img/keras_big.png" alt="MXNet based">
* [imgaug\_extension](https://github.com/cadenai/imgaug_extension) - Additional augmentations for imgaug.

## Time Series

* [Prophet](https://github.com/facebook/prophet) ⭐ 20,012 | 🐛 460 | 🌐 Python | 📅 2026-02-09 - Automatic Forecasting Procedure.
* [sktime](https://github.com/alan-turing-institute/sktime) ⭐ 9,505 | 🐛 1,771 | 🌐 Python | 📅 2026-02-16 - A unified framework for machine learning with time series. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [darts](https://github.com/unit8co/darts) ⭐ 9,207 | 🐛 252 | 🌐 Python | 📅 2026-02-15 - A python library for easy manipulation and forecasting of time series.
* [statsforecast](https://github.com/Nixtla/statsforecast) ⭐ 4,685 | 🐛 137 | 🌐 Python | 📅 2026-02-16 - Lightning fast forecasting with statistical and econometric models.
* [neuralforecast](https://github.com/Nixtla/neuralforecast) ⭐ 3,969 | 🐛 101 | 🌐 Python | 📅 2026-02-16 - Scalable machine learning-based time series forecasting.
* [maya](https://github.com/timofurrer/maya) ⭐ 3,418 | 🐛 21 | 🌐 Python | 📅 2024-07-19 - makes it very easy to parse a string and for changing timezones
* [tslearn](https://github.com/rtavenar/tslearn) ⭐ 3,100 | 🐛 95 | 🌐 Python | 📅 2026-02-12 - Machine learning toolkit dedicated to time-series data. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [PyFlux](https://github.com/RJT1990/pyflux) ⭐ 2,139 | 🐛 93 | 🌐 Python | 📅 2023-10-24 - Open source time series library for Python.
* [greykite](https://github.com/linkedin/greykite) ⭐ 1,854 | 🐛 13 | 🌐 Python | 📅 2025-02-20 - A flexible, intuitive, and fast forecasting library next.
* [skforecast](https://github.com/JoaquinAmatRodrigo/skforecast) ⭐ 1,443 | 🐛 18 | 🌐 Python | 📅 2026-02-14 - Time series forecasting with machine learning models
* [luminol](https://github.com/linkedin/luminol) ⭐ 1,230 | 🐛 35 | 🌐 Python | 📅 2025-08-22 - Anomaly Detection and Correlation library.
* [mlforecast](https://github.com/Nixtla/mlforecast) ⭐ 1,162 | 🐛 17 | 🌐 Python | 📅 2026-02-16 - Scalable machine learning-based time series forecasting.
* [Chaos Genius](https://github.com/chaos-genius/chaos_genius) ⚠️ Archived - ML powered analytics engine for outlier/anomaly detection and root cause analysis
* [tick](https://github.com/X-DataInitiative/tick) ⭐ 538 | 🐛 87 | 🌐 Python | 📅 2024-11-27 - Module for statistical learning, with a particular emphasis on time-dependent modeling.  <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [bayesloop](https://github.com/christophmark/bayesloop) ⭐ 168 | 🐛 4 | 🌐 Python | 📅 2026-02-09 - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models.
* [dateutil](https://dateutil.readthedocs.io/en/stable/) - Powerful extensions to the standard datetime module

## Reinforcement Learning

* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) ⭐ 12,749 | 🐛 79 | 🌐 Python | 📅 2026-02-08 - A set of improved implementations of reinforcement learning algorithms based on OpenAI Baselines.
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) ⭐ 11,334 | 🐛 84 | 🌐 Python | 📅 2026-02-10 - An API standard for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly [Gym](https://github.com/openai/gym) ⭐ 37,041 | 🐛 127 | 🌐 Python | 📅 2024-10-11).
* [Dopamine](https://github.com/google/dopamine) ⭐ 10,843 | 🐛 110 | 🌐 Jupyter Notebook | 📅 2024-11-04 - A research framework for fast prototyping of reinforcement learning algorithms.
* [Tianshou](https://github.com/thu-ml/tianshou/#comprehensive-functionality) ⭐ 10,217 | 🐛 134 | 🌐 Python | 📅 2025-12-01 - An elegant PyTorch deep reinforcement learning library. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐ 9,110 | 🐛 91 | 🌐 Python | 📅 2025-07-08 - High-quality single file implementation of Deep Reinforcement Learning algorithms with research-friendly features (PPO, DQN, C51, DDPG, TD3, SAC, PPG).
* [keras-rl](https://github.com/keras-rl/keras-rl) ⭐ 5,554 | 🐛 48 | 🌐 Python | 📅 2023-09-17 - Deep Reinforcement Learning for Keras. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Acme](https://github.com/google-deepmind/acme) ⭐ 3,921 | 🐛 94 | 🌐 Python | 📅 2026-02-16 - A library of reinforcement learning components and agents.
* [Horizon](https://github.com/facebookresearch/Horizon) ⭐ 3,682 | 🐛 85 | 🌐 Python | 📅 2026-02-12 - A platform for Applied Reinforcement Learning.
* [DI-engine](https://github.com/opendilab/DI-engine) ⭐ 3,591 | 🐛 25 | 🌐 Python | 📅 2025-12-07 - OpenDILab Decision AI Engine. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) ⭐ 3,310 | 🐛 59 | 🌐 Python | 📅 2026-02-06 - An API standard for multi-agent reinforcement learning environments, with popular reference environments and related utilities.
* [TensorForce](https://github.com/reinforceio/tensorforce) ⭐ 3,310 | 🐛 44 | 🌐 Python | 📅 2024-07-31 - A TensorFlow library for applied reinforcement learning. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [TRFL](https://github.com/deepmind/trfl) ⭐ 3,135 | 🐛 6 | 🌐 Python | 📅 2022-12-08 - TensorFlow Reinforcement Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TF-Agents](https://github.com/tensorflow/agents) ⭐ 2,988 | 🐛 211 | 🌐 Python | 📅 2026-01-16 - A library for Reinforcement Learning in TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [rlpyt](https://github.com/astooke/rlpyt) ⭐ 2,275 | 🐛 63 | 🌐 Python | 📅 2021-01-04 - Reinforcement Learning in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [garage](https://github.com/rlworkgroup/garage) ⭐ 2,078 | 🐛 234 | 🌐 Python | 📅 2023-05-04 - A toolkit for reproducible reinforcement learning research.
* [Imitation](https://github.com/HumanCompatibleAI/imitation) ⭐ 1,689 | 🐛 95 | 🌐 Python | 📅 2025-01-07 - Clean PyTorch implementations of imitation and reward learning algorithms. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [d3rlpy](https://github.com/takuseno/d3rlpy) ⭐ 1,643 | 🐛 59 | 🌐 Python | 📅 2025-09-10 - An offline deep reinforcement learning library.
* [EnvPool](https://github.com/sail-sg/envpool) ⭐ 1,268 | 🐛 72 | 🌐 C++ | 📅 2024-08-12 - C++-based high-performance parallel environment execution engine (vectorized env) for general RL environments.
* [SKRL](https://github.com/Toni-SM/skrl) ⭐ 999 | 🐛 26 | 🌐 Python | 📅 2026-02-14 - Modular reinforcement learning library (on PyTorch and JAX) with support for NVIDIA Isaac Gym, Isaac Orbit and Omniverse Isaac Gym. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Machin](https://github.com/iffiX/machin) ⭐ 419 | 🐛 1 | 🌐 Python | 📅 2021-08-08 -  A reinforcement library designed for pytorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [MAgent2](https://github.com/Farama-Foundation/MAgent2) ⭐ 320 | 🐛 22 | 🌐 C++ | 📅 2025-11-16 - An engine for high performance multi-agent environments with very large numbers of agents, along with a set of reference environments.
* [Shimmy](https://github.com/Farama-Foundation/Shimmy) ⭐ 203 | 🐛 10 | 🌐 Python | 📅 2025-12-15 - An API conversion tool for popular external reinforcement learning environments.
* [Catalyst-RL](https://github.com/catalyst-team/catalyst-rl) ⭐ 48 | 🐛 4 | 🌐 Python | 📅 2021-09-13 - PyTorch framework for RL research. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) - Scalable Reinforcement Learning.

## Graph Machine Learning

* [pytorch\_geometric](https://github.com/rusty1s/pytorch_geometric) ⭐ 23,471 | 🐛 1,249 | 🌐 Python | 📅 2026-02-16 - Geometric Deep Learning Extension Library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [dgl](https://github.com/dmlc/dgl) ⭐ 14,231 | 🐛 600 | 🌐 Python | 📅 2025-07-31 - Python package built to ease deep learning on graph, on top of existing DL frameworks. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/tf_big2.png" alt="TensorFlow"> <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [Graph Nets](https://github.com/google-deepmind/graph_nets) ⭐ 5,392 | 🐛 9 | 🌐 Python | 📅 2022-12-12 - Build Graph Nets in Tensorflow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) ⚠️ Archived - Generate embeddings from large-scale graph-structured data. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [StellarGraph](https://github.com/stellargraph/stellargraph) ⭐ 3,044 | 🐛 326 | 🌐 Python | 📅 2024-04-10 - Machine Learning on Graphs. <img height="20" src="img/tf_big2.png" alt="TensorFlow">  <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [pytorch\_geometric\_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) ⭐ 2,943 | 🐛 29 | 🌐 Python | 📅 2025-09-18 - Temporal Extension Library for PyTorch Geometric. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Spektral](https://github.com/danielegrattarola/spektral) ⭐ 2,394 | 🐛 74 | 🌐 Python | 📅 2024-01-21 - Deep learning on graphs. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Karate Club](https://github.com/benedekrozemberczki/karateclub) ⭐ 2,277 | 🐛 12 | 🌐 Python | 📅 2024-07-17 - An unsupervised machine learning library for graph-structured data.
* [TensorFlow GNN](https://github.com/tensorflow/gnn) ⭐ 1,510 | 🐛 44 | 🌐 Python | 📅 2026-02-14 - A library to build Graph Neural Networks on the TensorFlow platform. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [Jraph](https://github.com/google-deepmind/jraph) ⚠️ Archived - A Graph Neural Network Library in Jax.
* [Auto Graph Learning](https://github.com/THUMNLab/AutoGL) ⭐ 1,135 | 🐛 20 | 🌐 Python | 📅 2025-11-20 -An autoML framework & toolkit for machine learning on graphs.
* [Auto Graph Learning](https://github.com/THUMNLab/AutoGL) ⭐ 1,135 | 🐛 20 | 🌐 Python | 📅 2025-11-20 - An autoML framework & toolkit for machine learning on graphs.
* [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) ⭐ 713 | 🐛 7 | 🌐 Python | 📅 2025-12-20 - A library for sampling graph structured data.
* [GRAPE](https://github.com/AnacletoLAB/grape/tree/main) ⭐ 622 | 🐛 37 | 🌐 Jupyter Notebook | 📅 2024-02-24 - GRAPE is a Rust/Python Graph Representation Learning library for Predictions and Evaluations
* [PyTorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed) ⭐ 144 | 🐛 0 | 🌐 Python | 📅 2025-02-09 -  A signed/directed graph neural network extension library for PyTorch Geometric. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [GreatX](https://github.com/EdisonLeeeee/GreatX) ⭐ 90 | 🐛 3 | 🌐 Python | 📅 2024-10-15 - A graph reliability toolbox based on PyTorch and PyTorch Geometric (PyG). <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">

## Graph Manipulation

* [Networkx](https://github.com/networkx/networkx) ⭐ 16,620 | 🐛 349 | 🌐 Python | 📅 2026-02-15 - Network Analysis in Python.
* [Rustworkx](https://github.com/Qiskit/rustworkx) ⭐ 1,587 | 🐛 128 | 🌐 Rust | 📅 2026-02-17 - A high performance Python graph library implemented in Rust.
* [igraph](https://github.com/igraph/python-igraph) ⭐ 1,435 | 🐛 60 | 🌐 Python | 📅 2026-01-12 - Python interface for igraph.
* [graph-tool](https://graph-tool.skewed.de/) - an efficient Python module for manipulation and statistical analysis of graphs (a.k.a. networks).

## Learning-to-Rank & Recommender Systems

* [Surprise](https://github.com/NicolasHug/Surprise) ⭐ 6,765 | 🐛 92 | 🌐 Python | 📅 2025-07-24 - A Python scikit for building and analyzing recommender systems.
* [LightFM](https://github.com/lyst/lightfm) ⭐ 5,069 | 🐛 165 | 🌐 Python | 📅 2024-07-24 - A Python implementation of LightFM, a hybrid recommendation algorithm.
* [RecBole](https://github.com/RUCAIBox/RecBole) ⭐ 4,271 | 🐛 354 | 🌐 Python | 📅 2025-02-24 - A unified, comprehensive and efficient recommendation library. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [TensorFlow Ranking](https://github.com/tensorflow/ranking) ⚠️ Archived - Learning to Rank in TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [TensorFlow Recommenders](https://github.com/tensorflow/recommenders) ⭐ 1,997 | 🐛 282 | 🌐 Python | 📅 2026-01-23 - A library for building recommender system models using TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow"> <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [allRank](https://github.com/allegro/allRank) ⭐ 990 | 🐛 18 | 🌐 Python | 📅 2024-08-06 - allRank is a framework for training learning-to-rank neural models based on PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Spotlight](https://maciejkula.github.io/spotlight/) - Deep recommender models using PyTorch.

## Probabilistic Graphical Models

* [pomegranate](https://github.com/jmschrei/pomegranate) ⭐ 3,506 | 🐛 40 | 🌐 Python | 📅 2025-03-06 - Probabilistic and graphical models for Python. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pgmpy](https://github.com/pgmpy/pgmpy) ⭐ 3,152 | 🐛 444 | 🌐 Python | 📅 2026-02-10 - A python library for working with Probabilistic Graphical Models.
* [pyAgrum](https://agrum.gitlab.io/) - A GRaphical Universal Modeler.

## Probabilistic Methods

* [PyMC](https://github.com/pymc-devs/pymc) ⭐ 9,478 | 🐛 461 | 🌐 Python | 📅 2026-02-16 - Bayesian Stochastic Modelling in Python.
* [pyro](https://github.com/uber/pyro) ⭐ 8,980 | 🐛 274 | 🌐 Python | 📅 2025-07-09 - A flexible, scalable deep probabilistic programming library built on PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [GPyTorch](https://github.com/cornellius-gp/gpytorch) ⭐ 3,829 | 🐛 402 | 🌐 Python | 📅 2026-02-13 - A highly efficient and modular implementation of Gaussian Processes in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [emcee](https://github.com/dfm/emcee) ⭐ 1,566 | 🐛 67 | 🌐 Python | 📅 2026-01-19 - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [pyhsmm](https://github.com/mattjj/pyhsmm) ⭐ 572 | 🐛 46 | 🌐 Python | 📅 2025-01-25 - Bayesian inference in HSMMs and HMMs.
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes) ⭐ 522 | 🐛 20 | 🌐 Jupyter Notebook | 📅 2021-09-22 - Python package for Bayesian Machine Learning with scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) ⭐ 433 | 🐛 42 | 🌐 Python | 📅 2026-02-10 - A scikit-learn-inspired API for CRFsuite. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [PyVarInf](https://github.com/ctallec/pyvarinf) ⭐ 362 | 🐛 4 | 🌐 Python | 📅 2019-10-12 - Bayesian Deep Learning methods with Variational Inference for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PyStan](https://github.com/stan-dev/pystan) ⭐ 361 | 🐛 15 | 🌐 Python | 📅 2024-07-03 - Bayesian inference using the No-U-Turn sampler (Python interface).
* [skpro](https://github.com/alan-turing-institute/skpro) ⭐ 296 | 🐛 111 | 🌐 Python | 📅 2026-02-15 - Supervised domain-agnostic prediction framework for probabilistic modelling by [The Alan Turing Institute](https://www.turing.ac.uk/). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [InferPy](https://github.com/PGM-Lab/InferPy) ⭐ 148 | 🐛 65 | 🌐 Jupyter Notebook | 📅 2024-08-02 - Deep Probabilistic Modelling Made Easy.  <img height="20" src="img/tf_big2.png" alt="sklearn">
* [hsmmlearn](https://github.com/jvkersch/hsmmlearn) ⚠️ Archived - A library for hidden semi-Markov models with explicit durations.
* [ZhuSuan](http://zhusuan.readthedocs.io/en/latest/) - Bayesian Deep Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [GPflow](http://gpflow.readthedocs.io/en/latest/?badge=latest) - Gaussian processes in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">

## Model Explanation

* [Netron](https://github.com/lutzroeder/Netron) ⭐ 32,404 | 🐛 19 | 🌐 JavaScript | 📅 2026-02-16 - Visualizer for deep learning and machine learning models (no Python code, but visualizes models from most Python Deep Learning frameworks).
* [shap](https://github.com/slundberg/shap) ⭐ 25,031 | 🐛 588 | 🌐 Jupyter Notebook | 📅 2026-02-14 - A unified approach to explain the output of any machine learning model. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Lime](https://github.com/marcotcr/lime) ⭐ 12,100 | 🐛 131 | 🌐 JavaScript | 📅 2024-07-25 - Explaining the predictions of any machine learning classifier. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) ⭐ 7,989 | 🐛 91 | 🌐 Python | 📅 2026-02-05 - Tensorboard for PyTorch (and chainer, mxnet, numpy, ...).
* [InterpretML](https://github.com/interpretml/interpret) ⭐ 6,793 | 🐛 112 | 🌐 C++ | 📅 2026-02-17 - InterpretML implements the Explainable Boosting Machine (EBM), a modern, fully interpretable machine learning model based on Generalized Additive Models (GAMs). This open-source package also provides visualization tools for EBMs, other glass-box models, and black-box explanations. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [lucid](https://github.com/tensorflow/lucid) ⚠️ Archived - A collection of infrastructure and tools for research in neural network interpretability.
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) ⭐ 4,394 | 🐛 109 | 🌐 Python | 📅 2025-02-19 - Visual analysis and diagnostic tools to facilitate machine learning model selection. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ELI5](https://github.com/TeamHG-Memex/eli5) ⭐ 2,771 | 🐛 162 | 🌐 Jupyter Notebook | 📅 2026-02-10 - A library for debugging/inspecting machine learning classifiers and explaining their predictions.
* [Alibi](https://github.com/SeldonIO/alibi) ⭐ 2,607 | 🐛 157 | 🌐 Python | 📅 2025-10-17 - Algorithms for monitoring and explaining machine learning models.
* [scikit-plot](https://github.com/reiinakano/scikit-plot) ⭐ 2,431 | 🐛 31 | 🌐 Python | 📅 2024-08-20 - An intuitive library to add plotting functionality to scikit-learn objects. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AI Explainability 360](https://github.com/IBM/AIX360) ⭐ 1,761 | 🐛 55 | 🌐 Python | 📅 2025-02-26 - Interpretability and explainability of data and machine learning models.
* [dalex](https://github.com/ModelOriented/DALEX) ⭐ 1,456 | 🐛 30 | 🌐 Python | 📅 2026-01-20 - moDel Agnostic Language for Exploration and explanation. <img height="20" src="img/sklearn_big.png" alt="sklearn"><img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [model-analysis](https://github.com/tensorflow/model-analysis) ⭐ 1,267 | 🐛 37 | 🌐 Python | 📅 2025-08-06 - Model analysis tools for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PDPbox](https://github.com/SauceCat/PDPbox) ⭐ 863 | 🐛 33 | 🌐 Jupyter Notebook | 📅 2024-09-03 - Partial dependence plot toolbox.
* [anchor](https://github.com/marcotcr/anchor) ⭐ 814 | 🐛 26 | 🌐 Jupyter Notebook | 📅 2022-07-19 - Code for "High-Precision Model-Agnostic Explanations" paper.
* [treeinterpreter](https://github.com/andosa/treeinterpreter) ⭐ 761 | 🐛 26 | 🌐 Python | 📅 2023-07-18 - Interpreting scikit-learn's decision tree and random forest predictions. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [aequitas](https://github.com/dssg/aequitas) ⭐ 748 | 🐛 55 | 🌐 Python | 📅 2026-02-03 - Bias and Fairness Audit Toolkit.
* [CapsNet-Visualization](https://github.com/bourdakos1/CapsNet-Visualization) ⭐ 395 | 🐛 16 | 🌐 Python | 📅 2021-10-05 - A visualization of the CapsNet layers to better understand how it works.
* [FairML](https://github.com/adebayoj/fairml) ⭐ 366 | 🐛 14 | 🌐 Python | 📅 2021-05-10 - FairML is a python toolbox auditing the machine learning models for bias. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Shapley](https://github.com/benedekrozemberczki/shapley) ⭐ 222 | 🐛 1 | 🌐 Python | 📅 2026-01-01 - A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
* [PyCEbox](https://github.com/AustinRochford/PyCEbox) ⭐ 164 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2020-05-29 - Python Individual Conditional Expectation Plot Toolbox.
* [themis-ml](https://github.com/cosmicBboy/themis-ml) ⚠️ Archived - A library that implements fairness-aware machine learning algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [L2X](https://github.com/Jianbo-Lab/L2X) ⭐ 125 | 🐛 3 | 🌐 Python | 📅 2021-05-10 - Code for replicating the experiments in the paper *Learning to Explain: An Information-Theoretic Perspective on Model Interpretation*.
* [Contrastive Explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) ⭐ 45 | 🐛 2 | 🌐 Python | 📅 2023-01-31 - Contrastive Explanation (Foil Trees). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Auralisation](https://github.com/keunwoochoi/Auralisation) ⭐ 42 | 🐛 0 | 🌐 Python | 📅 2017-03-11 - Auralisation of learned features in CNN (for audio).
* [Skater](https://github.com/datascienceinc/Skater) - Python Library for Model Interpretation.
* [FlashLight](https://github.com/dlguys/flashlight) - Visualization Tool for your NeuralNetwork.

## Genetic Programming

* [DEAP](https://github.com/DEAP/deap) ⭐ 6,330 | 🐛 284 | 🌐 Python | 📅 2025-11-16 - Distributed Evolutionary Algorithms in Python.
* [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) ⭐ 2,166 | 🐛 102 | 🌐 Python | 📅 2025-07-09 - Genetic Algorithm in Python. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/keras_big.png" alt="keras">
* [gplearn](https://github.com/trevorstephens/gplearn) ⭐ 1,810 | 🐛 17 | 🌐 Python | 📅 2026-01-10 - Genetic Programming in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic) ⭐ 325 | 🐛 10 | 🌐 Python | 📅 2024-01-20 - Genetic feature selection module for scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [karoo\_gp](https://github.com/kstaats/karoo_gp) ⭐ 164 | 🐛 22 | 🌐 Python | 📅 2022-10-31 - A Genetic Programming platform for Python with GPU support. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [monkeys](https://github.com/hchasestevens/monkeys) ⭐ 124 | 🐛 4 | 🌐 Python | 📅 2018-06-13 - A strongly-typed genetic programming framework for Python.

<a name="opt"></a>

## Optimization

* [Optuna](https://github.com/optuna/optuna) ⭐ 13,523 | 🐛 34 | 🌐 Python | 📅 2026-02-16 - A hyperparameter optimization framework.
* [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) ⭐ 8,549 | 🐛 9 | 🌐 Python | 📅 2025-12-27 - A Python implementation of global optimization with gaussian processes.
* [hyperopt](https://github.com/hyperopt/hyperopt) ⭐ 7,619 | 🐛 3 | 🌐 Python | 📅 2026-02-08 - Distributed Asynchronous Hyperparameter Optimization in Python.
* [scikit-opt](https://github.com/guofei9987/scikit-opt) ⭐ 6,370 | 🐛 70 | 🌐 Python | 📅 2025-08-31 - Heuristic Algorithms for optimization.
* [BoTorch](https://github.com/pytorch/botorch) ⭐ 3,462 | 🐛 95 | 🌐 Jupyter Notebook | 📅 2026-02-14 - Bayesian optimization in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) ⚠️ Archived - Sequential model-based optimization with a `scipy.optimize` interface.
* [pymoo](https://github.com/anyoptimization/pymoo) ⭐ 2,777 | 🐛 9 | 🌐 Python | 📅 2025-11-26 - Multi-objective Optimization in Python.
* [POT](https://github.com/rflamary/POT) ⭐ 2,745 | 🐛 57 | 🌐 Python | 📅 2026-02-13 - Python Optimal Transport library.
* [nlopt](https://github.com/stevengj/nlopt) ⭐ 2,179 | 🐛 99 | 🌐 C | 📅 2026-02-09 - Library for nonlinear optimization (global and local, constrained or unconstrained).
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) ⭐ 1,643 | 🐛 78 | 🌐 Python | 📅 2025-04-15 - Hyper-parameter optimization for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Talos](https://github.com/autonomio/talos) ⭐ 1,639 | 🐛 13 | 🌐 Python | 📅 2024-04-22 - Hyperparameter Optimization for Keras Models.
* [Spearmint](https://github.com/HIPS/Spearmint) ⭐ 1,564 | 🐛 77 | 🌐 Python | 📅 2019-12-27 - Bayesian optimization.
* [PySwarms](https://github.com/ljvmiranda921/pyswarms) ⭐ 1,380 | 🐛 41 | 🌐 Python | 📅 2024-08-06 - A research toolkit for particle swarm optimization in Python.
* [pycma](https://github.com/CMA-ES/pycma?tab=readme-ov-file) ⭐ 1,281 | 🐛 87 | 🌐 Jupyter Notebook | 📅 2026-01-25 - Python implementation of CMA-ES.
* [SMAC3](https://github.com/automl/SMAC3) ⭐ 1,215 | 🐛 122 | 🌐 Python | 📅 2026-02-16 - Sequential Model-based Algorithm Configuration.
* [sklearn-deap](https://github.com/rsteca/sklearn-deap) ⭐ 770 | 🐛 28 | 🌐 Jupyter Notebook | 📅 2024-02-10 - Use evolutionary algorithms instead of gridsearch in scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Platypus](https://github.com/Project-Platypus/Platypus) ⭐ 643 | 🐛 0 | 🌐 Python | 📅 2026-01-21 - A Free and Open Source Python Library for Multiobjective Optimization.
* [Solid](https://github.com/100/Solid) ⭐ 579 | 🐛 7 | 🌐 Python | 📅 2019-07-19 - A comprehensive gradient-free optimization framework written in Python.
* [Optunity](https://github.com/claesenm/optunity) ⭐ 425 | 🐛 52 | 🌐 Jupyter Notebook | 📅 2023-11-25 - Is a library containing various optimizers for hyperparameter tuning.
* [sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt) ⭐ 355 | 🐛 4 | 🌐 Python | 📅 2025-09-13 - Hyperparameters tuning and feature selection using evolutionary algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [GPflowOpt](https://github.com/GPflow/GPflowOpt) ⭐ 274 | 🐛 29 | 🌐 Python | 📅 2020-12-02 - Bayesian Optimization using GPflow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [SafeOpt](https://github.com/befelix/SafeOpt) ⭐ 150 | 🐛 2 | 🌐 Python | 📅 2022-11-14 - Safe Bayesian Optimization.
* [sigopt\_sklearn](https://github.com/sigopt/sigopt_sklearn) ⚠️ Archived - SigOpt wrappers for scikit-learn methods. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [OR-Tools](https://developers.google.com/optimization) - An open-source software suite for optimization by Google; provides a unified programming interface to a half dozen solvers: SCIP, GLPK, GLOP, CP-SAT, CPLEX, and Gurobi.

## Feature Engineering

### General

* [tsfresh](https://github.com/blue-yonder/tsfresh) ⭐ 9,116 | 🐛 71 | 🌐 Jupyter Notebook | 📅 2025-11-15 - Automatic extraction of relevant features from time series. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Featuretools](https://github.com/Featuretools/featuretools) ⭐ 7,609 | 🐛 163 | 🌐 Python | 📅 2026-02-03 - Automated feature engineering.
* [Feature Engine](https://github.com/feature-engine/feature_engine) ⭐ 2,197 | 🐛 70 | 🌐 Python | 📅 2026-02-11 - Feature engineering package with sklearn-like functionality. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) ⭐ 861 | 🐛 24 | 🌐 Python | 📅 2024-05-27 - Automated feature generation with expert-level performance.
* [Feature Forge](https://github.com/machinalis/featureforge) ⭐ 385 | 🐛 11 | 🌐 Python | 📅 2017-12-26 - A set of tools for creating and testing machine learning features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-mdr](https://github.com/EpistasisLab/scikit-mdr) ⭐ 126 | 🐛 11 | 🌐 Python | 📅 2025-06-10 - A sklearn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [NitroFE](https://github.com/NITRO-AI/NitroFE) ⭐ 108 | 🐛 0 | 🌐 Python | 📅 2022-05-04 - Moving window features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [few](https://github.com/lacava/few) ⭐ 52 | 🐛 9 | 🌐 Python | 📅 2020-06-11 - A feature engineering wrapper for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [skl-groups](https://github.com/dougalsutherland/skl-groups) ⭐ 41 | 🐛 25 | 🌐 Python | 📅 2016-08-08 - A scikit-learn addon to operate on set/"group"-based features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [dirty\_cat](https://github.com/dirty-cat/dirty_cat) ⭐ 20 | 🐛 0 | 🌐 Python | 📅 2025-03-12 - Machine learning on dirty tabular data (especially: string-based variables for classifcation and regression). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sk-transformer](https://github.com/chrislemke/sk-transformers) ⚠️ Archived - A collection of various pandas & scikit-learn compatible transformers for all kinds of preprocessing and feature engineering steps <img height="20" src="img/pandas_big.png" alt="pandas compatible">

### Feature Selection

* [boruta\_py](https://github.com/scikit-learn-contrib/boruta_py) ⭐ 1,620 | 🐛 48 | 🌐 Python | 📅 2025-11-13 - Implementations of the Boruta all-relevant feature selection method. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-feature](https://github.com/jundongl/scikit-feature) ⭐ 1,565 | 🐛 45 | 🌐 Python | 📅 2024-07-11 - Feature selection repository in Python.
* [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) ⭐ 422 | 🐛 18 | 🌐 Python | 📅 2023-02-10 - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [zoofs](https://github.com/jaswinder9051998/zoofs) ⭐ 251 | 🐛 22 | 🌐 Python | 📅 2026-01-09 - A feature selection library based on evolutionary algorithms.
* [BoostARoota](https://github.com/chasedehan/BoostARoota) ⭐ 232 | 🐛 7 | 🌐 Python | 📅 2021-04-01 - A fast xgboost feature selection algorithm. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Visualization

### General Purposes

* [Matplotlib](https://github.com/matplotlib/matplotlib) ⭐ 22,426 | 🐛 1,542 | 🌐 Python | 📅 2026-02-16 - Plotting with Python.
* [seaborn](https://github.com/mwaskom/seaborn) ⭐ 13,725 | 🐛 206 | 🌐 Python | 📅 2026-01-22 - Statistical data visualization using matplotlib.
* [missingno](https://github.com/ResidentMario/missingno) ⭐ 4,188 | 🐛 15 | 🌐 Python | 📅 2024-05-14 - Missing data visualization module for Python.
* [chartify](https://github.com/spotify/chartify/) ⭐ 3,621 | 🐛 54 | 🌐 Python | 📅 2024-10-16 - Python library that makes it easy for data scientists to create charts.
* [prettyplotlib](https://github.com/olgabot/prettyplotlib) ⭐ 1,707 | 🐛 11 | 🌐 Python | 📅 2019-01-31 - Painlessly create beautiful matplotlib plots.
* [python-ternary](https://github.com/marcharper/python-ternary) ⭐ 778 | 🐛 37 | 🌐 Python | 📅 2024-06-12 - Ternary plotting library for Python with matplotlib.
* [physt](https://github.com/janpipek/physt) ⭐ 137 | 🐛 40 | 🌐 Python | 📅 2026-01-07 - Improved histograms.

### Interactive plots

* [Bokeh](https://github.com/bokeh/bokeh) ⭐ 20,351 | 🐛 870 | 🌐 TypeScript | 📅 2026-02-15 - Interactive Web Plotting for Python.
* [pyecharts](https://github.com/pyecharts/pyecharts) ⭐ 15,719 | 🐛 1 | 🌐 Python | 📅 2026-02-10 - Migrated from [Echarts](https://github.com/apache/echarts) ⭐ 65,719 | 🐛 1,796 | 🌐 TypeScript | 📅 2026-02-05, a charting and visualization library, to Python's interactive visual drawing library.<img height="20" src="img/pyecharts.png" alt="pyecharts"> <img height="20" src="img/echarts.png" alt="echarts">
* [bqplot](https://github.com/bqplot/bqplot) ⭐ 3,682 | 🐛 277 | 🌐 TypeScript | 📅 2026-01-23 - Plotting library for IPython/Jupyter notebooks
* [animatplot](https://github.com/t-makaro/animatplot) ⭐ 416 | 🐛 17 | 🌐 Python | 📅 2024-08-29 - A python package for animating plots built on matplotlib.
* [plotly](https://plot.ly/python/) - A Python library that makes interactive and publication-quality graphs.
* [Altair](https://altair-viz.github.io/) - Declarative statistical visualization library for Python. Can easily do many data transformation within the code to create graph

### Map

* [folium](https://python-visualization.github.io/folium/quickstart.html#Getting-Started) - Makes it easy to visualize data on an interactive open street map
* [geemap](https://github.com/giswqs/geemap) ⭐ 3,875 | 🐛 31 | 🌐 Python | 📅 2026-02-16 - Python package for interactive mapping with Google Earth Engine (GEE)

### Automatic Plotting

* [SweetViz](https://github.com/fbdesignpro/sweetviz) ⭐ 3,079 | 🐛 56 | 🌐 Python | 📅 2024-08-06: Visualize and compare datasets, target values and associations, with one line of code.
* [HoloViews](https://github.com/ioam/holoviews) ⭐ 2,879 | 🐛 1,066 | 🌐 Python | 📅 2026-02-05 - Stop plotting your data - annotate your data and let it visualize itself.
* [AutoViz](https://github.com/AutoViML/AutoViz) ⭐ 1,882 | 🐛 2 | 🌐 Python | 📅 2024-06-10: Visualize data automatically with 1 line of code (ideal for machine learning)

### NLP

* [pyLDAvis](https://github.com/bmabey/pyLDAvis) ⭐ 1,846 | 🐛 81 | 🌐 Jupyter Notebook | 📅 2025-12-04: Visualize interactive topic model

## Deployment

* [gradio](https://github.com/gradio-app/gradio) ⭐ 41,730 | 🐛 479 | 🌐 Python | 📅 2026-02-16 - Create UIs for your machine learning model in Python in 3 minutes.
* [Vizro](https://github.com/mckinsey/vizro) ⭐ 3,575 | 🐛 54 | 🌐 Python | 📅 2026-02-16 - A toolkit for creating modular data visualization applications.
* [Deepnote](https://github.com/deepnote/deepnote) ⭐ 2,649 | 🐛 13 | 🌐 TypeScript | 📅 2026-02-16 - Deepnote is a drop-in replacement for Jupyter with an AI-first design, sleek UI, new blocks, and native data integrations. Use Python, R, and SQL locally in your favorite IDE, then scale to Deepnote cloud for real-time collaboration, Deepnote agent, and deployable data apps.
* [streamsync](https://github.com/streamsync-cloud/streamsync) ⭐ 1,439 | 🐛 9 | 🌐 Python | 📅 2026-02-05 - No-code in the front, Python in the back. An open-source framework for creating data apps.
* [fastapi](https://fastapi.tiangolo.com/) - Modern, fast (high-performance), a web framework for building APIs with Python
* [streamlit](https://www.streamlit.io/) - Make it easy to deploy the machine learning model
* [datapane](https://datapane.com/) - A collection of APIs to turn scripts and notebooks into interactive reports.
* [binder](https://mybinder.org/) - Enable sharing and execute Jupyter Notebooks

## Statistics

* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) ⭐ 13,376 | 🐛 293 | 🌐 Python | 📅 2026-02-02 - Create HTML profiling reports from pandas DataFrame objects. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [statsmodels](https://github.com/statsmodels/statsmodels) ⭐ 11,249 | 🐛 2,970 | 🌐 Python | 📅 2026-01-13 - Statistical modeling and econometrics in Python.
* [Alphalens](https://github.com/quantopian/alphalens) ⭐ 4,135 | 🐛 49 | 🌐 Jupyter Notebook | 📅 2024-02-12 - Performance analysis of predictive (alpha) stock factors.
* [stockstats](https://github.com/jealous/stockstats) ⭐ 1,443 | 🐛 13 | 🌐 Python | 📅 2026-02-16 - Supply a wrapper `StockDataFrame` based on the `pandas.DataFrame` with inline stock statistics/indicators support.
* [pandas\_summary](https://github.com/mouradmourafiq/pandas-summary) ⭐ 528 | 🐛 6 | 🌐 Python | 📅 2026-02-11 - Extension to pandas dataframes describe function. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) ⭐ 380 | 🐛 3 | 🌐 Python | 📅 2026-02-11 - Pairwise Multiple Comparisons Post-hoc Tests.
* [weightedcalcs](https://github.com/jsvine/weightedcalcs) ⭐ 113 | 🐛 3 | 🌐 Python | 📅 2024-11-10 - A pandas-based utility to calculate weighted means, medians, distributions, standard deviations, and more.

## Data Manipulation

### Data Frames

* [polars](https://github.com/pola-rs/polars) ⭐ 37,443 | 🐛 2,750 | 🌐 Rust | 📅 2026-02-16 - A fast multi-threaded, hybrid-out-of-core DataFrame library.
* [pandas\_profiling](https://github.com/pandas-profiling/pandas-profiling) ⭐ 13,376 | 🐛 293 | 🌐 Python | 📅 2026-02-02 - Create HTML profiling reports from pandas DataFrame objects
* [modin](https://github.com/modin-project/modin) ⭐ 10,358 | 🐛 708 | 🌐 Python | 📅 2026-02-10 - Speed up your pandas workflows by changing a single line of code. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [cuDF](https://github.com/rapidsai/cudf) ⭐ 9,490 | 🐛 1,193 | 🌐 C++ | 📅 2026-02-16 - GPU DataFrame Library. <img height="20" src="img/pandas_big.png" alt="pandas compatible"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [vaex](https://github.com/vaexio/vaex) ⭐ 8,468 | 🐛 548 | 🌐 Python | 📅 2026-02-05 - Out-of-Core DataFrames for Python, ML, visualize and explore big tabular data at a billion rows per second.
* [xarray](https://github.com/pydata/xarray) ⭐ 4,087 | 🐛 1,337 | 🌐 Python | 📅 2026-02-16 - Xarray combines the best features of NumPy and pandas for multidimensional data selection by supplementing numerical axis labels with named dimensions for more intuitive, concise, and less error-prone indexing routines.
* [blaze](https://github.com/blaze/blaze) ⭐ 3,198 | 🐛 267 | 🌐 Python | 📅 2023-09-29 - NumPy and pandas interface to Big Data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Arctic](https://github.com/manahl/arctic) ⭐ 3,086 | 🐛 97 | 🌐 Python | 📅 2024-04-08 - High-performance datastore for time series and tick data.
* [swifter](https://github.com/jmcarpenter2/swifter) ⭐ 2,643 | 🐛 25 | 🌐 Python | 📅 2024-03-20 - A package that efficiently applies any function to a pandas dataframe or series in the fastest available manner.
* [datatable](https://github.com/h2oai/datatable) ⭐ 1,882 | 🐛 182 | 🌐 C++ | 📅 2025-03-17 - Data.table for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [pandasql](https://github.com/yhat/pandasql) ⭐ 1,348 | 🐛 59 | 🌐 Python | 📅 2024-07-24 -  Allows you to query pandas DataFrames using SQL syntax. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pandas-gbq](https://github.com/pydata/pandas-gbq) ⭐ 489 | 🐛 5 | 🌐 Python | 📅 2026-01-21 - pandas Google Big Query. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pysparkling](https://github.com/svenkreiss/pysparkling) ⭐ 270 | 🐛 11 | 🌐 Python | 📅 2024-09-03 - A pure Python implementation of Apache Spark's RDD and DStream interfaces. <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [pandas-log](https://github.com/eyaltrabelsi/pandas-log) ⭐ 218 | 🐛 11 | 🌐 Python | 📅 2021-06-26 - A package that allows providing feedback about basic pandas operations and finds both business logic and performance issues.
* [xpandas](https://github.com/alan-turing-institute/xpandas) ⭐ 26 | 🐛 10 | 🌐 Python | 📅 2022-06-21 - Universal 1d/2d data containers with Transformers .functionality for data analysis by [The Alan Turing Institute](https://www.turing.ac.uk/).
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - Powerful Python data analysis toolkit.

### Pipelines

* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) ⭐ 2,850 | 🐛 43 | 🌐 Python | 📅 2023-06-08 - pandas integration with sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Hamilton](https://github.com/DAGWorks-Inc/hamilton) ⭐ 2,402 | 🐛 172 | 🌐 Jupyter Notebook | 📅 2026-02-13 - A microframework for dataframe generation that applies Directed Acyclic Graphs specified by a flow of lazily evaluated Python functions.
* [pyjanitor](https://github.com/ericmjl/pyjanitor) ⭐ 1,480 | 🐛 108 | 🌐 Python | 📅 2026-02-13 - Clean APIs for data cleaning. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Dplython](https://github.com/dodger487/dplython) ⭐ 761 | 🐛 28 | 🌐 Python | 📅 2016-12-30 - Dplyr for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [pdpipe](https://github.com/shaypal5/pdpipe) ⭐ 725 | 🐛 22 | 🌐 Jupyter Notebook | 📅 2026-01-05 - Sasy pipelines for pandas DataFrames.
* [dopanda](https://github.com/dovpanda-dev/dovpanda) ⭐ 482 | 🐛 29 | 🌐 Python | 📅 2024-12-01 -  Hints and tips for using pandas in an analysis environment. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [meza](https://github.com/reubano/meza) ⭐ 421 | 🐛 12 | 🌐 Python | 📅 2025-02-27 - A Python toolkit for processing tabular data.
* [Dataset](https://github.com/analysiscenter/dataset) ⭐ 204 | 🐛 27 | 🌐 Python | 📅 2025-10-23 - Helps you conveniently work with random or sequential batches of your data and define data processing.
* [pandas-ply](https://github.com/coursera/pandas-ply) ⭐ 198 | 🐛 4 | 🌐 HTML | 📅 2015-08-27 - Functional data manipulation for pandas. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Prodmodel](https://github.com/prodmodel/prodmodel) ⭐ 58 | 🐛 7 | 🌐 Python | 📅 2022-06-21 - Build system for data science pipelines.
* [SSPipe](https://sspipe.github.io/) - Python pipe (|) operator with support for DataFrames and Numpy, and Pytorch.

### Data-centric AI

* [cleanlab](https://github.com/cleanlab/cleanlab) ⭐ 11,318 | 🐛 100 | 🌐 Python | 📅 2026-01-13 - The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.
* [snorkel](https://github.com/snorkel-team/snorkel) ⭐ 5,937 | 🐛 16 | 🌐 Python | 📅 2024-05-02 - A system for quickly generating training data with weak supervision.
* [dataprep](https://github.com/sfu-db/dataprep) ⭐ 2,235 | 🐛 165 | 🌐 Python | 📅 2024-06-27 - Collect, clean, and visualize your data in Python with a few lines of code.

### Synthetic Data

* [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) ⭐ 1,612 | 🐛 63 | 🌐 Jupyter Notebook | 📅 2026-02-16 - A package to generate synthetic tabular and time-series data leveraging the state-of-the-art generative models. <img height="20" src="img/pandas_big.png" alt="pandas compatible">

## Distributed Computing

* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ⭐ 23,631 | 🐛 1,596 | 🌐 C++ | 📅 2026-02-15 - PArallel Distributed Deep LEarning.
* [Horovod](https://github.com/uber/horovod) ⭐ 14,671 | 🐛 406 | 🌐 Python | 📅 2025-12-01 - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [DMTK](https://github.com/Microsoft/DMTK) ⚠️ Archived - Microsoft Distributed Machine Learning Toolkit.
* [Distributed](https://github.com/dask/distributed) ⭐ 1,667 | 🐛 1,503 | 🌐 Python | 📅 2026-02-16 - Distributed computation in Python.
* [dask-ml](https://github.com/dask/dask-ml) ⭐ 945 | 🐛 282 | 🌐 Python | 📅 2025-09-27 - Distributed and parallel machine learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Veles](https://github.com/Samsung/veles) ⭐ 915 | 🐛 24 | 🌐 C++ | 📅 2023-11-21 - Distributed machine learning platform.
* [Jubatus](https://github.com/jubatus/jubatus) ⭐ 708 | 🐛 4 | 🌐 C++ | 📅 2019-05-16 - Framework and Library for Distributed Online Machine Learning.
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Exposes the Spark programming model to Python. <img height="20" src="img/spark_big.png" alt="Apache Spark based">

## Experimentation

* [mlflow](https://github.com/mlflow/mlflow) ⭐ 24,187 | 🐛 2,108 | 🌐 Python | 📅 2026-02-17 - Open source platform for the machine learning lifecycle.
* [dvc](https://github.com/iterative/dvc) ⭐ 15,368 | 🐛 164 | 🌐 Python | 📅 2026-02-16 - Data Version Control | Git for Data & Models | ML Experiments Management.
* [Sacred](https://github.com/IDSIA/sacred) ⭐ 4,357 | 🐛 105 | 🌐 Python | 📅 2025-10-22 - A tool to help you configure, organize, log, and reproduce experiments.
* [Ax](https://github.com/facebook/Ax) ⭐ 2,704 | 🐛 169 | 🌐 Python | 📅 2026-02-15 - Adaptive Experimentation Platform. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [envd](https://github.com/tensorchord/envd) ⭐ 2,181 | 🐛 124 | 🌐 Go | 📅 2026-02-09 - 🏕️ machine learning development environment for data science and AI/ML engineering teams.
* [Neptune](https://neptune.ai) - A lightweight ML experiment tracking, results visualization, and management tool.

## Data Validation

* [great\_expectations](https://github.com/great-expectations/great_expectations) ⭐ 11,149 | 🐛 74 | 🌐 Python | 📅 2026-02-14 - Always know what to expect from your data.
* [evidently](https://github.com/evidentlyai/evidently) ⭐ 7,123 | 🐛 250 | 🌐 Jupyter Notebook | 📅 2026-02-15 - Evaluate and monitor ML models from validation to production.
* [pandera](https://github.com/unionai-oss/pandera) ⭐ 4,204 | 🐛 469 | 🌐 Python | 📅 2026-02-15 - A lightweight, flexible, and expressive statistical data testing library.
* [deepchecks](https://github.com/deepchecks/deepchecks) ⭐ 3,977 | 🐛 259 | 🌐 Python | 📅 2025-12-28 - Validation & testing of ML models and data during model development, deployment, and production. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [TensorFlow Data Validation](https://github.com/tensorflow/data-validation) ⭐ 779 | 🐛 39 | 🌐 Python | 📅 2025-06-23 - Library for exploring and validating machine learning data.
* [DataComPy](https://github.com/capitalone/datacompy) ⭐ 632 | 🐛 12 | 🌐 Python | 📅 2026-02-12- A library to compare Pandas, Polars, and Spark data frames. It provides stats and lets users adjust for match accuracy.

## Evaluation

* [AI Fairness 360](https://github.com/IBM/AIF360) ⭐ 2,746 | 🐛 212 | 🌐 Python | 📅 2025-11-13 - Fairness metrics for datasets and ML models, explanations, and algorithms to mitigate bias in datasets and models.
* [alibi-detect](https://github.com/SeldonIO/alibi-detect) ⭐ 2,488 | 🐛 141 | 🌐 Jupyter Notebook | 📅 2025-12-11 - Algorithms for outlier, adversarial and drift detection.<img height="20" src="img/alibi-detect.png" alt="sklearn">
* [Metrics](https://github.com/benhamner/Metrics) ⭐ 1,656 | 🐛 36 | 🌐 Python | 📅 2023-01-11 - Machine learning evaluation metric.
* [recmetrics](https://github.com/statisticianinstilettos/recmetrics) ⭐ 582 | 🐛 20 | 🌐 Jupyter Notebook | 📅 2024-01-11 - Library of useful metrics and plots for evaluating recommender systems.
* [sklearn-evaluation](https://github.com/edublancas/sklearn-evaluation) ⭐ 3 | 🐛 0 | 📅 2023-01-15 - Model evaluation made easy: plots, tables, and markdown reports. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Computations

* [Dask](https://github.com/dask/dask) ⭐ 13,743 | 🐛 1,222 | 🌐 Python | 📅 2026-02-16 - Parallel computing with task scheduling. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [CuPy](https://github.com/cupy/cupy) ⭐ 10,789 | 🐛 624 | 🌐 Python | 📅 2026-02-17 - NumPy-like API accelerated with CUDA.
* [NumExpr](https://github.com/pydata/numexpr) ⭐ 2,399 | 🐛 6 | 🌐 Python | 📅 2025-12-04 - A fast numerical expression evaluator for NumPy that comes with an integrated computing virtual machine to speed calculations up by avoiding memory allocation for intermediate results.
* [adaptive](https://github.com/python-adaptive/adaptive) ⭐ 1,215 | 🐛 106 | 🌐 Python | 📅 2026-02-16 - Tools for adaptive and parallel samping of mathematical functions.
* [bottleneck](https://github.com/kwgoodman/bottleneck) ⭐ 1,162 | 🐛 46 | 🌐 Python | 📅 2026-02-06 - Fast NumPy array functions written in C.
* [quaternion](https://github.com/moble/quaternion) ⭐ 656 | 🐛 14 | 🌐 Python | 📅 2025-12-15 - Add built-in support for quaternions to numpy.
* [scikit-tensor](https://github.com/mnick/scikit-tensor) ⭐ 407 | 🐛 27 | 🌐 Python | 📅 2018-08-23 - Python library for multilinear algebra and tensor factorizations.
* [numdifftools](https://github.com/pbrod/numdifftools) ⭐ 279 | 🐛 6 | 🌐 Python | 📅 2026-01-06 - Solve automatic numerical differentiation problems in one or more variables.
* [NumPy](https://numpy.org/) - The fundamental package for scientific computing with Python

## Web Scraping

* [Pattern](https://github.com/clips/pattern) ⭐ 8,855 | 🐛 176 | 🌐 Python | 📅 2024-06-10: High level scraping for well-establish websites such as Google, Twitter, and Wikipedia. Also has NLP, machine learning algorithms, and visualization
* [twitterscraper](https://github.com/taspinar/twitterscraper) ⭐ 2,456 | 🐛 144 | 🌐 Python | 📅 2022-10-05: Efficient library to scrape Twitter
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): The easiest library to scrape static websites for beginners
* [Scrapy](https://scrapy.org/): Fast and extensible scraping library. Can write rules and create customized scraper without touching the core
* [Selenium](https://selenium-python.readthedocs.io/installation.html#introduction): Use Selenium Python API to access all functionalities of Selenium WebDriver in an intuitive way like a real user.

## Spatial Analysis

* [GeoPandas](https://github.com/geopandas/geopandas) ⭐ 5,043 | 🐛 430 | 🌐 Python | 📅 2026-02-10 - Python tools for geographic data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [PySal](https://github.com/pysal/pysal) ⭐ 1,470 | 🐛 32 | 🌐 Python | 📅 2026-02-01 - Python Spatial Analysis Library.

## Quantum Computing

* [qiskit](https://github.com/Qiskit/qiskit) ⭐ 7,047 | 🐛 1,113 | 🌐 Python | 📅 2026-02-16 - Qiskit is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.
* [cirq](https://github.com/quantumlib/Cirq) ⭐ 4,871 | 🐛 137 | 🌐 Python | 📅 2026-02-17 - A python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits.
* [PennyLane](https://github.com/XanaduAI/pennylane) ⭐ 3,077 | 🐛 445 | 🌐 Python | 📅 2026-02-16 - Quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
* [QML](https://github.com/qmlcode/qml) ⚠️ Archived - A Python Toolkit for Quantum Machine Learning.

## Conversion

* [ONNX](https://github.com/onnx/onnx) ⭐ 20,336 | 🐛 288 | 🌐 Python | 📅 2026-02-16 - Open Neural Network Exchange.
* [MMdnn](https://github.com/Microsoft/MMdnn) ⭐ 5,818 | 🐛 337 | 🌐 Python | 📅 2025-08-07 -  A set of tools to help users inter-operate among different deep learning frameworks.
* [sklearn-porter](https://github.com/nok/sklearn-porter) ⭐ 1,307 | 🐛 47 | 🌐 Python | 📅 2024-06-12 - Transpile trained scikit-learn estimators to C, Java, JavaScript, and others.
* [treelite](https://github.com/dmlc/treelite) ⭐ 812 | 🐛 13 | 🌐 C++ | 📅 2026-02-16 - Universal model exchange and serialization format for decision tree forests.

## Contributing

Contributions are welcome! :sunglasses: </br>
Read the \<a href=[https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution](https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution) ⭐ 3,346 | 🐛 6 | 📅 2026-02-13 guideline</a>.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License - [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
