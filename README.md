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
  * [Kernel Methods](#kernel-methods)

* [Deep Learning](#deep-learning)
  * [PyTorch](#pytorch)
  * [TensorFlow](#tensorflow)
  * [Keras](#keras)
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

* [TabGAN](https://github.com/Diyago/Tabular-data-generation) ⭐ 565 | 🐛 0 | 🌐 Python | 📅 2026-03-29 - Synthetic tabular data generation using GANs, Diffusion Models, and LLMs. <img height="16" width="16" src="https://github.com/krzjoa/awesome-python-data-science/raw/master/img/sklearn_big.png" alt="sklearn">

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

* [dlib](https://github.com/davisking/dlib) ⭐ 14,362 | 🐛 44 | 🌐 C++ | 📅 2026-03-29 - Toolkit for making real-world machine learning and data analysis applications in C++ (Python bindings).
* [PyCaret](https://github.com/pycaret/pycaret) ⭐ 9,743 | 🐛 423 | 🌐 Jupyter Notebook | 📅 2025-04-21 - An open-source, low-code machine learning library in Python.  <img height="20" src="img/R_big.png" alt="R inspired lib">
* [causalml](https://github.com/uber/causalml) ⭐ 5,814 | 🐛 36 | 🌐 Python | 📅 2026-03-21 - Uplift modeling and causal inference with machine learning algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [mlpack](https://github.com/mlpack/mlpack) ⭐ 5,623 | 🐛 21 | 🌐 C++ | 📅 2026-04-15 - A scalable C++ machine learning library (Python bindings).
* [cuML](https://github.com/rapidsai/cuml) ⭐ 5,177 | 🐛 850 | 🌐 C++ | 📅 2026-04-14 - RAPIDS Machine Learning Library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [MLxtend](https://github.com/rasbt/mlxtend) ⭐ 5,131 | 🐛 159 | 🌐 Python | 📅 2026-01-24 - Extension and helper modules for Python's data analysis and machine learning libraries. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [xLearn](https://github.com/aksnzhy/xlearn) ⭐ 3,097 | 🐛 194 | 🌐 C++ | 📅 2023-08-28 - High Performance, Easy-to-use, and Scalable Machine Learning Package.
* [Shogun](https://github.com/shogun-toolbox/shogun) ⭐ 3,069 | 🐛 423 | 🌐 C++ | 📅 2023-12-19 - Machine learning toolbox.
* [hyperlearn](https://github.com/danielhanchen/hyperlearn) ⭐ 2,433 | 🐛 2 | 🌐 Jupyter Notebook | 📅 2024-11-19 - 50%+ Faster, 50%+ less RAM usage, GPU support re-written Sklearn, Statsmodels. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [modAL](https://github.com/cosmic-cortex/modAL) ⭐ 2,345 | 🐛 108 | 🌐 Python | 📅 2024-02-26 - Modular active learning framework for Python3. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [metric-learn](https://github.com/all-umass/metric-learn) ⭐ 1,433 | 🐛 51 | 🌐 Python | 📅 2026-03-19 - Metric learning algorithms in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Sparkit-learn](https://github.com/lensacom/sparkit-learn) ⭐ 1,149 | 🐛 35 | 🌐 Python | 📅 2020-12-31 - PySpark + scikit-learn = Sparkit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [pyGAM](https://github.com/dswah/pyGAM) ⭐ 993 | 🐛 239 | 🌐 Python | 📅 2026-01-22 - Generalized Additive Models in Python.
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) ⭐ 955 | 🐛 92 | 🌐 Python | 📅 2024-02-01 - Multi-label classification for python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [seqlearn](https://github.com/larsmans/seqlearn) ⭐ 706 | 🐛 33 | 🌐 Python | 📅 2023-03-24 - Sequence classification toolkit for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Reproducible Experiment Platform (REP)](https://github.com/yandex/rep) ⭐ 700 | 🐛 32 | 🌐 Jupyter Notebook | 📅 2024-07-31 - Machine Learning toolbox for Humans. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pystruct](https://github.com/pystruct/pystruct) ⭐ 670 | 🐛 108 | 🌐 Python | 📅 2021-09-23 - Simple structured learning framework for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) ⭐ 490 | 🐛 5 | 🌐 Python | 📅 2017-08-11 - Highly interpretable classifiers for scikit learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [RuleFit](https://github.com/christophM/rulefit) ⭐ 444 | 🐛 28 | 🌐 Python | 📅 2023-10-08 - Implementation of the rulefit. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [SciPy](https://scipy.org/) - Fundamental algorithms for scientific computing in Python
* [scikit-learn](https://scikit-learn.org/stable/) - Machine learning in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Gradient Boosting

* [XGBoost](https://github.com/dmlc/xgboost) ⭐ 28,276 | 🐛 469 | 🌐 C++ | 📅 2026-04-15 - Scalable, Portable, and Distributed Gradient Boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [LightGBM](https://github.com/Microsoft/LightGBM) ⭐ 18,257 | 🐛 498 | 🌐 C++ | 📅 2026-04-14 - A fast, distributed, high-performance gradient boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [CatBoost](https://github.com/catboost/catboost) ⭐ 8,896 | 🐛 693 | 🌐 C++ | 📅 2026-04-14 - An open-source gradient boosting on decision trees library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [NGBoost](https://github.com/stanfordmlgroup/ngboost) ⭐ 1,863 | 🐛 51 | 🌐 Jupyter Notebook | 📅 2026-03-24 - Natural Gradient Boosting for Probabilistic Prediction.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) ⭐ 712 | 🐛 39 | 🌐 C++ | 📅 2025-03-19 - Fast GBDTs and Random Forests on GPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests) ⭐ 694 | 🐛 51 | 🌐 Python | 📅 2026-03-30 - A collection of state-of-the-art algorithms for the training, serving and interpretation of Decision Forest models in Keras. <img height="20" src="img/keras_big.png" alt="keras"> <img height="20" src="img/tf_big2.png" alt="TensorFlow">

### Ensemble Methods

* [vecstack](https://github.com/vecxoz/vecstack) ⭐ 700 | 🐛 0 | 🌐 Python | 📅 2025-11-01 - Python package for stacking (machine learning technique). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Stacking](https://github.com/ikki407/stacking) ⭐ 231 | 🐛 20 | 🌐 Python | 📅 2017-12-21 - Simple and useful stacking library written in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [stacked\_generalization](https://github.com/fukatani/stacked_generalization) ⭐ 119 | 🐛 3 | 🌐 Python | 📅 2019-05-02 - Library for machine learning stacking generalization. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ML-Ensemble](http://ml-ensemble.com/) - High performance ensemble learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Imbalanced Datasets

* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) ⭐ 7,095 | 🐛 65 | 🌐 Python | 📅 2026-04-13 - Module to perform under-sampling and over-sampling with various techniques. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) ⭐ 241 | 🐛 1 | 🌐 Python | 📅 2022-01-29 - Python-based implementations of algorithms for learning on imbalanced data. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">

### Kernel Methods

* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) ⭐ 1,622 | 🐛 87 | 🌐 C++ | 📅 2024-04-01 - A fast SVM Library on GPUs and CPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [fastFM](https://github.com/ibayer/fastFM) ⭐ 1,088 | 🐛 51 | 🌐 Python | 📅 2022-07-17 - A library for Factorization Machines. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyFM](https://github.com/coreylynch/pyFM) ⭐ 926 | 🐛 44 | 🌐 Python | 📅 2020-10-01 - Factorization machines in python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tffm](https://github.com/geffy/tffm) ⭐ 779 | 🐛 19 | 🌐 Jupyter Notebook | 📅 2022-01-17 - TensorFlow implementation of an arbitrary order Factorization Machine. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">
* [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) ⭐ 236 | 🐛 13 | 🌐 Python | 📅 2025-08-14 - Relevance Vector Machine implementation using the scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [liquidSVM](https://github.com/liquidSVM/liquidSVM) ⭐ 71 | 🐛 17 | 🌐 C++ | 📅 2020-02-20 - An implementation of SVMs.

## Deep Learning

### PyTorch

* [PyTorch](https://github.com/pytorch/pytorch) ⭐ 99,163 | 🐛 18,515 | 🌐 Python | 📅 2026-04-16 - Tensors and Dynamic neural networks in Python with strong GPU acceleration. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pytorch-lightning](https://github.com/Lightning-AI/lightning) ⭐ 31,051 | 🐛 980 | 🌐 Python | 📅 2026-04-13 - PyTorch Lightning is just organized PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [skorch](https://github.com/dnouri/skorch) ⭐ 6,155 | 🐛 66 | 🌐 Jupyter Notebook | 📅 2026-03-27 - A scikit-learn compatible neural network library that wraps PyTorch. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ignite](https://github.com/pytorch/ignite) ⭐ 4,753 | 🐛 186 | 🌐 Python | 📅 2026-04-15 - High-level library to help with training neural networks in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Catalyst](https://github.com/catalyst-team/catalyst) ⭐ 3,375 | 🐛 3 | 🌐 Python | 📅 2025-06-27 - High-level utils for PyTorch DL & RL research. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ChemicalX](https://github.com/AstraZeneca/chemicalx) ⭐ 776 | 🐛 10 | 🌐 Python | 📅 2023-09-11 - A PyTorch-based deep learning library for drug pair scoring. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">

### TensorFlow

* [TensorFlow](https://github.com/tensorflow/tensorflow) ⭐ 194,753 | 🐛 4,426 | 🌐 C++ | 📅 2026-04-16 - Computation using data flow graphs for scalable machine learning by Google. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Ludwig](https://github.com/uber/ludwig) ⭐ 11,670 | 🐛 5 | 🌐 Python | 📅 2026-04-15 - A toolbox that allows one to train and test deep learning models without the need to write code. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Sonnet](https://github.com/deepmind/sonnet) ⭐ 9,916 | 🐛 40 | 🌐 Python | 📅 2026-02-10 - TensorFlow-based neural network library. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TFLearn](https://github.com/tflearn/tflearn) ⭐ 9,590 | 🐛 579 | 🌐 Python | 📅 2024-05-06 - Deep learning library featuring a higher-level API for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) ⭐ 7,391 | 🐛 36 | 🌐 Python | 📅 2023-02-18 - Deep Learning and Reinforcement Learning Library for Researcher and Engineer. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorpack](https://github.com/ppwwyyxx/tensorpack) ⭐ 6,295 | 🐛 14 | 🌐 Python | 📅 2023-08-06 - A Neural Net Training Interface on TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorFlow Fold](https://github.com/tensorflow/fold) ⚠️ Archived - Deep learning with dynamic computation graphs in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Mesh TensorFlow](https://github.com/tensorflow/mesh) ⚠️ Archived - Model Parallelism Made Easier. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorflow-upstream](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream) ⭐ 702 | 🐛 95 | 🌐 C++ | 📅 2026-03-20 - TensorFlow ROCm port. <img height="20" src="img/tf_big2.png" alt="sklearn"> <img height="20" src="img/amd_big.png" alt="Possible to run on AMD GPU">
* [tfdeploy](https://github.com/riga/tfdeploy) ⭐ 355 | 🐛 12 | 🌐 Python | 📅 2025-01-04 - Deploy TensorFlow graphs for fast evaluation and export to TensorFlow-less environments running numpy. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLight](https://github.com/bsautermeister/tensorlight) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2022-10-06 - A high-level framework for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">

### JAX

* [JAX](https://github.com/google/jax) ⭐ 35,401 | 🐛 2,232 | 🌐 Python | 📅 2026-04-16 - Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
* [FLAX](https://github.com/google/flax) ⭐ 7,160 | 🐛 479 | 🌐 Jupyter Notebook | 📅 2026-04-15 - A neural network library for JAX that is designed for flexibility.
* [Optax](https://github.com/google-deepmind/optax) ⭐ 2,235 | 🐛 82 | 🌐 Python | 📅 2026-04-15 - A gradient processing and optimization library for JAX.

### Keras

* [Hyperas](https://github.com/maxpumperla/hyperas) ⭐ 2,177 | 🐛 97 | 🌐 Python | 📅 2023-01-05 - Keras + Hyperopt: A straightforward wrapper for a convenient hyperparameter. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [keras-contrib](https://github.com/keras-team/keras-contrib) ⚠️ Archived - Keras community contributions. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Elephas](https://github.com/maxpumperla/elephas) ⭐ 1,578 | 🐛 9 | 🌐 Python | 📅 2023-05-01 - Distributed Deep learning with Keras & Spark. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [qkeras](https://github.com/google/qkeras) ⭐ 580 | 🐛 49 | 🌐 Python | 📅 2026-02-23 - A quantization deep learning library. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Keras](https://keras.io) - A high-level neural networks API running on top of TensorFlow.  <img height="20" src="img/keras_big.png" alt="Keras compatible">

### Others

* [transformers](https://github.com/huggingface/transformers) ⭐ 159,454 | 🐛 2,345 | 🌐 Python | 📅 2026-04-16 - State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Caffe](https://github.com/BVLC/caffe) ⭐ 34,656 | 🐛 1,184 | 🌐 C++ | 📅 2024-07-31 - A fast open framework for deep learning.
* [autograd](https://github.com/HIPS/autograd) ⭐ 7,479 | 🐛 190 | 🌐 Python | 📅 2026-04-13 - Efficiently computes derivatives of numpy code.
* [nnabla](https://github.com/sony/nnabla) ⭐ 2,777 | 🐛 34 | 🌐 Python | 📅 2025-08-29 - Neural Network Libraries by Sony.
* [Tangent](https://github.com/google/tangent) ⚠️ Archived - Source-to-Source Debuggable Derivatives in Pure Python.

## Automated Machine Learning

* [AutoGluon](https://github.com/awslabs/autogluon) ⭐ 10,223 | 🐛 399 | 🌐 Python | 📅 2026-04-15 - AutoML for Image, Text, Tabular, Time-Series, and MultiModal Data.
* [TPOT](https://github.com/rhiever/tpot) ⭐ 10,041 | 🐛 309 | 🌐 Jupyter Notebook | 📅 2025-09-11 - AutoML tool that optimizes machine learning pipelines using genetic programming. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AutoKeras](https://github.com/keras-team/autokeras) ⭐ 9,315 | 🐛 158 | 🌐 Python | 📅 2025-11-25 - AutoML library for deep learning. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [auto-sklearn](https://github.com/automl/auto-sklearn) ⭐ 8,081 | 🐛 208 | 🌐 Python | 📅 2026-01-20 - An AutoML toolkit and a drop-in replacement for a scikit-learn estimator. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) ⭐ 2,536 | 🐛 75 | 🌐 Python | 📅 2024-04-09 - Automatic architecture search and hyperparameter optimization for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [MLBox](https://github.com/AxeldeRomblay/MLBox) ⭐ 1,530 | 🐛 28 | 🌐 Python | 📅 2023-08-06 - A powerful Automated Machine Learning python library.

## Natural Language Processing

* [NLTK](https://github.com/nltk/nltk) ⭐ 14,589 | 🐛 287 | 🌐 Python | 📅 2026-04-10 -  Modules, data sets, and tutorials supporting research and development in Natural Language Processing.
* [flair](https://github.com/zalandoresearch/flair) ⭐ 14,370 | 🐛 34 | 🌐 Python | 📅 2025-10-27 - Very simple framework for state-of-the-art NLP.
* [torchtext](https://github.com/pytorch/text) ⚠️ Archived - Data loaders and abstractions for text and NLP. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Phonemizer](https://github.com/bootphon/phonemizer) ⭐ 1,532 | 🐛 47 | 🌐 Python | 📅 2024-09-26 - Simple text-to-phonemes converter for multiple languages.
* [KerasNLP](https://github.com/keras-team/keras-nlp) ⭐ 973 | 🐛 239 | 🌐 Python | 📅 2026-04-15 - Modular Natural Language Processing workflows with Keras. <img height="20" src="img/keras_big.png" alt="Keras based/compatible">
* [CLTK](https://github.com/cltk/cltk) ⭐ 903 | 🐛 5 | 🌐 Python | 📅 2026-02-12 - The Classical Language Toolkik.
* [skift](https://github.com/shaypal5/skift) ⭐ 233 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2022-06-07 - Scikit-learn wrappers for Python fastText. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyMorfologik](https://github.com/dmirecki/pyMorfologik) ⭐ 18 | 🐛 1 | 🌐 Python | 📅 2015-08-15 - Python binding for <a href="https://github.com/morfologik/morfologik-stemming">Morfologik</a>.
* [spaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing.
* [gensim](https://radimrehurek.com/gensim/) - Topic Modelling for Humans.

## Computer Audition

* [librosa](https://github.com/librosa/librosa) ⭐ 8,329 | 🐛 79 | 🌐 Python | 📅 2026-03-24 - Python library for audio and music analysis.
* [aubio](https://github.com/aubio/aubio) ⭐ 3,686 | 🐛 154 | 🌐 C | 📅 2026-04-10 - A library for audio and music analysis.
* [Essentia](https://github.com/MTG/essentia) ⭐ 3,517 | 🐛 416 | 🌐 C++ | 📅 2026-02-09 - Library for audio and music analysis, description, and synthesis.
* [torchaudio](https://github.com/pytorch/audio) ⭐ 2,865 | 🐛 322 | 🌐 Python | 📅 2026-04-15 - An audio library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [madmom](https://github.com/CPJKU/madmom) ⭐ 1,623 | 🐛 79 | 🌐 Python | 📅 2026-03-20 - Python audio and music signal processing library.
* [Marsyas](https://github.com/marsyas/marsyas) ⭐ 424 | 🐛 36 | 🌐 C++ | 📅 2023-04-19 - Music Analysis, Retrieval, and Synthesis for Audio Signals.
* [Yaafe](https://github.com/Yaafe/Yaafe) ⭐ 248 | 🐛 17 | 🌐 C++ | 📅 2021-06-21 - Audio features extraction.
* [muda](https://github.com/bmcfee/muda) ⭐ 237 | 🐛 9 | 🌐 Python | 📅 2021-05-03 - A library for augmenting annotated audio data.
* [LibXtract](https://github.com/jamiebullock/LibXtract) ⭐ 230 | 🐛 1 | 🌐 C++ | 📅 2026-04-09 - A simple, portable, lightweight library of audio feature extraction functions.

## Computer Vision

* [OpenCV](https://github.com/opencv/opencv) ⭐ 87,133 | 🐛 2,739 | 🌐 C++ | 📅 2026-04-15 - Open Source Computer Vision Library.
* [torchvision](https://github.com/pytorch/vision) ⭐ 17,635 | 🐛 1,197 | 🌐 Python | 📅 2026-04-15 - Datasets, Transforms, and Models specific to Computer Vision. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [albumentations](https://github.com/albu/albumentations) ⚠️ Archived - Fast image augmentation library and easy-to-use wrapper around other libraries.
* [imgaug](https://github.com/aleju/imgaug) ⭐ 14,733 | 🐛 309 | 🌐 Python | 📅 2024-07-30 - Image augmentation for machine learning experiments.
* [LAVIS](https://github.com/salesforce/LAVIS) ⭐ 11,198 | 🐛 498 | 🌐 Jupyter Notebook | 📅 2024-11-18 - A One-stop Library for Language-Vision Intelligence.
* [PyTorch3D](https://github.com/facebookresearch/pytorch3d) ⭐ 9,853 | 🐛 317 | 🌐 Python | 📅 2026-03-18 - PyTorch3D is FAIR's library of reusable components for deep learning with 3D data. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [scikit-image](https://github.com/scikit-image/scikit-image) ⭐ 6,495 | 🐛 899 | 🌐 Python | 📅 2026-04-14 - Image Processing SciKit (Toolbox for SciPy).
* [Augmentor](https://github.com/mdbloice/Augmentor) ⭐ 5,138 | 🐛 140 | 🌐 Python | 📅 2024-03-21 - Image augmentation library in Python for machine learning.
* [Decord](https://github.com/dmlc/decord) ⭐ 2,460 | 🐛 216 | 🌐 C++ | 📅 2024-07-17 - An efficient video loader for deep learning with smart shuffling that's super easy to digest.
* [MMEngine](https://github.com/open-mmlab/mmengine) ⭐ 1,466 | 🐛 257 | 🌐 Python | 📅 2025-12-23 - OpenMMLab Foundational Library for Training Deep Learning Models. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [KerasCV](https://github.com/keras-team/keras-cv) ⚠️ Archived - Industry-strength Computer Vision workflows with Keras. <img height="20" src="img/keras_big.png" alt="MXNet based">
* [imgaug\_extension](https://github.com/cadenai/imgaug_extension) - Additional augmentations for imgaug.

## Time Series

* [Prophet](https://github.com/facebook/prophet) ⭐ 20,125 | 🐛 461 | 🌐 Python | 📅 2026-04-08 - Automatic Forecasting Procedure.
* [sktime](https://github.com/alan-turing-institute/sktime) ⭐ 9,735 | 🐛 2,138 | 🌐 Python | 📅 2026-04-15 - A unified framework for machine learning with time series. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [darts](https://github.com/unit8co/darts) ⭐ 9,327 | 🐛 244 | 🌐 Python | 📅 2026-04-15 - A python library for easy manipulation and forecasting of time series.
* [statsforecast](https://github.com/Nixtla/statsforecast) ⭐ 4,760 | 🐛 140 | 🌐 Python | 📅 2026-04-15 - Lightning fast forecasting with statistical and econometric models.
* [neuralforecast](https://github.com/Nixtla/neuralforecast) ⭐ 4,050 | 🐛 63 | 🌐 Python | 📅 2026-04-15 - Scalable machine learning-based time series forecasting.
* [maya](https://github.com/timofurrer/maya) ⭐ 3,415 | 🐛 21 | 🌐 Python | 📅 2024-07-19 - makes it very easy to parse a string and for changing timezones
* [tslearn](https://github.com/rtavenar/tslearn) ⭐ 3,139 | 🐛 92 | 🌐 Python | 📅 2026-04-10 - Machine learning toolkit dedicated to time-series data. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [PyFlux](https://github.com/RJT1990/pyflux) ⭐ 2,141 | 🐛 93 | 🌐 Python | 📅 2023-10-24 - Open source time series library for Python.
* [greykite](https://github.com/linkedin/greykite) ⭐ 1,854 | 🐛 13 | 🌐 Python | 📅 2025-02-20 - A flexible, intuitive, and fast forecasting library next.
* [skforecast](https://github.com/JoaquinAmatRodrigo/skforecast) ⭐ 1,477 | 🐛 21 | 🌐 Python | 📅 2026-04-15 - Time series forecasting with machine learning models
* [luminol](https://github.com/linkedin/luminol) ⭐ 1,229 | 🐛 35 | 🌐 Python | 📅 2025-08-22 - Anomaly Detection and Correlation library.
* [mlforecast](https://github.com/Nixtla/mlforecast) ⭐ 1,210 | 🐛 15 | 🌐 Python | 📅 2026-04-16 - Scalable machine learning-based time series forecasting.
* [Chaos Genius](https://github.com/chaos-genius/chaos_genius) ⚠️ Archived - ML powered analytics engine for outlier/anomaly detection and root cause analysis
* [tick](https://github.com/X-DataInitiative/tick) ⭐ 541 | 🐛 87 | 🌐 Python | 📅 2024-11-27 - Module for statistical learning, with a particular emphasis on time-dependent modeling.  <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [bayesloop](https://github.com/christophmark/bayesloop) ⭐ 169 | 🐛 4 | 🌐 Python | 📅 2026-02-09 - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models.
* [dateutil](https://dateutil.readthedocs.io/en/stable/) - Powerful extensions to the standard datetime module

## Reinforcement Learning

* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) ⭐ 13,089 | 🐛 81 | 🌐 Python | 📅 2026-04-01 - A set of improved implementations of reinforcement learning algorithms based on OpenAI Baselines.
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) ⭐ 11,718 | 🐛 84 | 🌐 Python | 📅 2026-04-12 - An API standard for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly [Gym](https://github.com/openai/gym) ⚠️ Archived).
* [Dopamine](https://github.com/google/dopamine) ⭐ 10,872 | 🐛 109 | 🌐 Jupyter Notebook | 📅 2026-03-24 - A research framework for fast prototyping of reinforcement learning algorithms.
* [Tianshou](https://github.com/thu-ml/tianshou/#comprehensive-functionality) ⭐ 10,546 | 🐛 137 | 🌐 Python | 📅 2026-04-03 - An elegant PyTorch deep reinforcement learning library. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐ 9,579 | 🐛 97 | 🌐 Python | 📅 2026-04-15 - High-quality single file implementation of Deep Reinforcement Learning algorithms with research-friendly features (PPO, DQN, C51, DDPG, TD3, SAC, PPG).
* [keras-rl](https://github.com/keras-rl/keras-rl) ⭐ 5,558 | 🐛 48 | 🌐 Python | 📅 2023-09-17 - Deep Reinforcement Learning for Keras. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Acme](https://github.com/google-deepmind/acme) ⭐ 3,968 | 🐛 98 | 🌐 Python | 📅 2026-04-08 - A library of reinforcement learning components and agents.
* [Horizon](https://github.com/facebookresearch/Horizon) ⭐ 3,695 | 🐛 85 | 🌐 Python | 📅 2026-04-13 - A platform for Applied Reinforcement Learning.
* [DI-engine](https://github.com/opendilab/DI-engine) ⭐ 3,618 | 🐛 26 | 🌐 Python | 📅 2025-12-07 - OpenDILab Decision AI Engine. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) ⭐ 3,373 | 🐛 59 | 🌐 Python | 📅 2026-02-06 - An API standard for multi-agent reinforcement learning environments, with popular reference environments and related utilities.
* [TensorForce](https://github.com/reinforceio/tensorforce) ⭐ 3,309 | 🐛 44 | 🌐 Python | 📅 2024-07-31 - A TensorFlow library for applied reinforcement learning. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [TRFL](https://github.com/deepmind/trfl) ⭐ 3,134 | 🐛 6 | 🌐 Python | 📅 2022-12-08 - TensorFlow Reinforcement Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TF-Agents](https://github.com/tensorflow/agents) ⭐ 3,007 | 🐛 211 | 🌐 Python | 📅 2026-01-16 - A library for Reinforcement Learning in TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [rlpyt](https://github.com/astooke/rlpyt) ⭐ 2,274 | 🐛 63 | 🌐 Python | 📅 2021-01-04 - Reinforcement Learning in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [garage](https://github.com/rlworkgroup/garage) ⭐ 2,094 | 🐛 234 | 🌐 Python | 📅 2023-05-04 - A toolkit for reproducible reinforcement learning research.
* [Imitation](https://github.com/HumanCompatibleAI/imitation) ⭐ 1,721 | 🐛 95 | 🌐 Python | 📅 2025-01-07 - Clean PyTorch implementations of imitation and reward learning algorithms. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [d3rlpy](https://github.com/takuseno/d3rlpy) ⭐ 1,657 | 🐛 59 | 🌐 Python | 📅 2025-09-10 - An offline deep reinforcement learning library.
* [EnvPool](https://github.com/sail-sg/envpool) ⭐ 1,324 | 🐛 20 | 🌐 C++ | 📅 2026-04-16 - C++-based high-performance parallel environment execution engine (vectorized env) for general RL environments.
* [SKRL](https://github.com/Toni-SM/skrl) ⭐ 1,027 | 🐛 26 | 🌐 Python | 📅 2026-04-08 - Modular reinforcement learning library (on PyTorch and JAX) with support for NVIDIA Isaac Gym, Isaac Orbit and Omniverse Isaac Gym. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Machin](https://github.com/iffiX/machin) ⭐ 419 | 🐛 1 | 🌐 Python | 📅 2021-08-08 -  A reinforcement library designed for pytorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [MAgent2](https://github.com/Farama-Foundation/MAgent2) ⭐ 329 | 🐛 23 | 🌐 C++ | 📅 2025-11-16 - An engine for high performance multi-agent environments with very large numbers of agents, along with a set of reference environments.
* [Shimmy](https://github.com/Farama-Foundation/Shimmy) ⭐ 207 | 🐛 5 | 🌐 Python | 📅 2026-04-10 - An API conversion tool for popular external reinforcement learning environments.
* [Catalyst-RL](https://github.com/catalyst-team/catalyst-rl) ⭐ 48 | 🐛 4 | 🌐 Python | 📅 2021-09-13 - PyTorch framework for RL research. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) - Scalable Reinforcement Learning.

## Graph Machine Learning

* [pytorch\_geometric](https://github.com/rusty1s/pytorch_geometric) ⭐ 23,675 | 🐛 1,260 | 🌐 Python | 📅 2026-04-08 - Geometric Deep Learning Extension Library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [TRL](https://github.com/huggingface/trl) ⭐ 18,060 | 🐛 688 | 🌐 Python | 📅 2026-04-15 - Train transformer language models with reinforcement learning.
* [dgl](https://github.com/dmlc/dgl) ⭐ 14,265 | 🐛 603 | 🌐 Python | 📅 2025-07-31 - Python package built to ease deep learning on graph, on top of existing DL frameworks. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/tf_big2.png" alt="TensorFlow"> <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [Graph Nets](https://github.com/google-deepmind/graph_nets) ⭐ 5,397 | 🐛 9 | 🌐 Python | 📅 2022-12-12 - Build Graph Nets in Tensorflow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) ⚠️ Archived - Generate embeddings from large-scale graph-structured data. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [StellarGraph](https://github.com/stellargraph/stellargraph) ⭐ 3,052 | 🐛 326 | 🌐 Python | 📅 2024-04-10 - Machine Learning on Graphs. <img height="20" src="img/tf_big2.png" alt="TensorFlow">  <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [pytorch\_geometric\_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) ⭐ 2,980 | 🐛 30 | 🌐 Python | 📅 2025-09-18 - Temporal Extension Library for PyTorch Geometric. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Spektral](https://github.com/danielegrattarola/spektral) ⭐ 2,396 | 🐛 74 | 🌐 Python | 📅 2024-01-21 - Deep learning on graphs. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Karate Club](https://github.com/benedekrozemberczki/karateclub) ⭐ 2,275 | 🐛 12 | 🌐 Python | 📅 2024-07-17 - An unsupervised machine learning library for graph-structured data.
* [TensorFlow GNN](https://github.com/tensorflow/gnn) ⭐ 1,526 | 🐛 45 | 🌐 Python | 📅 2026-03-30 - A library to build Graph Neural Networks on the TensorFlow platform. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [Jraph](https://github.com/google-deepmind/jraph) ⚠️ Archived - A Graph Neural Network Library in Jax.
* [Auto Graph Learning](https://github.com/THUMNLab/AutoGL) ⭐ 1,137 | 🐛 20 | 🌐 Python | 📅 2025-11-20 -An autoML framework & toolkit for machine learning on graphs.
* [Auto Graph Learning](https://github.com/THUMNLab/AutoGL) ⭐ 1,137 | 🐛 20 | 🌐 Python | 📅 2025-11-20 - An autoML framework & toolkit for machine learning on graphs.
* [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) ⭐ 713 | 🐛 7 | 🌐 Python | 📅 2025-12-20 - A library for sampling graph structured data.
* [GRAPE](https://github.com/AnacletoLAB/grape/tree/main) ⭐ 626 | 🐛 37 | 🌐 Jupyter Notebook | 📅 2024-02-24 - GRAPE is a Rust/Python Graph Representation Learning library for Predictions and Evaluations
* [Cleora](https://github.com/BaseModelAI/cleora) ⭐ 537 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-04-02 - The Graph Embedding Engine.
* [PyTorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed) ⭐ 147 | 🐛 0 | 🌐 Python | 📅 2026-04-15 -  A signed/directed graph neural network extension library for PyTorch Geometric. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [GreatX](https://github.com/EdisonLeeeee/GreatX) ⭐ 89 | 🐛 3 | 🌐 Python | 📅 2024-10-15 - A graph reliability toolbox based on PyTorch and PyTorch Geometric (PyG). <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">

## Graph Manipulation

* [Networkx](https://github.com/networkx/networkx) ⭐ 16,820 | 🐛 346 | 🌐 Python | 📅 2026-04-15 - Network Analysis in Python.
* [Rustworkx](https://github.com/Qiskit/rustworkx) ⭐ 1,637 | 🐛 134 | 🌐 Rust | 📅 2026-04-15 - A high performance Python graph library implemented in Rust.
* [igraph](https://github.com/igraph/python-igraph) ⭐ 1,442 | 🐛 64 | 🌐 Python | 📅 2026-04-06 - Python interface for igraph.
* [graph-tool](https://graph-tool.skewed.de/) - an efficient Python module for manipulation and statistical analysis of graphs (a.k.a. networks).

## Learning-to-Rank & Recommender Systems

* [Surprise](https://github.com/NicolasHug/Surprise) ⭐ 6,776 | 🐛 92 | 🌐 Python | 📅 2025-07-24 - A Python scikit for building and analyzing recommender systems.
* [LightFM](https://github.com/lyst/lightfm) ⭐ 5,076 | 🐛 165 | 🌐 Python | 📅 2024-07-24 - A Python implementation of LightFM, a hybrid recommendation algorithm.
* [RecBole](https://github.com/RUCAIBox/RecBole) ⭐ 4,385 | 🐛 358 | 🌐 Python | 📅 2025-02-24 - A unified, comprehensive and efficient recommendation library. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [TensorFlow Ranking](https://github.com/tensorflow/ranking) ⚠️ Archived - Learning to Rank in TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow">
* [TensorFlow Recommenders](https://github.com/tensorflow/recommenders) ⭐ 2,007 | 🐛 281 | 🌐 Python | 📅 2026-01-23 - A library for building recommender system models using TensorFlow. <img height="20" src="img/tf_big2.png" alt="TensorFlow"> <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [allRank](https://github.com/allegro/allRank) ⭐ 1,000 | 🐛 18 | 🌐 Python | 📅 2024-08-06 - allRank is a framework for training learning-to-rank neural models based on PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Spotlight](https://maciejkula.github.io/spotlight/) - Deep recommender models using PyTorch.

## Probabilistic Graphical Models

* [pomegranate](https://github.com/jmschrei/pomegranate) ⭐ 3,517 | 🐛 41 | 🌐 Python | 📅 2025-03-06 - Probabilistic and graphical models for Python. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pgmpy](https://github.com/pgmpy/pgmpy) ⭐ 3,250 | 🐛 551 | 🌐 Python | 📅 2026-04-14 - A python library for working with Probabilistic Graphical Models.
* [pyAgrum](https://agrum.gitlab.io/) - A GRaphical Universal Modeler.

## Probabilistic Methods

* [PyMC](https://github.com/pymc-devs/pymc) ⭐ 9,576 | 🐛 506 | 🌐 Python | 📅 2026-04-15 - Bayesian Stochastic Modelling in Python.
* [pyro](https://github.com/uber/pyro) ⭐ 8,994 | 🐛 275 | 🌐 Python | 📅 2025-07-09 - A flexible, scalable deep probabilistic programming library built on PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [GPyTorch](https://github.com/cornellius-gp/gpytorch) ⭐ 3,870 | 🐛 405 | 🌐 Python | 📅 2026-04-12 - A highly efficient and modular implementation of Gaussian Processes in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [emcee](https://github.com/dfm/emcee) ⭐ 1,576 | 🐛 67 | 🌐 Python | 📅 2026-03-16 - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [pyhsmm](https://github.com/mattjj/pyhsmm) ⭐ 575 | 🐛 46 | 🌐 Python | 📅 2025-01-25 - Bayesian inference in HSMMs and HMMs.
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes) ⭐ 523 | 🐛 20 | 🌐 Jupyter Notebook | 📅 2021-09-22 - Python package for Bayesian Machine Learning with scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) ⭐ 436 | 🐛 42 | 🌐 Python | 📅 2026-04-08 - A scikit-learn-inspired API for CRFsuite. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [PyStan](https://github.com/stan-dev/pystan) ⭐ 364 | 🐛 15 | 🌐 Python | 📅 2026-03-12 - Bayesian inference using the No-U-Turn sampler (Python interface).
* [PyVarInf](https://github.com/ctallec/pyvarinf) ⭐ 362 | 🐛 4 | 🌐 Python | 📅 2019-10-12 - Bayesian Deep Learning methods with Variational Inference for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [skpro](https://github.com/alan-turing-institute/skpro) ⭐ 323 | 🐛 257 | 🌐 Python | 📅 2026-04-12 - Supervised domain-agnostic prediction framework for probabilistic modelling by [The Alan Turing Institute](https://www.turing.ac.uk/). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [InferPy](https://github.com/PGM-Lab/InferPy) ⭐ 148 | 🐛 65 | 🌐 Jupyter Notebook | 📅 2024-08-02 - Deep Probabilistic Modelling Made Easy.  <img height="20" src="img/tf_big2.png" alt="sklearn">
* [hsmmlearn](https://github.com/jvkersch/hsmmlearn) ⚠️ Archived - A library for hidden semi-Markov models with explicit durations.
* [ZhuSuan](https://zhusuan.readthedocs.io/en/latest/) - Bayesian Deep Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [GPflow](https://gpflow.readthedocs.io/en/latest/?badge=latest) - Gaussian processes in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">

## Model Explanation

* [Netron](https://github.com/lutzroeder/Netron) ⭐ 32,758 | 🐛 18 | 🌐 JavaScript | 📅 2026-04-14 - Visualizer for deep learning and machine learning models (no Python code, but visualizes models from most Python Deep Learning frameworks).
* [shap](https://github.com/slundberg/shap) ⭐ 25,301 | 🐛 764 | 🌐 Jupyter Notebook | 📅 2026-04-15 - A unified approach to explain the output of any machine learning model. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Lime](https://github.com/marcotcr/lime) ⭐ 12,118 | 🐛 132 | 🌐 JavaScript | 📅 2024-07-25 - Explaining the predictions of any machine learning classifier. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) ⭐ 7,986 | 🐛 84 | 🌐 Python | 📅 2026-04-10 - Tensorboard for PyTorch (and chainer, mxnet, numpy, ...).
* [InterpretML](https://github.com/interpretml/interpret) ⭐ 6,830 | 🐛 113 | 🌐 C++ | 📅 2026-04-16 - InterpretML implements the Explainable Boosting Machine (EBM), a modern, fully interpretable machine learning model based on Generalized Additive Models (GAMs). This open-source package also provides visualization tools for EBMs, other glass-box models, and black-box explanations. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [lucid](https://github.com/tensorflow/lucid) ⚠️ Archived - A collection of infrastructure and tools for research in neural network interpretability.
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) ⭐ 4,399 | 🐛 113 | 🌐 Python | 📅 2025-02-19 - Visual analysis and diagnostic tools to facilitate machine learning model selection. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ELI5](https://github.com/TeamHG-Memex/eli5) ⭐ 2,777 | 🐛 162 | 🌐 Jupyter Notebook | 📅 2026-04-08 - A library for debugging/inspecting machine learning classifiers and explaining their predictions.
* [Alibi](https://github.com/SeldonIO/alibi) ⭐ 2,623 | 🐛 157 | 🌐 Python | 📅 2025-10-17 - Algorithms for monitoring and explaining machine learning models.
* [scikit-plot](https://github.com/reiinakano/scikit-plot) ⭐ 2,433 | 🐛 32 | 🌐 Python | 📅 2024-08-20 - An intuitive library to add plotting functionality to scikit-learn objects. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AI Explainability 360](https://github.com/IBM/AIX360) ⭐ 1,774 | 🐛 56 | 🌐 Python | 📅 2026-03-18 - Interpretability and explainability of data and machine learning models.
* [dalex](https://github.com/ModelOriented/DALEX) ⭐ 1,463 | 🐛 30 | 🌐 Python | 📅 2026-01-20 - moDel Agnostic Language for Exploration and explanation. <img height="20" src="img/sklearn_big.png" alt="sklearn"><img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [model-analysis](https://github.com/tensorflow/model-analysis) ⭐ 1,265 | 🐛 38 | 🌐 Python | 📅 2025-08-06 - Model analysis tools for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PDPbox](https://github.com/SauceCat/PDPbox) ⭐ 861 | 🐛 33 | 🌐 Jupyter Notebook | 📅 2024-09-03 - Partial dependence plot toolbox.
* [anchor](https://github.com/marcotcr/anchor) ⭐ 813 | 🐛 26 | 🌐 Jupyter Notebook | 📅 2022-07-19 - Code for "High-Precision Model-Agnostic Explanations" paper.
* [treeinterpreter](https://github.com/andosa/treeinterpreter) ⭐ 761 | 🐛 26 | 🌐 Python | 📅 2023-07-18 - Interpreting scikit-learn's decision tree and random forest predictions. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [aequitas](https://github.com/dssg/aequitas) ⭐ 758 | 🐛 55 | 🌐 Python | 📅 2026-02-03 - Bias and Fairness Audit Toolkit.
* [CapsNet-Visualization](https://github.com/bourdakos1/CapsNet-Visualization) ⭐ 395 | 🐛 16 | 🌐 Python | 📅 2021-10-05 - A visualization of the CapsNet layers to better understand how it works.
* [FairML](https://github.com/adebayoj/fairml) ⭐ 366 | 🐛 14 | 🌐 Python | 📅 2021-05-10 - FairML is a python toolbox auditing the machine learning models for bias. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Shapley](https://github.com/benedekrozemberczki/shapley) ⭐ 224 | 🐛 1 | 🌐 Python | 📅 2026-01-01 - A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
* [PyCEbox](https://github.com/AustinRochford/PyCEbox) ⭐ 163 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2020-05-29 - Python Individual Conditional Expectation Plot Toolbox.
* [themis-ml](https://github.com/cosmicBboy/themis-ml) ⚠️ Archived - A library that implements fairness-aware machine learning algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [L2X](https://github.com/Jianbo-Lab/L2X) ⭐ 124 | 🐛 3 | 🌐 Python | 📅 2021-05-10 - Code for replicating the experiments in the paper *Learning to Explain: An Information-Theoretic Perspective on Model Interpretation*.
* [Contrastive Explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) ⭐ 45 | 🐛 2 | 🌐 Python | 📅 2023-01-31 - Contrastive Explanation (Foil Trees). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Auralisation](https://github.com/keunwoochoi/Auralisation) ⭐ 42 | 🐛 0 | 🌐 Python | 📅 2017-03-11 - Auralisation of learned features in CNN (for audio).
* [Skater](https://github.com/datascienceinc/Skater) - Python Library for Model Interpretation.
* [FlashLight](https://github.com/dlguys/flashlight) - Visualization Tool for your NeuralNetwork.

## Genetic Programming

* [DEAP](https://github.com/DEAP/deap) ⭐ 6,378 | 🐛 286 | 🌐 Python | 📅 2025-11-16 - Distributed Evolutionary Algorithms in Python.
* [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) ⭐ 2,189 | 🐛 101 | 🌐 Python | 📅 2026-04-13 - Genetic Algorithm in Python. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible"> <img height="20" src="img/keras_big.png" alt="keras">
* [gplearn](https://github.com/trevorstephens/gplearn) ⭐ 1,834 | 🐛 17 | 🌐 Python | 📅 2026-01-10 - Genetic Programming in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic) ⭐ 325 | 🐛 10 | 🌐 Python | 📅 2024-01-20 - Genetic feature selection module for scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [karoo\_gp](https://github.com/kstaats/karoo_gp) ⭐ 164 | 🐛 22 | 🌐 Python | 📅 2022-10-31 - A Genetic Programming platform for Python with GPU support. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [monkeys](https://github.com/hchasestevens/monkeys) ⭐ 125 | 🐛 4 | 🌐 Python | 📅 2018-06-13 - A strongly-typed genetic programming framework for Python.

<a name="opt"></a>

## Optimization

* [Optuna](https://github.com/optuna/optuna) ⭐ 13,969 | 🐛 29 | 🌐 Python | 📅 2026-04-15 - A hyperparameter optimization framework.
* [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) ⭐ 8,598 | 🐛 6 | 🌐 Python | 📅 2026-04-01 - A Python implementation of global optimization with gaussian processes.
* [hyperopt](https://github.com/hyperopt/hyperopt) ⭐ 7,590 | 🐛 7 | 🌐 Python | 📅 2026-03-16 - Distributed Asynchronous Hyperparameter Optimization in Python.
* [scikit-opt](https://github.com/guofei9987/scikit-opt) ⭐ 6,404 | 🐛 70 | 🌐 Python | 📅 2026-03-25 - Heuristic Algorithms for optimization.
* [BoTorch](https://github.com/pytorch/botorch) ⭐ 3,503 | 🐛 92 | 🌐 Jupyter Notebook | 📅 2026-04-13 - Bayesian optimization in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pymoo](https://github.com/anyoptimization/pymoo) ⭐ 2,843 | 🐛 16 | 🌐 Python | 📅 2026-02-22 - Multi-objective Optimization in Python.
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) ⚠️ Archived - Sequential model-based optimization with a `scipy.optimize` interface.
* [POT](https://github.com/rflamary/POT) ⭐ 2,781 | 🐛 51 | 🌐 Python | 📅 2026-03-11 - Python Optimal Transport library.
* [nlopt](https://github.com/stevengj/nlopt) ⭐ 2,205 | 🐛 100 | 🌐 C | 📅 2026-03-13 - Library for nonlinear optimization (global and local, constrained or unconstrained).
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) ⭐ 1,647 | 🐛 78 | 🌐 Python | 📅 2025-04-15 - Hyper-parameter optimization for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Talos](https://github.com/autonomio/talos) ⭐ 1,636 | 🐛 13 | 🌐 Python | 📅 2024-04-22 - Hyperparameter Optimization for Keras Models.
* [Spearmint](https://github.com/HIPS/Spearmint) ⭐ 1,568 | 🐛 77 | 🌐 Python | 📅 2019-12-27 - Bayesian optimization.
* [PySwarms](https://github.com/ljvmiranda921/pyswarms) ⭐ 1,388 | 🐛 41 | 🌐 Python | 📅 2024-08-06 - A research toolkit for particle swarm optimization in Python.
* [pycma](https://github.com/CMA-ES/pycma?tab=readme-ov-file) ⭐ 1,298 | 🐛 90 | 🌐 Jupyter Notebook | 📅 2026-02-25 - Python implementation of CMA-ES.
* [SMAC3](https://github.com/automl/SMAC3) ⭐ 1,219 | 🐛 120 | 🌐 Python | 📅 2026-04-10 - Sequential Model-based Algorithm Configuration.
* [sklearn-deap](https://github.com/rsteca/sklearn-deap) ⭐ 773 | 🐛 28 | 🌐 Jupyter Notebook | 📅 2024-02-10 - Use evolutionary algorithms instead of gridsearch in scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Platypus](https://github.com/Project-Platypus/Platypus) ⭐ 648 | 🐛 0 | 🌐 Python | 📅 2026-01-21 - A Free and Open Source Python Library for Multiobjective Optimization.
* [Solid](https://github.com/100/Solid) ⭐ 585 | 🐛 7 | 🌐 Python | 📅 2019-07-19 - A comprehensive gradient-free optimization framework written in Python.
* [Optunity](https://github.com/claesenm/optunity) ⭐ 425 | 🐛 52 | 🌐 Jupyter Notebook | 📅 2023-11-25 - Is a library containing various optimizers for hyperparameter tuning.
* [sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt) ⭐ 360 | 🐛 2 | 🌐 Python | 📅 2026-03-31 - Hyperparameters tuning and feature selection using evolutionary algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [GPflowOpt](https://github.com/GPflow/GPflowOpt) ⭐ 274 | 🐛 29 | 🌐 Python | 📅 2020-12-02 - Bayesian Optimization using GPflow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [SafeOpt](https://github.com/befelix/SafeOpt) ⭐ 150 | 🐛 2 | 🌐 Python | 📅 2022-11-14 - Safe Bayesian Optimization.
* [sigopt\_sklearn](https://github.com/sigopt/sigopt_sklearn) ⚠️ Archived - SigOpt wrappers for scikit-learn methods. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [OR-Tools](https://developers.google.com/optimization) - An open-source software suite for optimization by Google; provides a unified programming interface to a half dozen solvers: SCIP, GLPK, GLOP, CP-SAT, CPLEX, and Gurobi.

## Feature Engineering

### General

* [tsfresh](https://github.com/blue-yonder/tsfresh) ⭐ 9,173 | 🐛 72 | 🌐 Jupyter Notebook | 📅 2025-11-15 - Automatic extraction of relevant features from time series. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Featuretools](https://github.com/Featuretools/featuretools) ⭐ 7,629 | 🐛 161 | 🌐 Python | 📅 2026-02-03 - Automated feature engineering.
* [Feature Engine](https://github.com/feature-engine/feature_engine) ⭐ 2,226 | 🐛 80 | 🌐 Python | 📅 2026-03-28 - Feature engineering package with sklearn-like functionality. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) ⭐ 872 | 🐛 24 | 🌐 Python | 📅 2024-05-27 - Automated feature generation with expert-level performance.
* [Feature Forge](https://github.com/machinalis/featureforge) ⭐ 385 | 🐛 11 | 🌐 Python | 📅 2017-12-26 - A set of tools for creating and testing machine learning features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-mdr](https://github.com/EpistasisLab/scikit-mdr) ⭐ 126 | 🐛 11 | 🌐 Python | 📅 2025-06-10 - A sklearn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [NitroFE](https://github.com/NITRO-AI/NitroFE) ⭐ 108 | 🐛 0 | 🌐 Python | 📅 2022-05-04 - Moving window features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tubular](https://github.com/azukds/tubular) ⭐ 100 | 🐛 75 | 🌐 Python | 📅 2026-04-14 - Collection of scikit-learn compatible transformers written in [narwhals](https://github.com/narwhals-dev/narwhals) ⭐ 1,590 | 🐛 196 | 🌐 Python | 📅 2026-04-15, which can accept either polars/pandas inputs and utilise the chosen library under the hood. <img height="20" src="img/sklearn_big.png" alt="sklearn"><img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [few](https://github.com/lacava/few) ⭐ 52 | 🐛 9 | 🌐 Python | 📅 2020-06-11 - A feature engineering wrapper for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [skl-groups](https://github.com/dougalsutherland/skl-groups) ⭐ 41 | 🐛 25 | 🌐 Python | 📅 2016-08-08 - A scikit-learn addon to operate on set/"group"-based features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [dirty\_cat](https://github.com/dirty-cat/dirty_cat) ⭐ 20 | 🐛 0 | 🌐 Python | 📅 2025-03-12 - Machine learning on dirty tabular data (especially: string-based variables for classifcation and regression). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sk-transformer](https://github.com/chrislemke/sk-transformers) ⚠️ Archived - A collection of various pandas & scikit-learn compatible transformers for all kinds of preprocessing and feature engineering steps <img height="20" src="img/pandas_big.png" alt="pandas compatible">

### Feature Selection

* [boruta\_py](https://github.com/scikit-learn-contrib/boruta_py) ⭐ 1,622 | 🐛 48 | 🌐 Python | 📅 2025-11-13 - Implementations of the Boruta all-relevant feature selection method. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-feature](https://github.com/jundongl/scikit-feature) ⭐ 1,572 | 🐛 45 | 🌐 Python | 📅 2024-07-11 - Feature selection repository in Python.
* [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) ⭐ 422 | 🐛 18 | 🌐 Python | 📅 2023-02-10 - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [zoofs](https://github.com/jaswinder9051998/zoofs) ⭐ 254 | 🐛 22 | 🌐 Python | 📅 2026-01-09 - A feature selection library based on evolutionary algorithms.
* [BoostARoota](https://github.com/chasedehan/BoostARoota) ⭐ 234 | 🐛 7 | 🌐 Python | 📅 2021-04-01 - A fast xgboost feature selection algorithm. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Visualization

### General Purposes

* [Matplotlib](https://github.com/matplotlib/matplotlib) ⭐ 22,718 | 🐛 1,504 | 🌐 Python | 📅 2026-04-13 - Plotting with Python.
* [seaborn](https://github.com/mwaskom/seaborn) ⭐ 13,812 | 🐛 214 | 🌐 Python | 📅 2026-01-22 - Statistical data visualization using matplotlib.
* [missingno](https://github.com/ResidentMario/missingno) ⭐ 4,199 | 🐛 15 | 🌐 Python | 📅 2024-05-14 - Missing data visualization module for Python.
* [chartify](https://github.com/spotify/chartify/) ⭐ 3,628 | 🐛 54 | 🌐 Python | 📅 2024-10-16 - Python library that makes it easy for data scientists to create charts.
* [prettyplotlib](https://github.com/olgabot/prettyplotlib) ⭐ 1,705 | 🐛 11 | 🌐 Python | 📅 2019-01-31 - Painlessly create beautiful matplotlib plots.
* [python-ternary](https://github.com/marcharper/python-ternary) ⭐ 781 | 🐛 37 | 🌐 Python | 📅 2024-06-12 - Ternary plotting library for Python with matplotlib.
* [physt](https://github.com/janpipek/physt) ⭐ 136 | 🐛 40 | 🌐 Python | 📅 2026-03-19 - Improved histograms.

### Interactive plots

* [Bokeh](https://github.com/bokeh/bokeh) ⭐ 20,379 | 🐛 864 | 🌐 TypeScript | 📅 2026-04-15 - Interactive Web Plotting for Python.
* [pyecharts](https://github.com/pyecharts/pyecharts) ⭐ 15,751 | 🐛 3 | 🌐 Python | 📅 2026-04-14 - Migrated from [Echarts](https://github.com/apache/echarts) ⭐ 66,156 | 🐛 1,721 | 🌐 TypeScript | 📅 2026-04-11, a charting and visualization library, to Python's interactive visual drawing library.<img height="20" src="img/pyecharts.png" alt="pyecharts"> <img height="20" src="img/echarts.png" alt="echarts">
* [bqplot](https://github.com/bqplot/bqplot) ⭐ 3,688 | 🐛 281 | 🌐 TypeScript | 📅 2026-01-23 - Plotting library for IPython/Jupyter notebooks
* [animatplot](https://github.com/t-makaro/animatplot) ⭐ 417 | 🐛 17 | 🌐 Python | 📅 2024-08-29 - A python package for animating plots built on matplotlib.
* [plotly](https://plot.ly/python/) - A Python library that makes interactive and publication-quality graphs.
* [Altair](https://altair-viz.github.io/) - Declarative statistical visualization library for Python. Can easily do many data transformation within the code to create graph

### Map

* [folium](https://python-visualization.github.io/folium/quickstart.html#Getting-Started) - Makes it easy to visualize data on an interactive open street map
* [geemap](https://github.com/giswqs/geemap) ⭐ 3,926 | 🐛 39 | 🌐 Python | 📅 2026-04-16 - Python package for interactive mapping with Google Earth Engine (GEE)

### Automatic Plotting

* [SweetViz](https://github.com/fbdesignpro/sweetviz) ⭐ 3,091 | 🐛 41 | 🌐 Python | 📅 2026-04-11: Visualize and compare datasets, target values and associations, with one line of code.
* [HoloViews](https://github.com/ioam/holoviews) ⭐ 2,888 | 🐛 1,067 | 🌐 Python | 📅 2026-04-14 - Stop plotting your data - annotate your data and let it visualize itself.
* [AutoViz](https://github.com/AutoViML/AutoViz) ⭐ 1,894 | 🐛 2 | 🌐 Python | 📅 2024-06-10: Visualize data automatically with 1 line of code (ideal for machine learning)

### NLP

* [pyLDAvis](https://github.com/bmabey/pyLDAvis) ⭐ 1,846 | 🐛 81 | 🌐 Jupyter Notebook | 📅 2025-12-04: Visualize interactive topic model

## Deployment

* [gradio](https://github.com/gradio-app/gradio) ⭐ 42,340 | 🐛 465 | 🌐 Python | 📅 2026-04-15 - Create UIs for your machine learning model in Python in 3 minutes.
* [Vizro](https://github.com/mckinsey/vizro) ⭐ 3,661 | 🐛 73 | 🌐 Python | 📅 2026-04-15 - A toolkit for creating modular data visualization applications.
* [Deepnote](https://github.com/deepnote/deepnote) ⭐ 2,827 | 🐛 19 | 🌐 TypeScript | 📅 2026-04-14 - Deepnote is a drop-in replacement for Jupyter with an AI-first design, sleek UI, new blocks, and native data integrations. Use Python, R, and SQL locally in your favorite IDE, then scale to Deepnote cloud for real-time collaboration, Deepnote agent, and deployable data apps.
* [streamsync](https://github.com/streamsync-cloud/streamsync) ⭐ 1,439 | 🐛 12 | 🌐 Python | 📅 2026-04-07 - No-code in the front, Python in the back. An open-source framework for creating data apps.
* [fastapi](https://fastapi.tiangolo.com/) - Modern, fast (high-performance), a web framework for building APIs with Python
* [streamlit](https://www.streamlit.io/) - Make it easy to deploy the machine learning model
* [datapane](https://datapane.com/) - A collection of APIs to turn scripts and notebooks into interactive reports.
* [binder](https://mybinder.org/) - Enable sharing and execute Jupyter Notebooks

## Statistics

* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) ⭐ 13,507 | 🐛 319 | 🌐 Python | 📅 2026-04-13 - Create HTML profiling reports from pandas DataFrame objects. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [statsmodels](https://github.com/statsmodels/statsmodels) ⭐ 11,364 | 🐛 2,960 | 🌐 Python | 📅 2026-04-15 - Statistical modeling and econometrics in Python.
* [Alphalens](https://github.com/quantopian/alphalens) ⭐ 4,219 | 🐛 49 | 🌐 Jupyter Notebook | 📅 2024-02-12 - Performance analysis of predictive (alpha) stock factors.
* [stockstats](https://github.com/jealous/stockstats) ⭐ 1,458 | 🐛 15 | 🌐 Python | 📅 2026-03-30 - Supply a wrapper `StockDataFrame` based on the `pandas.DataFrame` with inline stock statistics/indicators support.
* [pandas\_summary](https://github.com/mouradmourafiq/pandas-summary) ⭐ 531 | 🐛 7 | 🌐 Python | 📅 2026-04-13 - Extension to pandas dataframes describe function. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) ⭐ 380 | 🐛 1 | 🌐 Python | 📅 2026-02-17 - Pairwise Multiple Comparisons Post-hoc Tests.
* [weightedcalcs](https://github.com/jsvine/weightedcalcs) ⭐ 113 | 🐛 3 | 🌐 Python | 📅 2024-11-10 - A pandas-based utility to calculate weighted means, medians, distributions, standard deviations, and more.

## Data Manipulation

### Data Frames

* [polars](https://github.com/pola-rs/polars) ⭐ 38,193 | 🐛 2,722 | 🌐 Rust | 📅 2026-04-15 - A fast multi-threaded, hybrid-out-of-core DataFrame library.
* [pandas\_profiling](https://github.com/pandas-profiling/pandas-profiling) ⭐ 13,507 | 🐛 319 | 🌐 Python | 📅 2026-04-13 - Create HTML profiling reports from pandas DataFrame objects
* [modin](https://github.com/modin-project/modin) ⭐ 10,380 | 🐛 709 | 🌐 Python | 📅 2026-02-10 - Speed up your pandas workflows by changing a single line of code. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [cuDF](https://github.com/rapidsai/cudf) ⭐ 9,605 | 🐛 1,238 | 🌐 C++ | 📅 2026-04-16 - GPU DataFrame Library. <img height="20" src="img/pandas_big.png" alt="pandas compatible"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [vaex](https://github.com/vaexio/vaex) ⭐ 8,498 | 🐛 552 | 🌐 Python | 📅 2026-04-01 - Out-of-Core DataFrames for Python, ML, visualize and explore big tabular data at a billion rows per second.
* [xarray](https://github.com/pydata/xarray) ⭐ 4,131 | 🐛 1,343 | 🌐 Python | 📅 2026-04-13 - Xarray combines the best features of NumPy and pandas for multidimensional data selection by supplementing numerical axis labels with named dimensions for more intuitive, concise, and less error-prone indexing routines.
* [blaze](https://github.com/blaze/blaze) ⭐ 3,194 | 🐛 268 | 🌐 Python | 📅 2023-09-29 - NumPy and pandas interface to Big Data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Arctic](https://github.com/manahl/arctic) ⭐ 3,091 | 🐛 97 | 🌐 Python | 📅 2024-04-08 - High-performance datastore for time series and tick data.
* [swifter](https://github.com/jmcarpenter2/swifter) ⭐ 2,640 | 🐛 26 | 🌐 Python | 📅 2024-03-20 - A package that efficiently applies any function to a pandas dataframe or series in the fastest available manner.
* [datatable](https://github.com/h2oai/datatable) ⭐ 1,881 | 🐛 182 | 🌐 C++ | 📅 2025-03-17 - Data.table for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [pandasql](https://github.com/yhat/pandasql) ⚠️ Archived -  Allows you to query pandas DataFrames using SQL syntax. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pandas-gbq](https://github.com/pydata/pandas-gbq) ⚠️ Archived - pandas Google Big Query. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pysparkling](https://github.com/svenkreiss/pysparkling) ⭐ 271 | 🐛 11 | 🌐 Python | 📅 2024-09-03 - A pure Python implementation of Apache Spark's RDD and DStream interfaces. <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [pandas-log](https://github.com/eyaltrabelsi/pandas-log) ⭐ 218 | 🐛 11 | 🌐 Python | 📅 2021-06-26 - A package that allows providing feedback about basic pandas operations and finds both business logic and performance issues.
* [xpandas](https://github.com/alan-turing-institute/xpandas) ⭐ 26 | 🐛 10 | 🌐 Python | 📅 2022-06-21 - Universal 1d/2d data containers with Transformers .functionality for data analysis by [The Alan Turing Institute](https://www.turing.ac.uk/).
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - Powerful Python data analysis toolkit.

### Pipelines

* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) ⭐ 2,852 | 🐛 43 | 🌐 Python | 📅 2023-06-08 - pandas integration with sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Hamilton](https://github.com/DAGWorks-Inc/hamilton) ⭐ 2,453 | 🐛 148 | 🌐 Jupyter Notebook | 📅 2026-04-16 - A microframework for dataframe generation that applies Directed Acyclic Graphs specified by a flow of lazily evaluated Python functions.
* [pyjanitor](https://github.com/ericmjl/pyjanitor) ⭐ 1,487 | 🐛 108 | 🌐 Python | 📅 2026-04-11 - Clean APIs for data cleaning. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Dplython](https://github.com/dodger487/dplython) ⭐ 760 | 🐛 28 | 🌐 Python | 📅 2016-12-30 - Dplyr for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [pdpipe](https://github.com/shaypal5/pdpipe) ⭐ 724 | 🐛 13 | 🌐 Jupyter Notebook | 📅 2026-04-06 - Sasy pipelines for pandas DataFrames.
* [dopanda](https://github.com/dovpanda-dev/dovpanda) ⭐ 480 | 🐛 29 | 🌐 Python | 📅 2024-12-01 -  Hints and tips for using pandas in an analysis environment. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [meza](https://github.com/reubano/meza) ⭐ 421 | 🐛 12 | 🌐 Python | 📅 2025-02-27 - A Python toolkit for processing tabular data.
* [Dataset](https://github.com/analysiscenter/dataset) ⭐ 205 | 🐛 40 | 🌐 Python | 📅 2026-04-14 - Helps you conveniently work with random or sequential batches of your data and define data processing.
* [pandas-ply](https://github.com/coursera/pandas-ply) ⭐ 197 | 🐛 4 | 🌐 HTML | 📅 2015-08-27 - Functional data manipulation for pandas. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Prodmodel](https://github.com/prodmodel/prodmodel) ⭐ 58 | 🐛 8 | 🌐 Python | 📅 2026-04-13 - Build system for data science pipelines.
* [SSPipe](https://sspipe.github.io/) - Python pipe (|) operator with support for DataFrames and Numpy, and Pytorch.

### Data-centric AI

* [cleanlab](https://github.com/cleanlab/cleanlab) ⭐ 11,431 | 🐛 103 | 🌐 Python | 📅 2026-01-13 - The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.
* [snorkel](https://github.com/snorkel-team/snorkel) ⭐ 5,951 | 🐛 16 | 🌐 Python | 📅 2026-04-10 - A system for quickly generating training data with weak supervision.
* [dataprep](https://github.com/sfu-db/dataprep) ⭐ 2,238 | 🐛 165 | 🌐 Python | 📅 2024-06-27 - Collect, clean, and visualize your data in Python with a few lines of code.

### Synthetic Data

* [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) ⭐ 1,620 | 🐛 66 | 🌐 Jupyter Notebook | 📅 2026-04-15 - A package to generate synthetic tabular and time-series data leveraging the state-of-the-art generative models. <img height="20" src="img/pandas_big.png" alt="pandas compatible">

## Distributed Computing

* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ⭐ 23,827 | 🐛 1,503 | 🌐 C++ | 📅 2026-04-16 - PArallel Distributed Deep LEarning.
* [Horovod](https://github.com/uber/horovod) ⭐ 14,691 | 🐛 406 | 🌐 Python | 📅 2025-12-01 - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [DMTK](https://github.com/Microsoft/DMTK) ⚠️ Archived - Microsoft Distributed Machine Learning Toolkit.
* [Distributed](https://github.com/dask/distributed) ⭐ 1,668 | 🐛 1,510 | 🌐 Python | 📅 2026-04-15 - Distributed computation in Python.
* [dask-ml](https://github.com/dask/dask-ml) ⭐ 944 | 🐛 283 | 🌐 Python | 📅 2025-09-27 - Distributed and parallel machine learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Veles](https://github.com/Samsung/veles) ⭐ 916 | 🐛 36 | 🌐 C++ | 📅 2023-11-21 - Distributed machine learning platform.
* [Jubatus](https://github.com/jubatus/jubatus) ⭐ 709 | 🐛 4 | 🌐 C++ | 📅 2019-05-16 - Framework and Library for Distributed Online Machine Learning.
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Exposes the Spark programming model to Python. <img height="20" src="img/spark_big.png" alt="Apache Spark based">

## Experimentation

* [mlflow](https://github.com/mlflow/mlflow) ⭐ 25,392 | 🐛 2,004 | 🌐 Python | 📅 2026-04-16 - Open source platform for the machine learning lifecycle.
* [dvc](https://github.com/iterative/dvc) ⭐ 15,541 | 🐛 167 | 🌐 Python | 📅 2026-04-14 - Data Version Control | Git for Data & Models | ML Experiments Management.
* [Sacred](https://github.com/IDSIA/sacred) ⭐ 4,360 | 🐛 106 | 🌐 Python | 📅 2025-10-22 - A tool to help you configure, organize, log, and reproduce experiments.
* [Ax](https://github.com/facebook/Ax) ⭐ 2,734 | 🐛 191 | 🌐 Python | 📅 2026-04-15 - Adaptive Experimentation Platform. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [envd](https://github.com/tensorchord/envd) ⭐ 2,195 | 🐛 131 | 🌐 Go | 📅 2026-04-10 - 🏕️ machine learning development environment for data science and AI/ML engineering teams.
* [Neptune](https://neptune.ai) - A lightweight ML experiment tracking, results visualization, and management tool.

## Data Validation

* [great\_expectations](https://github.com/great-expectations/great_expectations) ⭐ 11,406 | 🐛 54 | 🌐 Python | 📅 2026-04-16 - Always know what to expect from your data.
* [evidently](https://github.com/evidentlyai/evidently) ⭐ 7,390 | 🐛 263 | 🌐 Jupyter Notebook | 📅 2026-03-30 - Evaluate and monitor ML models from validation to production.
* [pandera](https://github.com/unionai-oss/pandera) ⭐ 4,308 | 🐛 447 | 🌐 Python | 📅 2026-04-16 - A lightweight, flexible, and expressive statistical data testing library.
* [deepchecks](https://github.com/deepchecks/deepchecks) ⭐ 4,002 | 🐛 261 | 🌐 Python | 📅 2025-12-28 - Validation & testing of ML models and data during model development, deployment, and production. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [TensorFlow Data Validation](https://github.com/tensorflow/data-validation) ⭐ 779 | 🐛 38 | 🌐 Python | 📅 2026-03-23 - Library for exploring and validating machine learning data.
* [DataComPy](https://github.com/capitalone/datacompy) ⭐ 639 | 🐛 11 | 🌐 Python | 📅 2026-04-09- A library to compare Pandas, Polars, and Spark data frames. It provides stats and lets users adjust for match accuracy.

## Evaluation

* [AI Fairness 360](https://github.com/IBM/AIF360) ⭐ 2,786 | 🐛 218 | 🌐 Python | 📅 2025-11-13 - Fairness metrics for datasets and ML models, explanations, and algorithms to mitigate bias in datasets and models.
* [alibi-detect](https://github.com/SeldonIO/alibi-detect) ⭐ 2,512 | 🐛 140 | 🌐 Jupyter Notebook | 📅 2025-12-11 - Algorithms for outlier, adversarial and drift detection.<img height="20" src="img/alibi-detect.png" alt="sklearn">
* [Metrics](https://github.com/benhamner/Metrics) ⭐ 1,650 | 🐛 37 | 🌐 Python | 📅 2023-01-11 - Machine learning evaluation metric.
* [recmetrics](https://github.com/statisticianinstilettos/recmetrics) ⭐ 582 | 🐛 20 | 🌐 Jupyter Notebook | 📅 2024-01-11 - Library of useful metrics and plots for evaluating recommender systems.
* [sklearn-evaluation](https://github.com/edublancas/sklearn-evaluation) ⭐ 3 | 🐛 0 | 📅 2023-01-15 - Model evaluation made easy: plots, tables, and markdown reports. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Computations

* [Dask](https://github.com/dask/dask) ⭐ 13,804 | 🐛 1,243 | 🌐 Python | 📅 2026-04-13 - Parallel computing with task scheduling. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [CuPy](https://github.com/cupy/cupy) ⭐ 10,895 | 🐛 673 | 🌐 Python | 📅 2026-04-15 - NumPy-like API accelerated with CUDA.
* [NumExpr](https://github.com/pydata/numexpr) ⭐ 2,412 | 🐛 2 | 🌐 Python | 📅 2026-03-02 - A fast numerical expression evaluator for NumPy that comes with an integrated computing virtual machine to speed calculations up by avoiding memory allocation for intermediate results.
* [adaptive](https://github.com/python-adaptive/adaptive) ⭐ 1,218 | 🐛 105 | 🌐 Python | 📅 2026-04-13 - Tools for adaptive and parallel samping of mathematical functions.
* [bottleneck](https://github.com/kwgoodman/bottleneck) ⭐ 1,177 | 🐛 59 | 🌐 Python | 📅 2026-04-14 - Fast NumPy array functions written in C.
* [quaternion](https://github.com/moble/quaternion) ⭐ 656 | 🐛 14 | 🌐 Python | 📅 2025-12-15 - Add built-in support for quaternions to numpy.
* [scikit-tensor](https://github.com/mnick/scikit-tensor) ⭐ 405 | 🐛 27 | 🌐 Python | 📅 2018-08-23 - Python library for multilinear algebra and tensor factorizations.
* [numdifftools](https://github.com/pbrod/numdifftools) ⭐ 280 | 🐛 6 | 🌐 Python | 📅 2026-01-06 - Solve automatic numerical differentiation problems in one or more variables.
* [NumPy](https://numpy.org/) - The fundamental package for scientific computing with Python

## Web Scraping

* [Pattern](https://github.com/clips/pattern) ⭐ 8,859 | 🐛 178 | 🌐 Python | 📅 2024-06-10: High level scraping for well-establish websites such as Google, Twitter, and Wikipedia. Also has NLP, machine learning algorithms, and visualization
* [twitterscraper](https://github.com/taspinar/twitterscraper) ⭐ 2,458 | 🐛 144 | 🌐 Python | 📅 2022-10-05: Efficient library to scrape Twitter
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): The easiest library to scrape static websites for beginners
* [Scrapy](https://scrapy.org/): Fast and extensible scraping library. Can write rules and create customized scraper without touching the core
* [Selenium](https://selenium-python.readthedocs.io/installation.html#introduction): Use Selenium Python API to access all functionalities of Selenium WebDriver in an intuitive way like a real user.

## Spatial Analysis

* [GeoPandas](https://github.com/geopandas/geopandas) ⭐ 5,100 | 🐛 416 | 🌐 Python | 📅 2026-04-13 - Python tools for geographic data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [PySal](https://github.com/pysal/pysal) ⭐ 1,489 | 🐛 23 | 🌐 Python | 📅 2026-04-15 - Python Spatial Analysis Library.

## Quantum Computing

* [qiskit](https://github.com/Qiskit/qiskit) ⭐ 7,260 | 🐛 1,152 | 🌐 Python | 📅 2026-04-15 - Qiskit is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.
* [cirq](https://github.com/quantumlib/Cirq) ⭐ 4,927 | 🐛 124 | 🌐 Python | 📅 2026-04-15 - A python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits.
* [PennyLane](https://github.com/XanaduAI/pennylane) ⭐ 3,156 | 🐛 445 | 🌐 Python | 📅 2026-04-16 - Quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
* [QML](https://github.com/qmlcode/qml) ⚠️ Archived - A Python Toolkit for Quantum Machine Learning.

## Conversion

* [ONNX](https://github.com/onnx/onnx) ⭐ 20,660 | 🐛 315 | 🌐 Python | 📅 2026-04-15 - Open Neural Network Exchange.
* [MMdnn](https://github.com/Microsoft/MMdnn) ⭐ 5,812 | 🐛 337 | 🌐 Python | 📅 2025-08-07 -  A set of tools to help users inter-operate among different deep learning frameworks.
* [sklearn-porter](https://github.com/nok/sklearn-porter) ⭐ 1,309 | 🐛 47 | 🌐 Python | 📅 2024-06-12 - Transpile trained scikit-learn estimators to C, Java, JavaScript, and others.
* [treelite](https://github.com/dmlc/treelite) ⭐ 814 | 🐛 15 | 🌐 C++ | 📅 2026-04-13 - Universal model exchange and serialization format for decision tree forests.

## Contributing

Contributions are welcome! :sunglasses: </br>
Read the \<a href=[https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution](https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution) ⭐ 3,400 | 🐛 6 | 📅 2026-04-13 guideline</a>.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License - [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
