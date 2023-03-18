Bayesian Deep Learning for learned augmented data
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Advanced deep network architectures, robust computation, and access to big data are few of the factors that have contributed to the rapid progress of deep neural networks compare to conventional machine learning methods. However, the avail- ability of large annotated data is scarce in specific domains. As a result the training of such models becomes insuﬀicient.

The most popular technique to increase the size of a dataset is data augmentation. Such a process has been proven to be cumbersome and time consuming. Other methods have been developed where utilize the capabilites of generative adversarial networks (GANs) and reinforcement learning to automatically generate new samples that resemble the training data. These methods yet promising are unstable and diﬀicult to train.

One can perform data augmentation by finding the right transformations of the input images. The development of spatial transformer network (STN) which esti- mates image transformations has shown remarkable results. However, STNs are also diﬀicult to train and can be sensitive to incorrect transformations. In the study we apply the Bayesian framework using Laplace approximation to turn the STN into a probabilistic model to perform learned data augmentation while overcoming its aforementioned limitations.

The Laplace approximation method provides the flexibility to apply Bayesian in- ference solely on a small part of the model which reduces the computational cost significantly. The use of marginal likelihood allows for automatic parameter tuning neglecting any manual adjustments. This is a major advantage compared to other Bayesian approximation methods.

The probabilistic manner of STN gives us the capability to estimate numerous trans- formations rather than a deterministic one. Thus, we can evaluate every image in multiple postures, which will enhance the robustness of the training. The major benefit though is that those transformations behave as learned data augmentation scheme that improves the predictive performance of the model.

We demonstrate through a series of experiments on conventional and challenging image datasets that the probabilistic STN using Laplace approximation leads indeed to improved classification performance and better model calibration compared to the deterministic one.
