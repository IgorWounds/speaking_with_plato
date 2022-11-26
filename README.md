# Exploring Plato's philosophy with AI - A Data Spiral blog article

<img src="https://th.bing.com/th/id/R.bb69a3e9d1613b4e039a0a3ef1a1a9f1?rik=c8DvRKRZiGeBkA&riu=http%3a%2f%2fetc.usf.edu%2fclipart%2f1200%2f1247%2fPlato_1_lg.gif&ehk=GGD4%2fvhA%2bQIJ3lJ6nUfS2hSl11F%2fqg7eipz9%2f%2fNbre0%3d&risl=&pid=ImgRaw&r=0" width="150" heigh="200"/> 

[![GitHub license](https://badgen.net/github/license/IgorWounds/speaking_with_plato)](https://github.com/IgorWounds/speaking_with_plat/blob/master/LICENSE)
[![GitHub stars](https://badgen.net/github/stars/IgorWounds/speaking_with_plato)](https://github.com/IgorWounds/speaking_with_plato)


## Table of Contents

- [Exploring Plato's philosophy with AI - A Data Spiral blog article](#exploring-platos-philosophy-with-ai---a-data-spiral-blog-article)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Contact](#contact)

## Introduction

This repository contains the code for the article [Speaking with Plato - A Deep Learning Approach to Philosophy](https://dataspiral.blog/speaking-with-plato/).

It features two deep learning models that are trained on Plato's works. The first one is a chatbot that can be used to have a conversation with Socrates. The second one is a text generator that can be used to generate new philosophical texts.

You can read more about the project in the article linked above.

## Project Structure
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models
    │   └── GPT-2          <- GPT-2 model trained on Plato's works
    |   └── small-dialouGPT <- DialoGPT model trained on Plato's dialogues
    │
    ├── notebooks          <- Jupyter notebooks (e.g. EDA)
    │   └── plato_complete_eda_11_18_22 <- Notebook used for data exploration
    |
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    |   └── files          <- Other files to be used in reporting (e.g. Embedding, HTML, etc.)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be |   imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py <- Script to download and process data
        |   └── prepare_data.py <- Script to prepare data for modeling
        |   └── file_names.json <- JSON file containing file names for each work
        |   └── __init__.py     <- Makes data a Python module
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── build_features.py <- Script to build features for Chatbot model
        |   └── download.py       <- Script to download models
        │   └── predict_chatbot_model.py
        |   └── predict_writebot_model.py
        |   └── train_chatbot_model.py
        |   └── train_writebot_model.py
        |   └── __init__.py <- Makes models a Python module
        │
        ├── visualization  <- Scripts to create exploratory and results oriented visualizations
        |    └── visualize.py
        |    └── __init__.py <- Makes visualization a Python module
        |
        └── app.py         <- Tkinter app for running the Chatbot and Writebot models
--------

## Setup

To run the app locally, clone the repository and run the following commands:

```bash
$ cd speaking_with_plato
$ pip install -r requirements.txt
$ pip install -U --no-cache-dir gdown --pre
$ python src/app.py
```

The first time you run the app, it will download the GPT-2 and DialoGPT models. This may take a few minutes depending on your internet connection. Make sure you have at least 7 GB of free space.

If you face issues with downloading the models, please download them manually from the following links and place them in their respective folders found inside the `models` folder and extract them:
* [DialoGPT](https://drive.google.com/file/d/1QLEF2KVXKvfroAmqrNQK4Q6G8qFcgSmG/view?usp=sharing) -> `models/small-dialouGPT/`
* [GPT-2](https://drive.google.com/file/d/15aQCUMY_UAD3bikl1MARxi7j4dTNY4Ia/view?usp=sharing) -> `models/GPT-2/`

## Usage

The app will open in a new window. You can use the main menu to select the model you want to run or change the parameters for. The Chatbot model is based on DialoGPT and the Writebot model is based on GPT-2. The Chatbot model is trained on Plato's dialogues and the Writebot model is trained on Plato's complete works. The Chatbot model is more conversational and the Writebot model is more creative.

To navigate within the app, simply click on the buttons. When you insert your prompt in either bot press the "Enter" key on your keyboard to generate a response. To go back to the main menu (this also resets the bot), click the "Back" button. To exit the app, click the "Exit" button.

## Contact

If you have any questions or comments, feel free to send me an email at [dataspiralblog@gmail.com](mailto:dataspiralblog@gmail.com).
