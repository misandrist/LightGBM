# LightGBM Model Manager

## Overview

This package provides a suite of command-line tools for training,
saving, and then reusing LightGBM models. The datasets, models, and
parameters are all managed through a declarative interface, and
multiple sets of models can be active at any time without conflict.

## Command Processing

All of the commands are run as subcommands of the `modelmgr`
script. The first argument is the command, and each command has it’s
own arguments after that.

In the following sections, the `command`s mentioned should be the
first argument to `modelmgr`. For example this command will train and cross-validate  models
model from the `CSV` file `input.csv`, validate it against the `CSV`
file `test.csv`. It will use a class named `SimpleModel`, which should
be the only class with its name in this case. The trained model data
is written to `simple-model-<metadata-hash>.mod.gz` as a [compressed
`joblib`
pickle](https://pythonhosted.org/joblib/persistence.html#compressed-joblib-pickles)
using [`gzip`
compression](https://pythonhosted.org/joblib/persistence.html#compressed-joblib-pickles). It
also persists the full module and class name of the model found, along
with the model class’s [feature flags](#feature-flags):

    modelmgr train \
        --dataset=data.csv \
        --crossvalidation=5 \
        --model=SimpleModel \
        --outfile=simple-model

## Training Models

To train a model, the `train` command will
