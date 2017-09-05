"""Manages loading data, training, persisting trained models, and using them.

The modules here implement the core workflow of the model
management. The Model metaclass maps pairs of (PRED, callable) into
separate graphs so several different trees of models can coexist and
be managed by the same persistence and feature tag machinery.

Feature Tags essentially are a set of values describing the features
implemented by a model subclass. The simple implementation in the base
template is to ensure that the target class has a superset of feature
tags, but this can readily be overridden by your own base class in the
check_forward_features method.

"""
