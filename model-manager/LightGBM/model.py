# coding: utf-8
# pylint: disable = invalid-name, C0111

import sys
import abc
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

class Dataset:
    def __init__(self, data):
        """Represents a data set."""

        self._y = data[0].values
        self._X = data.drop(0, axis=1).values

    @property
    def X(self):
        """The inputs."""
        return self._X

    @property
    def y(self):
        """The decision."""
        return self._y

    @property
    def dataset(self):
        """An lgb.Dataset from our data."""

        return lgb.Dataset(self.X, self.y)

class Evalset(Dataset):
    """A Dataset referencing another one for evaluating a model.

    The reference set is accessible through the X and y properties,
    while the evaluation set is accessible through the Xt and yt
    properties.

    """

    def __init__(self, test, data):
        """Create an evaluation Dataset referencing data."""

        super(Dataset, self).__init__(data)
        self._test=Dataset(test)

    @property
    def Xt(self):
        """The evaulation inputs."""
        return self._test._X

    @property
    def yt(self):
        """The evaluation decisions."""
        return self._test._y

    @property
    def evalset(self):
        """The evaluation lgb.Dataset"""

        ds = self._test
        lgb.Dataset(ds.X, ds.y, reference=self.dataset)

class Model(type):
    """The metaclass that manages the Model machinery.

    This defines the interface and state machinery for pickling and
    unpickling, the lifecycle that a Model passes through.

    """

    # Feature flags that get persisted with the pickle. We can
    # determine later whether a pickle is compatible with another.
    _features = ("STATES", "PICKLING", "TRAINING", "PARAMETERS")

    # State transition table. None represents the null state prior to
    # the initial state, or the null stat following the final state.
    _trans = {None, "UNTRAINED",
              "UNTRAINED": "TRAINED",
              "TRAINED": None}

    # This is a map between (state, pred_cls) pairs and target
    # classes. The target class state will be set automatically
    # according to _trans.
    _transfns = {}

    @classmethod
    def __prepare__(metacls, name, bases, pred, cpred, **kwds):
        """Copy the dictionary of our template class."""

        # Copy the template class dictionary.
        ns = metacls.MT.__dict__.copy()

        tstate = metacls._trans.get(pred, None)
        if tstate = None:
            raise Exception("%s: Invalid transition from %s to %s specified." % (name, pred))
        # Set the state it represents
        ns["state"] =

    def __new__(cls, name, bases, namespace, pred, cpred, **kwds):
        """Create a transition handler.

        Transition handlers are keyed by both the predecessor state
        and class so that multiple Model representations can use the
        same underlying mechanisms.

        """

        # First, get a new class instance.
        c = type.__new__(cls, name, bases, dict(namespace))
        Model._transfns[pred][cpred] =

    class MT:
        """Our template class that has the pickle mechanisms and the state
        machinery."""

        @property
        def tag(self):
            """The pickle version tag and state.

            We have this on the instance so that we're capturing the
            actual class used and not the base Model class.

            """

            c = self.__class__
            return (c.__module__, c.__name__, self._features, self._state)

        def next(self):
            """Transition to the next state in _states.

            The general mechanism is that a transition is effected by
            passing the callable of the next state the current object. If
            there is no next state, the current object is returned.

            """

            nst = self._trans.get(self.state, None)
            if nst is None:
                return self

            return self._transfns(self.state)(self)

        def __getstate__(self):
            """Prepare a versioned pickle state."""

            return (self.tag, self._model.__getstate__())

        def __setstate__(self, state):
            """Load a saved, versioned model."""

            (module, nm, vers, st), model = state

            tag = (module, nm, vers, st)

            if any((n != e) for n, e in zip(tag, self.tag)):
                raise Exception("%s tag doesn't match saved %s" % (self.tag, tag))

            m = object.__new__(lgb.Booster)
            m.__setstate__(model)

            self._model = m

class UntrainedModel(Model):
    def __init__(self, train_data, test_data, parameters):
        """Prepare a Model for training with the parameters.

        The model will not be trained initially so that the data and
        parameters can be saved independently of the tranied
        model.

        The default trained model representation is Trained, but this
        can be overridden by passing another callable as the trained
        parameter.

        """

        self._d_train = train_data
        self._d_test = test_data
        self._params = dict(parameters)
        self._trained = trained

    def train(self):
        """Train the model, return a TrainedModel."""

        return self._trained(
            lgb.train(self.params,
                      self.train,
                      num_boost_round=20,
                      valid_sets=self.test,
                      early_stopping_rounds=5)
        )

    @property
    def train(self):
        """The training Dataset."""
        return self._d_train.dataset

    @property
    def test(self):
        """The holdout test Dataset."""
        return self._d_test.dataset

    @property
    def model(self):
        """Return the trained model."""

        return self._model

def main():

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

Dataset(pd.read_csv('../regression/regression.train', header=None, sep='\t'))
Dataset(pd.read_csv('../regression/regression.test', header=None, sep='\t'))
