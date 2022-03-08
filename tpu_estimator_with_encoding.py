from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import tempfile

import numpy as np
import six

from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator import util
from tensorflow.python.estimator.export.export import build_all_signature_defs
from tensorflow.python.estimator.export.export import get_temp_export_dir
from tensorflow.python.estimator.export.export import get_timestamped_export_dir
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.summary import summary
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import device_setter
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.util import compat
from tensorflow.python.util import compat_internal
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.contrib import tpu as contrib_tpu

class TPUEstimatorWithEncoding(contrib_tpu.TPUEstimator):
    def __init__(self, sequence_output, *args, **kwargs):
        super(TPUEstimatorWithEncoding, self).__init__(*args, **kwargs)
        self.sequence_output = sequence_output

    def predict_with_encoding(self,
            input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None,
            yield_single_examples=True):
        """Yields predictions for given features.

        Args:
        input_fn: A function that constructs the features. Prediction continues
        until `input_fn` raises an end-of-input exception (`OutOfRangeError` or
        `StopIteration`).
        See @{$get_started/premade_estimators#create_input_functions} for more
        information. The function should construct and return one of
        the following:

            * A 'tf.data.Dataset' object: Outputs of `Dataset` object must have
            same constraints as below.
            * features: A `Tensor` or a dictionary of string feature name to
            `Tensor`. features are consumed by `model_fn`. They should satisfy
            the expectation of `model_fn` from inputs.
            * A tuple, in which case the first item is extracted as features.

        predict_keys: list of `str`, name of the keys to predict. It is used if
        the `EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used
        then rest of the predictions will be filtered from the dictionary. If
        `None`, returns all.
        hooks: List of `SessionRunHook` subclass instances. Used for callbacks
        inside the prediction call.
        checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
        latest checkpoint in `model_dir` is used.
        yield_single_examples: If False, yield the whole batch as returned by the
        `model_fn` instead of decomposing the batch into individual elements.
        This is useful if `model_fn` returns some tensors whose first dimension
        is not equal to the batch size.

        Yields:
        Evaluated values of `predictions` tensors.

        Raises:
        ValueError: Could not find a trained model in `model_dir`.
        ValueError: If batch length of predictions is not the same and
        `yield_single_examples` is True.
        ValueError: If there is a conflict between `predict_keys` and
        `predictions`. For example if `predict_keys` is not `None` but
        `EstimatorSpec.predictions` is not a `dict`.
        """
        hooks = _check_hooks_type(hooks)
        # Check that model has been trained.
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(self._model_dir)
        if not checkpoint_path:
            raise ValueError('Could not find trained model in model_dir: {}.'.format(
                self._model_dir))

        with ops.Graph().as_default() as g:
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, input_hooks = self._get_features_from_input_fn(
                input_fn, model_fn_lib.ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(
                features, None, model_fn_lib.ModeKeys.PREDICT, self.config)
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)
            all_hooks = list(input_hooks)
            all_hooks.extend(hooks)
            all_hooks.extend(list(estimator_spec.prediction_hooks or []))
            with training.MonitoredSession(
                session_creator=training.ChiefSessionCreator(
                    checkpoint_filename_with_path=checkpoint_path,
                    master=self._config.master,
                    scaffold=estimator_spec.scaffold,
                    config=self._session_config),
                hooks=all_hooks) as mon_sess:
                    while not mon_sess.should_stop():
                        preds_evaluated = mon_sess.run(predictions)
                        if not yield_single_examples:
                            yield preds_evaluated
                        elif not isinstance(predictions, dict):
                            for pred in preds_evaluated:
                                yield pred
                        else:
                            for i in range(self._extract_batch_length(preds_evaluated)):
                                yield {
                                    key: value[i]
                                    for key, value in six.iteritems(preds_evaluated)
                                }
