# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf

import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn
import pprint

flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)

DGX_SERVER = False

def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  tests_models = [
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'parameter_server', 'model': 'inception3'},
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'parameter_server', 'model': 'resnet50'},
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'parameter_server', 'model': 'resnet152'},
    {'num_gpus': None, 'batch_size': 512, 'variable_update': 'replicated', 'model': 'alexnet'},
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'replicated', 'model': 'vgg16'}
  ]
  test_gpus = [1, 4, 8]

  stats = []
  for test in tests_models:
    for num_gpus in test_gpus:
      test['num_gpus'] = num_gpus

      params = benchmark_cnn.make_params_from_flags()
      params = benchmark_cnn.setup(params)

      # params._replace(num_gpus=1, batch_size=32, model='resnet50', variable_update='cpu')
      params = params._replace(num_gpus=test['num_gpus'],
                               batch_size=test['batch_size'],
                               model=test['model'],
                               variable_update=test['variable_update'],
                               hierarchical_copy=DGX_SERVER)

      bench = benchmark_cnn.BenchmarkCNN(params)

      tfversion = cnn_util.tensorflow_version_tuple()
      log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

      bench.print_info()
      results = bench.run()

      stats.append({'test': test,
                    'result': results})

  # summary
  print('summary:')
  pprint.pprint(stats)
  # todo save results into a json file




if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
