#!/usr/bin/env python

"""
Benchmark script for TensorFlow.
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


def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.

  # For DGX servers use hierarchical_copy=True argument

  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  tests_models = [
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'parameter_server', 'model': 'inception3'},
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'parameter_server', 'model': 'resnet50'},
    {'num_gpus': None, 'batch_size': 32, 'variable_update': 'parameter_server', 'model': 'resnet152'}, #batch=64 crashes
    {'num_gpus': None, 'batch_size': 64, 'variable_update': 'replicated', 'model': 'vgg16'},
    {'num_gpus': None, 'batch_size': 512, 'variable_update': 'replicated', 'model': 'alexnet'}
  ]

  test_gpus = [1, 2, 4, 8]

  stats = []
  for test in tests_models:
    for num_gpus in test_gpus:
      test['num_gpus'] = num_gpus

      params = benchmark_cnn.make_params_from_flags()
      params = benchmark_cnn.setup(params)

      # force --hierarchical_copy to False when using 1 GPU
      if num_gpus == 1:
        params = params._replace(hierarchical_copy=False)

      params = params._replace(num_gpus=test['num_gpus'],
                               batch_size=test['batch_size'],
                               model=test['model'],
                               variable_update=test['variable_update']
                               )

      bench = benchmark_cnn.BenchmarkCNN(params)

      tfversion = cnn_util.tensorflow_version_tuple()
      log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

      bench.print_info()
      results = bench.run()
      # result
      # {
      #     'average_wall_time': 0.6646941304206848,
      #     'images_per_sec': 385.1395525908701,
      #     'last_average_loss': 7.256145,
      #     'num_steps': 100,
      #     'num_workers': 1
      # }
      stats.append({'test': test.copy(),
                    'result': results})


  # summary
  print('summary:')
  print('==========')
  pprint.pprint(stats)

  print('==========')
  s = ''
  for i in range(len(test_gpus)):
    for j in range(len(tests_models)):
      s += str(stats[i + j * len(test_gpus)]['result']['images_per_sec'])
      s += ', '
    s += '\n'
  print(s)
  print('==========')


if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
