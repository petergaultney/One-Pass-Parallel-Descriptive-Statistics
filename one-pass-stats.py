#!/usr/bin/env python
# @author petergaultney
# Copyright 2015 Peter Gaultney
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from math import sqrt

class OnePassParallelDescriptiveStats(object):

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.min = float('inf')
        self.max = -float('inf')
        self.M2 = 0.0

    def addValue(self, value):
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

        self.count += 1
        delta = value - self.mean
        self.mean = self.mean + delta / self.count
        self.M2 = self.M2 + delta * (value - self.mean)

    def getVariance(self):
        if (self.count < 2):
            return 0.0
        else:
            return self.M2 / (self.count - 1) # sample variance

    def getStddev(self):
        return sqrt(self.getVariance())

    def mergeStats(self, B):
        """
        This is a non-destructive merge operation - it does not 
        alter the values in the calling object or the supplied object.
        It could just as easily be written as a class function.
        """
        merged = OnePassParallelDescriptiveStats()
        delta = B.mean - self.mean
        merged.count = self.count + B.count
        if merged.count > 0:
            merged.mean = (self.count * self.mean + B.count * B.mean) / merged.count
            merged.M2 = self.M2 + B.M2 + delta * delta * ((self.count * B.count) / merged.count)
        else:
            merged.mean = 0.0
            merged.M2 = 0.0

        # mins, maxes
        if B.max > self.max:
            merged.max = B.max
        else:
            merged.max = self.max
        if B.min < self.min:
            merged.min = B.min
        else:
            merged.min = self.min

        return merged

def op_stats_to_str(stats):
    return 'Mean {} | Count {} | Stddev {} | Min {} | Max {}'.format(
        stats.mean, stats.count, stats.getStddev(), stats.min, stats.max)

def test_single_set(norm_dist):
    import numpy as np
    print('Calculating descriptive statistics using the one-pass algorithm...')
    onepass_stats = OnePassParallelDescriptiveStats()
    for number in norm_dist:
        onepass_stats.addValue(number)

    print('The following OnePass and batched mean values should match: ')
    print('one-pass: ' + str(onepass_stats.mean))
    print('numpy:    ' + str(np.mean(norm_dist)))

    print('The following OnePass and batched stddev values should match: ')
    print('one-pass: ' + str(onepass_stats.getStddev()))
    print('numpy:    ' + str(np.std(norm_dist, ddof=1)))

    print('all one-pass stats: ' + op_stats_to_str(onepass_stats))
    return onepass_stats

def make_stats(dist):
    stats = OnePassParallelDescriptiveStats()
    for value in dist:
        stats.addValue(value)
    return stats
        
def test_merge(original_dist, parallel_nb):
    import numpy as np
    dists = np.split(original_dist, parallel_nb)
    stats_list = list()

    from multiprocessing import Pool

    pool = Pool(processes=parallel_nb,)
    
    stats_list = pool.map(make_stats, dists)
    
    merged_stats = OnePassParallelDescriptiveStats()
    for stats in stats_list:
        merged_stats = merged_stats.mergeStats(stats)

    print('split ({}) & merged: '.format(parallel_nb) + op_stats_to_str(merged_stats))
        
if __name__ == '__main__':
    import numpy as np

    # make a normal distribution to test the algorithm
    dist_size = 1000000
    expected_mean = 42.3
    expected_sigma = 67.3
    parallel_nb = 10
    
    import sys

    if len(sys.argv) > 1:
        dist_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        expected_mean = float(sys.argv[2])
    if len(sys.argv) > 3:
        expected_sigma = float(sys.argv[3])
    if len(sys.argv) > 4:
        parallel_nb = int(sys.argv[4])
    
    print('Testing normal distribution of size {} with '.format(dist_size) +
          'expected mean {} and expected stddev {}.'.format(
              expected_mean, expected_sigma))
    norm_dist = np.random.normal(expected_mean, expected_sigma, dist_size)
    import time
    t0 = time.time()
    test_single_set(norm_dist)
    single = time.time() - t0
    print('Single threaded test run in ' + str(single) + ' seconds')

    print('Now testing a map-reduce version using the same algorithm, but parallelized for speedup...')
    t0 = time.time()
    test_merge(norm_dist, parallel_nb)
    multi = time.time() - t0
    print('Multi-threaded test run in ' + str(multi) + ' seconds, for a speedup of ' + str(single/multi))
