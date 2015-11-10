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
from __future__ import division
from math import sqrt

class OnePassParallelDescriptiveStats(object):

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.min = float('inf')
        self.max = -float('inf')
        self.M2 = 0.0
        self.delta = 0.0 # this is useful for covariance calc

    def addValue(self, value):
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

        self.count += 1
        self.delta = value - self.mean
        self.mean = self.mean + self.delta / self.count
        self.M2 = self.M2 + self.delta * (value - self.mean)

    def getVariance(self):
        if (self.count < 2):
            return 0.0
        else:
            return self.M2 / (self.count - 1) # sample variance

    def getStddev(self):
        return sqrt(self.getVariance())

    def merge(self, B):
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

class OnePassCovariance(object):
    def __init__(self):
        self.X = OnePassParallelDescriptiveStats()
        self.Y = OnePassParallelDescriptiveStats()
        self.M12 = 0.0
        self.sample = False
    def add_pair(self, x, y):
        self.X.addValue(x)
        self.Y.addValue(y)
        self.M12 += ((self.X.count - 1)
                     * (self.X.delta / self.X.count)
                     * (self.Y.delta / self.Y.count)
                     - (self.M12 / self.X.count))
    def covariance(self):
        if self.X.count >= 2:
            sample_factor = self.X.count / (self.X.count - 1)
            if not self.sample:
                sample_factor = 1.0
            return sample_factor * self.M12
        else:
            return 0.0
    def pearson(self):
        div = self.X.getStddev() * self.Y.getStddev()
        if div != 0.0:
            return self.covariance() / div
        else:
            return 0.0
    # takes two computed covariances representing parts of the same
    # X and Y random variables and finds the overall covariance.
    def merge(covA, covB):
        # CA + CB + (MXA - MXB)(MYA - MYB)(nA*nB/nX)
        xdiffmean = covA.X.mean - covB.X.mean
        ydiffmean = covA.Y.mean - covB.Y.mean
        print('diffmeans', xdiffmean, ydiffmean)
        combinedX = covA.X.merge(covB.X)
        combinedY = covA.Y.merge(covB.Y)
        print(op_stats_to_str(combinedX), op_stats_to_str(combinedY))
        if combinedX.count != combinedY.count:
            print('uh oh! the merge is impossible!')
        if combinedX.count > 0:
            print(covA.X.count, covB.X.count, combinedY.count)
            n_ab_x = (covA.X.count * covB.X.count) / combinedY.count
        else:
            n_ab_x = 0.0
        print('n_ab_x', n_ab_x)
        covX = covA.covariance() + covB.covariance() + xdiffmean * ydiffmean * n_ab_x
        print('internal covX', covX)
        new_covar_obj = OnePassCovariance()
        new_covar_obj.X = combinedX
        new_covar_obj.Y = combinedY
        if combinedX.count >= 2:
            sample_factor = 1.0
            if covA.sample and covB.sample:
                sample_factor = combinedX.count / (combinedX.count - 1.0)
                new_covar_obj.sample = True
            new_covar_obj.M12 = covX / sample_factor
        return new_covar_obj

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

def make_covar((distX, distY)):
    covar = OnePassCovariance()
    print('entering make covar')
    for x, y in zip(distX, distY):
        print('adding pair: ' + str(x) + ' ' + str(y))
        covar.add_pair(x, y)
    print('cov: ' + str(covar.covariance()) + ' pears: ' + str(covar.pearson()))
    return covar

def test_merge(original_dist, parallel_nb):
    import numpy as np
    dists = np.split(original_dist, parallel_nb)
    stats_list = list()

    from multiprocessing import Pool

    pool = Pool(processes=parallel_nb,)
    
    stats_list = pool.map(make_stats, dists)
    
    merged_stats = OnePassParallelDescriptiveStats()
    for stats in stats_list:
        merged_stats = merged_stats.merge(stats)

    print('split ({}) & merged: '.format(parallel_nb) + op_stats_to_str(merged_stats))

def test_covar(distX, distY):
    print('distX = ' + str(distX))
    print('distY = ' + str(distY))
    
    covar = OnePassCovariance()
    for x, y in zip(distX, distY):
        covar.add_pair(x, y)
    print('OnePass Covariance: ' + str(covar.covariance()) + '; Pearson: ' + str(covar.pearson()))

    ox = make_stats(distX)
    oy = make_stats(distY)
    
    print('Naive, two-pass Covariance: '
          + str(naive_covariance(distX, distY))
          + '; Pearson: ' + str(naive_covariance(distX, distY) / (ox.getStddev() * oy.getStddev())))

    import numpy as np
    split_xs = np.split(np.array(distX), 2)
    split_ys = np.split(np.array(distY), 2)
    
    from multiprocessing import Pool

    pool = Pool(processes=parallel_nb,)
    
    covars_list = pool.map(make_covar, zip(split_xs, split_ys))
    naive_list = pool.map(naive_pop_wrapper, zip(split_xs, split_ys))
    print(naive_list)

    merged_covar = OnePassCovariance()
    for covar in covars_list:
        print('A: ' + str(merged_covar.covariance()))
        print('B: ' + str(covar.covariance()))
        merged_covar = merged_covar.merge(covar)
        print('C: ' + str(merged_covar.covariance()))

    print(merged_covar.covariance())
    print(merged_covar.pearson())

def naive_covariance(data1, data2, sample=False):
    n = len(data1)
    sum12 = 0
    sum1 = sum(data1)
    sum2 = sum(data2)
    for i in range(n):
        sum12 += data1[i]*data2[i]

    sample_div = n - 1
    if not sample:
        sample_div = n
    return (sum12 - sum1*sum2 / n) / sample_div

def naive_pop_wrapper((data1, data2)):
    return naive_covariance(data1, data2)
    
if __name__ == '__main__':
    import numpy as np

    # make a normal distribution to test the algorithm
    dist_size = 100000
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

    x = [1, 3, 5, 7, 9, 11]
    y = [4, 17, 24, 33, 77, 119]
    
    test_covar(x, y)
