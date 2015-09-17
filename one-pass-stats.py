#!/usr/bin/env python
# @author pgaultney

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

    def generateAggregate(self, B):
        combined = OnePassParallelDescriptiveStats()
        delta = B.mean - self.mean
        combined.count = self.count + B.count
        if combined.count > 0:
            combined.mean = (self.count * self.mean + B.count * B.mean) / combined.count
            combined.M2 = self.M2 + B.M2 + delta * delta * ((self.count * B.count) / combined.count)
        else:
            combined.mean = 0.0
            combined.M2 = 0.0

        # mins, maxes
        if B.max > self.max:
            combined.max = B.max
        else:
            combined.max = self.max
        if B.min < self.min:
            combined.min = B.min
        else:
            combined.min = self.min

        return combined

def op_stats_to_str(stats):
    return 'Mean {} | Count {} | Stddev {} | Min {} | Max {}'.format(
        stats.mean, stats.count, stats.getStddev(), stats.min, stats.max)
    

if __name__ == '__main__':
    import numpy as np

    # make a normal distribution to test the algorithm
    dist_size = 1000000
    expected_mean = 42.3
    expected_sigma = 67.3

    import sys

    if len(sys.argv) > 1:
        dist_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        expected_mean = float(sys.argv[2])
    if len(sys.argv) > 3:
        expected_sigma = float(sys.argv[3])
    
    print('Generating normal distribution of size {} with '.format(dist_size) +
          'expected mean {} and expected stddev {}.'.format(
              expected_mean, expected_sigma))
    norm_dist = np.random.normal(expected_mean, expected_sigma, dist_size)

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
