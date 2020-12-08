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
from __future__ import print_function
from math import sqrt


# from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
class ParallelDescriptiveStats(object):
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.min = float("inf")
        self.max = -float("inf")
        self.M2 = 0.0  # second moment
        self.delta = 0.0  # this is useful for a separate covariance calculation

    def addValue(self, value):
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

        self.count += 1
        self.delta = value - self.mean
        self.mean = self.mean + self.delta / self.count
        self.M2 = self.M2 + self.delta * (value - self.mean)

    def getVariance(self, sample=True):
        if self.count < 2:
            return 0.0
        else:
            sample_factor = 0
            if sample:
                sample_factor = 1
            return self.M2 / (self.count - sample_factor)

    def getStddev(self):
        return sqrt(self.getVariance())

    def merge(A, B):
        """
        This is a non-destructive merge operation - it does not
        alter the values in the calling object or the supplied object.
        """
        C = ParallelDescriptiveStats()
        delta = B.mean - A.mean
        C.count = A.count + B.count
        if C.count > 0:
            C.mean = (A.count * A.mean + B.count * B.mean) / C.count
            C.M2 = A.M2 + B.M2 + delta * delta * A.count * B.count / C.count
        else:
            C.mean = 0.0
            C.M2 = 0.0

        # mins, maxes
        if B.max > A.max:
            C.max = B.max
        else:
            C.max = A.max
        if B.min < A.min:
            C.min = B.min
        else:
            C.min = A.min

        return C


# from http://prod.sandia.gov/techlib/access-control.cgi/2008/086212.pdf
class ParallelCovariance(object):
    def __init__(self):
        self.co2 = 0.0  # 2nd comoment
        self.X = ParallelDescriptiveStats()
        self.Y = ParallelDescriptiveStats()

    def add_pair(self, x, y):
        self.X.addValue(x)
        self.Y.addValue(y)
        self.co2 = self.co2 + (self.X.count - 1) * self.X.delta * self.Y.delta / self.X.count

    def covariance(self, sample=True):
        div_factor = self.X.count
        if sample:
            div_factor = self.X.count - 1
        if self.X.count > 1:
            return self.co2 / div_factor
        else:
            return 0.0

    def pearson(self, sample=True):
        return self.covariance(sample) / (self.X.getStddev() * self.Y.getStddev())

    def merge(A, B):
        C = ParallelCovariance()
        C.X = A.X.merge(B.X)
        C.Y = A.Y.merge(B.Y)
        dx21 = B.X.mean - A.X.mean
        dy21 = B.Y.mean - A.Y.mean
        C.co2 = A.co2 + B.co2 + dx21 * dy21 * A.X.count * B.X.count / C.X.count
        return C
