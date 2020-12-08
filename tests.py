#!/usr/bin/env python
from one_pass_stats import ParallelCovariance, ParallelDescriptiveStats


def op_stats_to_str(stats):
    return "Mean {} | Count {} | Stddev {} | Min {} | Max {}".format(
        stats.mean, stats.count, stats.getStddev(), stats.min, stats.max
    )


def test_single_set(norm_dist):
    import numpy as np

    print("Calculating descriptive statistics using the one-pass algorithm...")
    onepass_stats = make_stats(norm_dist)

    print("The following OnePass and numpy-calculated means should match: ")
    print("one-pass: " + str(onepass_stats.mean))
    print("numpy:    " + str(np.mean(norm_dist)))

    print("The following OnePass and numpy-calculated stddevs should match: ")
    print("one-pass: " + str(onepass_stats.getStddev()))
    print("numpy:    " + str(np.std(norm_dist, ddof=1)))

    print("all one-pass stats: " + op_stats_to_str(onepass_stats))
    return onepass_stats


def make_stats(dist):
    stats = ParallelDescriptiveStats()
    for value in dist:
        stats.addValue(value)
    return stats


def make_covar(*args):
    distX, distY = args[0]
    covar = ParallelCovariance()
    for x, y in zip(distX, distY):
        covar.add_pair(x, y)
    print("make cov: " + str(covar.covariance()) + " pears: " + str(covar.pearson()))
    return covar


def test_merge(original_dist, parallel_nb):
    import numpy as np

    dists = np.split(original_dist, parallel_nb)
    stats_list = list()

    from multiprocessing import Pool

    with Pool(
        processes=parallel_nb,
    ) as pool:
        stats_list = pool.map(make_stats, dists)

        merged_stats = ParallelDescriptiveStats()
        for stats in stats_list:
            merged_stats = merged_stats.merge(stats)

        print("split ({}) & merged: ".format(parallel_nb) + op_stats_to_str(merged_stats))


def test_covar(distX, distY):
    print("distX = " + str(distX))
    print("distY = " + str(distY))

    covar = ParallelCovariance()
    for x, y in zip(distX, distY):
        covar.add_pair(x, y)
    print("OnePass Covariance: " + str(covar.covariance()) + "; Pearson: " + str(covar.pearson()))

    ox = make_stats(distX)
    oy = make_stats(distY)
    print("correct stats: ", op_stats_to_str(ox), op_stats_to_str(oy))

    print(
        "Naive, two-pass Covariance: "
        + str(naive_covariance(distX, distY))
        + "; Pearson: "
        + str(naive_covariance(distX, distY) / (ox.getStddev() * oy.getStddev()))
    )

    import numpy as np

    split_xs = np.split(np.array(distX), 4)
    split_ys = np.split(np.array(distY), 4)

    print("Testing covariance merge algorithm...")

    from multiprocessing import Pool

    pool = Pool(
        processes=4,
    )

    covars_list = pool.map(make_covar, zip(split_xs, split_ys))
    # naive_list = pool.map(naive_pop_wrapper, zip(split_xs, split_ys))
    # print(naive_list)

    merged_covar = ParallelCovariance()
    for covar in covars_list:
        # print('A: ' + str(merged_covar.covariance()))
        # print('B: ' + str(covar.covariance()))
        merged_covar = merged_covar.merge(covar)
        # print('C: ' + str(merged_covar.covariance()))

    print(merged_covar.covariance())
    print(merged_covar.pearson())


def naive_covariance(data1, data2, sample=False):
    n = len(data1)
    sum12 = 0
    sum1 = sum(data1)
    sum2 = sum(data2)
    for i in range(n):
        sum12 += data1[i] * data2[i]

    sample_div = n - 1
    if not sample:
        sample_div = n
    return (sum12 - sum1 * sum2 / n) / sample_div


def naive_pop_wrapper(*args):
    data1, data2 = args[0]
    return naive_covariance(data1, data2)


def make_comom2(data1, data2):
    comom = ParallelCovariance()
    for x, y in zip(data1, data2):
        comom.add_pair(x, y)
    return comom


if __name__ == "__main__":
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

    print(
        "Testing normal distribution of size {} with ".format(dist_size)
        + "expected mean {} and expected stddev {}.".format(expected_mean, expected_sigma)
    )
    norm_dist = np.random.normal(expected_mean, expected_sigma, dist_size)
    import time

    t0 = time.time()
    test_single_set(norm_dist)
    single = time.time() - t0
    print("Single threaded test run in " + str(single) + " seconds")

    print(
        "Now testing a map-reduce version using the same algorithm, but parallelized for speedup..."
    )
    t0 = time.time()
    test_merge(norm_dist, parallel_nb)
    multi = time.time() - t0
    print(
        "Multi-threaded test run in "
        + str(multi)
        + " seconds, for a speedup of "
        + str(single / multi)
    )

    x = [1, 3, 5, 7, 9, 11, 13, 15]
    y = [4, 17, 24, 33, 77, 119, 200, 270]

    test_covar(x, y)

    blah = ParallelCovariance()
    for i in range(len(x)):
        blah.add_pair(x[i], y[i])
    print(blah.covariance())

    blah1 = make_comom2(x[0:4], y[0:4])
    blah2 = make_comom2(x[4:8], y[4:8])
    blah = blah1.merge(blah2)
    print(blah.covariance(sample=True))
    print(naive_covariance(x, y, sample=True))
