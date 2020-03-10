Contained are a tested/working Python and C++ implementation of the parallel version of Welford's algorithm for calculating simple descriptive statistics in a single pass over the data. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

The advantage of this approach is twofold: Not only can you calculate these statistics as your data points arrive, for, e.g., a live dashboard, but you can actually distribute your calculation across multiple 'groups', whether phyiscal hardware (servers) or time spans (e.g. in 5 minute 'windows'), and merge those groups at arbitrary times to get the overall statistics for your full population. 

In the long run, this provides the advantage of allowing reduced data storage, as you can aggregate and store the data in whatever granularity you need, allowing you to trade off storage, compute, and precision within a large parameter space.
