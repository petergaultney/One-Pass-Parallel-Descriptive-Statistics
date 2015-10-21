#pragma once

#include <limits>

// This class is not thread safe.
class OnePassDescriptiveStats
{
public:
	void addValue(double value);
	double getVariance() const;
	double getStddev() const;

	OnePassDescriptiveStats aggregateWithSet(const OnePassDescriptiveStats& B) const;

	unsigned long long count = 0;
	double mean = 0.0;

	double min = std::numeric_limits<double>::max();
	double max = std::numeric_limits<double>::min();
		
	double M2 = 0.0;

public:
	static OnePassDescriptiveStats aggregateSets(const OnePassDescriptiveStats& A,
												 const OnePassDescriptiveStats& B);
};

/*
 * Implementation sourced from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
 * which is itself (mostly) due to Donald Knuth, according to Wikipedia.
 */
#include <cmath>

inline void OnePassDescriptiveStats::addValue(double value)
{
	if (value > this->max) {
		this->max = value;
	}
	if (value < this->min) {
		this->min = value;
	}

	++ this->count;
	double delta = value - this->mean;
	this->mean = this->mean + delta / this->count;
	this->M2 = this->M2 + delta * (value - this->mean);
}

/**
 * This is the sample, not population, variance.
 */
inline double OnePassDescriptiveStats::getVariance() const
{
	if (this->count < 2) {
		return 0.0;
	} else {
		return this->M2 / (this->count - 1);
	}
}

inline double OnePassDescriptiveStats::getStddev() const
{
	return sqrt(this->getVariance());
}

inline OnePassDescriptiveStats OnePassDescriptiveStats::aggregateWithSet(
	const OnePassDescriptiveStats& B) const
{
	return OnePassDescriptiveStats::aggregateSets(*this, B);
}

inline OnePassDescriptiveStats OnePassDescriptiveStats::aggregateSets(const OnePassDescriptiveStats& A,
																	  const OnePassDescriptiveStats& B)
{
	OnePassDescriptiveStats combined;
	// this algorithm due to Chan et al., also from Wikipedia page above
	double delta = B.mean - A.mean;
	combined.count = A.count + B.count;
	if (combined.count > 0) {
		combined.mean = (A.count * A.mean + B.count * B.mean) / combined.count;
		combined.M2 = A.M2 + B.M2 + delta * delta * ((double)(A.count * B.count) / combined.count);
	} else { // if we're combining two empty sets, we don't want floating point exceptions.
		combined.mean = 0;
		combined.M2 = 0;
	}

	// mins, maxes
	if (B.max > A.max) {
		combined.max = B.max;
	} else {
		combined.max = A.max;
	}
	if (B.min < A.min) {
		combined.min = B.min;
	} else {
		combined.min = A.min;
	}

	return combined;
}
