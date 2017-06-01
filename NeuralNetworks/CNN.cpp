#include "stdafx.h"
#include "CNN.h"
Matrix& CNN::Xt()
{
	if (bpX.empty()) {
		return Matrix(LENGTH, WIDTH);
	}
	else {
		return bpX.back();
	}
}

CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride) :
	LENGTH(input_length), WIDTH(input_width),
	FEATURE_LENGTH(filter_length), FEATURE_WIDTH(filter_length),
	STRIDE(stride),
	DEPTH(1),
	NUMB_FEATURES(numb_filters),
	POOLED_LENGTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s
	POOLED_WIDTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s 
	Layer(input_length * input_width, ((floor(input_length - filter_length / stride) + 1) * (floor(input_width - filter_length) / 1) + 1))
{
	filters.reserve(NUMB_FEATURES * DEPTH);
	for (int i = 0; i < NUMB_FEATURES; ++i) {
		filters.emplace_back(Filter(LENGTH, WIDTH, FEATURE_LENGTH, FEATURE_WIDTH));
	}
}

Vector concatenate(std::vector<Matrix>& m) {
	int l = 0;
	for (int i = 0; i < m.size(); ++i) {
		l += m[i].size();
	}
	Vector r = Vector(l);
	
	int index = 0;
	for (int i = 0; i < m.size(); ++i) {
		for (int x = 0; x < m[i].length(); ++x) {
			for (int y = 0; y < m[i].width(); ++y) {
				r[index] = m[i][x][y]; 
			}
		}
	}
	return r;
}
Vector CNN::forwardPropagation_express(const Vector &input)
{
	std::vector<Vector> collection; 
	collection.reserve(NUMB_FEATURES);

	for (int i = 0; i < NUMB_FEATURES; ++i) {
		collection.push_back(filters[i].forwardPropagation_express(input));
	}
	if (next != nullptr) {
		return next->forwardPropagation_express(Vector::concat(collection));
	}
	else {
		return Vector::concat(collection);
	}
}

Vector CNN::forwardPropagation(const Vector & input)
{
	std::vector<Vector> collection;
	collection.reserve(NUMB_FEATURES);

	for (int i = 0; i < NUMB_FEATURES; ++i) {
		collection.push_back(filters[i].forwardPropagation(input));
	}

	if (next != nullptr) {
		return next->forwardPropagation(Vector::concat(collection));
	}
	else {
		return Vector::concat(collection);
	}
}
static double sigmoid(double d) {
	return 1 / (1 + pow(2.71828, -d));
}
static double sigmoid_deriv(double d) {
	return d *= (1 - d);
}
Vector CNN::backwardPropagation(const Vector & dy)
{
	//method not working for multiple layers (only works as input layer)
	for (int i = 0; i < NUMB_FEATURES; i += DEPTH * POOLED_LENGTH * POOLED_WIDTH) {
		filters[i].backwardPropagation(dy.sub_Vector(i, DEPTH * POOLED_LENGTH));
	}
	return dy; 
}

Vector CNN::backwardPropagation_ThroughTime(const Vector & dy)
{
	std::cout << " BPTT attempt FAILED  __ I HAVENT WRITTEN THIS YET SORRY" << std::endl;
	throw std::invalid_argument("not supported");
}

void CNN::clearBPStorage()
{
	bpX.clear();
	for (int i = 0; i < NUMB_FEATURES; ++i) {
		filters[i].clearBPStorage();
	}
	Layer::clearBPStorage();
}

void CNN::clearGradients()
{
	for (int i = 0; i < NUMB_FEATURES; ++i) {
		filters[i].clearGradients();
	}

	Layer::clearGradients();

}

void CNN::updateGradients()
{
	for (int i = 0; i < NUMB_FEATURES; ++i) {
		filters[i].updateGradients();
	}
	Layer::updateGradients();

}

void CNN::printFilterWeights() {
	for (int i = 0; i < filters.size(); ++i) {
		std::cout << " layer = " << i << std::endl;
		filters[i].printWeights();
		std::cout << std::endl;
	}
}