

#include "stdafx.h"
#include "CNN.h"


Vector CNN::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	else {
		return bpX.back();
	}
}

CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride) :
	LENGTH(input_length), WIDTH(input_width),
	STRIDE(stride),
	NUMB_FILTERS(numb_filters),
	FEATURE_LENGTH(filter_length),
	FEATURE_WIDTH(filter_length),
	POOLED_LENGTH(floor(LENGTH - filter_length / STRIDE) + 1),
	POOLED_WIDTH(floor(WIDTH - filter_length / STRIDE) + 1),
	DEPTH(1),
	Layer(LENGTH * WIDTH, POOLED_LENGTH * POOLED_WIDTH * NUMB_FILTERS * DEPTH)
{
//	features = std::vector<Filter>(NUMB_FILTERS); 
	features.clear();
	for (int i = 0; i < NUMB_FILTERS; ++i) {
		features.push_back(Filter(LENGTH, WIDTH, FEATURE_LENGTH, STRIDE));
	}
}
Vector CNN::combine(std::vector<Vector>& stack) {
	Vector r = NUMB_FILTERS * DEPTH * POOLED_LENGTH * POOLED_WIDTH;

	int r_index = 0;
	for (int i = 0; i < stack.size(); ++i) {
		for (int j = 0; j < stack[i].length(); ++j) {
			r[r_index] = stack[i][j];
			++r_index;
		}
	}
	return r;
}
Vector CNN::forwardPropagation_express(const Vector & x)
{
	std::vector<Vector> collection = std::vector<Vector>(0);
	for (int i = 0; i < NUMB_FILTERS; ++i) {
		collection.push_back(features[i].forwardPropagation_express(x));
	}
	if (next != nullptr) {
		return next->forwardPropagation_express(combine(collection));
	}
	else {
		return combine(collection);
	}
}

Vector CNN::forwardPropagation(const Vector & x)
{
	std::vector<Vector> collection = std::vector<Vector>(0);
	for (int i = 0; i < NUMB_FILTERS; ++i) {
		collection.push_back(features[i].forwardPropagation(x));
	}
	if (next != nullptr) {
		return next->forwardPropagation(combine(collection));
	}
	else {
		return combine(collection);
	}
}

Vector CNN::backwardPropagation(const Vector & dy)
{
	for (int i = 0; i < NUMB_FILTERS * DEPTH; ++i) {
		features[i].backwardPropagation(dy.sub_Vector(i, POOLED_LENGTH * POOLED_WIDTH));
	}
	return dy;
}

Vector CNN::backwardPropagation_ThroughTime(const Vector & dy)
{
	std::cout << " cnn no bptt" << std::endl;
	throw std::invalid_argument("not supported");
}

void CNN::clearBPStorage()
{
	Layer::clearBPStorage();
}

void CNN::clearGradients()
{
	Layer::clearGradients();
}

void CNN::updateGradients()
{
	Layer::updateGradients();
}
