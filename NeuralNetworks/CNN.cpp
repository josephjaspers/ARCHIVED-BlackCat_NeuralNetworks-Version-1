#include "stdafx.h"
#include "CNN.h"
/*
CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride) :
	NUMB_FILTERS(numb_filters),
	LENGTH(input_length), WIDTH(input_width),
	DEPTH(1),
	STRIDE(stride),
	FEATURE_LENGTH(filter_length), FEATURE_WIDTH(filter_length),
	POOLED_LENGTH(floor(input_length - filter_length / stride) + 1),
	POOLED_WIDTH(floor(input_width - filter_length / stride) + 1),
	POOLED_SIZE(POOLED_LENGTH * POOLED_WIDTH),
	Layer(POOLED_SIZE, POOLED_SIZE)//?????fdjfldshbklfhdsjk
{
	filters_byDepth = std::vector<Filter>(DEPTH, Filter(LENGTH, WIDTH, NUMB_FILTERS, FEATURE_LENGTH, STRIDE));
	std::cout << filters_byDepth.size() << " f sze " << std::endl;
}

CNN::~CNN()
{
	//for (int i = 0; i < filters_byDepth.size(); ++i) {
	//	filters_byDepth[i].~Filter(); //call the destruction operator 
	//}
}

Vector CNN::forwardPropagation_express(const Vector & x)
{
	std::vector<Vector> outputs;
	outputs.reserve(DEPTH);

	for (int i = 0; i < DEPTH; ++i) {
		Vector& img = x.sub_Vector(i, x.length() / DEPTH);
		std::cout << img.length() << std::endl;
		outputs.emplace_back(filters_byDepth[i].forwardPropagation_express(img));
	}
	if (next != nullptr) {
		return next->forwardPropagation_express(Vector::concat(outputs));
	}
	else {
		return Vector::concat(outputs);
	}
}
Vector CNN::forwardPropagation(const Vector & x)
{
	std::vector<Vector> outputs;
	outputs.reserve(DEPTH);

	for (int i = 0; i < DEPTH; ++i) {
		Vector& img = x.sub_Vector(i, x.length() / DEPTH);
		outputs.emplace_back(filters_byDepth[i].forwardPropagation(img));
	}
	if (next != nullptr) {
		return next->forwardPropagation(Vector::concat(outputs));
	}
	else {
		return Vector::concat(outputs);
	}
}

Vector CNN::backwardPropagation(const Vector & dy)
{
	for (int i = 0; i < filters_byDepth.size(); ++i) {
		filters_byDepth[i].backwardPropagation(dy.sub_Vector(i, dy.length() / DEPTH));
	}




	return dy;
}

Vector CNN::backwardPropagation_ThroughTime(const Vector & dy)
{
	std::cout << "not supported " << std::endl;

	throw std::invalid_argument("error");
}

void CNN::clearBPStorage()
{
	for (Filter f : filters_byDepth) {
		f.clearBPStorage();
	}
	Layer::clearBPStorage();
}

void CNN::clearGradients()
{
	for (Filter f : filters_byDepth) {
		f.clearGradients();
	}
	Layer::clearGradients();
}

void CNN::updateGradients()
{
	for (Filter f : filters_byDepth) {
		f.updateGradients();
	}
	Layer::updateGradients();
}
*/