

#include "stdafx.h"
#include "CNN.h"


Convolution::Convolution(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length) :
	LENGTH(input_length),
	WIDTH(input_width),
	STRIDE(1),
	DEPTH(1),
	FEATURE_MAP_LENGTH(filter_length),
	FEATURE_MAP_WIDTH(filter_length),
	NUMB_FEATURES(numb_filters),
	POOLED_LENGTH((input_length - filter_length / STRIDE) + 1),
	POOLED_WIDTH((input_width - filter_length / STRIDE) + 1),
	Layer(input_length, POOLED_LENGTH * POOLED_WIDTH) {
	
	if (input_length != input_width) {
	std::cout << " input_length and input_width must be equal -- cnn layers currently only support square dimensions [require manual 0 padding]" << std::endl;
	throw std::out_of_range("error");
	}

	w = std::vector<std::vector<Matrix>>(NUMB_FEATURES); 
	b = std::vector<std::vector<Matrix>>(NUMB_FEATURES);

	//initialize bias and w
	for (int f = 0; f < NUMB_FEATURES; ++f) {
		std::vector<Matrix> w_depth;
		std::vector<Matrix> b_depth;

		for (int d = 0; d < DEPTH; ++d) {
			w_depth.push_back(Matrix(FEATURE_MAP_LENGTH, FEATURE_MAP_WIDTH));
			b_depth.push_back(Matrix(FEATURE_MAP_LENGTH, FEATURE_MAP_WIDTH));
		}
	}
}
//does not support more than 1 channel currently
Vector Convolution::forwardPropagation_express(const Vector &input)
{
	//((LENGTH - FEATURE_MAP_LENGTH / STRIDE) + 1) * NUMB_FEATURES;
	std::vector<std::vector<Matrix>> feature_map_storage = std::vector<std::vector<Matrix>>(0);

	std::vector<Matrix> img; //3d image (IE rgb)

	//convolve around the image
	for (int i = 0; i < NUMB_FEATURES; ++i) {			//For the number of filters
		std::vector<Matrix> depth_storage(DEPTH);		//
		for (int d = 0; d < DEPTH; ++d) {				//For each depth index
			for (int x = 0; x < LENGTH; ++x) {			//For each length index
				for (int y = 0; y < WIDTH; ++y) {		//For each width index 
														//pointwise multiply filter and the img at given indexes
					Matrix& a = w[i][d] & img[i].sub_Matrix(x, y, FEATURE_MAP_LENGTH, FEATURE_MAP_WIDTH);
					depth_storage.push_back(a);
				}
			}
		}
		feature_map_storage.push_back(depth_storage);
	}
}

Vector Convolution::forwardPropagation(const Vector & input)
{

}

Vector Convolution::backwardPropagation(const Vector & dy)
{

}

Vector Convolution::backwardPropagation_ThroughTime(const Vector & dy)
{
}

void Convolution::clearBPStorage()
{
}

void Convolution::clearGradients()
{
}

void Convolution::updateGradients()
{
}
