#pragma once
#ifndef ConvolutionalNeural_Layer_h
#define ConvolutionalNeural_Layer_h
#include "Layer.h"
#include <functional>

class Convolution : public Layer {	


	//Currently supporting only 2-d convolution depth = 1

	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length

	const unsigned STRIDE;					//stride distance

	const unsigned NUMB_FEATURES;			//number of filters 

	const unsigned DEPTH;					//depth of img or numb_channels

	const unsigned FEATURE_MAP_LENGTH;		//feature map length dimension
	const unsigned FEATURE_MAP_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling
	
	std::vector<std::vector<Matrix>> bpX; //storage of inputs 

	std::vector<std::vector<Matrix>> w;	//filter weights	||Feature<filter_depth<filter-weight>>
	std::vector<std::vector<Matrix>> b;	//bias				||Feature<filter_depth<filter-weight>>

	std::vector<std::vector<Matrix>> f_gradientStorage;		//weight gradient storage
	std::vector<std::vector<Matrix>> b_gradientStorage;		//bias gradient storage
	
public:
Convolution(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length);

//@Override methods
Vector forwardPropagation_express(const Vector& x);						//forward propagation [Express does not store activations for BPPT]
Vector forwardPropagation(const Vector& x);								//forward propagation [Stores activations for BP & BPPT]
Vector backwardPropagation(const Vector& dy);							//backward propagation[Initial BP]
Vector backwardPropagation_ThroughTime(const Vector& dy);				//BPPT [Regular BP must be called before BPTT]

void clearBPStorage();
void clearGradients();
void updateGradients();

//not supported yet
void write(std::ofstream& os) {};
void writeClass(std::ofstream& os) {};
static Convolution read(std::ifstream& is) {};
};


class Filter : Layer{
	//Forward Propagation Essentials 
	Matrix* curr_img; 
	Matrix* delta_img;


	std::vector<Matrix> a; //curr activations
	Matrix x;	//curr inputs 
	Matrix w;	//filter
	Matrix b;	//filter bias

	std::vector<int> x_pos_Max;
	std::vector<int> y_pos_Max;

	const int INPUT_DEPTH;
	const int OUT_DEPTH;

	const int FILTER_LENGTH;
	const int FILTER_WIDTH;

	const int POOLED_LENGTH;
	const int POOLED_WIDTH;

public:

	Vector forwardPropagation_express(const Vector& x) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Vector();
	}
	Vector forwardPropagation(const Vector& x) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Vector();
	}
	Vector backwardPropagation(const Vector& dy) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Vector();
	}
	Vector backwardPropagation_ThroughTime(const Vector& dy) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Vector();
	};

	Matrix forwardPropagation_express(const Matrix& img) {
		a.clear(); //clear the current activations matrix;

		for (int x = 0; x < img.length(); ++x) {
			for (int y = 0; y < img.width(); ++y) {
				Matrix& activation = w * img.sub_Matrix(x, y, FILTER_LENGTH, FILTER_WIDTH) + b;
				a.push_back(activation);
			}
		}
	}
	Matrix forwardPropagation(const Matrix& x) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Matrix();
	}
	Matrix backwardPropagation(const Matrix& dy) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Matrix();
	}
	Matrix backwardPropagation_ThroughTime(const Matrix& dy) {
		std::cout << "Vector implementations not supported in convulsion layer" << std::endl;
		throw std::bad_function_call();

		return Matrix();
	};

	void clearBPStorage();
	void clearGradients();
	void updateGradients();

};
#endif
