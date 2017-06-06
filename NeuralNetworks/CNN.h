#pragma once
#ifndef CNN_h
#define CNN_h
#include "Layer.h"
#include "Filter.h"
class CNN : public Layer{

	const unsigned NUMB_FILTERS;

	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length

	const unsigned DEPTH;					
	const unsigned STRIDE;					//stride distance

	const unsigned FEATURE_LENGTH;		//feature map length dimension
	const unsigned FEATURE_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling
	const unsigned POOLED_SIZE;

	std::vector<Filter> filters_byDepth;

public:
	CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride);
	~CNN();

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
	static Filter read(std::ifstream& is) {};
};
#endif