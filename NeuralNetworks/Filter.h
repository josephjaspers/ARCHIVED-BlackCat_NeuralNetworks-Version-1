#pragma once
#ifndef Filter_jas_h
#define Filter_jas_h
#include "Layer.h"
class Filter : public Layer {
	//CNN Layer --> seperate by channels --> seperate by filters 

	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length

	const unsigned STRIDE;					//stride distance

	const unsigned FEATURE_LENGTH;		//feature map length dimension
	const unsigned FEATURE_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling

	std::vector<int> bp_max_index_x;
	std::vector<int> bp_max_index_y;
	std::vector<Matrix> bpX;
	Matrix w;
	Matrix b;
	Matrix w_gradientStorage;
	Matrix b_gradientStorage;

	Matrix Xt();
	int index_Xt();
	int index_Yt();
public:
	Filter(unsigned input_length, unsigned input_width, unsigned filter_length, unsigned stride);
	Filter() : Layer(0, 0), LENGTH(0), WIDTH(0), FEATURE_LENGTH(0), FEATURE_WIDTH(0), POOLED_LENGTH(0), POOLED_WIDTH(0), STRIDE(0) {};
	double findMax(Matrix & img, int & x_store, int & y_store);

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