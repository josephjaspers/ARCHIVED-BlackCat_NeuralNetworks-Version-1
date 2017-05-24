#pragma once
#ifndef ConvolutionalNeural_Layer_h
#define ConvolutionalNeural_Layer_h
#include "Layer.h"

class CNN : public Layer {	
	//Currently supporting only 2-d convolution depth = 1
	struct max_indexes {
		max_indexes(int x, int y) {
			max_indexes::x = x;
			max_indexes::y = y;
		}
		int x;
		int y;
	};
	struct Feature {
		Matrix w;											//this is the actual feature Matrix;
		Matrix b;											//bias 
		std::vector<Matrix> x;								//these are the convolved sub_matrices of the given matrix input 
		std::vector<std::vector<Vector>> bpX;				//backpropagation x -- these are the stored activations for 
	};


protected:
	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length

	const unsigned STRIDE;					//stride distance

	const unsigned FEATURE_MAP_LENGTH;		//feature map length dimension
	const unsigned FEATURE_MAP_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling
	
	//currently developing for only 1 filter
	std::vector<max_indexes> maxes;
	std::vector<Matrix> bpX; //storage of inputs 
	Matrix x; //last input 
	Matrix f; //filter 
	Matrix b; //bias

	Matrix f_gradientStorage;
	Matrix b_gradientStorage; 
	Matrix fd;//delta f
	Matrix bf;//delta 


public:
CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length);

//@Override methods
Vector forwardPropagation_express(const Vector& x);						//forward propagation [Express does not store activations for BPPT]
Vector forwardPropagation(const Vector& x);								//forward propagation [Stores activations for BP & BPPT]
Vector backwardPropagation(const Vector& dy);							//backward propagation[Initial BP]
Vector backwardPropagation_ThroughTime(const Vector& dy);				//BPPT [Regular BP must be called before BPTT]

void clearBPStorage();
void clearGradients();
void updateGradients();

void write(std::ofstream& os) {

}
void writeClass(std::ofstream& os) {

}
static CNN read(std::ifstream& is) {
return CNN(1, 1, 1, 1);
}
private:

Matrix convolution(const Matrix & x, const Matrix & feature);//returns a feature_map (multiple vectors) of size n where n = (n_length - filter_ length) / (stride + 1)
Matrix maxPooling(std::vector<Matrix> convolved_mat);						//accepts a feature map storage, finds the max value of each and returns an n dimensional vector
Vector concatenate(std::vector<Matrix>& sto);								//concatenates a set of vectors, feeds these vectors to the output layer (the next layer is feedforward) //Will implement a design to chain together multiple cnn layers


};

#endif
