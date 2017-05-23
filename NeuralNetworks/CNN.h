#pragma once
#ifndef ConvolutionalNeural_Layer_h
#define ConvolutionalNeural_Layer_h
#include "Layer.h"

namespace CNN_names { typedef std::vector<Matrix> featureMapStorage; }
//DOES NOT WORK
using namespace CNN_names;
class CNN : public Layer {	
	featureMapStorage f_maps;				//Storage for the all the feature maps
	featureMapStorage delta_f_maps;			//error storage for all the feature maps
	std::vector<featureMapStorage> bp_maps;	//The stored activations of the feature maps for BP (ability to store multiple to combine with recurrent models)
	bpStorage bpX;							//Input storage (for BP)

protected:
	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length
	const unsigned DEPTH;					//Number of channels per vectorized picture || 3-d convolution not supported yet (only 2d/1d)

	const unsigned STRIDE;					//stride distance
	const unsigned NUMB_FEATURES;			//number of filters

	const unsigned FEATURE_MAP_LENGTH;		//feature map length dimension
	const unsigned FEATURE_MAP_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling

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

featureMapStorage convolution(const Matrix & x, const Matrix & feature);//returns a feature_map (multiple vectors) of size n where n = (n_length - filter_ length) / (stride + 1)
Vector maxPooling(featureMapStorage convolved_mat);						//accepts a feature map storage, finds the max value of each and returns an n dimensional vector
Vector concatenate(featureMapStorage& sto);								//concatenates a set of vectors, feeds these vectors to the output layer (the next layer is feedforward) //Will implement a design to chain together multiple cnn layers

const Vector& Xt();

};

#endif
