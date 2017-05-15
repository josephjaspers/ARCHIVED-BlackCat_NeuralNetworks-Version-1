#pragma once
#ifndef Layer_h
#define Layer_h
#include "stdafx.h"
#include "Matrices.h"
#include "nonLinearityFunction.h"
using namespace Matrices;
class Layer {
	//super class of all sub class layers
protected:
	nonLinearityFunct g;								//An object that applies non linearity functions and derivatives to apropriate Vectors

public:
	const Vector INPUT_ZEROS;							//An all 0's Vector of size n = numb_inputs
	const Vector OUTPUT_ZEROS;							//An all 0's Vector of size n = numb_outputs
	const int NUMB_INPUTS;								//number of inputs
	const int NUMBER_OUTPUTS;							//number of outputs
protected:
														//superclass constructor -- initializes constants
	Layer(int inputs, int outputs) : INPUT_ZEROS(inputs), OUTPUT_ZEROS(outputs), NUMB_INPUTS(inputs), NUMBER_OUTPUTS(outputs) {

	}				
	typedef std::vector<Vector> bpStorage;				//type defination -- this format is used consistently to store the activations for backprop through time 
	Layer* next;										//pointer to next layer in network --similair to LinkedList 
	Layer* prev;										//pointer to previous layer in network --similair to LinkedList
	double lr = .03;									//Learning Rate
	double mr = .01;									//Momentum Rate (currently not supported)
	
public:
														//Method for linking two layers together
	void link(Layer& l) {
		next = &l; //next = linked layer 
		l.prev = this; //next layers "prev" is set to this 
	}

	double getLearningRate() { return lr; }				//accessors for lr/mr
	double getMomentumRate() { return mr; }
	double setLearningRate(double lr) { Layer::lr = lr; }	//mutators for lr/mr
	double setMomentumRate(double mr) { Layer::mr = mr; }
	int getInputs() { return NUMB_INPUTS; }				//accessors for inputs/outputs
	int getOutputs() { return NUMBER_OUTPUTS; }
	
	virtual Vector forwardPropagation_express(const Vector& x) = 0;						//forward propagation [Express does not store activations for BPPT]
	virtual Vector forwardPropagation(const Vector& x) = 0;								//forward propagation [Stores activations for BP & BPPT]
	virtual Vector backwardPropagation(const Vector& dy) = 0;							//backward propagation[Initial BP] 
	virtual Vector backwardPropagation_ThroughTime(const Vector& dy) = 0;				//BPPT [Regular BP must be called before BPTT]
												
	void setSigmoid() { g.setSigmoid(); }				//set the crush function of the layer to sigmoid (default = sigmoid)
	void setTanh() { g.setTanh();  }					//set the crush function of the layer to tanh

	virtual void clearBPStorage() = 0;					//clears the stored activations used in BP of the layers --> Bpstorage will accumulate after a while 
	virtual void clearGradients() = 0;					//clears the stored gradients of the past back propagation --> Clears the store gradients 
	virtual void updateGradients() = 0;					//updates the weights using the stored gradients. 
};
#endif
