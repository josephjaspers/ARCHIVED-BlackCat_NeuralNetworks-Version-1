#pragma once
#ifndef CNN_Laye_interface_h
#define CNN_Laye_interface_h
#include "Layer.h"
class CNN_Layer : public Layer {
protected:
	CNN_Layer* conv_next;
	CNN_Layer* conv_prev;
public:
	virtual void link(CNN_Layer* l) {
		conv_next = l;
		l->prev = this;
	}
	CNN_Layer(int inputs, int outputs) : Layer(inputs, outputs) {};

	virtual Vector forwardPropagation_express(const Vector& x) {
		std::cout << "Vectorized implementations not supported for CNN's " << std::endl;
		throw std::invalid_argument("error");
		return x;
	}
	virtual Vector forwardPropagation(const Vector& x) {
		std::cout << "Vectorized implementations not supported for CNN's " << std::endl;
		throw std::invalid_argument("error");
		return x;
	}
	virtual Vector backwardPropagation(const Vector& dy) {
		std::cout << "Vectorized implementations not supported for CNN's " << std::endl;
		throw std::invalid_argument("error");
		return dy;
	}
	virtual Vector backwardPropagation_ThroughTime(const Vector& dy) {
		std::cout << "Vectorized implementations not supported for CNN's " << std::endl;
		throw std::invalid_argument("error");
		return dy;
	}

	virtual Stack_Matrix forwardPropagation_express(const Stack_Matrix& x) = 0;
	virtual Stack_Matrix forwardPropagation(const Stack_Matrix& x) = 0;
	virtual Stack_Matrix backwardPropagation(const Stack_Matrix& dy) = 0;
	virtual Stack_Matrix backwardPropagation_ThroughTime(const Stack_Matrix& dy) = 0;

	virtual void clearBPStorage() = 0;
	virtual void clearGradients() = 0;
	virtual void updateGradients() = 0;

	//static CNN_Layer* read(std::ifstream& is);
	virtual void write(std::ofstream& os) = 0;
	virtual void writeClass(std::ofstream& os) = 0;

};

#endif