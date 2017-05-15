#pragma once
#ifndef NeuralNetwork_h
#define NeuralNetwork_h
#include "stdafx.h"
#include "Layer.h"

class NeuralNetwork : public Layer {

	std::vector<Layer*> network;
	Vector output; 

public:
	void push_back(Layer* l) {
		if (!network.empty()) {
			network.back()->link(*l);
		}
		network.push_back(l);
	}
	NeuralNetwork() : Layer(0,0) {

	}
	~NeuralNetwork() {
		for (int i = 0; i < network.size(); ++i) {
			delete network[i];
		}
	}

	Vector forwardPropagation_express(const Vector& input) {
		return network.front()->forwardPropagation_express(input);
	}
	Vector forwardPropagation(const Vector& input) {
		output = network[0]->forwardPropagation(input);
		return output;
	}
	Vector backwardPropagation(const Vector& y) {
		Vector dy = output - y; //get residual
		return network.back()->backwardPropagation(dy);
	}
	Vector backwardPropagation_ThroughTime(const Vector& dy) {
		return network.back()->backwardPropagation_ThroughTime(dy);
	}

	void train(std::vector<Vector>& x, Vector& y) {
		clearGradients();
		for (Vector input : x) {
		forwardPropagation(input);
		}
		backwardPropagation(y);
		for (int i = 0; i < x.size() - 1; ++i) {
			backwardPropagation_ThroughTime(network.back()->OUTPUT_ZEROS);
		}

		updateGradients();
		clearBPStorage();
	}
	void train(Vector x, Vector y) {
		clearGradients();
		forwardPropagation(x);
		backwardPropagation(y);
		updateGradients();
		clearBPStorage();
	}
	void train(std::vector<std::vector<double>> x, std::vector<double> y) {
		clearGradients();
		for (std::vector<double> input : x) {
			forwardPropagation(Vector(input));
		}
		backwardPropagation(Vector(y));
		for (int i = 0; i < x.size() - 1; ++i) {
			backwardPropagation_ThroughTime(network.back()->OUTPUT_ZEROS);
		}
		updateGradients();
		clearBPStorage();
	}
	void train(std::vector<double> x, std::vector<double> y) {
		clearGradients();
		forwardPropagation(Vector(x));
		backwardPropagation(Vector(y));
		updateGradients();
		clearBPStorage();
	}

	Vector predict(std::vector<Vector> x) {
		for (int i = 0; i < x.size() - 1; ++i) {
			forwardPropagation_express(x[i]);
		}
		return forwardPropagation_express(x.back());
	}
	Vector predict(Vector x) {
		return forwardPropagation_express(x);
	}
	Vector predict(std::vector<std::vector<double>> x) {
		for (int i = 0; i < x.size() - 1; ++i) {
			forwardPropagation_express(Vector(x[i]));
		}
		return forwardPropagation_express(Vector(x.back()));
	}
	Vector predict(std::vector<double> x) {
		return forwardPropagation_express(Vector(x));
	}

	void clearBPStorage() {
		for (Layer* l : network) {
			l->clearBPStorage();
		}
	}
	void clearGradients() {
		for (Layer* l : network) {
			l->clearGradients();
		}
	}
	void updateGradients() {
		for (Layer* l : network) {
			l->updateGradients();
		}
	}
};

#endif 