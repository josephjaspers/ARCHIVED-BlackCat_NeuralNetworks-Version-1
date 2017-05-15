#include "stdafx.h"
#include "RecurrentUnit.h"
const Vector & RecurrentUnit::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

const Vector & RecurrentUnit::Ct()
{
	if (bpC.empty()) {
		return OUTPUT_ZEROS;
	}
	return bpC.back();
}


RecurrentUnit::RecurrentUnit(int inputs, int outputs) : Layer(inputs, outputs)
{
	b_gradientStorage = Vector(outputs);
	w_gradientStorage = Matrix(outputs, inputs);
	r_gradientStorage = Matrix(outputs, outputs);

	c = Vector(outputs);
	b = Vector(outputs);
	w = Matrix(outputs, inputs);
	r = Matrix(outputs, outputs);

	Matrices::randomize(b, -4, 4);
	Matrices::randomize(w, -4, 4);
	Matrices::randomize(r, -4, 0); //Initialize the recurrent weights as having 0 impact initially
}

Vector RecurrentUnit::forwardPropagation_express(const Vector & x)
{
	c = g(w * x + r * c + b);
	if (next != nullptr) 
		return next->forwardPropagation_express(c);
	else
		return c;
}

Vector RecurrentUnit::forwardPropagation(const Vector & x)
{
	//store the current activations and cellstate
	bpX.push_back(x);
	bpC.push_back(c);
	//set the cellstate (c is the output)
	c = g(w * x + r * c + b);
	//continue backprop
	if (next != nullptr)
		return next->forwardPropagation(c);
	else
		return c;
}

Vector RecurrentUnit::backwardPropagation(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	r_gradientStorage -= dy * c; 
	//get input error
	Vector& dx = (w.T() * dy) & g.d(Xt());
	//update backprop storage
	bpX.pop_back();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation(dx);
	}
	else
		return dx;
}

Vector RecurrentUnit::backwardPropagation_ThroughTime(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	r_gradientStorage -= dy * Ct();
	//get input error
	Vector& dx = (w.T() * dy) & g.d(Xt());
	//update backprop storage
	bpX.pop_back();
	bpC.pop_back();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation_ThroughTime(dx);
	}
	else
		return dx;
}

void RecurrentUnit::clearBPStorage()
{
	bpX.clear();
	bpC.clear();
}

void RecurrentUnit::clearGradients()
{
	Matrix::fill(w_gradientStorage, 0);
	Matrix::fill(r_gradientStorage, 0);
	Vector::fill(b_gradientStorage, 0);
}

void RecurrentUnit::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
	r += r_gradientStorage & lr;
}
