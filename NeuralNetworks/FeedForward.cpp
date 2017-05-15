#include "stdafx.h"
#include "FeedForward.h"
const Vector & FeedForward::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

FeedForward::FeedForward(int inputs, int outputs) : Layer(inputs, outputs)
{
	b_gradientStorage = Vector(outputs);
	w_gradientStorage = Matrix(outputs, inputs);

	b = Vector(outputs);
	w = Matrix(outputs, inputs);

	Matrices::randomize(b, -4, 4);
	Matrices::randomize(w, -4, 4);
}

Vector FeedForward::forwardPropagation_express(const Vector & x)
{
	if (next != nullptr)
		return next->forwardPropagation_express(g(w * x + b));
	else
		return g(w * x + b);
}

Vector FeedForward::forwardPropagation(const Vector & x)
{
	bpX.push_back(x); //store the inputs

	if (next != nullptr)
		return next->forwardPropagation(g(w * x + b));
	else
		return g(w * x + b);
}

Vector FeedForward::backwardPropagation(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	//input delta
	Vector& dx = w.T() * dy & g.d(Xt());
	//update storage
	bpX.pop_back();
	//continue backprop
	if (prev != nullptr)
		return prev->backwardPropagation(dx);
	else
		return dx;
}

Vector FeedForward::backwardPropagation_ThroughTime(const Vector & dy)
{	

	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	//input delta
	Vector& dx = w.T() * dy & g.d(Xt());
	//update storage
	bpX.pop_back();
	//continue backprop
	if (prev != nullptr)
		return prev->backwardPropagation_ThroughTime(dx);
	else
		return dx;
}

void FeedForward::clearBPStorage()
{
	bpX.clear();
}

void FeedForward::clearGradients()
{
	Matrix::fill(w_gradientStorage, 0);
	Vector::fill(b_gradientStorage, 0);
}

void FeedForward::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
}
