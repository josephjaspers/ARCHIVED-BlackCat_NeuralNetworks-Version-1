#include "stdafx.h"
#include "GRU.h"
const Vector & GRU::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

const Vector & GRU::Ct()
{
	if (bpC.empty()) {
		return OUTPUT_ZEROS;
	}
	return bpC.back();
}

const Vector & GRU::Ct_1()
{
	if (bpC.size() < 2) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpC[bpC.size() - 2]; //return second to last 
	}
}

const Vector & GRU::Ft()
{
	if (bpF.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpF.back();
	}
}

const Vector & GRU::Zt()
{
	if (bpZ.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpZ.back();
	}
}


GRU::GRU(int inputs, int outputs) : Layer(inputs, outputs)
{
	bz_gradientStorage = Vector(outputs);
	wz_gradientStorage = Matrix(outputs, inputs);
	rz_gradientStorage = Matrix(outputs, outputs);

	bf_gradientStorage = Vector(outputs);
	wf_gradientStorage = Matrix(outputs, inputs);
	rf_gradientStorage = Matrix(outputs, outputs);

	c = Vector(outputs);
	dc = Vector(outputs);

	x = Vector(inputs);

	z = Vector(outputs);
	bz = Vector(outputs);
	dz = Vector(outputs);
	wz = Matrix(outputs, inputs);
	rz = Matrix(outputs, outputs);
	
	f = Vector(outputs);
	bf = Vector(outputs);
	df = Vector(outputs);
	wf = Matrix(outputs, inputs);
	rf = Matrix(outputs, outputs);

	Matrices::randomize(bz, -4, 4);
	Matrices::randomize(wz, -4, 4);
	Matrices::randomize(rz, -4, 4);

	//initialize forget gate in negative range so network must be trained to "remember"
	Matrices::randomize(bf, -4, 0);
	Matrices::randomize(wf, -4, 0);
	Matrices::randomize(rf, -4, 0); 

}

Vector GRU::forwardPropagation_express(const Vector & input)
{
	x = input;

	f = f_g(wf * x + rf * c + bf);
	z = z_g(wz * x + rz * c + bz);

	c &= f;
	c += z;

	if (next != nullptr)
		return next->forwardPropagation_express(c);
	else
		return c;
}

Vector GRU::forwardPropagation(const Vector & input)
{
	//store the current activations and cellstate
	bpX.push_back(x);
	bpC.push_back(c);
	bpF.push_back(f);
	bpZ.push_back(z);
	//Math 
	x = input;
	f = f_g(wf * x + rf * c + bf);
	z = z_g(wz * x + rz * c + bz);

	c &= f;
	c += z;

	//continue forwardprop
	if (next != nullptr)
		return next->forwardPropagation(c);
	else
		return c;
}

Vector GRU::backwardPropagation(const Vector & dy)
{
	//calculate gate errors
	dc = dy;
	dz = dc & z_g.d(z);
	df = dc & Ct() & f_g.d(f);
	//Store gradients
	wz_gradientStorage -= (dz * x);
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * c;

	wf_gradientStorage -= (df * x);
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * c;
	//calculate input error
	Vector& dx = (wz.T() * dz + wf.T() * df);// &g.d(Xt());
	//send error through forget gate 
	dc &= f;
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation(dx);
	}
	else
		return dx;
}

Vector GRU::backwardPropagation_ThroughTime(const Vector & dy)
{
	//calculate delta c error 
	dc += dy + rz.T() * dz + rf.T() * df;
	df = dc & Ct_1() & f_g.d(Ft());
	dz = dc & g.d(Zt());
	//Store gradients 
	wz_gradientStorage -= dz * Xt();
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * Ct();

	wf_gradientStorage -= df * Xt();
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * Ct();
	//get input error
	Vector& dx = (wz.T() * dz + wf.T() * df);
	//send the error through the gate 
	dc &= Ft();
	//update backprop storage
	bpX.pop_back();
	bpC.pop_back();
	bpF.pop_back();
	bpZ.pop_back();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation_ThroughTime(dx);
	}
	else
		return dx;
}

void GRU::clearBPStorage()
{
	bpF.clear();
	bpZ.clear();
	bpX.clear();
	bpC.clear();
}

void GRU::clearGradients()
{
	Matrix::fill(wz_gradientStorage, 0);
	Matrix::fill(rz_gradientStorage, 0);
	Vector::fill(bz_gradientStorage, 0);

	Matrix::fill(wf_gradientStorage, 0);
	Matrix::fill(rf_gradientStorage, 0);
	Vector::fill(bf_gradientStorage, 0);
}

void GRU::updateGradients()
{
	wz += wz_gradientStorage & lr;
	bz += bz_gradientStorage & lr;
	rz += rz_gradientStorage & lr;

	wf += wf_gradientStorage & lr;
	bf += bf_gradientStorage & lr;
	rf += rf_gradientStorage & lr;

}

