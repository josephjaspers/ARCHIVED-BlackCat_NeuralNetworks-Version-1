#include "stdafx.h"
#include "LSTM.h"
const Vector & LSTM::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

const Vector & LSTM::Ct()
{
	if (bpC.empty()) {
		return OUTPUT_ZEROS;
	}
	return bpC.back();
}

const Vector & LSTM::Ct_1()
{
	if (bpC.size() < 2) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpC[bpC.size() - 2]; //return second to last 
	}
}

const Vector & LSTM::Ft()
{
	if (bpF.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpF.back();
	}
}

const Vector & LSTM::Zt()
{
	if (bpZ.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpZ.back();
	}
}

const Vector & LSTM::It()
{
	if (bpI.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpI.back();
	}
}

const Vector & LSTM::Ot()
{
	if (bpO.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpO.back();
	}
}

const Vector & LSTM::Yt()
{
	if (bpY.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpY.back();
	}
}


LSTM::LSTM(int inputs, int outputs) : Layer(inputs, outputs)
{
	z_g.setTanh();
	i_g.setSigmoid();
	f_g.setSigmoid();
	o_g.setSigmoid();
	g.setTanh();

	bz_gradientStorage = Vector(outputs);
	wz_gradientStorage = Matrix(outputs, inputs);
	rz_gradientStorage = Matrix(outputs, outputs);

	bi_gradientStorage = Vector(outputs);
	wi_gradientStorage = Matrix(outputs, inputs);
	ri_gradientStorage = Matrix(outputs, outputs);

	bf_gradientStorage = Vector(outputs);
	wf_gradientStorage = Matrix(outputs, inputs);
	rf_gradientStorage = Matrix(outputs, outputs);

	bo_gradientStorage = Vector(outputs);
	wo_gradientStorage = Matrix(outputs, inputs);
	ro_gradientStorage = Matrix(outputs, outputs);

	c = Vector(outputs);
	dc = Vector(outputs);

	x = Vector(inputs);
	y = Vector(outputs);

	z = Vector(outputs);
	bz = Vector(outputs);
	dz = Vector(outputs);
	wz = Matrix(outputs, inputs);
	rz = Matrix(outputs, outputs);

	i = Vector(outputs);
	bi = Vector(outputs);
	di = Vector(outputs);
	wi = Matrix(outputs, inputs);
	ri = Matrix(outputs, outputs);

	f = Vector(outputs);
	bf = Vector(outputs);
	df = Vector(outputs);
	wf = Matrix(outputs, inputs);
	rf = Matrix(outputs, outputs);

	o = Vector(outputs);
	bo = Vector(outputs);
	od = Vector(outputs);
	wo = Matrix(outputs, inputs);
	ro = Matrix(outputs, outputs);

	Matrices::randomize(bz, 0, 4);
	Matrices::randomize(wz, 0, 4);
	Matrices::randomize(rz, 0, 4);

	Matrices::randomize(bi, -4, 4);
	Matrices::randomize(wi, -4, 4);
	Matrices::randomize(ri, -4, 4);

	//initialize forget gate in negative range so network must be trained to "remember"
	Matrices::randomize(bf, -4, 0);
	Matrices::randomize(wf, -4, 0);
	Matrices::randomize(rf, -4, 0);
	//start in positive range (output everything)
	Matrices::randomize(bo, 0, 5);
	Matrices::randomize(wo, 0, 5);
	Matrices::randomize(ro, 0, 5);
}

Vector LSTM::forwardPropagation_express(const Vector & input)
{
	x = input;

	f = f_g(wf * x + rf * y + bf);
	z = z_g(wz * x + rz * y + bz);
	i = i_g(wi * x + ri * y + bi);
	o = o_g(wo * x + ro * y + bo);

	c &= f;
	c += (z & i);

	y = (g.nonLin(c) & o); //apply non linearity to a copy of the cell state (g(Vector& x) -- accepts a reference to x while g.nonLin() returns a crushed copy

	if (next != nullptr)
		return next->forwardPropagation_express(y);
	else
		return y;
}

Vector LSTM::forwardPropagation(const Vector & input)
{
	updateBPStorage(); //stores all the current activations 
	//Math 
	x = input;

	f = f_g(wf * x + rf * y + bf);
	z = z_g(wz * x + rz * y + bz);
	i = i_g(wi * x + ri * y + bi);
	o = o_g(wo * x + ro * y + bo);

	c &= f;
	c += (z & i);

	y = g.nonLin(c) & o;

	//continue forwardprop
	if (next != nullptr)
		return next->forwardPropagation(y);
	else
		return y;
}

Vector LSTM::backwardPropagation(const Vector & dy)
{
	//calculate gate errors
	dc = dy & o & g.d(c);
	od = dy & g.nonLin(c) & o_g.d(o);
	df = dc & Ct() & f_g.d(f);
	dz = dc & i & z_g.d(z);
	di = dc & z & i_g.d(i);
	dc &= f; //update cell error
	//Store gradients
	storeGradients();
	//calculate input error
	Vector& dx = (wz.T() * dz + wf.T() * df + wi.T() * di + wo.T() * od);


	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation(dx);
	}
	else
		return dx;
}

Vector LSTM::backwardPropagation_ThroughTime(const Vector & deltaError)
{
	//calculate delta 
	Vector& dy = deltaError + rz.T() * dz + ri.T() * di + rf.T() * df + ro.T() * od;
	//math of error 
	dc += dy & g.d(y) & Ot();
	od = dc & g.nonLin(Ct()) & o_g.d(Ot());
	df = dc & Ct_1() & f_g.d(Ft());
	dz = dc & It() & z_g.d(Zt());
	di = dc & Zt() & i_g.d(It());
	//Store gradients 

	//get input error
	Vector& dx = (wz.T() * dz) + (wf.T() * df) + (wi.T() * di) + (wo.T() * od);
	//send the error through the gate 
	dc &= Ft();
	//update backprop storage
	bpStorage_pop_back_all();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation_ThroughTime(dx);
	}
	else
		return dx;
}

void LSTM::clearBPStorage()
{
	bpF.clear();
	bpZ.clear();
	bpX.clear();
	bpC.clear();
	bpI.clear();
	bpO.clear();
}

void LSTM::clearGradients()
{
	Matrix::fill(wz_gradientStorage, 0);
	Matrix::fill(rz_gradientStorage, 0);
	Vector::fill(bz_gradientStorage, 0);

	Matrix::fill(wi_gradientStorage, 0);
	Matrix::fill(ri_gradientStorage, 0);
	Vector::fill(bi_gradientStorage, 0);

	Matrix::fill(wf_gradientStorage, 0);
	Matrix::fill(rf_gradientStorage, 0);
	Vector::fill(bf_gradientStorage, 0);
}

void LSTM::updateGradients()
{
	wz += wz_gradientStorage & lr;
	bz += bz_gradientStorage & lr;
	rz += rz_gradientStorage & lr;

	wi += wi_gradientStorage & lr;
	bi += bi_gradientStorage & lr;
	ri += ri_gradientStorage & lr;

	wf += wf_gradientStorage & lr;
	bf += bf_gradientStorage & lr;
	rf += rf_gradientStorage & lr;

}

void LSTM::storeGradients()
{
	wz_gradientStorage -= dz * x;
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * c;

	wf_gradientStorage -= df * x;
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * c;

	wi_gradientStorage -= di * x;
	bi_gradientStorage -= di;
	ri_gradientStorage -= di * c;

	wo_gradientStorage -= od * x;
	bo_gradientStorage -= od;
	ro_gradientStorage -= od * c;
}

void LSTM::storeGradients_BPTT()
{
	wz_gradientStorage -= dz * Xt();
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * Yt();

	wi_gradientStorage -= di * Xt();
	bi_gradientStorage -= di;
	ri_gradientStorage -= di * Yt();

	wi_gradientStorage -= od * Xt();
	bi_gradientStorage -= od;
	ri_gradientStorage -= od * Yt();

	wf_gradientStorage -= df * Xt();
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * Yt();
}

void LSTM::bpStorage_pop_back_all()
{
	bpX.pop_back();
	bpC.pop_back();
	bpF.pop_back();
	bpZ.pop_back();
	bpI.pop_back();
	bpY.pop_back();
}

void LSTM::updateBPStorage()
{
	bpX.push_back(x);
	bpC.push_back(c);
	bpF.push_back(f);
	bpZ.push_back(z);
	bpI.push_back(i);
	bpO.push_back(o);
	bpY.push_back(y);
}

