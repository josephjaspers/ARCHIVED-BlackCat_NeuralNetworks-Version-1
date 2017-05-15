#ifndef GRULayer_h
#define GRULayer_h
#include "Layer.h"

class GRU : public Layer {

	nonLinearityFunct f_g; //forget nonlinearity funct 
	nonLinearityFunct z_g; //input gate nonlinearity function 

	Vector c;
	Vector dc;

	Vector x;
	Vector f;
	Vector z;

	Vector bz;
	Vector dz;
	Matrix wz;
	Matrix rz;

	Vector bf;
	Vector df;
	Matrix wf;
	Matrix rf;

	Vector bz_gradientStorage;
	Matrix wz_gradientStorage;
	Matrix rz_gradientStorage;

	Vector bf_gradientStorage;
	Matrix wf_gradientStorage;
	Matrix rf_gradientStorage;

	bpStorage bpX;
	bpStorage bpC;
	bpStorage bpF;
	bpStorage bpZ;

	const Vector& Xt(); //inputs
	const Vector& Ct(); 
	const Vector& Ct_1();
	const Vector& Ft();
	const Vector& Zt();

public:
	GRU(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();
};
#endif


