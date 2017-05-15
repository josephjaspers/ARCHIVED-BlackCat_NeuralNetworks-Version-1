#ifndef LSTMLayer_h
#define LSTMLayer_h
#include "Layer.h"

class LSTM : public Layer {

	nonLinearityFunct f_g; //forget nonlinearity funct 
	nonLinearityFunct z_g; //input gate nonlinearity function 
	nonLinearityFunct i_g; //input gate nonlinearity function 
	nonLinearityFunct o_g; //output gate nonLinearity function

	Vector c;
	Vector dc;

	Vector y;

	Vector x;
	Vector f;
	Vector z;
	Vector i;
	Vector o;

	Vector bz;
	Vector dz;
	Matrix wz;
	Matrix rz;

	Vector bi;
	Vector di;
	Matrix wi;
	Matrix ri;

	Vector bf;
	Vector df;
	Matrix wf;
	Matrix rf;

	Vector bo;
	Vector od; //inversed as "do" is in stdlib
	Matrix wo;
	Matrix ro;

	Vector bz_gradientStorage;
	Matrix wz_gradientStorage;
	Matrix rz_gradientStorage;

	Vector bi_gradientStorage;
	Matrix wi_gradientStorage;
	Matrix ri_gradientStorage;

	Vector bf_gradientStorage;
	Matrix wf_gradientStorage;
	Matrix rf_gradientStorage;

	Vector bo_gradientStorage;
	Matrix wo_gradientStorage;
	Matrix ro_gradientStorage;

	bpStorage bpX;
	bpStorage bpC;
	bpStorage bpF;
	bpStorage bpZ;
	bpStorage bpI;
	bpStorage bpO;
	bpStorage bpY;


	const Vector& Xt(); //inputs
	const Vector& Ct();
	const Vector& Ct_1();
	const Vector& Ft();
	const Vector& Zt();
	const Vector& It();
	const Vector& Ot();
	const Vector& Yt();

public:
	LSTM(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();


private:
	void storeGradients();
	void storeGradients_BPTT();
	void bpStorage_pop_back_all();
	void updateBPStorage();
};
#endif


