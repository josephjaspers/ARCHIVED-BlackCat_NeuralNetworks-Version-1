#include "stdafx.h"
#include "FeedForward.h"
#include "GRU.h"
#include "RecurrentUnit.h"
#include "NeuralNetwork.h"
#include "LSTM.h"
using namespace std;


using namespace std;

namespace testClass {
	void print(vector<double> v) {
		cout.precision(1);
		for (double dz : v) {
			cout << dz << " ";
		}
		cout << endl;

	}
	void printConf(vector<double> v) {
		cout.precision(2);
		int index = -1;
		double best = 0;
		for (int i = 0; i < v.size(); ++i) {
			if (v[i] > best) {
				best = v[i];
				index = i;
			}
		}
		cout << "(" << index << ")" << " conf (" << best << ")" << endl << endl;
	}

	vector<vector<double>> getZero() {
		vector<double> r1 = { 0,0,1,0,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,0,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,0,1,0,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getOne() {
		vector<double> r1 = { 0,1,1,0,0 };
		vector<double> r2 = { 0,0,1,0,0 };
		vector<double> r3 = { 0,0,1,0,0 };
		vector<double> r4 = { 0,0,1,0,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getTwo() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,0,1,0,0 };
		vector<double> r4 = { 0,1,0,0,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getThree() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getFour() {
		vector<double> r1 = { 0,1,0,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,1 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,0,0,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getFive() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,0,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getSix() {
		vector<double> r1 = { 0,1,0,0,0 };
		vector<double> r2 = { 0,1,0,0,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getSeven() {
		vector<double> r1 = { 0,1,1,0,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,0,1,1,0 };
		vector<double> r4 = { 0,1,0,0,0 };
		vector<double> r5 = { 0,1,0,0,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getEight() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> getNine() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,0,0,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
}

using namespace testClass;

void printConf(Vector& x) {
	x.print();

	int best = -1;
	double bestVal = -1;
	for (int i = 0; i < x.length(); ++i) {
		if (bestVal < x[i]) {
			bestVal = x[i];
			best = i;
		}
	}
	cout << "~~ (" << best << ") " << " conf: " << bestVal;
}

void drTest() {


	NeuralNetwork network;

	network.push_back(new LSTM(5, 25));
	network.push_back(new FeedForward(25, 25));
	network.push_back(new GRU(25, 10));
	network.push_back(new FeedForward(10, 10));

	//	network.push_back(new RecurrentUnit(10, 10));


	int train = 1;
	while (train > 0) {
		cout.precision(1);
		cout << " testing 0 " << endl;
		Vector& t0 = network.predict(getZero());
		printConf(t0);
		cout << endl << " testing 1 " << endl;
		Vector& t1 = network.predict(getOne());
		printConf(t1);

		cout << endl << " testing 2 " << endl;
		Vector& t2 = network.predict(getTwo());
		printConf(t2);

		cout << endl << " testing 3 " << endl;
		Vector& t3 = network.predict(getThree());
		printConf(t3);

		cout << endl << " testing 4 " << endl;
		Vector& t4 = network.predict(getFour());
		printConf(t4);

		cout << endl << " testing 5 " << endl;
		Vector& t5 = network.predict(getFive());
		printConf(t5);

		cout << endl << " testing 6 " << endl;
		Vector& t6 = network.predict(getSix());
		printConf(t6);

		cout << endl << " testing 7 " << endl;
		Vector& t7 = network.predict(getSeven());
		printConf(t7);

		cout << endl << " testing 8 " << endl;
		Vector& t8 = network.predict(getEight());
		printConf(t8);

		cout << endl << " testing 9" << endl;
		Vector& t9 = network.predict(getNine());
		printConf(t9);

		cout << endl;

		cout << " input training iterations " << endl;
		cin >> train;
		for (int i = 0; i < train; ++i) {
			network.train(getZero(), vector<double> {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(getOne(), vector<double>  {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(getTwo(), vector<double>  {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
			network.train(getThree(), vector<double>{0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
			network.train(getFour(), vector<double> {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
			network.train(getFive(), vector<double> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
			network.train(getSix(), vector<double>  {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
			network.train(getSeven(), vector<double>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
			network.train(getEight(), vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
			network.train(getNine(), vector<double> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
		}
	}
}

int main() {
	drTest();
	//initialize training set 
	Vector i1(std::vector<double> {0, 0});
	Vector i2(std::vector<double> {1, 1});
	Vector i3(std::vector<double> {1, 0});
	Vector i4(std::vector<double> {0, 1});
	Vector o1(std::vector<double> {1});
	Vector o2(std::vector<double> {1});
	Vector o3(std::vector<double> {0});
	Vector o4(std::vector<double> {0});
	//initialize network
	NeuralNetwork network;
	network.push_back(new FeedForward(2, 12)); //generate a feedforward layer with 2 inputs 5 outputs
	network.push_back(new FeedForward(12, 10));
	network.push_back(new FeedForward(10, 1)); //generate a feedforwad layer with 5 inputs 1 outputs 

	int train = 1;
	while (train > 0) {
	
	cout << " testing 1, 1 " << endl;
	network.predict(i1).print();
	cout << endl << " testing 0, 0 " << endl;
	network.predict(i2).print();
	cout << endl << " testing 1, 0 " << endl;
	network.predict(i3).print();
	cout << endl << " testing 0, 1 " << endl;
	network.predict(i4).print();
	cout << endl;

	cout << " input training iterations " << endl;
	cin >> train;
	for (int i = 0; i < train; ++i) {
		network.train(i1, o1);
		network.train(i2, o2);
		network.train(i3, o3);
		network.train(i4, o4);
	}
	}
}
