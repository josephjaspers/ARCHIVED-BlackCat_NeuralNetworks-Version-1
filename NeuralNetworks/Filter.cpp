#include "stdafx.h"
#include "Filter.h"
Matrix Filter::Xt()
{
	if (bpX.empty()) {
		return Matrix(LENGTH, WIDTH);
	}
	else {
		return bpX.back();
	}
}

int Filter::index_Xt()
{
	if (bp_max_index_x.empty()) {
		return -1;
	}
	else {
		return bp_max_index_x.back();
	}
}

int Filter::index_Yt()
{
	if (bp_max_index_y.empty()) {
		return -1;
	}
	else {
		return bp_max_index_y.back();
	}
}

Filter::Filter(unsigned input_length, unsigned input_width, unsigned filter_length, unsigned stride) :
	LENGTH(input_length), WIDTH(input_width),
	FEATURE_LENGTH(filter_length), FEATURE_WIDTH(filter_length),
	STRIDE(stride),
	POOLED_LENGTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s
	POOLED_WIDTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s 
	Layer(input_length * input_width, ((floor(input_length - filter_length / stride) + 1) * (floor(input_width - filter_length) / 1) + 1))
{
	w = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);

	w_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
		
	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);
	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);

	randomize(w, -4, 4);
	randomize(b, -4, 4);


	g.setNonLinearityFunction(0);
}
double Filter::findMax(Matrix& img, int& x_store, int& y_store) {
	double max = img[0][0];
	x_store = 0;
	y_store = 0;
	for (int x = 0; x < img.length(); ++x) {
		for (int y = 0; y < img.width(); ++y) {
			if (max < img[x][y]) {
				max = img[x][y];
				x_store = x;
				y_store = y;
			}
		}
	}
	return max;
}
//doesnt work????
Vector Filter::forwardPropagation_express(const Vector &input)
{
	Matrix img = vec_toMatrix(input, LENGTH, WIDTH);
	Matrix pooled = Matrix(POOLED_LENGTH, POOLED_WIDTH);
	bpX.push_back(img);

	for (int x = 0; x < img.length() - FEATURE_LENGTH + 1; x += STRIDE) {
		for (int y = 0; y < img.width() - FEATURE_WIDTH + 1; y += STRIDE) {
			Matrix a = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
			a = img.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH);
			Matrix conv = g(w * a + b);

			pooled[x / STRIDE][y / STRIDE] = conv.max();
		}
	}

	if (next != nullptr) {
		return next->forwardPropagation_express(mat_toVector(pooled));
	}
	else {
		return mat_toVector(pooled);
	}
}
Vector Filter::forwardPropagation(const Vector & input)
{
	bp_max_index_x.clear();
	bp_max_index_y.clear();

	Matrix img = vec_toMatrix(input, LENGTH, WIDTH);
	Matrix pooled = Matrix(POOLED_LENGTH, POOLED_WIDTH);
	bpX.push_back(img);

	for (int x = 0; x < img.length() - FEATURE_LENGTH + 1; x += STRIDE) {
		for (int y = 0; y < img.width() - FEATURE_WIDTH + 1; y += STRIDE) {
			Matrix a = img.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH);

			Matrix conv = g(w * a + b);

			int max_index_x;
			int max_index_y;
			pooled[x / STRIDE][y / STRIDE] = findMax(conv, max_index_x, max_index_y);

			bp_max_index_x.push_back(max_index_x);
			bp_max_index_y.push_back(max_index_y);
		}
	}

	if (next != nullptr) {
		return next->forwardPropagation(mat_toVector(pooled));
	}
	else {
		return mat_toVector(pooled);
	}
}

Vector Filter::backwardPropagation(const Vector & dy)
{
	Matrix& img_xt = Xt();

	int dy_index = 0;
	for (int x = 0; x < img_xt.length() - FEATURE_LENGTH; x += STRIDE) {
		for (int y = 0; y < img_xt.width() - FEATURE_WIDTH; y += STRIDE) {
			Matrix delta = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
			//delta[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]] = dy.get(dy_index);
			//w_gradientStorage -= delta; &g.d(g(img_xt.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH)));
			//b_gradientStorage -= delta;
			w_gradientStorage[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]]-= dy.get(dy_index); //multiply by derivative?????
			b_gradientStorage[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]] -= dy.get(dy_index);

			dy_index++;
		}
	}
	bpX.pop_back();

	return dy;
}

Vector Filter::backwardPropagation_ThroughTime(const Vector & dy)
{
	std::cout << " BPTT attempt FAILED  __ I HAVENT WRITTEN THIS YET SORRY" << std::endl;
	throw std::invalid_argument("not supported");
}

void Filter::clearBPStorage()
{
	bpX.clear();
	Layer::clearBPStorage();

}

void Filter::clearGradients()
{
	w_gradientStorage.fill(0);
	b_gradientStorage.fill(0);

	Layer::clearGradients();

}

void Filter::updateGradients()
{
	std::cout << std::endl;
	w_gradientStorage.print();
	std::cout << std::endl;
	w.print();
	std::cout << std::endl;

	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
	Layer::updateGradients();

}
