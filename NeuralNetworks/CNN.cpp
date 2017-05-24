

#include "stdafx.h"
#include "CNN.h"


CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length) :
	LENGTH(input_length),
	WIDTH(input_width),
	STRIDE(1),
	FEATURE_MAP_LENGTH(filter_length),
	FEATURE_MAP_WIDTH(filter_length),
	POOLED_LENGTH((input_length - filter_length / STRIDE) + 1),
	POOLED_WIDTH((input_width - filter_length / STRIDE) + 1),
	Layer(input_length, POOLED_LENGTH * POOLED_WIDTH) {
	
	if (input_length != input_width) {
	std::cout << " input_length and input_width must be equal -- cnn layers currently only support square dimensions [require manual 0 padding]" << std::endl;
	throw std::out_of_range("error");
	}

	f = Matrix(filter_length, filter_length);
	x = Matrix(input_length, input_width);
}

Vector CNN::forwardPropagation_express(const Vector &input)
{
	x = vec_toMatrix(input, LENGTH, WIDTH); //convert the vector to a matrix
	Matrix& pooled = convolution(x, f);
	
	if (next != nullptr) {
		next->forwardPropagation_express(mat_toVector(x)); //flatten the matrix send to next layer
	}
	else return (mat_toVector(x));
}

Vector CNN::forwardPropagation(const Vector & input)
{
	x = vec_toMatrix(input, LENGTH, WIDTH); //convert the vector to a matrix
	Matrix& pooled = convolution(x, f);

	bpX.push_back(x); //store the activations 
	//bpX_indexes.push_back(indexes);

	if (next != nullptr) {
		next->forwardPropagation_express(mat_toVector(x));
	}
	else return (mat_toVector(x));
}

Vector CNN::backwardPropagation(const Vector & dy)
{
	//dy at index 
	//w_gradientStorage -= dy * a;
	//create a Matrix[] set delta at apropriate indexes

	//for (int i = 0; i; i < matrix[].length; ++i) {
	//f_gradientStorage -= matrix_delta[i] * a; //need to store activations ?? don't actually need to store the big x
	//d_gradientSor
	//}
	//
}

Vector CNN::backwardPropagation_ThroughTime(const Vector & dy)
{
return Vector();
}

void CNN::clearBPStorage()
{
}

void CNN::clearGradients()
{
}

void CNN::updateGradients()
{
}

Matrix CNN::convolution(const Matrix & img, const Matrix & feature)
{
	std::vector<Matrix> convolved_imgs; 

	for (int x = 0; x < img.length(); ++x) {
		for (int y = 0; y < img.width(); ++y) {
			Matrix& conv = img.sub_Matrix(x, y, FEATURE_MAP_LENGTH, FEATURE_MAP_WIDTH);
			Matrix& filtered = feature & conv + b; 
			convolved_imgs.push_back(filtered);
		}
	}
	//apply relu on the convolved_imgs 
	return maxPooling(convolved_imgs);
}

Matrix CNN::maxPooling(std::vector<Matrix> convolved_mat)
{
	//get the maximums 
	Matrix pooled = Matrix(POOLED_LENGTH, POOLED_WIDTH);

	int index = 0;
	for (int x = 0; x < POOLED_LENGTH; ++x) {
		for (int y = 0; y < POOLED_WIDTH; ++y) {
			pooled[x][y] = max(convolved_mat[index]);
			++index;
		}
	}
	return pooled;
}
