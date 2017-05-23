

#include "stdafx.h"
#include "CNN.h"

using namespace CNN_names;

CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length) : 
Layer(input_length * input_width, (input_length-filter_length)/2), 
LENGTH(input_length), WIDTH(input_width), 
NUMB_FEATURES(numb_filters), FEATURE_MAP_LENGTH(filter_length), FEATURE_MAP_WIDTH(filter_length), 
POOLED_LENGTH(((LENGTH - FEATURE_MAP_LENGTH) / STRIDE) + 1),
POOLED_WIDTH(((WIDTH - FEATURE_MAP_WIDTH) / STRIDE) + 1),
STRIDE(1), DEPTH(1)
{
if (input_length != input_width) {
std::cout << " input_length and input_width must be equal -- cnn layers currently only support square dimensions [require manual 0 padding]" << std::endl;
throw std::out_of_range("error");
}

//output dimensions = (n-f)/(stride + 1)
// (length-filterlength) / (stride + 1)
}

Vector CNN::forwardPropagation_express(const Vector & x)
{
	//convert the inputs to matrix format 
	Matrix img = Matrices::vec_toMatrix(x, LENGTH, WIDTH);


	for (int i = 0; i < NUMB_FEATURES; ++i) {
		featureMapStorage filtered_maps = convolution(f_maps[i], img);			//get the convolved images 
	//	bp_maps.push_back(filter)
	}
}

Vector CNN::forwardPropagation(const Vector & x)
{
return Vector();
}

Vector CNN::backwardPropagation(const Vector & dy)
{
return Vector();
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

featureMapStorage CNN::convolution(const Matrix & img, const Matrix & feature)
{
	featureMapStorage filtered_images; 
	for (int x = 0; x < img.length(); ++x) {
		for (int y = 0; y < img.width(); ++y) {
			Matrix& partition = img.sub_Matrix(x, y, FEATURE_MAP_LENGTH, FEATURE_MAP_WIDTH);
			Matrix& filtered_img = feature * partition;		//take dot product of filtered image
			filtered_images.push_back(filtered_img);		//add it to the collection 
		}
	}
}
