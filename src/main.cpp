#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/tracking.hpp>
#include "include/Calltech_Image_Matrix.h"
#include "include/KFoldValidation.h"
int main(void)
{
    // Test for correct integration of opencv + contrib
    std::cout << "Hello fellas!\n";

    cv::Vec2d vec = cv::Vec2d(0,0);
    std::cout << "CV Test: output of zero vector: x = " << vec[0] << ", y = " << vec[1] << std::endl;


    // test if loading works
    // data organization: make a new directory 'data' in the build directory and extract the
    // 'Caltech101'(101_ObjectCategories) data container into it
    cv::String path = "data/101_ObjectCategories/*.jpg";
    Calltech_Image_Matrix img_Matrix;
	if (img_Matrix.loadImagesFromPath(path) != 0) return 0;

	// put image matrix in the 10Fold
//	KFoldValidation Validation;
//	Validation.create10Fold(img_Matrix.getAllImages());
	
	return 0;
}
