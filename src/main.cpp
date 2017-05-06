#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/tracking.hpp>
#include "Calltech_Image_Matrix.h"

int main(void)
{
std::cout << "Hello fellas!\n";

cv::Vec2d vec = cv::Vec2d(0,0);
std::cout << "CV Test: output of zero vector: x = " << vec[0] << ", y = " << vec[1] << std::endl;


// test if loading works
cv::String path = "C:/101_ObjectCategories/*.jpg";
Calltech_Image_Matrix img_Matrix;
if (img_Matrix.loadImagesFromPath(path) != 0) return 0;

return 0;
}
