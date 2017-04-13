#include <iostream>
#include <opencv2/tracking.hpp>

int main(void)
{
std::cout << "Hello fellas!\n";

cv::Vec2d vec = cv::Vec2d(0,0);
std::cout << "CV Test: output of zero vector: x = " << vec[0] << ", y = " << vec[1] << std::endl;

return 0;
}
