#ifndef KFOLDVALIDATION_HPP
#define KFOLDVALIDATION_HPP

#include <opencv2/core/core.hpp>

class KFoldValidation
{
public:
	KFoldValidation();
	~KFoldValidation();
	int create10Fold(std::vector < std::vector < cv::Mat > > m_all_image_data_of_class);
};
#endif