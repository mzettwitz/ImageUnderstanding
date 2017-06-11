#ifndef KFOLDVALIDATION_HPP
#define KFOLDVALIDATION_HPP

#include <opencv2/core/core.hpp>

class KFoldValidation
{
public:
	KFoldValidation();
	~KFoldValidation();
	int create10Fold(std::vector < std::vector < cv::Mat > > m_all_image_data_of_class);
	int findImageNumberOfSmallestClass(std::vector < std::vector < cv::Mat > > m_all_image_data);
	std::vector <std::vector < std::vector < cv::Mat > > >   createInitialFolds(int Folds,int numberOfImages, std::vector < std::vector < cv::Mat > > all_image_data);
	
	std::vector <std::vector < std::vector < cv::Mat > > > m_initalFolds; // class + fold + image
	int smallestClassSize;
};
#endif