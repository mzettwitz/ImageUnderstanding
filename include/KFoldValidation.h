#pragma once

#ifndef KFOLDVALIDATION_HPP
#define KFOLDVALIDATION_HPP

 // #include <opencv2/core/core.hpp>
#include "include/ClassifierData.h"

class KFoldValidation
{
public:
	KFoldValidation();
	~KFoldValidation();
	int create10Fold(std::vector < std::vector <  cv::Mat  > > &m_all_image_data_of_class);
	int findImageNumberOfSmallestClass(std::vector < std::vector < cv::Mat > > &m_all_image_data);
	std::vector <std::vector < std::vector < cv::Mat > > >   createInitialFolds(int Folds,int numberOfImages, std::vector < std::vector < cv::Mat > > &all_image_data);
	
	std::vector <std::vector < std::vector < cv::Mat > > > m_initalFolds; // class + fold + image
	int smallestClassSize;
	std::vector< ClassifierData > getClassifierData() { return m_classifier; }
private:
	std::vector< ClassifierData > m_classifier; // vector of all classifieres
};
#endif