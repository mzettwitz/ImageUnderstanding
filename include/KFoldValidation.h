#pragma once

#ifndef KFOLDVALIDATION_HPP
#define KFOLDVALIDATION_HPP

 // #include <opencv2/core/core.hpp>
#include "include/ClassifierData.h"
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms/interpolation_abstract.h>

#include <iostream>
#include <fstream>
#include <dlib/pixel.h>
#include <dlib/image_processing/generic_image.h>

#include <mutex>

typedef dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >		img_array3ptr;
typedef dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >						img_array2;
typedef dlib::array < dlib::array2d < dlib::bgr_pixel>* >										img_arrayptr;
class KFoldValidation
{
public:
    KFoldValidation();
    KFoldValidation(int nrClasses);
	~KFoldValidation();
	int create10Fold(img_array2 &m_all_image_data_of_class);
	int findImageNumberOfSmallestClass(img_array2 &m_all_image_data);
	img_array3ptr   createInitialFolds(int Folds,int numberOfImages, img_array2 &all_image_data);
    void printErrorMatrix();
	
    std::vector< ClassifierData >& getClassifierData() { return m_classifier; }
    std::vector < std::vector < int > >& getErrorMatrix() {return m_error_matrix;}

private:
    std::mutex mute;
	int smallestClassSize;
	std::vector<int> testClasses;
    int m_NrClasses;
	std::vector < std::vector < int > > m_error_matrix;
	img_array3ptr	m_initalFolds; // class + fold + image
	std::vector< ClassifierData > m_classifier; // vector of all classifieres
    void prepareTraining(int class_i, ClassifierData &classi_data, int k, dlib::array<dlib::array<dlib::array2d<dlib::bgr_pixel> > > &m_all_image_data);
    void trainClass(int class_i, ClassifierData &classi_data, int k, int fold);
};
#endif
