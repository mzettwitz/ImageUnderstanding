#pragma once
#ifndef DLIB_JPEG_SUPPORT
#define DLIB_JPEG_SUPPORT
#endif
#ifndef CALLTECH_IMAGE_MATRIX_HPP
#define CALLTECH_IMAGE_MATRIX_HPP


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms/interpolation_abstract.h>

#include <iostream>
#include <fstream>
typedef dlib::matrix < float, 1116, 1 > sample_type;

class Calltech_Image_Matrix
{
public:
    Calltech_Image_Matrix();
    ~Calltech_Image_Matrix();

	int loadImagesFromPath(cv::String path, int width, int height, int cellsize, int rowpadding, int colpadding, const int featureSize);

    dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >&					getAllImages()							{ return m_all_image_data; }
    dlib::array< dlib::array2d < dlib::bgr_pixel > >&									getAllImagesOfIthClass(int i)			{ return m_all_image_data[i]; }
    dlib::array2d < dlib::bgr_pixel >&													getIthImageOfJthCategory(int i, int j)	{ return getAllImagesOfIthClass(j)[i]; }

	std::vector < std::vector < dlib::array2d < dlib::matrix<float, 31, 1> > > >&		getAllFeaturesOfImages()						{ return m_all_feature_data; }
	std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > >&							getAllFeaturesIFImagesOfIthClass(int i)			{ return m_all_feature_data[i]; }
	dlib::array2d < dlib::matrix<float, 31, 1> >&										getFeatureOfIthImageOfJthCategory(int i, int j) { return getAllFeaturesIFImagesOfIthClass(j)[i]; }
		
	std::vector < std::vector < sample_type > >&										getAllFlatFeaturesOfImages()						{ return m_all_flat_features; }
	std::vector < sample_type >&														getAllFlatFeaturesIFImagesOfIthClass(int i)			{ return m_all_flat_features[i]; }
	sample_type&																		getFlatFeatureOfIthImageOfJthCategory(int i, int j) { return getAllFlatFeaturesIFImagesOfIthClass(j)[i]; }


    std::vector<std::vector<dlib::rectangle> >&											getAllROIs()                            { return m_all_rois; }

    std::vector < cv::String >															getAllCategoriesNames()					{ return m_categories_names; }
    cv::String																			getIthCategoryName(int i)				{ return m_categories_names.at(i); }

    size_t																				getNrCategories()                       { return m_nr_categories;}


private:
    dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >					m_all_image_data;
	std::vector < std::vector <dlib::array2d<dlib::matrix<float, 31, 1> > > >			m_all_feature_data;
	std::vector < std::vector < sample_type > >											m_all_flat_features;
    std::vector<std::vector<dlib::rectangle> >											m_all_rois;
    std::vector < cv::String >															m_categories_names;

    size_t m_nr_categories;
};

#endif
