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

class Calltech_Image_Matrix
{
public:
	Calltech_Image_Matrix();
	~Calltech_Image_Matrix();

	int loadImagesFromPath(cv::String);

	dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >&		getAllImages()							{ return m_all_image_data; }
	dlib::array< dlib::array2d < dlib::bgr_pixel > >&						getAllImagesOfIthClass(int i)			{ return m_all_image_data[i]; }
	dlib::array2d < dlib::bgr_pixel >&										getIthImageOfJthCategory(int i, int j)	{ return getAllImagesOfIthClass(j)[i]; }

    std::vector < cv::String >												getAllCategoriesNames()					{ return m_categories_names; }
    cv::String																getIthCategoryName(int i)				{ return m_categories_names.at(i); }

    size_t																	getNrCategories()                       { return m_nr_categories;}

private:
	dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >	m_all_image_data;
	std::vector < cv::String >											m_categories_names;

	size_t m_nr_categories;
};

#endif
