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

typedef std::vector < std::vector < std::vector < dlib::array2d < dlib::bgr_pixel >* > > >		img_array3ptr;
typedef std::vector < std::vector < std::vector < dlib::array2d < dlib::bgr_pixel > > > >		img_array3;
typedef std::vector < std::vector < dlib::array2d < dlib::bgr_pixel > > >						img_array2;
typedef std::vector < dlib::array2d < dlib::bgr_pixel>* >										img_arrayptr;
typedef std::vector < dlib::array2d < dlib::bgr_pixel> >										img_array;
class Calltech_Image_Matrix
{
public:
	Calltech_Image_Matrix();
	~Calltech_Image_Matrix();

    int loadImagesFromPath(cv::String, int width, int height);

	img_array2&																getAllImages()							{ return m_all_image_data; }
	img_array&																getAllImagesOfIthClass(int i)			{ return m_all_image_data[i]; }
	dlib::array2d < dlib::bgr_pixel >&										getIthImageOfJthCategory(int i, int j)	{ return getAllImagesOfIthClass(j)[i]; }
    std::vector<std::vector<dlib::rectangle> >&                             getAllROIs()                            { return m_all_rois; }

    std::vector < cv::String >												getAllCategoriesNames()					{ return m_categories_names; }
    cv::String																getIthCategoryName(int i)				{ return m_categories_names.at(i); }

    size_t																	getNrCategories()                       { return m_nr_categories;}


private:
	img_array2															m_all_image_data;
    std::vector<std::vector<dlib::rectangle> >                          m_all_rois;
	std::vector < cv::String >											m_categories_names;

	size_t m_nr_categories;
};

#endif
