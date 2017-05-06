#ifndef CALLTECH_IMAGE_MATRIX_HPP
#define CALLTECH_IMAGE_MATRIX_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class Calltech_Image_Matrix
{
public:
	Calltech_Image_Matrix();
	~Calltech_Image_Matrix();

	int loadImagesFromPath(cv::String);

    std::vector < std::vector < cv::Mat > > getAllImages()	{ return m_all_image_data; }
    std::vector < cv::Mat > getAllImagesOfIthClass(int i)	{ return m_all_image_data.at(i); }
	cv::Mat getIthImageOfJthCategory(int i, int j)			{ return getAllImagesOfIthClass(j).at(i); }

    std::vector < cv::String > getAllCategoriesNames()		{ return m_categories_names; }
    cv::String getIthCategoryName(int i)					{ return m_categories_names.at(i); }

private:
	std::vector < std::vector < cv::Mat > > m_all_image_data;
	std::vector < cv::String > m_categories_names;

	size_t m_nr_categories;
};

#endif
