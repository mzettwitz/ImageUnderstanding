#include "include/Calltech_Image_Matrix.h"
#include <iostream>

Calltech_Image_Matrix::Calltech_Image_Matrix()
{
}


Calltech_Image_Matrix::~Calltech_Image_Matrix()
{
}

// load all images in a folder recursively
int Calltech_Image_Matrix::loadImagesFromPath(cv::String path)
{
	// get all image paths 
	std::vector < cv::String  > all_img_paths;
	cv::glob(path, all_img_paths, true);

	cv::String class_name_last = "X";
	size_t categories_nr = -1;

	// loop over all paths
	for (int i = 0; i < all_img_paths.size(); i++)
	{
		// find the class name from the right substring of the img path
		// hope this is plattform independet but not sure

		cv::String class_name_current;
		class_name_current = all_img_paths.at(i);
		size_t pos = class_name_current.find_last_of("/\\");
		class_name_current = class_name_current.substr(0, pos);
		pos = class_name_current.find_last_of("/\\");
		class_name_current = class_name_current.substr(pos + 1);

		// if its a new class, resize our all image vector and increase counter
		// also store the class name
		if (class_name_current.compare(class_name_last) != 0)
		{
			categories_nr++;
			m_categories_names.push_back(class_name_current);
			class_name_last = class_name_current;

			m_all_image_data.resize(categories_nr + 1);
		}

		// load the img and check if data was read in

		cv::Mat img = cv::imread(all_img_paths.at(i));
		if (img.data == NULL)
		{
			std::cout << "Fehler beim lesen des Bildes mit dem Pfad" << all_img_paths.at(i);
			return 1;
		}
		// resize the image, convert it to grayscale and make it flat 
		
		cv::Size size = { 400, 400 };
		cv::resize(img, img, size);
		/*
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		img.convertTo(img, CV_32F);
		img = img.reshape(1, 1);
		// save the image in the right categorie
		*/
		m_all_image_data.at(categories_nr).push_back(img);
	}

	// save the number of categories
	categories_nr++;
	m_nr_categories = categories_nr;

	// output statistics
	std::cout << "Read in " << all_img_paths.size() << " images in " << categories_nr << " different categories | folders" << std::endl;
	return 0;
}
