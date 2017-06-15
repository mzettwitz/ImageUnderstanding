#include "include/Calltech_Image_Matrix.h"
#include <iostream>


Calltech_Image_Matrix::Calltech_Image_Matrix()
{
}


Calltech_Image_Matrix::~Calltech_Image_Matrix()
{
}

// load all images in a folder recursively
int Calltech_Image_Matrix::loadImagesFromPath(cv::String path, int width, int height)
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

		dlib::array2d< dlib::bgr_pixel > img ;
		dlib::load_jpeg(img, all_img_paths.at(i));
		if (img.nc() == 0)
		{
			std::cout << "Fehler beim lesen des Bildes mit dem Pfad" << all_img_paths.at(i);
			return 1;
		}
		// resize the image
		dlib::array2d< dlib::bgr_pixel > img_resized(128,128);
		
		dlib::resize_image(img, img_resized);

		// save the image in the right categorie
		
		m_all_image_data[categories_nr].push_back(img_resized);
	}

    // setup the ROI vector
    for(int i = 0; i < m_all_image_data.size(); i++)
    {
        std::vector<dlib::rectangle> rois;
        for(int j = 0; j < m_all_image_data[i].size(); j++)
        {
            dlib::rectangle rect = dlib::rectangle(0,0,width, height);
            rois.push_back(rect);
        }
        m_all_rois.push_back(rois);
    }


	// save the number of categories
	categories_nr++;
	m_nr_categories = categories_nr;

	// output statistics
	std::cout << "Read in " << all_img_paths.size() << " images in " << categories_nr << " different categories | folders" << std::endl;
	return 0;
}
