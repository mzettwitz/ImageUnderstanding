#include "include/Calltech_Image_Matrix.h"
#include <iostream>


Calltech_Image_Matrix::Calltech_Image_Matrix()
{
}

// ---------------------------------------------------------------------------------------------------------------------

Calltech_Image_Matrix::~Calltech_Image_Matrix()
{
}

// ---------------------------------------------------------------------------------------------------------------------

// load all images in a folder recursively
int Calltech_Image_Matrix::loadImagesFromPath(cv::String path, int width, int height, int cellsize, int rowPadding, int colPadding, const int featureSize)
{
    // get all image paths
    std::vector < cv::String  > all_img_paths;
    cv::glob(path, all_img_paths, true);

    cv::String class_name_last = "X";
    size_t categories_nr = -1;
	size_t img_nr_in_cl = 0;
    // loop over all paths
    for (uint i = 0; i < all_img_paths.size(); i++)
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
			m_all_feature_data.resize(categories_nr + 1);
			m_all_flat_features.resize(categories_nr + 1);
			img_nr_in_cl = 0;
        }

        // load the img and check if data was read in

        dlib::array2d< dlib::bgr_pixel > img ;
        dlib::load_jpeg(img, all_img_paths.at(i));
        if (img.nc() == 0)
        {
            std::cout << "Fehler beim lesen des Bildes mit dem Pfad" << all_img_paths.at(i);
            return 1;
        }
		img_nr_in_cl++;
        // resize the image
        dlib::array2d< dlib::bgr_pixel > img_resized(width,height);

        dlib::resize_image(img, img_resized);

		//create hog feature
		m_all_feature_data[categories_nr].resize(img_nr_in_cl);

		dlib::array2d < dlib::matrix<double, 31, 1> > features;
		dlib::extract_fhog_features(img_resized, features, cellsize, rowPadding, colPadding);
		
		std::vector< double > flat_values;
		for (int j = 0; j < features.nc(); j++)
		{
			for (int k = 0; k < features.nr(); k++)
			{
				for (int l = 0; l < 31; l++)
				{
					flat_values.push_back(features[j][k](l));
				}
			}
		}
		dlib::matrix < double, 0, 1 > temp_mat;
		temp_mat.set_size(featureSize, 1);
		for (uint j = 0; j < flat_values.size(); j++)
		{
			temp_mat(j) = (flat_values.at(j));

		}
		m_all_flat_features[categories_nr].resize(img_nr_in_cl);
		m_all_flat_features[categories_nr][img_nr_in_cl - 1] = temp_mat;

        // save the image in the right categorie
        m_all_image_data[categories_nr].push_back(img_resized);

		
    }

    // setup the ROI vector
    for(uint i = 0; i < m_all_image_data.size(); i++)
    {
        std::vector<dlib::rectangle> rois;
        for(uint j = 0; j < m_all_image_data[i].size(); j++)
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
    std::cout << "Read " << all_img_paths.size() << " images in " << categories_nr << " different categories | folders" << std::endl;
    return 0;
}
