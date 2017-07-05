
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>

#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/svm.h>

#include "include/Calltech_Image_Matrix.h"
#include "include/KFoldValidation.h"
#include "include/error_metrics.h"


using namespace std;
using namespace std::chrono;

int main(void)
{
    // data organization: make a new directory 'data' in the build directory and extract the
    // 'Caltech101'(101_ObjectCategories) data container into it
    cv::String path = "data/101_ObjectCategories/*.jpg";
    Calltech_Image_Matrix img_Matrix;

    // SETUP
    int imagesize = 64;
    int cellsize = 8;
    int rowPadding, colPadding;
    rowPadding = colPadding = 1;
	int dim1 = std::max((int)std::round((float)imagesize / (float)cellsize) - 2, 0) + colPadding - 1;
	int dim2 = std::max((int)std::round((float)imagesize / (float)cellsize) - 2, 0) + rowPadding - 1;
	const int featureSize = dim1*dim2 * 31;

    double nu = 0.0043;//0.015;
    string type = "SVM";
    string setup = "imagesize ="  + std::to_string(imagesize) + " cellsize = " +
            std::to_string(cellsize) + " type = " + type + " nu = " + std::to_string(nu);

    if (img_Matrix.loadImagesFromPath(path,imagesize,imagesize,cellsize,rowPadding, colPadding, featureSize) != 0) return 0;
//	dlib::image_window test(img_Matrix.getIthImageOfJthCategory(0,0));

    //while(nu < 0.019)

    //string setup = "imagesize ="  + std::to_string(imagesize) + " cellsize = " +
    //        std::to_string(cellsize) + " type = " + type + " nu = " + std::to_string(nu);
    //std::cerr << nu << std::endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    KFoldValidation validation(img_Matrix.getNrCategories());
    validation.create10Fold(img_Matrix.getAllImages(), cellsize, imagesize, nu);

    //validation.printErrorMatrix();
    //std::cout << "\n\nPlain RESULTS\n";

    //auto confMat = confMatrix(validation,img_Matrix);
    //printResults(confMat);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>( t2 - t1 ).count();
    cerr << "time to compute: " << duration << " seconds" << std::endl;

    //storeMatrixOnDisk(confMat, duration, setup);

    //nu *= 1.1;
    //==============================================================================================================
    //==============================================================================================================
    //==============================================================================================================


    std::vector<std::vector<float> > matrix(img_Matrix.getNrCategories(), std::vector<float>(img_Matrix.getNrCategories()));

    // Perform classification

    //------------------------------------------------------------------------------------------------------------------
    // build test set for each fold
    std::vector < sample_type > testFeatures;

    int countImages = 0;
    for(uint i = 0;i < img_Matrix.getNrCategories(); i++)
    {
        for(uint j = 0; j < img_Matrix.getAllImagesOfIthClass(i).size(); j++)
        {
            countImages++;
            //images->push_back(img_Matrix.getIthImageOfJthCategory(i,j)));
        }
    }



    int count = 0;
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(countImages);
    for(uint j = 0; j < img_Matrix.getNrCategories(); j++)
    {
        for (int i = 0; i < countImages; i++)
        {
			img_Matrix.getFeatureOfIthImageOfJthCategory(i, j).swap(hog_test_features[count]);
    //        dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i,j), hog_test_features[count], cellsize, rowPadding, colPadding); // *(*images)[i]
#ifdef USE_BOOST
            cv::Mat flat_values;
            for (int j = 0; j < hog_test_features[0].nc(); j++)
            {
                for (int k = 0; k < hog_test_features[0].nr(); k++)
                {
                    for (int l = 0; l < 31; l++)
                    {
                        flat_values.push_back(hog_test_features[i][j][k](l));
                    }
                }
            }
            flat_values = flat_values.reshape(1, 1);
            flat_values.convertTo(flat_values, CV_32F);
            testFeatures.push_back(flat_values);
#else
            std::vector< float > flat_values;
            for (int j = 0; j < hog_test_features[0].nc(); j++)
            {
                for (int k = 0; k < hog_test_features[0].nr(); k++)
                {
                    for (int l = 0; l < 31; l++)
                    {
                        flat_values.push_back(hog_test_features[i][j][k](l));
                    }
                }
            }
            dlib::matrix < float, 0, 1 > temp_mat;
            temp_mat.set_size(featureSize,1);
            for (int j = 0; j < featureSize; j++)
            {
                temp_mat(j) = (flat_values.at(j));

            }
            testFeatures.push_back(temp_mat);
#endif
            count++;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    float prediction = 0.f;
    int counter = 0;
    for(uint cat_i = 0; cat_i < img_Matrix.getNrCategories(); cat_i++)   // classes
    {
        for(uint img_j = 0; img_j < img_Matrix.getAllImagesOfIthClass(cat_i).size(); img_j++)    // images
        {
            for(uint classifier_k = 0; classifier_k < validation.getClassifierData().size(); classifier_k++)
            {
                // naiv version: sum up predictions, normalize with predictions per class
                funct_type learnedF = validation.getClassifierData()[classifier_k].getLearnedFunction();
                prediction = learnedF(testFeatures.at(counter));

                if(prediction == 1.f)
                    matrix[cat_i][classifier_k]++;
            }
            counter++;
        }
    }

    auto confMat = confMatrix(matrix,img_Matrix);
    printResults(confMat);
    storeMatrixOnDisk(confMat, duration, setup);

    return 0;
}
