
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
    int cellsize = 64;
    double nu = 0.00384434;//0.015;
    string type = "SVM";
    string setup = "imagesize ="  + std::to_string(imagesize) + " cellsize = " +
            std::to_string(cellsize) + " type = " + type + " nu = " + std::to_string(nu);

    if (img_Matrix.loadImagesFromPath(path,imagesize,imagesize) != 0) return 0;

    while(nu < 0.019)
    {
        string setup = "imagesize ="  + std::to_string(imagesize) + " cellsize = " +
                std::to_string(cellsize) + " type = " + type + " nu = " + std::to_string(nu);
        std::cerr << nu << std::endl;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        KFoldValidation validation(img_Matrix.getNrCategories());
        validation.create10Fold(img_Matrix.getAllImages(), cellsize, imagesize, nu);

        //validation.printErrorMatrix();
        //std::cout << "\n\nPlain RESULTS\n";

        auto confMat = confMatrix(validation,img_Matrix);
        printResults(confMat);

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration = duration_cast<seconds>( t2 - t1 ).count();
        cerr << "time to compute: " << duration << " seconds" << std::endl;

        storeMatrixOnDisk(confMat, duration, setup);

        nu *= 1.1;
    }





    /*
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(30);
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(30);

    for (int i = 0; i < 30; i++)
    {
        dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i, 0), hog_training_features[i]);
        dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(0, i + 1), hog_test_features[i]);
        dlib::image_window win(img_Matrix.getIthImageOfJthCategory(i, 0));
        dlib::image_window winhog(draw_fhog(hog_training_features[i]));
        system("Pause");
     std::cout << std::endl << "Features generated";
            }
*/
    return 0;
}
