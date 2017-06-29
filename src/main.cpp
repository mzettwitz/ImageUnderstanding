
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
    if (img_Matrix.loadImagesFromPath(path,64,64) != 0) return 0;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    KFoldValidation validation(img_Matrix.getNrCategories());
    validation.create10Fold(img_Matrix.getAllImages());

    //validation.printErrorMatrix();
    //std::cout << "\n\nPlain RESULTS\n";

    auto confMat = confMatrix(validation,img_Matrix);
    printResults(confMat);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>( t2 - t1 ).count();
    cerr << "time to compute: " << duration << " seconds";


    return 0;
}
