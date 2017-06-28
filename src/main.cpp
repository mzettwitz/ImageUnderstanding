#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>

#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>

#include "include/Calltech_Image_Matrix.h"
#include "include/KFoldValidation.h"
#include "include/error_metrics.h"


using namespace std;
using namespace std::chrono;

int main(void)
{
    // test if loading works
    // data organization: make a new directory 'data' in the build directory and extract the
    // 'Caltech101'(101_ObjectCategories) data container into it
    cv::String path = "data/101_ObjectCategories/*.jpg";
    Calltech_Image_Matrix img_Matrix;
    if (img_Matrix.loadImagesFromPath(path,64,64) != 0) return 0;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    KFoldValidation validation(img_Matrix.getNrCategories());
    validation.create10Fold(img_Matrix.getAllImages());

    validation.printErrorMatrix();
    std::cout << "\n\nPlain RESULTS\n";

    auto confMat = confMatrix(validation,img_Matrix);
    printConfMatrix(confMat);


    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>( t2 - t1 ).count();
    cerr << "time to compute: " << duration << " seconds";



    /*
    std::cout << "\n\n\n==========================================\nRESULTS\n";
    std::cout << "\t";
    for(int i = 0; i< confMat.size();i++)
        std::cout << "in class " << i <<"\t";
    std::cout << std::endl;

    for(int i = 0; i < confMat.size();i++)
    {
        std::cout << "class " << i;
        for(int j = 0; j < confMat[i].size(); j++)
             std::cout << "\t" << j;
    }*/
    /*

    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(10);
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(10);

    for (int i = 0; i < 10; i++)
    {
        dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i,0), hog_training_features[i]);
    //	dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(0,i+1), hog_test_features[i]);
        dlib::image_window win(img_Matrix.getIthImageOfJthCategory(i, 0));
        dlib::image_window winhog(draw_fhog(hog_training_features[i]));
        system("Pause");
        std::cout << std::endl << "Features generated";
    }

    /*
    cv::Mat features,labels, testFeatures;
    for (int i = 0; i < 50; i++)
    {
        cv::Mat flat_values;
        for (int j = 0; j < hog_training_features[0].nc(); j++)
        {
            for (int k = 0; k < hog_training_features[0].nr(); k++)
            {
                for (int l = 0; l < 31; l++)
                {
                    flat_values.push_back(hog_training_features[i][j][k](l));
                }
            }
        }
        flat_values = flat_values.reshape(1, 1);
        flat_values.convertTo(flat_values, CV_32F);
        features.push_back(flat_values);
    }
    //===========================TEST
    for (int i = 0; i < 50; i++)
    {
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
        if (i > 47) features.push_back(flat_values);
    }
    // ==============================

    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);
    labels.push_back(1);

    labels.push_back(-1);
    labels.push_back(-1);


    cv::Ptr<cv::ml::Boost> model = model->create();
    cv::Ptr<cv::ml::TrainData> train_data;
    train_data = train_data->create(features, cv::ml::ROW_SAMPLE, labels);
    std::cout << std::endl << "Begin Train";
    model->train(train_data);
    std::cout << std::endl << "End Train";

    std::cout << model->isClassifier();
    std::cout << model->isTrained();

    cv::Mat results;
    model->predict(testFeatures,results);
    std::cout << results;
    */
    /*
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(6, 0)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(20, 0)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(2, 0)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(5, 0)));
    std::cout << std::endl;
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(1, 1)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(2, 3)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(5, 4)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(1, 2)));
    std::cout << std::endl << model->predict(dlib::toMat(img_Matrix.getIthImageOfJthCategory(4, 20)));
    */

    /*
    KFoldValidation Validation;
    Validation.create10Fold(img_Matrix.getAllImages());

    std::vector< ClassifierData > classes = Validation.getClassifierData();
    ClassifierData classi_class_1 = classes.at(0);
    ClassifierData classi_class_2 = classes.at(2);
    std::vector<int> test_1 = classi_class_1.getErrors();
    std::vector<int> test_2 = classi_class_2.getErrors();

    for (int i = 0; i < test_1.size(); i++)
    {
        std::cout << std::endl << test_1.at(i);
    }
    std::cout << std::endl;

    for (int i = 0; i < test_1.size(); i++)
    {
        std::cout << std::endl << test_2.at(i);
    }
    */
    return 0;
}
