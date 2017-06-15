#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/ml.hpp>
//#include <opencv2/tracking/feature.hpp>
#include "include/Calltech_Image_Matrix.h"
#include "include/KFoldValidation.h"
#include <dlib/opencv.h>

#include <dlib/gui_widgets.h>

int main(void)
{
    // Test for correct integration of opencv + contrib
    std::cout << "Hello fellas!\n";

    cv::Vec2d vec = cv::Vec2d(0,0);
    std::cout << "CV Test: output of zero vector: x = " << vec[0] << ", y = " << vec[1] << std::endl;


    // test if loading works
    // data organization: make a new directory 'data' in the build directory and extract the
    // 'Caltech101'(101_ObjectCategories) data container into it
    cv::String path = "data/101_ObjectCategories/*.jpg";
    Calltech_Image_Matrix img_Matrix;
	if (img_Matrix.loadImagesFromPath(path,128,128) != 0) return 0;
	
//	KFoldValidation Validation;
//	Validation.create10Fold(img_Matrix.getAllImages());
	
	std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(20);
	std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(20);

	for (int i = 0; i < 20; i++)
	{
		dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i,0), hog_training_features[i]);
		dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i,i+1), hog_test_features[i]);
	//	dlib::image_window win(img_Matrix.getIthImageOfJthCategory(i, 0));
	//	dlib::image_window winhog(draw_fhog(hog_training_features[i]));

		std::cout << std::endl << "Features generated";
	}

	cv::Mat features,labels, testFeatures;
	for (int i = 0; i < 20; i++)
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
	for (int i = 0; i < 20; i++)
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
	}
	// ==============================
	
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(-1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(-1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(-1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
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
