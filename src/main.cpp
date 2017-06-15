#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/ml.hpp>
//#include <opencv2/tracking/feature.hpp>
#include "include/Calltech_Image_Matrix.h"
#include "include/KFoldValidation.h"

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

	for (int i = 0; i < 20; i++)
	{
		dlib::extract_fhog_features(img_Matrix.getIthImageOfJthCategory(i,0), hog_training_features[i]);
		dlib::image_window win(img_Matrix.getIthImageOfJthCategory(i, 0));
		dlib::image_window winhog(draw_fhog(hog_training_features[i]));
		std::cout << std::endl << "Nr of row: " << hog_training_features[i].nr();
		std::cout << std::endl << "Nr of columns: " << hog_training_features[i].nc();
		system("Pause");
	}

	/*


	cv::Ptr< cv::CvFeatureParams > featureParams = cv::CvFeatureParams::create(0);
	featureParams->init(*featureParams);
	cv::Ptr < cv::CvHaarEvaluator > test_haar;
	cv::Ptr < cv::CvHaarEvaluator::FeatureHaar > test_2;
	test_2->eval(img1, cv::Rect(0, 0, 400, 400), feat);
	test_haar->setImage(img1, '1', 1);
	
	auto test = test_haar->getNumFeatures();
	cv::Mat features, labels;
	features.push_back(features_1);
	features.push_back(features_2);
	features.push_back(features_3);
	features.push_back(features_4);
	features.push_back(features_5);
	features.push_back(features_6);

	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(1);
	labels.push_back(-1);

	
	cv::CascadeClassifier test;
	test.load("cascade.xml");
	
	cv::Ptr<cv::ml::Boost> model = model->create();
	model = cv::Algorithm::load<cv::ml::Boost>("cascade.xml");

	std::cout << model->isClassifier();
	std::cout << model->isTrained();


	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(1, 1));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(1, 2));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(1, 3));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(1, 4));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(1, 5));
	std::cout << std::endl;
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(2, 2));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(3, 1));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(4, 2));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(5, 1));
	std::cout << std::endl << model->predict(img_Matrix.getIthImageOfJthCategory(6, 2));

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
