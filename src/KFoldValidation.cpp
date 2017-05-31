#include "include/KFoldValidation.h"
#include "include/Calltech_Image_Matrix.h"
#include <iostream>

//=====================================================================================================================
/*TODOS:
class Classifier(with int classNr + class Classifier + class ClassifierOutput)
class ClassifierOutput: std::vector<int> classifiedAs, std::vector<int> errorDistribution
reserve capacity of nr_categories in ClassifierOutput()
*/
//=====================================================================================================================
KFoldValidation::KFoldValidation()
{
}


KFoldValidation::~KFoldValidation()
{
}

int KFoldValidation::create10Fold(std::vector < std::vector < cv::Mat > > m_all_image_data)
{
	//iterativ steps over all classes 
	int sizes_of_all_classes = 101;

	for (int classes = 0; classes < 101; classes++)
	{
		// get number of  images in the class 
		std::vector < cv::Mat > all_images_of_class = m_all_image_data.at(classes);
		int sizeClass = static_cast<int>(all_images_of_class.size());
		int k = 10; // definition of 10 folds

		// create Lists for Trainig and Testing
		cv::Mat training;
		cv::Mat testing;
		for (int fold = 1; fold <= k; fold++)
		{
			// build training set for each fold
			for (int trainfold = 1; trainfold <= k; trainfold++)
			{
				if (trainfold == fold)
				{
					continue;
				}
				for (int c = 0; c < sizeClass; c++)
				{
					cv::Mat image = all_images_of_class.at(trainfold);
					training.push_back(image);
				}
			}
			// build test set for each fold
			for (int c = 0; c < sizeClass; c++)
			{
				cv::Mat image = all_images_of_class.at(fold);
				testing.push_back(image);
			}

			//=====================================================================================================================
			/*TODOS:
			- implement classifier to train the trainset and then test each object of testset
			- adding result to error_metrics
			*/
			//=====================================================================================================================

			// train and test

			/*
			for (int i = 0; i < static_cast<int>(testing.size().width); i++)
			{
			prediction = classify(testing.at(i));
			add prediction to errormetrics
			}
			*/

			

		}
	}


	return 0;
}