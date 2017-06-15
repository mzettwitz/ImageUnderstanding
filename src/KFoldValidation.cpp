#include "include/KFoldValidation.h"
#include <iostream>
#include <random>
#include <dlib/gui_widgets.h>

typedef dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >		img_array3ptr;
typedef dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >						img_array2;
typedef dlib::array < dlib::array2d < dlib::bgr_pixel>* >										img_arrayptr;

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

int KFoldValidation::create10Fold(img_array2& m_all_image_data)

{
	//get smallest imagenumber of all classes

	smallestClassSize = findImageNumberOfSmallestClass(m_all_image_data);
	int k = 10; // definition of 10 folds
	
	for (int classes = 0; classes < 101; classes++) 
	{
		std::cout << std::endl << "10Fold class: " << classes; 
		ClassifierData classi_data (101, classes, 1); // 1 for boost Classifier
		
		// create intial fold datastructure

		m_initalFolds = createInitialFolds(k, smallestClassSize, m_all_image_data);

		for (int fold = 1; fold <= k; fold++)
		{
			cv::Mat features, labels, testFeatures;
			// create Lists for Trainig and Testing
			std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(smallestClassSize);
			std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(smallestClassSize);
			// build training set for each fold
			for (int trainfold = 1; trainfold <= k; trainfold++)
			{
				if (trainfold == fold)
				{
					continue;
				}
				for (int c = 0; c < 101; c++)
				{
					
					img_arrayptr* foldimages = &m_initalFolds[c][trainfold];
					for (int i = 0; i < foldimages->size(); i++)
					{
						dlib::extract_fhog_features(*foldimages[i], hog_training_features[i]);
						signed int label = (classes == c) ? 1 : -1;
						labels.push_back(label);
						
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
				}				
			}

			// build test set for each fold
			for (int c = 0; c < 101; c++)
			{				
				img_arrayptr* foldimages = &m_initalFolds[c][fold];
				for (int i = 0; i < foldimages->size(); i++)
				{
					dlib::extract_fhog_features((*foldimages)[i], hog_test_features[i]);
					
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

			}



			cv::Ptr<cv::ml::Boost> classifier = (cv::Ptr<cv::ml::Boost>) (classi_data.getClassifier(1));
			std::cout << std::endl << "Starting training class: " << classes; 
			cv::Ptr<cv::ml::TrainData> train_data;
			train_data = train_data->create(features, cv::ml::ROW_SAMPLE, labels);
			classifier->train(train_data);
			std::cout << std::endl << "Finished training class: " << classes;
		

			cv::Mat results;
			classifier->predict(testFeatures, results);
			std::cout << std::endl << "Finished predicting class: " << classes;
			// test
			
			for (int i = 0; i < results.size().height; i++)
			{
				signed int prediction = results.at<int>(i,0);
				if (prediction == -1 && i == classes)
				{
					classi_data.addError(classes);
				}
				else
				if (prediction == 1 && i != classes)
				{
					classi_data.addError(classes);

				}

			}
			std::cout << std::endl << "Printing results for class " << classi_data.getNr() << std::endl;
			for (int i = 0; i < classi_data.getErrors().size(); i++)
			{
				std::cout << classi_data.getErrors()[i];
			}
			
		}
		m_classifier.push_back(classi_data);
	}


	return 0;
}

img_array3ptr   KFoldValidation::createInitialFolds(int Folds, int numberOfImages, img_array2 &all_image_data)
{
	img_array3ptr initialFolds;
	initialFolds.resize(101);// for each class produce folds

	// go over each class and gives each object an inital fold

	for (int classes = 0; classes < 101; classes++)
	{
		initialFolds[classes].resize(numberOfImages); // need to resize/reserve memory for the individual vectors too !
		int foldcounter = 0;
		// preparation to get random images for the classifier
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(1,static_cast<int>(all_image_data[classes].size()));

		for (int index = 0; index < numberOfImages; index++)
		{
			int randomNumber = dis(gen) % all_image_data[classes].size();  //otherwise out of range possible

			dlib::array2d < dlib::bgr_pixel>* tempImage = &all_image_data[classes][randomNumber];
			initialFolds[classes][foldcounter % Folds].push_back(tempImage);// save image for each class and the inital fold 
			foldcounter++;
		}

	}


	return initialFolds;
}

int  KFoldValidation::findImageNumberOfSmallestClass(img_array2 &m_all_image_data)
{
	int smallestnumber = 1000; // initial number of smallest class
	int tempnumber; // temp number of images per class

	for (int i = 0 ; i<101;i++)
	{
		tempnumber = static_cast<int>(m_all_image_data[i].size());

		if (tempnumber < smallestnumber)
		{
			smallestnumber = tempnumber;
		}
	}
	return smallestnumber;
}