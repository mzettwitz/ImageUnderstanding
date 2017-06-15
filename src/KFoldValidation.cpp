#include "include/KFoldValidation.h"
#include <iostream>
#include <random>
#include <dlib/gui_widgets.h>

typedef std::vector < std::vector < std::vector < dlib::array2d < dlib::bgr_pixel >* > > >		img_array3ptr;
typedef std::vector < std::vector < dlib::array2d < dlib::bgr_pixel > > >						img_array2;
typedef std::vector < dlib::array2d < dlib::bgr_pixel>* >										img_arrayptr;

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
			cv::Mat labels;
			// create Lists for Trainig and Testing
			img_arrayptr  training;
			img_arrayptr  testing;
			// build training set for each fold
			for (int trainfold = 1; trainfold <= k; trainfold++)
			{
				if (trainfold == fold)
				{
					continue;
				}
				for (int c = 0; c < 101; c++)
				{
					
					img_arrayptr foldimages = m_initalFolds[c][trainfold];
					for (int i = 0; i < foldimages.size(); i++)
					{
						training.push_back(foldimages[i]);
						signed int label = (classes == c) ? 1 : -1;
						labels.push_back(label);
						break;
					}
					break;
					
				}
				break;
			}
			// build test set for each fold
			for (int c = 0; c < 101; c++)
			{
				
				img_arrayptr foldimages = m_initalFolds[c][fold];
				for (int i = 0; i < foldimages.size(); i++)
				{
					testing.push_back(foldimages[i]);
					break;
				}
				break;
			}


			// train
			std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features;
			hog_training_features.reserve(training.size());
			for (int i = 0; i < training.size(); i++)
			{
				dlib::extract_fhog_features(*training[i], hog_training_features[i]);
				dlib::image_window win(*training[i]);
				dlib::image_window winhog(draw_fhog(hog_training_features[i]));
				int test = 0;
				break;
			}
			//cv::Ptr<cv::ml::Boost> classifier = (cv::Ptr<cv::ml::Boost>) (classi_data.getClassifier(1));
			
			//classifier->train(training, cv::ml::ROW_SAMPLE, labels);
		//	classi_data.m_detector = classi_data.m_trainer->train(training, )

			
			// test
			/*
			for (int i = 0; i < testing.size(); i++)
			{
				signed int prediction = classifier->predict(testing.at<int>(i));
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
			*/
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