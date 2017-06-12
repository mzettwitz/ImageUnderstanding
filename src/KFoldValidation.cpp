#include "include/KFoldValidation.h"
#include <iostream>
#include <random>

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

int KFoldValidation::create10Fold(std::vector < std::vector < cv::Mat > > &m_all_image_data)
{
	//get smallest imagenumber of all classes

	smallestClassSize = findImageNumberOfSmallestClass(m_all_image_data);
	int k = 10; // definition of 10 folds
	
	for (int classes = 0; classes < 101; classes++) 
	{
		std::cout << std::endl << "10Fold class: " << classes; 
		ClassifierData classi_data = ClassifierData(101, classes, 1); // 1 for bosst Classifier
		
		// create intial fold datastructure

		m_initalFolds = createInitialFolds(k, smallestClassSize, m_all_image_data);

		for (int fold = 1; fold <= k; fold++)
		{
			cv::Mat labels;
			// create Lists for Trainig and Testing
			cv::Mat     training;
			cv::Mat     testing;
			// build training set for each fold
			for (int trainfold = 1; trainfold <= k; trainfold++)
			{
				if (trainfold == fold)
				{
					continue;
				}
				for (int c = 0; c < 101; c++)
				{
					std::vector < cv::Mat > foldimages = m_initalFolds.at(c).at(trainfold);
					for (int i = 0; i < foldimages.size(); i++)
					{
						training.push_back(foldimages.at(i));
						signed int label = (classes == c) ? 1 : -1;
						labels.push_back(label);
					}
					
					
				}
			}
			// build test set for each fold
			for (int c = 0; c < 101; c++)
			{
				std::vector < cv::Mat > foldimages = m_initalFolds.at(c).at(fold);
				for (int i = 0; i < foldimages.size(); i++)
				{
					testing.push_back(foldimages.at(i));
				}
			}


			// train
			cv::Ptr<cv::ml::Boost> classifier = (cv::Ptr<cv::ml::Boost>) (classi_data.getClassifier(1));
			
			classifier->train(training, cv::ml::ROW_SAMPLE, labels);
			
			// test
			for (int i = 0; i < testing.size().width; i++)
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
		}
		m_classifier.push_back(classi_data);
	}


	return 0;
}

std::vector <std::vector < std::vector < cv::Mat > > >   KFoldValidation::createInitialFolds(int Folds, int numberOfImages, std::vector < std::vector < cv::Mat > > &all_image_data)
{
	std::vector <std::vector < std::vector < cv::Mat > > > initialFolds;
	initialFolds.resize(101);// for each class produce folds

	// go over each class and gives each object an inital fold

	for (int classes = 0; classes < 101; classes++)
	{
		initialFolds.at(classes).resize(numberOfImages); // need to resize/reserve memory for the individual vectors too !
		int foldcounter = 0;
		// preparation to get random images for the classifier
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(1,static_cast<int>(all_image_data.at(classes).size()));

		for (int index = 0; index < numberOfImages; index++)
		{
			int randomNumber = dis(gen) % all_image_data.at(classes).size();  //otherwise out of range possible

			auto tempImage = all_image_data.at(classes).at(randomNumber);
			initialFolds.at(classes).at(foldcounter % Folds).push_back(tempImage);// save image for each class and the inital fold 
			foldcounter++;
		}

	}


	return initialFolds;
}

int  KFoldValidation::findImageNumberOfSmallestClass(std::vector < std::vector < cv::Mat > > &m_all_image_data)
{
	int smallestnumber = 1000; // initial number of smallest class
	int tempnumber; // temp number of images per class

	for (int i = 0 ; i<101;i++)
	{
		tempnumber = static_cast<int>(m_all_image_data.at(i).size());

		if (tempnumber < smallestnumber)
		{
			smallestnumber = tempnumber;
		}
	}
	return smallestnumber;
}