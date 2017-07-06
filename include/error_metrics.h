/* Header file for error metrics
   Contains methods to compute simple error metrics (false-negative) and a confusion matrix
 */

#pragma once

// Includes
#include "include/Calltech_Image_Matrix.h"
#include "include/ClassifierData.h"
#include "include/KFoldValidation.h"
#include "include/Calltech_Image_Matrix.h"

#include <iostream>
//#include <fstab.h>
#include <time.h>

// Error per class
inline float classError(int classNr, std::vector<int> &errorDistribution)
{
    float err = 0.f;
    // summed errors / total errors
    for(unsigned int i = 0; i < errorDistribution.size(); i++)
        err += (float)errorDistribution.at(i);

    return err/(float)errorDistribution.size();
}

// ---------------------------------------------------------------------------------------------------------------------

// Print confusion matrix
inline void printConfMatrix(std::vector< std::vector <float> > &confMat)
{
    std::cout << "\n\n\n==========================================\nRESULTS\n";
    std::cout << " ";
    for(unsigned int i = 0; i < confMat.size();i++)
        std::cout << /*"in class "*/" " << i <<" ";

    for(unsigned int i = 0; i < confMat.size();i++)
    {
        std::cout << std::endl <</*"class "*/ i << " ";
        for(unsigned int j = 0; j < confMat[i].size(); j++)
        {
            if(j == confMat[i].size()-1)
                std::cout << "\t" << confMat[i][j];
            else
                std::cout << " " << confMat[i][j];
        }

    }

    float avgError = 0.f;
    for(uint i = 0; i < confMat.size(); i++)
        avgError += confMat[i][confMat.size()];
    avgError /= confMat.size();
    std::cout << "\n\nAverage Error: " << avgError;
    std::cout << "\nAverage Prediction: " << 1.f - avgError;
}

// ---------------------------------------------------------------------------------------------------------------------

inline void printResults(std::vector< std::vector <float> > &confMat)
{
    float avgError = 0.f;
    for(uint i = 0; i < confMat.size(); i++)
        avgError += confMat[i][confMat.size()];
    avgError /= confMat.size();
    std::cout << "\n\nAverage Error: " << avgError;
    std::cout << "\nAverage Prediction: " << 1.f - avgError;
}

// ---------------------------------------------------------------------------------------------------------------------

inline bool storeMatrixOnDisk(std::vector< std::vector <float> > confMat, int time, std::string setup, Calltech_Image_Matrix &img_mat)
{
    std::ofstream file;
    std::string filename = "confMatrix" + setup + ".csv";
    file.open(filename);

    file << "\nSetup: " << setup << std::endl;
    file << "\nComputation time in seconds: \t" << time;
    file << "\n\n\n==========================================\nRESULTS\n\t";
    for(unsigned int i = 0; i < confMat.size();i++)
        file << img_mat.getIthCategoryName(i) << "\t";
    file << "error";

    for(unsigned int i = 0; i < confMat.size();i++)
    {
        file << std::endl << img_mat.getIthCategoryName(i);
        for(unsigned int j = 0; j < confMat[i].size(); j++)
        {
            if(j == confMat[i].size()-1)
                file << "\t" << confMat[i][j];
            else
                file << "\t" << confMat[i][j];
        }
    }

    float avgError = 0.f;
    for(uint i = 0; i < confMat.size(); i++)
        avgError += confMat[i][confMat.size()];
    avgError /= confMat.size();
    file << "\n\nAverage Error: \t" << avgError;
    file << "\nAverage Prediction: \t" << 1.f - avgError;

    file.close();
    return true;
}
// ---------------------------------------------------------------------------------------------------------------------

// overload Function for confusion Matrix with std vector float 

inline std::vector< std::vector <float>> confMatrix(std::vector <std::vector<float>> validation,Calltech_Image_Matrix &imageMat)//, int smallestClass)
{
	int nrCats = imageMat.getNrCategories();
	// reserve capacity for all N classes and all N errors + average error per class: NxN+1 matrix
	std::vector< std::vector <float> > confMat(nrCats, std::vector<float>(nrCats + 1));

	// iterate over all classes
	for (int i = 0; i < nrCats; i++)
	{
		float totalPredictions_i = 0.f;
		for (int j = 0; j < nrCats; j++)
		{
            //if (j == i)
                //totalPredictions_i += smallestClass;
            //else
				totalPredictions_i += validation.at(i).at(j);
		}


		// iterate over all classes and normalize predicted results
		for (int j = 0; j < nrCats; j++)
		{
			if (totalPredictions_i != 0)
				confMat[i][j] = (validation.at(i).at(j)) / totalPredictions_i;
		}

		// average error for class i
		confMat[i][nrCats] = 1.f - confMat[i][i];//classError(i, validation.getErrorMatrix().at(i));
	}



	return confMat;
}
// ---------------------------------------------------------------------------------------------------------------------

// Compute confusion Matrix
inline std::vector< std::vector <float> > confMatrix(KFoldValidation &validation, Calltech_Image_Matrix &imageMat)
{

    int nrCats = imageMat.getNrCategories();

    // reserve capacity for all N classes and all N errors + average error per class: NxN+1 matrix
    std::vector< std::vector <float> > confMat(nrCats,std::vector<float>(nrCats+1));
    //confMat.resize( nrCats , std::vector<double>( nrCats+1, 0.f));

    // iterate over all classes
    for(int i = 0; i < nrCats; i++)
    {
        float totalPredictions_i = 0.f;
        for(int j = 0; j < nrCats; j++)
        {
            if(j==i)
                totalPredictions_i += validation.findImageNumberOfSmallestClass(imageMat.getAllImages());
            else
                totalPredictions_i += (float)validation.getErrorMatrix().at(i).at(j);
        }


        // iterate over all classes and normalize predicted results
        for(int j = 0; j < nrCats; j++)
        {
            if(totalPredictions_i!= 0)
                confMat[i][j] = (float)(validation.getErrorMatrix().at(i).at(j))/totalPredictions_i;
        }

        // average error for class i
        confMat[i][nrCats] = 1.f - confMat[i][i];//classError(i, validation.getErrorMatrix().at(i));
    }
    return confMat;
}


