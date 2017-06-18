/* Header file for error metrics
   Contains methods to compute simple error metrics (false-negative) and a confusion matrix
 */

#pragma once

// Includes
#include "include/Calltech_Image_Matrix.h"
#include "include/ClassifierData.h"


//=====================================================================================================================
/*TODOS:
 class Classifier(with int classNr + class Classifier + class ClassifierOutput)
 class ClassifierOutput: std::vector<int> classifiedAs, std::vector<int> errorDistribution
 reserve capacity of nr_categories in ClassifierOutput()
*/
//=====================================================================================================================



// Error per class
inline float classError(int classNr, std::vector<int> &errorDistribution)
{
    float err = 0.f;
    // summed errors / total errors
    for(int i = 0; i < errorDistribution.size(); i++)
        err += (float)errorDistribution.at(i);

    return err/(float)errorDistribution.size();
}


// Print confusion matrix
inline void printConfMatrix(std::vector< std::vector <float> > confMat)
{
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
    }
}


// Confusion Matrix
inline std::vector< std::vector <float> > confMatrix(std::vector<ClassifierData> &classifiers, Calltech_Image_Matrix &imageMat)
{

    int nrCats = imageMat.getNrCategories();

    // reserve capacity for all N classes and all N errors + average error per class: NxN+1 matrix
    std::vector< std::vector <float> > confMat(nrCats,std::vector<float>(nrCats+1));
    //confMat.resize( nrCats , std::vector<double>( nrCats+1, 0.f));

    // iterate over all classes
    for(int i = 0; i < nrCats; i++)
    {
        float totalErrors = 0.f;
        for(int j = 0; j < nrCats; j++)
            totalErrors += (float)classifiers.at(i).getErrors().at(j);

        // iterate over all class errors
        for(int j = 0; i < nrCats; j++)
            confMat[i][j] = (float)(classifiers.at(i).getErrors().at(j))/totalErrors;

        // average error for class j
        confMat[i][nrCats] = classError(classifiers.at(i).getNr(), classifiers.at(i).getErrors());
    }
    return confMat;
}


