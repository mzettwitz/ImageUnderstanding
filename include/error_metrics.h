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
float classError(int classNr, std::vector<int> &errorDistribution)
{
    float err = 0.f;
    // summed errors / total errors
    for(int i = 0; i < errorDistribution.size(); i++)
        err += (float)errorDistribution.at(i);

    return err/(float)errorDistribution.size();
}


// Confusion Matrix
std::vector< std::vector <float> > confMatrix(std::vector<ClassifierData> &classifiers, Calltech_Image_Matrix &imageMat)
{

    int nrCats = imageMat.getNrCategories();

    // reserve capacity for all N classes and all N errors + average error per class: NxN+1 matrix
    std::vector< std::vector <float> > confMatrix;
    confMatrix.reserve(nrCats);
    for(std::vector<float> singleClass : confMatrix)
        singleClass.reserve(nrCats+1);

    // iterate over all classes
    for(int i = 0; i < nrCats; i++)
    {
        float totalErrors = 0.f;
        for(int j = 0; j < nrCats; j++)
            totalErrors += (float)classifiers.at(i).getErrors().at(j);

        // iterate over all class errors
        for(int j = 0; i < nrCats; j++)
            confMatrix[i][j] = (float)(classifiers.at(i).getErrors().at(j))/totalErrors;

        // average error for class j
        confMatrix[i][nrCats] = classError(classifiers.at(i).getNr(), classifiers.at(i).getError());
    }
}
