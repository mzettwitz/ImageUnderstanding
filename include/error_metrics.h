/* Header file for error metrics
   Contains methods to compute simple error metrics (false-negative) and a confusion matrix
 */

#pragma once

// Includes
#include "include/Calltech_Image_Matrix.h"


//=====================================================================================================================
/*TODOS:
 class Classifier(with int classNr + class Classifier + class ClassifierOutput)
 class ClassifierOutput: std::vector<int> classifiedAs, std::vector<int> errorDistribution
 reserve capacity of nr_categories in ClassifierOutput()
*/
//=====================================================================================================================


class Classifier;
class ClassifierOutput;


// Assignments per class
void classAssignments(int classNr, std::vector<int> &classifiedAs, std::vector<int> &errorDistribution)
{
    // count assignments for each class (false-negative and true-positive)
    for(i = 0; i < classifiedAs.size(); i++)
    {
       if(classifiedAs[i] != classNr)
           errorDistribution[classifiedAs[i]]++;
       else
           errorDistribution[classNr]++;
    }
}


// Error per class
float classError(int classNr, std::vector<int> &errorDistribution)
{
    // correct assignments / total number of classes
   return (float)errorDistribution[classNr]/(float)(errorDistribution.capacity()-errorDistribution[classNr]);
}


// Confusion Matrix
std::vector< std::vector <int> > confMatrix(std::vector<Classifier> &classifiers, Calltech_Image_Matrix &imageMat)
{

    int nrCats = imageMat.getNrCategories();

    // reserve capacity for all N classes and all N errors + average error per class: NxN+1 matrix
    std::vector< std::vector <int> > confMatrix;
    confMatrix.reserve(nrCats);
    for(std::vector<int> singleClass : confMatrix)
        singleClass.reserve(nrCats+1);

    // iterate over all classes
    for(int i = 0; i < nrCats; i++)
    {
        // iterate over all class errors
        for(int j = 0; i < nrCats; j++)
            confMatrix[i][j] = classifiers[i].classifierOutput.errorDistribution[j];

        // average error for class j
        confMatrix[i][nrCats] = classError(classifiers[i].classifierOutput.errorDistribtion);
    }
}
