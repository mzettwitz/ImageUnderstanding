/* Header file for classfier data structure
   Stores information for classifier relevant data
 */

#pragma once

#include <dlib/svm_threaded.h>
#include <dlib/svm.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>


//#define USE_BOOST   //define if u want to use BOOST Classifier  change also in KFoldValidation.h
#define USE_LBP
typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
typedef dlib::matrix < float, 1 * 1 * 59, 1 > sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::decision_function<kernel_type> funct_type;
class ClassifierData
{
private:
    unsigned int        m_classNumber;
    std::vector<int>    m_errors;

    // dlib::svm_c_trainer //TODO: try different trainers and not ones for object detection ;)
#ifdef USE_BOOST	
    cv::Ptr<cv::ml::Boost>          m_BoostClassifier;
#else
    dlib::svm_nu_trainer<kernel_type> m_svmClassifier;
#endif


public:
    ClassifierData ();
    ClassifierData (unsigned int totalClasses, unsigned int id, unsigned int type);
    ~ClassifierData ();



    // increments error at target position
    void addError(unsigned int i);
    // returns void pointer to underlying classifier
#ifdef USE_BOOST
    cv::Ptr<cv::ml::Boost>  getClassifier(unsigned int type);
#else
    dlib::svm_nu_trainer<kernel_type> getClassifier(unsigned int type);
#endif
    unsigned int getNr();
    std::vector<int>& getErrors();

};  // end class ClassifierData
