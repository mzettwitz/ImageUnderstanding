/* Header file for classfier data structure
   Stores information for classifier relevant data
 */

#pragma once

#include <dlib/svm_threaded.h>

#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>

#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
class ClassifierData
{
private:
    unsigned int        m_classNumber;
    std::vector<int>    m_errors;

    // cv classifier
  //  union
  //  {


	// dlib::svm_c_trainer //TODO: try different trainers and not ones for object detection ;) 
	
        cv::Ptr<cv::ml::Boost>          m_BoostClassifier;

 //   };

public:
    ClassifierData ();
    ClassifierData (unsigned int totalClasses, unsigned int id, unsigned int type);
    ~ClassifierData ();



    // increments error at target position
    void addError(unsigned int i);
    // returns void pointer to underlying classifier
	cv::Ptr<cv::ml::Boost>  getClassifier(unsigned int type);
    unsigned int getNr();
    std::vector<int>& getErrors();

};  // end class ClassifierData
