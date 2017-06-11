/* Header file for classfier data structure
   Stores information for classifier relevant data
 */

#pragma once

#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

class ClassifierData
{
private:
    unsigned int        m_classNumber;
    std::vector<int>    m_errors;

    // cv classifier
  //  union
  //  {
        cv::Ptr<cv::ml::Boost>          m_BoostClassifier;
        cv::Ptr<cv::CascadeClassifier>  m_CascadeClassifier;
 //   };

public:
    ClassifierData ();
    ClassifierData (unsigned int totalClasses, unsigned int id, unsigned int type);
    ~ClassifierData ();

    // increments error at target position
    void addError(unsigned int i);
    // returns void pointer to underlying classifier
    cv::Ptr<void> getClassifier(unsigned int type);
    unsigned int getNr();
    std::vector<int>& getErrors();

};  // end class ClassifierData
