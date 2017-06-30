#include "include/ClassifierData.h"
#include <thread>



ClassifierData::ClassifierData()
{
    m_classNumber = -1;
    for (int i = 0; i < 102; i++)
        m_errors.push_back(0);
}

// ---------------------------------------------------------------------------------------------------------------------

ClassifierData::ClassifierData(unsigned int totalClasses, unsigned int id, unsigned int type)
{
    /*
    m_scanner.set_detection_window_size(128, 128); // TODO: make this as a parameter
    m_trainer = &dlib::structural_object_detection_trainer<image_scanner_type>(m_scanner);
    m_trainer->set_num_threads(std::thread::hardware_concurrency());
    m_trainer->set_c(1); // TODO: make this as a parameter  IMPORTANT
    m_trainer->be_verbose();
    m_trainer->set_epsilon(0.01);	// TODO: make this as a parameter  IMPORTANT
    */
#ifdef USE_BOOST
    m_BoostClassifier = cv::ml::Boost::create();
#else
#endif
    m_classNumber = id;
    for (uint i = 0; i < totalClasses; i++)
        m_errors.push_back(0);
}

// ---------------------------------------------------------------------------------------------------------------------

ClassifierData::~ClassifierData()
{
}

// ---------------------------------------------------------------------------------------------------------------------

void ClassifierData::addError(unsigned int i)
{
    m_errors.at(i)++;
}

// ---------------------------------------------------------------------------------------------------------------------

#ifdef USE_BOOST 
cv::Ptr<cv::ml::Boost>  ClassifierData::getClassifier(unsigned int type)
{
    return m_BoostClassifier;
#else
dlib::svm_nu_trainer<kernel_type> ClassifierData::getClassifier(unsigned int type)
{
    return m_svmClassifier;
#endif

}

// ---------------------------------------------------------------------------------------------------------------------

unsigned int ClassifierData::getNr()
{
    return m_classNumber;
}

// ---------------------------------------------------------------------------------------------------------------------

std::vector<int>& ClassifierData::getErrors()
{
    return m_errors;
}
