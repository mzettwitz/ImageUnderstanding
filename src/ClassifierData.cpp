#include "include/ClassifierData.h"
#include <thread>

ClassifierData::ClassifierData()
{
    m_classNumber = -1;
    for (int i = 0; i < 102; i++)
		m_errors.push_back(0);
	
  //  m_BoostClassifier = cv::ml::Boost::create();

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
	m_BoostClassifier = cv::ml::Boost::create();

    m_classNumber = id;
    for (uint i = 0; i < totalClasses; i++)
		m_errors.push_back(0);
    switch(type)
    {
    case 1:
       // m_BoostClassifier = cv::ml::Boost::create();
        break;
    case 2:
//        m_CascadeClassifier = cv::Ptr<cv::CascadeClassifier>();
        break;
    default:
        break;
    }
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

cv::Ptr<cv::ml::Boost>  ClassifierData::getClassifier(unsigned int type)
{
	return m_BoostClassifier;
    switch (type) {
    case 1:
    //    return m_BoostClassifier;
        break;
    case 2:
   //     return m_CascadeClassifier;
    default:
//        return cv::Ptr<void>();
        break;
    }
	
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
