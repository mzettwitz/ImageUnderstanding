#include "include/ClassifierData.h"

ClassifierData::ClassifierData()
{
    m_classNumber = -1;
    m_errors.resize(101);
    m_BoostClassifier = cv::ml::Boost.create();
}

ClassifierData::ClassifierData(unsigned int totalClasses, unsigned int id, unsigned int type)
{
    m_classNumber = id;
    m_errors.reserve(totalClasses);
    switch(type)
    {
    case 1:
        m_BoostClassifier = cv::ml::Boost::create();
        break;
    case 2:
        m_CascadeClassifier = cv::Ptr<cv::CascadeClassifier>();
        break;
    default:
        break;
    }
}

ClassifierData::~ClassifierData()
{
}

void ClassifierData::addError(unsigned int i)
{
    m_errors.at(i)++;
}

cv::Ptr<void> ClassifierData::getClassifier(unsigned int type)
{
    switch (type) {
    case 1:
        return m_BoostClassifier;
        break;
    case 2:
        return m_CascadeClassifier;
    default:
        return nullptr;
        break;
    }
}

unsigned int ClassifierData::getNr()
{
    return m_classNumber;
}

std::vector<int>& ClassifierData::getErrors()
{
    return &m_errors;
}
