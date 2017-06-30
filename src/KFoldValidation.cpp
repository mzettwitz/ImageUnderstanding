#include <iostream>
#include <random>
#include <thread>

#include <dlib/gui_widgets.h>

#include "include/KFoldValidation.h"

//typedef dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >		dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >;
//typedef dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >						dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >;
//typedef dlib::array < dlib::array2d < dlib::bgr_pixel>* >										dlib::array < dlib::array2d < dlib::bgr_pixel>* >;

KFoldValidation::KFoldValidation()
{
    m_NrClasses = 102;
    m_error_matrix.resize(m_NrClasses);
    for (int class_i = 0; class_i < m_NrClasses; class_i++)
    {
        m_error_matrix[class_i].resize(m_NrClasses);
    }
}


// ---------------------------------------------------------------------------------------------------------------------

KFoldValidation::KFoldValidation(int nrClasses)
{
    m_NrClasses = nrClasses;
    m_error_matrix = std::vector<std::vector<int> > (m_NrClasses,std::vector<int>(m_NrClasses));
}

// ---------------------------------------------------------------------------------------------------------------------

KFoldValidation::~KFoldValidation()
{
}

// ---------------------------------------------------------------------------------------------------------------------

int KFoldValidation::create10Fold(dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >& m_all_image_data)
{
    //get smallest imagenumber of all classes
    smallestClassSize = findImageNumberOfSmallestClass(m_all_image_data);
    int k = 10; // definition of 10 folds

    int numThreads = std::thread::hardware_concurrency();
    std::vector<ClassifierData> threadedClassifiers(numThreads);
    int numOps = 0;
    std::vector<std::thread> threadArr(numThreads);

    m_initalFolds = createInitialFolds(k, smallestClassSize, m_all_image_data);

    std::cout << "progress: 0 % => " << std::flush;

    int class_i = 0;
    while (class_i < m_NrClasses)
    {
        // more operations/iterations than threads => use all threads
        if((m_NrClasses-class_i)/numThreads  > 0)
        {
            numOps = numThreads;
            for(int i = 0; i < numOps; i++)
            {
                threadedClassifiers[i] = ClassifierData (m_NrClasses, class_i, 1); // 1 for boost Classifier
                threadArr[i] = std::thread(&KFoldValidation::prepareTraining,this, class_i++,std::ref(threadedClassifiers[i]),k, std::ref(m_all_image_data)); //function pointer, object, param

#ifdef __unix__     // load balancing on unix systems
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i%8, &cpuset);
                pthread_setaffinity_np(threadArr[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
            }
        }
        // less operations/iterations than threads => use remaining number of threads
        else
        {
            numOps = (m_NrClasses-class_i) % numThreads;
            for(int i = 0; i < numOps; i++)
            {
                threadArr[i] = std::thread(&KFoldValidation::prepareTraining,this, class_i++,std::ref(threadedClassifiers[i]),k, std::ref(m_all_image_data)); //function pointer, object, param

#ifdef __unix__     // load balancing on unix systems
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i%8, &cpuset);
                pthread_setaffinity_np(threadArr[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
            }
        }

        // join threads
        for(int i = 0; i < numOps; i++)
            threadArr[i].join();
        for(int i = 0; i < numOps; i++)
            m_classifier.push_back(threadedClassifiers[i]);

        // print progress
        std::cout << (int)(100.f * ((float)class_i/(float)m_NrClasses)) << " % => " << std::flush;
    }
    //std::cout << endl;
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------

dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >
KFoldValidation::createInitialFolds
(int Folds, int numberOfImages, dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > > &all_image_data)
{
    dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > > initialFolds;
    initialFolds.resize(m_NrClasses);// for each class produce folds

    // go over each class and gives each object an inital fold
    for (int class_i = 0; class_i < m_NrClasses; class_i++)
    {
        initialFolds[class_i].resize(numberOfImages); // need to resize/reserve memory for the individual vectors too !
        int foldcounter = 0;
        // preparation to get random images for the classifier
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1,static_cast<int>(all_image_data[class_i].size()));

        for (int index = 0; index < numberOfImages; index++)
        {
            testClasses.push_back(class_i);	// store class information

            int randomNumber = dis(gen) % all_image_data[class_i].size();  //otherwise out of range possible
            dlib::array2d < dlib::bgr_pixel>* tempImage = &all_image_data[class_i][randomNumber];
            initialFolds[class_i][foldcounter % Folds].push_back(tempImage);// save image for each class and the inital fold
            foldcounter++;
        }

    }


    return initialFolds;
}

// ---------------------------------------------------------------------------------------------------------------------

int  KFoldValidation::findImageNumberOfSmallestClass(dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > > &m_all_image_data)
{
    int smallestnumber = INT32_MAX; // initial number of smallest class
    int tempnumber; // temp number of images per class

    for (int i = 0 ; i<m_NrClasses;i++)
    {
        tempnumber = static_cast<int>(m_all_image_data[i].size());

        if (tempnumber < smallestnumber)
        {
            smallestnumber = tempnumber;
        }
    }
    return smallestnumber;
}

// ---------------------------------------------------------------------------------------------------------------------

void KFoldValidation::trainClass(int class_i, ClassifierData& classi_data, int k, int fold)
{
#ifdef USE_BOOST 
    cv::Mat features, labels, testFeatures;
#else
    std::vector < sample_type > features, testFeatures;
    std::vector < float > labels;
#endif
    // create Lists for Trainig and Testing
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(smallestClassSize);
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(smallestClassSize);
    int cellsize = 8;
    int colPadding = 1;
    int rowPadding = 1;

    // build training set for each fold
    for (int trainfold = 1; trainfold <= k; trainfold++)
    {
        if (trainfold == fold)
        {
            continue;
        }
        mute.lock();
        for (int c = 0; c < m_NrClasses; c++)
        {

            dlib::array < dlib::array2d < dlib::bgr_pixel>* >* foldimages = &m_initalFolds[c][trainfold];
            for (unsigned int i = 0; i < foldimages->size(); i++)
            {

                dlib::extract_fhog_features(*(*foldimages)[i], hog_training_features[i],cellsize, rowPadding, colPadding);
                signed int label = (class_i == c) ? 1 : -1;
                labels.push_back(label);

#ifdef USE_BOOST
                cv::Mat flat_values;
                for (int j = 0; j < hog_training_features[0].nc(); j++)
                {
                    for (int k = 0; k < hog_training_features[0].nr(); k++)
                    {
                        for (int l = 0; l < 31; l++)
                        {
                            flat_values.push_back(hog_training_features[i][j][k](l));
                        }
                    }
                }
                flat_values = flat_values.reshape(1, 1);
                flat_values.convertTo(flat_values, CV_32F);
                features.push_back(flat_values);
#else
                std::vector< float > flat_values ;
                for (int j = 0; j < hog_training_features[0].nc(); j++)
                {
                    for (int k = 0; k < hog_training_features[0].nr(); k++)
                    {
                        for (int l = 0; l < 31; l++)
                        {
                            flat_values.push_back(hog_training_features[i][j][k](l));
                        }
                    }
                }
                dlib::matrix < float, 1116, 1 > temp_mat;
                for (uint j = 0; j < flat_values.size() ; j++)
                {
                    temp_mat(j) = (flat_values.at(j));

                }

                features.push_back(temp_mat);
#endif
            }

        }
        mute.unlock();
    }

    // build test set for each fold
    for (int c = 0; c < m_NrClasses; c++)
    {
        mute.lock();
        dlib::array < dlib::array2d < dlib::bgr_pixel>* >* foldimages = &m_initalFolds[c][fold];
        for (unsigned int i = 0; i < foldimages->size(); i++)
        {
            dlib::extract_fhog_features(*(*foldimages)[i], hog_test_features[i],cellsize, rowPadding, colPadding);
#ifdef USE_BOOST
            cv::Mat flat_values;
            for (int j = 0; j < hog_test_features[0].nc(); j++)
            {
                for (int k = 0; k < hog_test_features[0].nr(); k++)
                {
                    for (int l = 0; l < 31; l++)
                    {
                        flat_values.push_back(hog_test_features[i][j][k](l));
                    }
                }
            }
            flat_values = flat_values.reshape(1, 1);
            flat_values.convertTo(flat_values, CV_32F);
            testFeatures.push_back(flat_values);
#else
            std::vector< float > flat_values;
            for (int j = 0; j < hog_test_features[0].nc(); j++)
            {
                for (int k = 0; k < hog_test_features[0].nr(); k++)
                {
                    for (int l = 0; l < 31; l++)
                    {
                        flat_values.push_back(hog_test_features[i][j][k](l));
                    }
                }
            }
            dlib::matrix < float, 1116, 1 > temp_mat;
            for (int j = 0; j < 1116; j++)
            {
                temp_mat(j) = (flat_values.at(j));

            }
            testFeatures.push_back(temp_mat);
#endif
        }
        mute.unlock();
    }
#ifdef USE_BOOST
    cv::Ptr<cv::ml::Boost> classifier = (cv::Ptr<cv::ml::Boost>) (classi_data.getClassifier(1));
    cv::Ptr<cv::ml::TrainData> train_data;
    train_data = train_data->create(features, cv::ml::ROW_SAMPLE, labels);
    classifier->train(train_data);

    cv::Mat results;
    classifier->predict(testFeatures, results);

    // validate
    for (int i = 0; i < results.size().height; i++)
    {
        float prediction = results.at<float>(i);
        if (prediction == 1.f && i / ((int)results.size().height / m_NrClasses) == class_i)
        {
            classi_data.addError(class_i);
            m_error_matrix[i / ((int)results.size().height / m_NrClasses)][class_i]++;
        }
        else if (prediction == 1.f && i / ((int)results.size().height / m_NrClasses) != class_i)
        {
            classi_data.addError(class_i);
            m_error_matrix[i / ((int)results.size().height / m_NrClasses)][class_i]++;
        }

#else
    const double max_nu = dlib::maximum_nu(labels);
    double nu = 0.015;
    dlib::svm_nu_trainer<kernel_type> classifier = classi_data.getClassifier(1);
    classifier.set_nu(nu);
    classifier.set_kernel(kernel_type(nu));
    //std::cout << "nu was set to : " << classifier.get_nu() << std::endl;
    //std::cout << "Max nu: " << max_nu << std::endl;
    funct_type learned_function;
    try
    {
        learned_function = classifier.train(features, labels);
    }
    catch (dlib::error e )
    {
        std::cout << std::endl << e.what() << std::endl;
    }
    std::vector < float > results;
    for (uint i = 0; i < testFeatures.size(); i++)
    {
        float prediction = learned_function(testFeatures.at(i));
        if (prediction > 0)
        {
            results.push_back(1.f);
        }
        else
        {
            results.push_back(-1.f);
        }
    }
    for (uint i = 0; i < results.size(); i++)
    {
        float prediction = results.at(i);

        //	std::cout << std::endl << "Prediction " << prediction;
        if (prediction == 1.f && i/((int)results.size()/m_NrClasses) == class_i)
        {
            classi_data.addError(class_i);
            m_error_matrix[i/((int)results.size()/m_NrClasses)][class_i]++;
        }
        else if (prediction == 1.f && i/((int)results.size()/m_NrClasses) != class_i)
        {
            classi_data.addError(class_i);
            m_error_matrix[i/((int)results.size()/m_NrClasses)][class_i]++;
        }
#endif
    }
    /*for (int i = 0; i < m_NrClasses; i++)
    {
        std::cout << " " << m_error_matrix[class_i][i];
    }*/
}

// ---------------------------------------------------------------------------------------------------------------------

void KFoldValidation::prepareTraining(int class_i, ClassifierData &classi_data, int k, dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > > &m_all_image_data)
{    
    // create initial fold datastructure;
    for(int fold = 1; fold <= k; fold++)
        trainClass(class_i, classi_data, k, fold);
}

// ---------------------------------------------------------------------------------------------------------------------

void KFoldValidation::printErrorMatrix()
{
    for(uint i = 0 ; i < m_error_matrix.size(); i++)
    {
        std:: cout << std::endl;
        for(uint j = 0; j < m_error_matrix[i].size(); j++)
            std:: cout << m_error_matrix[i][j] << " ";
    }
}
