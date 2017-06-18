#include "include/KFoldValidation.h"
#include <iostream>
#include <random>
#include <dlib/gui_widgets.h>
#include <thread>

//typedef dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >		dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >;
//typedef dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >						dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >;
//typedef dlib::array < dlib::array2d < dlib::bgr_pixel>* >										dlib::array < dlib::array2d < dlib::bgr_pixel>* >;

//=====================================================================================================================
/*TODOS:
class Classifier(with int classNr + class Classifier + class ClassifierOutput)
class ClassifierOutput: std::vector<int> classifiedAs, std::vector<int> errorDistribution
reserve capacity of nr_categories in ClassifierOutput()
*/
//=====================================================================================================================
KFoldValidation::KFoldValidation()
{
    m_error_matrix.resize(101);
    for (int class_i = 0; class_i < 101; class_i++)
    {
        m_error_matrix[class_i].resize(101);
    }
}


KFoldValidation::~KFoldValidation()
{
}

int KFoldValidation::create10Fold(dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > >& m_all_image_data)
{
    //get smallest imagenumber of all classes
    smallestClassSize = findImageNumberOfSmallestClass(m_all_image_data);
    int k = 10; // definition of 10 folds

    ClassifierData threadedClassifiers [std::thread::hardware_concurrency()];

    int class_i = 0;
    while (class_i < 101)
    {
        std::cout << std::endl << "10Fold class: " << class_i;
        ClassifierData classi_data (101, class_i, 1); // 1 for boost Classifier

        // create initial fold datastructure
        m_initalFolds = createInitialFolds(k, smallestClassSize, m_all_image_data);

        for(int fold = 1; fold <= k; fold++)
            trainClass(class_i, classi_data, k, fold);

        m_classifier.push_back(classi_data);
        class_i++;
    }


    return 0;
}

dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > >
KFoldValidation::createInitialFolds
(int Folds, int numberOfImages, dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > > &all_image_data)
{
    dlib::array < dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel >* > > > initialFolds;
    initialFolds.resize(101);// for each class produce folds

    // go over each class and gives each object an inital fold

    for (int class_i = 0; class_i < 101; class_i++)
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

int  KFoldValidation::findImageNumberOfSmallestClass(dlib::array < dlib::array < dlib::array2d < dlib::bgr_pixel > > > &m_all_image_data)
{
    int smallestnumber = 1000; // initial number of smallest class
    int tempnumber; // temp number of images per class

    for (int i = 0 ; i<101;i++)
    {
        tempnumber = static_cast<int>(m_all_image_data[i].size());

        if (tempnumber < smallestnumber)
        {
            smallestnumber = tempnumber;
        }
    }
    return smallestnumber;
}

void KFoldValidation::trainClass(int class_i, ClassifierData& classi_data, int k, int fold)
{
    cv::Mat features, labels, testFeatures;
    // create Lists for Trainig and Testing
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_training_features(smallestClassSize);
    std::vector < dlib::array2d<dlib::matrix<float, 31, 1> > > hog_test_features(smallestClassSize);
    int cellsize = 16;
    int colPadding = 4;
    int rowPadding = 4;
    // build training set for each fold
    for (int trainfold = 1; trainfold <= k; trainfold++)
    {
        if (trainfold == fold)
        {
            continue;
        }
        for (int c = 0; c < 101; c++)
        {

            dlib::array < dlib::array2d < dlib::bgr_pixel>* >* foldimages = &m_initalFolds[c][trainfold];
            for (unsigned int i = 0; i < foldimages->size(); i++)
            {

                dlib::extract_fhog_features(*(*foldimages)[i], hog_training_features[i],cellsize, rowPadding, colPadding);
                signed int label = (class_i == c) ? 1 : -1;
                labels.push_back(label);

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
            }
        }
    }

    // build test set for each fold
    for (int c = 0; c < 101; c++)
    {
        dlib::array < dlib::array2d < dlib::bgr_pixel>* >* foldimages = &m_initalFolds[c][fold];
        for (unsigned int i = 0; i < foldimages->size(); i++)
        {
            dlib::extract_fhog_features(*(*foldimages)[i], hog_test_features[i],cellsize, rowPadding, colPadding);

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
        }
    }

    cv::Ptr<cv::ml::Boost> classifier = (cv::Ptr<cv::ml::Boost>) (classi_data.getClassifier(1));
    std::cout << std::endl << "Starting training class: " << class_i;
    cv::Ptr<cv::ml::TrainData> train_data;
    train_data = train_data->create(features, cv::ml::ROW_SAMPLE, labels);
    classifier->train(train_data);
    std::cout << std::endl << "Finished training class: " << class_i;


    cv::Mat results;
    classifier->predict(testFeatures, results);
    std::cout << std::endl << "Finished predicting class: " << class_i;
    // test

    for (int i = 0; i < results.size().height; i++)
    {
        float prediction = results.at<float>(i);
        //	std::cout << std::endl << "Prediction " << prediction;
        if (prediction == -1.f && testClasses[i] == class_i)
        {
            //classi_data.addError(classes);
            //	m_classifier[testClasses[i]].addError(classes);
            m_error_matrix[testClasses[i]][class_i]++;
        }
        else
            if (prediction == 1.f && testClasses[i] != class_i)
            {
                //classi_data.addError(classes);
                //	m_classifier[testClasses[i]].addError(classes);
                m_error_matrix[testClasses[i]][class_i]++;
            }

    }
    std::cout << std::endl << "Printing results for class " << classi_data.getNr() << std::endl;
    for (int i = 0; i < 101; i++)
    {
        std::cout << " " << m_error_matrix[class_i][i];
    }
}



/*
      // setup multithreading
        int numThreads = std::thread::hardware_concurrency();
        int numOps = 0;
        int fold = 1;
        std::thread threadArr[numThreads];
while (fold <= k)
{
    // more operations/iterations than threads => use all threads
    if((k - fold)/numThreads  > 0)
    {
        numOps = numThreads;
        for(int i = 0; i < numOps; i++)
        {
            threadArr[i] = std::thread(&KFoldValidation::trainClass,this, class_i,std::ref(classi_data),k,fold); //function pointer, object, param
            //numObjs--;
            fold++;

#ifdef __unix__     // load balancing on unix systems
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_setaffinity_np(threadArr[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
        }
    }
    // less operations/iterations than threads => use remaining number of threads
    else
    {
        numOps = (k - fold) % numThreads;
        for(int i = 0; i < numOps; i++)
        {
            threadArr[i] = std::thread(&KFoldValidation::trainClass,this, class_i,std::ref(classi_data),k,fold); //function pointer, object, param
            //numObjs--;
            fold++;

#ifdef __unix__     // load balancing on unix systems
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_setaffinity_np(threadArr[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
        }
    }
    // join threads
    for(int i = 0; i < numOps; i++)
        threadArr[i].join();
*/
