#define trainOrtest 2 //train == 1 or test == 2

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>


#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <io.h>
#include <stdlib.h>
#include <direct.h>
#include "corecrt_io.h"


using namespace cv;
using namespace std;

string pathName = "./101_ObjectCategories/101_ObjectCategories/*.*";
string pathName2 = "./101_ObjectCategories/101_ObjectCategories/";


void loaddata(string &pathName, vector<Mat> &trainCells, vector<int> &trainLabels, int trainortest) {
	int datanum = 0;
	if (trainortest == 1) {
		

		//-------------------------
		struct _finddata_t fd;

		intptr_t handle;

		if ((handle = _findfirst(pathName.c_str(), &fd)) == -1L)

			cout << "No file in directory!" << endl;
		int i = 0;
		do

		{
			if (i >= 2) {
				vector<String> filenames;
				cout << fd.name << endl;
				glob(pathName2 + fd.name, filenames);
				cout << filenames.size() << endl;
				for (int j = 0; j < (int)(filenames.size()*(0.8)); j++)
				{
					trainCells.push_back(imread(filenames[j], IMREAD_GRAYSCALE));
				}
				for (int z = 0; z < (int)(filenames.size()*(0.8)); z++) {

					trainLabels.push_back(i-1);
				}
			}
			i++;
			

		} while (_findnext(handle, &fd) == 0);
		cout << "num=" << trainCells.size() << "ddd" << trainLabels.size() << endl;
		_findclose(handle);

	}
	else {
		//-----------------
		datanum = 20;
		struct _finddata_t fd;

		intptr_t handle;

		if ((handle = _findfirst(pathName.c_str(), &fd)) == -1L)

			cout << "No file in directory!" << endl;
		int i = 0;
		do

		{
			if (i >= 2) {

				vector<String> filenames;
				cout << fd.name << endl;
				glob(pathName2 + fd.name, filenames);
				cout << filenames.size() << endl;
				
				int tnum = 0;
				for (int j = 0; j < filenames.size(); j++)
				{
					if (j >= (int)(filenames.size()*(0.8))) {
						trainCells.push_back(imread(filenames[j], IMREAD_GRAYSCALE));
						tnum++;
					}
				}
				for (int z = 0; z < tnum; z++) {

					trainLabels.push_back(i);
				}
				datanum = 20;
			}
			i++;
			
		} while (_findnext(handle, &fd) == 0);
		cout << "num=" << trainCells.size() << "ddd" << trainLabels.size() << endl;
		_findclose(handle);
	}


}
void SVMtrain(Mat &trainMat, vector<int> &trainLabels, Mat &testResponse, Mat &testMat) {
	String SVMfilename = "./mnist.xml";
	FileStorage fs(SVMfilename, FileStorage::READ);

	if (fs.isOpened()) {
		CvSVM svm;
		svm.load("./mnist.xml");
		svm.predict(testMat, testResponse);
	}
	else {
		CvSVMParams params = CvSVMParams(
			CvSVM::C_SVC,   // Type of SVM, here N classes (see manual)
			CvSVM::RBF,  // kernel type (see manual)
			0.0,			// kernel parameter (degree) for poly kernel only
			2,			// kernel parameter (gamma) for poly/rbf kernel only
			0.0,			// kernel parameter (coef0) for poly/sigmoid kernel only
			20,		// SVM optimization parameter C
			0,				// SVM optimization parameter nu (not used for N classe SVM)
			0,				// SVM optimization parameter p (not used for N classe SVM)
			NULL,			// class wieghts (or priors)
			// Optional weights, assigned to particular classes.
			// They are multiplied by C and thus affect the misclassification
			// penalty for different classes. The larger weight, the larger penalty
			// on misclassification of data from the corresponding class.

			// termination criteria for learning algorithm

			cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)

		);
		CvSVM svm;
		CvMat tryMat = trainMat;
		Mat trainLabelsMat(trainLabels.size(), 1, CV_32FC1);

		for (int i = 0; i < trainLabels.size(); i++) {
			cout << "\n_____" << trainLabels[i] << endl;
			trainLabelsMat.at<float>(i, 0) = trainLabels[i];
		}
		CvMat tryMat_2 = trainLabelsMat;
		cout << "\n------------" << trainMat.size() << "----------" << trainLabels.size() << endl;
		CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;

		
		svm.train_auto(&tryMat, &tryMat_2, Mat(), Mat(), params);

		svm.save("mnist.xml");
		svm.predict(testMat, testResponse);
	}


}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

	int num = 0;
	int accuracy1 = 0;
	for (int i = 0; i < testResponse.rows; i++)
	{
		//cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
		cout << "-----\n" << testResponse.at<float>(i, 0) << endl;
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
			
		}
		num++;
		if (testLabels[i - 1] != testLabels[i]) {
			accuracy1 = (count / num) * 100;
			cout << "\naccuracy=" << testLabels[i] << "=="<<accuracy1 << endl;
			num = 0;

		}
	}
	accuracy = (count / testResponse.rows) * 100;
	
}
Mat bowCell(BOWImgDescriptorExtractor bowExtractor, vector<Mat> trainCells) {
	Mat bowcell;
	for (int i = 0; i < trainCells.size(); i++) {
		Mat img = trainCells[i];
		Mat bowDescriptor = Mat(0, 32, CV_32F);

		vector<KeyPoint> keypoints;
		SiftFeatureDetector detector;

		detector.detect(img, keypoints);

		bowExtractor.compute(img, keypoints, bowDescriptor);
		bowcell.push_back(bowDescriptor);
		printf("%d\n", i);
	}
	return bowcell;
}

void bowtrain(int what) {
	vector<Mat> trainCells;
	vector<Mat> testCells;

	vector<int> trainLabels;
	vector<int> testLabels;

	loaddata(pathName, trainCells, trainLabels, what);
	// 사진 불러오기


	cout << "file open done\n" << endl;



	SiftFeatureDetector detector; //sift detector   
	SiftDescriptorExtractor extractor;// sift extractor

	vector<KeyPoint> keypoint; // 특징점 저장 변수

	Mat descriptor; // 반복문 내에서 각 train이미지들의 descriptor를 임시 저장

	Mat training_descriptors;// train이미지들의 descriptor들을 누적해서 저장


	for (int i = 0; i < trainCells.size(); i++) {


		detector.detect(trainCells[i], keypoint);

		extractor.compute(trainCells[i], keypoint, descriptor);

		training_descriptors.push_back(descriptor);

		cout << (int)((float)(i + 1) / trainCells.size() * 100) << "% processing" << endl;// 진행상황

	}

	Mat vocabulary;
	vocabulary.create(0, 1, CV_32F); // 인자 - (행, 열, 반환형식)   CV_32F는 처음 행렬을 영행렬로 초기화해줌

	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;// 

	int num_cluster = 200;

	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);// num_cluster: cluster의 수(=단어의 수)  flags


	bowtrainer.add(training_descriptors);
	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorExtractor> descExtractor = SIFT::create("SIFT");
	Ptr<DescriptorMatcher> descMatcher = BFMatcher::create("BruteForce");

	BOWImgDescriptorExtractor bowExtractor(descExtractor, descMatcher);

	bowExtractor.setVocabulary(vocabulary);

	cout << "\nvoca done\n" << endl;
	//vocabulary classifier done;

	Mat bowcell((int)trainCells.size(), bowExtractor.getVocabulary().rows, CV_32FC1); 
	bowcell= bowCell(bowExtractor, trainCells);
	//svm gogo
	cout << "\start svm\n" << endl;

	Mat testResponse;
	SVMtrain(bowcell, trainLabels, testResponse, bowcell);

	cout << "\nsvmdone-----------start__test____\n" << endl;
	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, trainLabels);

	cout << "Accuracy        : " << accuracy << "%" << endl;


}


int main() {


	String SVMfilename = "./mnist.xml";
	FileStorage fs(SVMfilename, FileStorage::READ);
	SVM svm;

	svm.load("./mnist.xml");
	bowtrain(1);
	bowtrain(2);



	return 0;
}