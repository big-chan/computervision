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

bool readVocabulary(const string& filename, Mat& vocabulary)
{
	cout << "Reading vocabulary...";
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
		cout << "done" << endl;
		return true;
	}
	return false;
}
bool writeVocabulary(const string& filename, const Mat& vocabulary)
{
	cout << "Saving vocabulary..." << endl;
	FileStorage fs(filename, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "vocabulary" << vocabulary;
		return true;
	}
	return false;
}
bool readdeccriptor(const string& filename, Mat& deccriptor)
{
	cout << "Reading deccriptor...";
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["deccriptor"] >> deccriptor;
		cout << "done" << endl;
		return true;
	}
	return false;
}
bool writedeccriptor(const string& filename, const Mat& deccriptor)
{
	cout << "Saving deccriptor..." << endl;
	FileStorage fs(filename, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "deccriptor" << deccriptor;
		return true;
	}
	return false;
}
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
				if (filenames.size() > 100)continue;
				//cout << filenames.size() << endl;
			
				for (int j = 0; j < 30; j++)
				{
					trainCells.push_back(imread(filenames[j], IMREAD_GRAYSCALE));
				}
				for (int z = 0; z <30; z++) {

					trainLabels.push_back(i - 1);
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
		int lnum = 0;
		do

		{
			if (i >= 2) {

				vector<String> filenames;
				cout << fd.name << endl;
				glob(pathName2 + fd.name, filenames);
				//cout << filenames.size() << endl;
				if (filenames.size() > 100)continue;
				lnum++;
				int tnum = 0;
				for (int j = 0; j < filenames.size(); j++)
				{
					
					if (j >=30) {
						trainCells.push_back(imread(filenames[j], IMREAD_GRAYSCALE));
						tnum++;
					}
				}
				for (int z = 0; z < tnum; z++) {

					trainLabels.push_back(i-1);
				}
				datanum = 20;
			}
			i++;
			
		} while (_findnext(handle, &fd) == 0);
		cout << "num=" << trainCells.size() << "ddd" << lnum << endl;
		_findclose(handle);
	}


}
void setSVMParams(CvSVMParams& svmParams, CvMat& class_wts_cv, vector<int> numzip)
{
	
	svmParams.svm_type = CvSVM::C_SVC;
	svmParams.kernel_type = CvSVM::RBF;
	Mat class_wts(numzip.size(), 1, CV_32FC1);
	for (int i = 0; i < numzip.size();i++) {
		
		class_wts.at<float>(i) = (float)numzip[i] / (float)numzip.size(); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
		
	}
	cout << "a" << endl;
	class_wts_cv = class_wts;
	svmParams.class_weights = &class_wts_cv;
}
static void setSVMTrainAutoParams(CvParamGrid& c_grid, CvParamGrid& gamma_grid,
	CvParamGrid& p_grid, CvParamGrid& nu_grid,
	CvParamGrid& coef_grid, CvParamGrid& degree_grid)
{
	c_grid = CvSVM::get_default_grid(CvSVM::C);

	gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

	p_grid = CvSVM::get_default_grid(CvSVM::P);
	p_grid.step = 0;

	nu_grid = CvSVM::get_default_grid(CvSVM::NU);
	nu_grid.step = 0;

	coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
	coef_grid.step = 0;

	degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
	degree_grid.step = 0;
}
vector<int> coutsize(Mat &trainMat, vector<int> &trainLabels) {
	vector<int> numzip;
	int num=0;
	for (int i = 0; i < trainLabels.size(); i++) {
		num++;
		if (i + 1!=trainLabels.size() &&trainLabels[i] != trainLabels[i + 1]) {
	
			numzip.push_back(num);
			num = 0;
		}
		else if (i + 1 == trainLabels.size()) {
			numzip.push_back(num);
			num = 0;
		}
	}
	return numzip;
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
	
		vector<int> numzip=coutsize(trainMat,trainLabels);
		CvSVMParams svmParams;
		CvMat class_wts_cv;
		CvSVM svm;
	
		setSVMParams(svmParams, class_wts_cv, numzip);
		CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
		setSVMTrainAutoParams(c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
		
	
	
		
		CvMat tryMat = trainMat;
		Mat trainLabelsMat(trainLabels.size(), 1, CV_32FC1);

		for (int i = 0; i < trainLabels.size(); i++) {
			//cout << "\n_____" << trainLabels[i] << endl;
			trainLabelsMat.at<float>(i, 0) = trainLabels[i];
		}
		CvMat tryMat_2 = trainLabelsMat;
		cout << "\n------------" << trainMat.size() << "----------" << trainLabels.size() << endl;
	

		svm.train_auto(&tryMat, &tryMat_2, Mat(), Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
		

		svm.save("mnist.xml");
		svm.predict(testMat, testResponse);
	}


}
void printevalute(vector<float> accuracyzip, vector<int> numzip) {
	for (int i = 0; i < numzip.size(); i++) {
		cout << "\nlabel=" << i + 1 << ",num=" << numzip[i] << ",accuracy=" << accuracyzip[i] << endl;

	}
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

	int num = 0;
	float accuracy1 = 0;
	float count2 = 0;
	vector<float> accuracyzip;
	vector<int> numzip;
	for (int i = 0; i < testResponse.rows; i++)
	{
		//cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
		//cout << "-----\n" << testResponse.at<float>(i, 0) << endl;
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
			count2++;

		}
		num++;
		if (i+1!= testResponse.rows&&testLabels[i] != testLabels[i+1]) {
			accuracy1 = (count2 /(float)num) * 100;
			cout << "\naccuracy=" << testLabels[i] << "==" << accuracy1 << endl;
			accuracyzip.push_back(accuracy1);
			numzip.push_back(num);
			num = 0;

			count2 = 0;
		}
		else if(i + 1 == testResponse.rows) {
			accuracy1 = (count2 / (float)num) * 100;
			cout << "\naccuracy=" << testLabels[i] << "==" << accuracy1 << endl;
			accuracyzip.push_back(accuracy1);
			numzip.push_back(num);
			num = 0;

			count2 = 0;
		}
	}
	
	accuracy = (count / testResponse.rows) * 100;
	printevalute(accuracyzip, numzip);

}
Mat bowCell(BOWImgDescriptorExtractor bowExtractor, vector<Mat> trainCells,int what) {
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
	String dfilename = "./bowdescriptortrain.xml";
	if (what == 2)dfilename = "./bowdescriptortest.xml";
	if (!writedeccriptor(dfilename, bowcell)) {
		cout << "\nbowdescriptor save error\n" << endl;
	}
	return bowcell;
}
void bowtrain(int what) {
	Mat vocabulary;

	string filename ;

	  filename = "./vocabulary.xml";

	vector<Mat> trainCells;
	vector<Mat> testCells;

	vector<int> trainLabels;
	vector<int> testLabels;

    loaddata(pathName, trainCells, trainLabels, what);
		// 사진 불러오기
	if (!readVocabulary(filename, vocabulary)) {

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

		
		vocabulary.create(0, 1, CV_32F); // 인자 - (행, 열, 반환형식)   CV_32F는 처음 행렬을 영행렬로 초기화해줌

		TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON;// 

		int num_cluster = 1000;

		BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);// num_cluster: cluster의 수(=단어의 수)  flags


		bowtrainer.add(training_descriptors);
		vocabulary = bowtrainer.cluster();
		if (!writeVocabulary(filename, vocabulary)) {
			cout << "error:can't open vocabulary\n" << endl;
		}
	}
	Ptr<DescriptorExtractor> descExtractor = SIFT::create("SIFT");
	Ptr<DescriptorMatcher> descMatcher = BFMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowExtractor(descExtractor, descMatcher);
	Mat bowcell((int)trainCells.size(), bowExtractor.getVocabulary().rows, CV_32FC1);
	bowExtractor.setVocabulary(vocabulary);
	String dfilename = "./bowdescriptortrain.xml";
	if (what == 2)dfilename = "./bowdescriptortest.xml";
	if (!readdeccriptor(dfilename,bowcell)) {
	
		cout << "\nvoca done\n" << endl;
		//vocabulary classifier done;


		bowcell = bowCell(bowExtractor, trainCells,what);
	}
	//svm gogo
	cout << "\nstart svm\n" << endl;

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