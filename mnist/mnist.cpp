
//This code will run in both opencv 2 and 3. Just change the first two macros in the code according to the requirement.
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string pathName = "./MNIST";
int SZ = 20;
float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;

Mat deskew(Mat& img) {
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	Moments m = moments(gray);

	if (abs(m.mu02) < 1e-2) {
		return img.clone();
	}
	float skew = m.mu11 / m.mu02;
	Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
	Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());

	warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);

	return imgOut;
}

void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<Mat> &testCells, vector<int> &trainLabels, vector<int> &testLabels) {

	vector<String> filenames;
	glob(pathName, filenames);
	int ImgCount = 0;
	int j = 0;
	for (size_t i = 0; i < filenames.size(); i++)
	{

		if (j < 450) {

			trainCells.push_back(imread(filenames[i], IMREAD_COLOR));
			j++;

		}
		else {

			testCells.push_back(imread(filenames[i], IMREAD_COLOR));
			j++;
			if (j == 500)j = 0;
		}
		ImgCount++;
	}


	cout << "Image Count : " << ImgCount << endl;
	float digitClassNumber = 0;

	for (int z = 0; z<int(0.9*ImgCount); z++) {
		if (z % 450 == 0 && z != 0) {
			digitClassNumber = digitClassNumber + 1;
		}
		trainLabels.push_back(digitClassNumber);
	}
	digitClassNumber = 0;
	for (int z = 0; z<int(0.1*ImgCount); z++) {
		if (z % 50 == 0 && z != 0) {
			digitClassNumber = digitClassNumber + 1;
		}
		testLabels.push_back(digitClassNumber);
	}
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells, vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells) {


	for (int i = 0; i < trainCells.size(); i++) {

		Mat deskewedImg = deskew(trainCells[i]);

		deskewedTrainCells.push_back(deskewedImg);
	}

	for (int i = 0; i < testCells.size(); i++) {

		Mat deskewedImg = deskew(testCells[i]);
		deskewedTestCells.push_back(deskewedImg);
	}

}

HOGDescriptor hog(
	Size(20, 20), //winSize
	Size(8, 8), //blocksize
	Size(4, 4), //blockStride,
	Size(8, 8), //cellSize,
	9, //nbins,
	1, //derivAper,
	-1, //winSigma,
	0, //histogramNormType,
	0.2, //L2HysThresh,
	0,//gammal correction,
	64//nlevels=64
);
void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells) {

	for (int y = 0; y < deskewedtrainCells.size(); y++) {
		vector<float> descriptors;
		hog.compute(deskewedtrainCells[y], descriptors);
		trainHOG.push_back(descriptors);
	}

	for (int y = 0; y < deskewedtestCells.size(); y++) {

		vector<float> descriptors;
		hog.compute(deskewedtestCells[y], descriptors);
		testHOG.push_back(descriptors);
	}
}
void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

	int descriptor_size = trainHOG[0].size();

	for (int i = 0; i < trainHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			trainMat.at<float>(i, j) = trainHOG[i][j];
		}
	}
	for (int i = 0; i < testHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			testMat.at<float>(i, j) = testHOG[i][j];
		}
	}
}
/*
void getSVMParams(SVM *svm)
{
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}*/

void SVMtrain(Mat &trainMat, vector<int> &trainLabels, Mat &testResponse, Mat &testMat) {
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 0.50625;
	params.C = 2.5;
	CvSVM svm;
	CvMat tryMat = trainMat;
	Mat trainLabelsMat(trainLabels.size(), 1, CV_32FC1);

	for (int i = 0; i < trainLabels.size(); i++) {
		trainLabelsMat.at<float>(i, 0) = trainLabels[i];
	}
	CvMat tryMat_2 = trainLabelsMat;
	svm.train(&tryMat, &tryMat_2, Mat(), Mat(), params);
	svm.save("mnist.xml");
	svm.predict(testMat, testResponse);

}
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

	for (int i = 0; i < testResponse.rows; i++)
	{
		//cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
		}
	}

	accuracy = (count / testResponse.rows) * 100;
}
void mnisttrain() {
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	loadTrainTestLabel(pathName, trainCells, testCells, trainLabels, testLabels);

	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;


	CreateDeskewedTrainTest(deskewedTrainCells, deskewedTestCells, trainCells, testCells);


	std::vector<std::vector<float> > trainHOG;
	std::vector<std::vector<float> > testHOG;
	CreateTrainTestHOG(trainHOG, testHOG, deskewedTrainCells, deskewedTestCells);

	int descriptor_size = trainHOG[0].size();
	cout << "Descriptor Size : " << descriptor_size << endl;

	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
	Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

	ConvertVectortoMatrix(trainHOG, testHOG, trainMat, testMat);

	Mat testResponse;
	SVMtrain(trainMat, trainLabels, testResponse, testMat);


	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, testLabels);

	cout << "Accuracy        : " << accuracy << "%" << endl;
}
void test() {
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	loadTrainTestLabel(pathName, trainCells, testCells, trainLabels, testLabels);

	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;


	CreateDeskewedTrainTest(deskewedTrainCells, deskewedTestCells, trainCells, testCells);

	std::vector<std::vector<float> > trainHOG;
	std::vector<std::vector<float> > testHOG;
	CreateTrainTestHOG(trainHOG, testHOG, deskewedTrainCells, deskewedTestCells);

	int descriptor_size = trainHOG[0].size();
	cout << "Descriptor Size : " << descriptor_size << endl;

	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
	Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

	ConvertVectortoMatrix(trainHOG, testHOG, trainMat, testMat);

	Mat testResponse;
	CvSVM svm;
	svm.load("./mnist.xml");
	svm.predict(testMat, testResponse);

	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, testLabels);

	cout << "Accuracy        : " << accuracy << "%" << endl;
}
int main() {
	String SVMfilename = "./mnist.xml";
	FileStorage fs(SVMfilename, FileStorage::READ);
	SVM svm;
	if (fs.isOpened()) {
		cout << "a" << endl;
		test();
	}
	else {
		cout << "b" << endl;
		mnisttrain();
	}
	return 0;
}
