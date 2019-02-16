

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string pathName = "./bowj/dataset";
int SZ = 20;
float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;

void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<int> &trainLabels) {

	vector<String> pfilenames;
	//pos image save
	glob(pathName + "/pos", pfilenames);
	int pnum = 0;
	int nnum = 0;
	for (size_t i = 0; i < pfilenames.size(); i++)
	{
		trainCells.push_back(imread(pfilenames[i], IMREAD_GRAYSCALE));
		pnum++;

	}
	//neg img save
	vector<String> nfilenames;

	glob(pathName + "/neg", nfilenames);
	for (size_t i = 0; i < nfilenames.size(); i++)
	{
		trainCells.push_back(imread(nfilenames[i], IMREAD_GRAYSCALE));
		nnum++;
	}
	cout << pnum + nnum << endl;
	float digitClassNumber = 0;
	//label save
	for (int z = 0; z < trainCells.size(); z++) {
		if (z == pnum && z != 0) {
			digitClassNumber = digitClassNumber + 1;
		}
		trainLabels.push_back(digitClassNumber);
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
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		CvSVMParams params = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 2.5, 2.1, 0.5, 0.1, NULL, criteria);
		CvSVM svm;
		CvMat tryMat = trainMat;
		Mat trainLabelsMat(trainLabels.size(), 1, CV_32FC1);

		for (int i = 0; i < trainLabels.size(); i++) {
			cout << "\n_____" << trainLabels[i] << endl;
			trainLabelsMat.at<float>(i, 0) = trainLabels[i];
		}
		CvMat tryMat_2 = trainLabelsMat;
		svm.train_auto(&tryMat, &tryMat_2, Mat(), Mat(), params);
		svm.save("mnist.xml");
		svm.predict(testMat, testResponse);
	}


}
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

	for (int i = 0; i < testResponse.rows; i++)
	{
		//cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
		cout << "-----\n" << testResponse.at<float>(i, 0) << endl;
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
		}
	}

	accuracy = (count / testResponse.rows) * 100;
}
Mat bowCell(BOWImgDescriptorExtractor bowextractor,vector<Mat> trainCell) {
	Mat bowcell;
	for (int i = 0; i < trainCell.size(); i++) {
		Mat img = trainCell[i];
		Mat bowDescriptor = Mat(0, 32, CV_32F);
		vector<KeyPoint> keypoints;
		SiftFeatureDetector detector;
		detector.detect(img,keypoints);
		bowextractor.compute(img, keypoints,bowDescriptor);
		bowcell.push_back(bowDescriptor);
		printf("%d\n", i);
	}
	return bowcell;
}
void loadTrainTestLabelt(string &pathName, vector<Mat> &trainCells, vector<int> &trainLabels) {

	vector<String> pfilenames;
	//pos image save
	glob(pathName + "/test", pfilenames);
	int pnum = 0;
	int nnum = 0;
	for (int i = 0; i < pfilenames.size(); i++)
	{
		trainCells.push_back(imread(pfilenames[i], IMREAD_GRAYSCALE));
		cout << pfilenames[i] << endl;
		pnum++;

	}
	//neg img save

	
	cout << pnum + nnum << endl;
	float digitClassNumber = 0;
	//label save
	for (int z = 0; z < trainCells.size(); z++) {
		if (z == 4) {
			digitClassNumber = digitClassNumber + 1;
		}
		trainLabels.push_back(digitClassNumber);
	}

}
void mnisttrain() {
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	loadTrainTestLabel(pathName, trainCells, trainLabels);
	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;
	//이미지 불러오기 현재는 pos와 neg만 저장함
	cout << "\file down done\n" << endl;
	//bow에 넣을 decriptor 뽑기
	SiftFeatureDetector detector;
	SiftDescriptorExtractor ex;

	Mat vocabulary;
	vocabulary.create(0, 1, CV_32F);
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;
	int num_cluster = 200;
	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);
	vector<Mat> descriptorzip;
	Mat training_descriptors_S;
	cout << "\file down done\n" << endl;

	vector<KeyPoint> keypoint;
	Mat descriptor;
	//

	for (int i = 0; i < trainCells.size(); i++) {


		detector.detect(trainCells[i], keypoint);

		ex.compute(trainCells[i], keypoint, descriptor);
		cout << i << endl;
		descriptorzip.push_back(descriptor);
		training_descriptors_S.push_back(descriptor);

	}
	bowtrainer.add(training_descriptors_S);
	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorExtractor> descExtractor=SIFT::create("SIFT");
	Ptr<DescriptorMatcher> descMatcher = BFMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowExtractor(descExtractor,descMatcher);
	
	bowExtractor.setVocabulary(vocabulary);
	cout << "\nvoca done\n" << endl;
	//vocabulary done;
	
	Mat bowcell=bowCell(bowExtractor,trainCells);
	//svm gogo
	cout << "\start svm\n" << endl;

	Mat testResponse;
	SVMtrain(bowcell, trainLabels, testResponse, bowcell);


	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, trainLabels);

	cout << "Accuracy        : " << accuracy << "%" << endl;
	
}
void test() {
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	loadTrainTestLabelt(pathName, trainCells, trainLabels);
	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;
	//이미지 불러오기 현재는 pos와 neg만 저장함
	cout << "\file down done\n" << endl;
	//bow에 넣을 decriptor 뽑기
	SiftFeatureDetector detector;
	SiftDescriptorExtractor ex;

	Mat vocabulary;
	vocabulary.create(0, 1, CV_32F);
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;
	int num_cluster = 200;
	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);
	vector<Mat> descriptorzip;
	Mat training_descriptors_S;
	cout << "\file down done\n" << endl;

	vector<KeyPoint> keypoint;
	Mat descriptor;
	//

	for (int i = 0; i < trainCells.size(); i++) {


		detector.detect(trainCells[i], keypoint);

		ex.compute(trainCells[i], keypoint, descriptor);
		cout << i << endl;
		descriptorzip.push_back(descriptor);
		training_descriptors_S.push_back(descriptor);

	}
	bowtrainer.add(training_descriptors_S);
	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorExtractor> descExtractor = SIFT::create("SIFT");
	Ptr<DescriptorMatcher> descMatcher = BFMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowExtractor(descExtractor, descMatcher);

	bowExtractor.setVocabulary(vocabulary);
	cout << "\nvoca done\n" << endl;
	//vocabulary done;

	Mat bowcell = bowCell(bowExtractor, trainCells);
	//svm gogo
	cout << "\start svm\n" << endl;

	Mat testResponse;
	SVMtrain(bowcell, trainLabels, testResponse, bowcell);


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
	mnisttrain();
	cout << "\ntest\n" << endl;
	test();
	/*
	if (fs.isOpened()) {
		cout << "a" << endl;
		test();
	}
	else {
		cout << "b" << endl;
		mnisttrain();
	}
	*/
	return 0;
}
