#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <iostream>

#include "opencv2/core.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;
typedef struct feat {
	Mat featcher;
	vector< KeyPoint > keypoint;
	Mat img;
}feat;
double find_matches(feat img1, feat img2)
{
	Mat  img1_descriptors, img2_descriptors;
	vector< KeyPoint > vtKeypoint_img1, vtKeypoint_img2;
	img1_descriptors = img1.featcher;
	img2_descriptors = img2.featcher;
	vtKeypoint_img1 = img1.keypoint;
	vtKeypoint_img2 = img2.keypoint;

	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	vector<DMatch> good_matches;
	vector<vector < DMatch >> matches;

	//Flann
	  //FlannBasedMatcher Matcher;
	  //Matcher.match(img1_descriptors, img2_descriptors, matches);

	  //BFMatch
	BFMatcher matcher;
	matcher.knnMatch(img1_descriptors, img2_descriptors, matches, 2); // Find two nearest matches

	//좋은 매칭 걸러내기 방법 1 -> Flann
	/*
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;


	for (int i = 0; i < img1_descriptors.rows; i++)
	{
	   dDistance = matches[i].distance;

	   if (dDistance < dMinDist) dMinDist = dDistance;
	   if (dDistance > dMaxDist) dMaxDist = dDistance;

	}
	cout << "Max :" << dMaxDist << endl;
	cout << "Min :" << dMinDist << endl;

	for (int i = 0; i < img1_descriptors.rows; i++)
	{
	   if (matches[i].distance < 3 * dMinDist)
	   {
		  good_matches.push_back(matches[i]);
	   }
	}
	*/

	//좋은 매칭 걸러내기 방법 2 -> BFMatcher

	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.7; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
		if (i == matches.size() - 1) break;
	}

	cout << "Good Match : " << good_matches.size() << endl;
	

	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(vtKeypoint_img1[good_matches[i].queryIdx].pt);
		img2_pt.push_back(vtKeypoint_img2[good_matches[i].trainIdx].pt);
	}

	Mat mask;
	Mat HomoMatrix = findHomography(img2_pt, img1_pt, CV_RANSAC, 5, mask);

	double outline_cnt = 0;
	double inline_cnt = 0;

	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<bool>(i) == 0)
		{
			outline_cnt++;
		}
		else
		{
			inline_cnt++;
		}
	}

	double percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100;
	int match_flag = 0;

	if (percentage >= 10)
	{
		match_flag = 1;
	}
	std::cout << "percentage : " << percentage << std::endl;

	return percentage;
}

Mat sift_panorama(feat img1, feat img2)
{
	
	vector< KeyPoint > vtKeypoint_img1, vtKeypoint_img2;
	Mat img1_descriptors, img2_descriptors;
	img1_descriptors = img1.featcher;
	img2_descriptors = img2.featcher;
	vtKeypoint_img1 = img1.keypoint;
	vtKeypoint_img2 = img2.keypoint;

	vector<DMatch> good_matches;
	vector<vector < DMatch >> matches;

	//Flann
	  //FlannBasedMatcher Matcher;
	  //Matcher.match(img1_descriptors, img2_descriptors, matches);

	  //BFMatch
	BFMatcher matcher;
	matcher.knnMatch(img1_descriptors, img2_descriptors, matches, 2); // Find two nearest matches

	//좋은 매칭 걸러내기 방법 1 -> Flann
	/*
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;


	for (int i = 0; i < img1_descriptors.rows; i++)
	{
	   dDistance = matches[i].distance;

	   if (dDistance < dMinDist) dMinDist = dDistance;
	   if (dDistance > dMaxDist) dMaxDist = dDistance;

	}
	cout << "Max :" << dMaxDist << endl;
	cout << "Min :" << dMinDist << endl;

	for (int i = 0; i < img1_descriptors.rows; i++)
	{
	   if (matches[i].distance < 3 * dMinDist)
	   {
		  good_matches.push_back(matches[i]);
	   }
	}
	*/

	//좋은 매칭 걸러내기 방법 2 -> BFMatcher

	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.7; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
		if (i == matches.size() - 1) break;
	}

	cout << "Good Match : " << good_matches.size() << endl;


	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(vtKeypoint_img1[good_matches[i].queryIdx].pt);
		img2_pt.push_back(vtKeypoint_img2[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(img2_pt, img1_pt, CV_RANSAC, 5);

	std::cout << HomoMatrix << endl;

	Mat matResult;
	Mat matPanorama;

	// 4개의 코너 구하기
	vector<Point2f> conerPt;

	conerPt.push_back(Point2f(0, 0));
	conerPt.push_back(Point2f(img2.img.size().width, 0));
	conerPt.push_back(Point2f(0, img2.img.size().height));
	conerPt.push_back(Point2f(img2.img.size().width, img2.img.size().height));

	Mat P_Trans_conerPt;
	perspectiveTransform(Mat(conerPt), P_Trans_conerPt, HomoMatrix);

	// 이미지의 모서리 계산
	double min_x, min_y, max_x, max_y;
	float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2;

	min_x1 = min(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	min_x2 = min(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	min_y1 = min(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	min_y2 = min(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	max_x1 = max(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	max_x2 = max(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	max_y1 = max(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	max_y2 = max(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	min_x = min(min_x1, min_x2);
	min_y = min(min_y1, min_y2);
	max_x = max(max_x1, max_x2);
	max_y = max(max_y1, max_y2);

	// Transformation matrix
	Mat Htr = Mat::eye(3, 3, CV_64F);
	if (min_x < 0)
	{
		max_x = img1.img.size().width - min_x;
		Htr.at<double>(0, 2) = -min_x;
	}
	else
	{
		if (max_x < img1.img.size().width) max_x = img1.img.size().width;
	}

	if (min_y < 0)
	{
		max_y = img1.img.size().height - min_y;
		Htr.at<double>(1, 2) = -min_y;
	}
	else
	{
		if (max_y < img1.img.size().height) max_y = img1.img.size().height;
	}


	// 파노라마 만들기
	matPanorama = Mat(Size(max_x, max_y), CV_32F);
	warpPerspective(img1.img, matPanorama, Htr, matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);
	warpPerspective(img2.img, matPanorama, (Htr*HomoMatrix), matPanorama.size(), INTER_CUBIC, BORDER_TRANSPARENT, 0);

	return matPanorama;

}
feat findf(Mat img1) {
	feat what;
	Mat imgg;
	cvtColor(img1,imgg,COLOR_BGR2GRAY);
	SiftDescriptorExtractor detector;
	vector< KeyPoint > keypoint;
	detector.detect(img1, keypoint);
	SiftDescriptorExtractor ex;
	Mat featcher;
	ex.compute(img1, keypoint, featcher);
	what.featcher = featcher;
	what.keypoint = keypoint;
	what.img = img1;
	cout << "featdone" << endl;
	return what;
}

int main()
{
	//사진 불러오기&체크
	vector<Mat> panorama;
	vector<feat> pf;
	panorama.push_back(imread("i1.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i3.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i4.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i5.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i6.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i7.jpg", IMREAD_COLOR));
	panorama.push_back(imread("i8.jpg", IMREAD_COLOR));
	feat what;
	
	Mat panoramaimg = imread("i2.jpg", IMREAD_COLOR);

	/**/
	std::cout << "################ matches ################" << std::endl;
	for (int i = 0; i < 7; i++) {
		what = findf(panorama[i]);
		pf.push_back(what);
		cout<<"feat===" << i << endl;
	}
	std::cout << "################ matches ################" << std::endl;
	double per;
	double maxp=0;
	int maxi;
	char buf[256];
	for (int i = 0; i < 7;i++) {
		what = findf(panoramaimg);
		for (int j = 0; j < pf.size(); j++) {
			per=find_matches(what,pf[j]);
			if (maxp <= per) {
				maxp = per;
				maxi = j;
				cout << maxi << endl;
			}
		}
		cout << maxi<<"__size__"<<pf.size()<<endl;
		panoramaimg = sift_panorama(what,pf[maxi]);
		//panoramaimg = crop(panoramaimg);
		sprintf(buf, "img%d.jpg", i);
		namedWindow(buf, CV_WINDOW_FREERATIO);
        imshow(buf, panoramaimg);
		
		pf.erase(pf.begin()+maxi);
		maxp = 0;
	}
	imshow("result", panoramaimg);

	//up
	imwrite("result.jpg", panoramaimg);

	//namedWindow("result1", WINDOW_FREERATIO);
	//imshow("result1", sum_123);

	//namedWindow("result2", WINDOW_FREERATIO);
	//imshow("result2", sum_567);

	
	//imshow("result3", sum_123567);
	/*
	Mat panorama_t_1 = imread("assignment3_data/hill/1.jpg", IMREAD_COLOR);
	Mat panorama_t_2 = imread("assignment3_data/hill/2.jpg", IMREAD_COLOR);
	Mat panorama_t_3 = imread("assignment3_data/hill/3.jpg", IMREAD_COLOR);

	Mat sum_t_12 = sift_panorama(panorama_t_1, panorama_t_2, 500);
	sum_t_12 = crop(sum_t_12);
	Mat sum_t_23 = sift_panorama(panorama_t_2, panorama_t_3, 500);
	sum_t_23 = crop(sum_t_23);
	Mat sum_t_123 = sift_panorama(sum_t_12, sum_t_23, 500);
	sum_t_123 = crop(sum_t_123);

	
	imshow("result_t", sum_t_123);
	*/
	waitKey(0);
}



