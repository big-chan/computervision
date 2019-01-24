#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|C:/opencv-3.4.3/build/etc/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|C:/opencv-3.4.3/build/etc/haarcascades/haarcascade_eye.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();
	String face_cascade_name = parser.get<String>("face_cascade");
	String eyes_cascade_name = parser.get<String>("eyes_cascade");
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};
	int camera_device = parser.get<int>("camera");
	Mat frame = imread("11.jpg");
	detectAndDisplay(frame);
	return 0;
}
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces ,1.1,2,0);
	cout <<"f"<< faces.size() << endl;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		Mat faceROI = frame_gray(faces[i]);
		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		cout << eyes.size() << endl;
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
	waitKey(0);
}