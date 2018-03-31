#include<opencv2/objdetect.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

String face_cascade_name = "resources/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

void detectAndDisplay(Mat frame, Mat4b meme)
{
    vector<Rect> faces; // Data structure to hold the list of all the detected faces
	Mat frame_gray; // Convert the current into grayscale
	Mat4b newSource, tmp_meme;
		
	float temp_width_factor, temp_height_factor; // Parameters for adjusting the size of the meme frame so that it covers the detected face

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);                                                     // Covert the image to grayscale (eases the calculations)
    equalizeHist(frame_gray, frame_gray);                                                            // improve the constrast of the image
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));   // detect faces

	cvtColor(frame, newSource, CV_BGR2BGRA, 4);
	for(size_t i = 0; i < faces.size(); i++)
	{
		temp_width_factor = (float) faces[i].width/meme.cols * 2.0;     // Adjust the meme frame for
		temp_height_factor = (float) faces[i].height/meme.rows * 2.0;   // every detected face.
		
		resize(meme, tmp_meme, Size(), temp_width_factor, temp_height_factor); // Resize the frame
		int x = faces[i].x - (tmp_meme.cols-faces[i].width)/2;
		int y = faces[i].y - (tmp_meme.rows-faces[i].height)/2;
		
		for(int m = 0; m < tmp_meme.rows; m++)
			for(int n = 0; n < tmp_meme.cols; n++)
				if((int) tmp_meme.at<Vec4b>(m, n)[3] != 0)
				{
                    newSource.at<Vec4b>(m + y, n + x)[0] = saturate_cast<uchar>(tmp_meme.at<Vec4b>(m, n)[0]);   // For every pixel in the ROI
                    newSource.at<Vec4b>(m + y, n + x)[1] = saturate_cast<uchar>(tmp_meme.at<Vec4b>(m, n)[1]);   // of the detected face, overwrite
                    newSource.at<Vec4b>(m + y, n + x)[2] = saturate_cast<uchar>(tmp_meme.at<Vec4b>(m, n)[2]);   // the pixel RGBa values
                    newSource.at<Vec4b>(m + y, n + x)[3] = saturate_cast<uchar>(tmp_meme.at<Vec4b>(m, n)[3]);   // with that of the adjusted meme frame
                }
		}
		namedWindow("Face Detection", WINDOW_AUTOSIZE);
		imshow("Face Detection", newSource); // Output
}

int main()
{
	VideoCapture cap(0); //Initialize the camera input. Using '0' for the default camera input device.
	Mat frame, meme;
	Mat4b newFrame;
	meme = imread("resources/Pepe.png", CV_LOAD_IMAGE_UNCHANGED); // RGBa meme image
	face_cascade.load(face_cascade_name);
	if(!cap.isOpened()) //For when camera can't be opened.
	{
		cout << "Can't get the camera to work...";
		return -1;
	}
	while(cap.read(frame))
	{
		 if( frame.empty() )
		 {
			 cout << " --(!) No captured frame -- Break!\n";
		     break;
		 }
		 cvtColor(frame, newFrame, CV_BGR2BGRA, 4); // Convert the current frame in RGB to RGBa
		 detectAndDisplay(frame, meme);
		if(waitKey(1)==27) // Press 'Esc' to terminate
			break;
	}
	return 0;
}
