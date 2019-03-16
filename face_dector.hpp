#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>  
#include <iostream>
#include <vector>


//simple face detector by opencv
//TODO implement on CUDA later
class face_detector {

	public:
		face_detector() = default;
		~face_detector() = default;

		explicit face_detector(const cv::Mat& image_):image(image_) noexcept {
			init();
		}
		explicit face_detector(const cv::Mat&& image_):image(std::move(image_)) noexcept {
			init();
		}
		
		bool init() {
			if(!face_cascade.load(face_cascade_name)) {
				std::cerr << "--(!)Error loading face cascade\n";	
				return false;
			}
			if(!eyes_cascade.load(eyes_cascade_name)) {
				std::cerr << "--(!)Error loading eyes cascade\n";
				return false;
			}
			return true;
		}

		std::vector<Rect> detect() {
			using namespace cv;
			std::vector<Rect> faces;  
    		Mat frame_gray;  
  
    		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);  
    		equalizeHist(frame_gray, frame_gray);  
  
    		//-- Detect faces  
    		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(70, 70),Size(100,100));  
 			return faces; 
		}

	private:
		cv::Mat image;
		cv::String face_cascade_name = "haarcascade_frontalface_default.xml";  
		cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";  
		cv::CascadeClassifier face_cascade;    
		cv::CascadeClassifier eyes_cascade;
};

#endif
