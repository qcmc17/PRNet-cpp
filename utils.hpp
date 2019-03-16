#ifndef UTILS_H
#define UTILS_H
#include <memory>
#include <fstream>
#include <string>
#include <iostream>
#include <array>
#include <cassert>
#include <limits>

//boost
#include <boost/filesystem.hpp>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "face_detector.hpp"

#include <tensorflow/cc/ops/const_op.h>
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace fs = boost::filesystem;

namespace util{
	
	struct face_index {
		std::vector<size_t> index;
	};

	struct triangle {
		std::vector<std::array<size_t, 3>> tri;	
	};

	struct mesh {
		std::vector<size_t> index;
		std::vector<std::array<float, 3>> color;
		std::vector<std::array<float, 3>> vertices;
		std::vector<std::array<size_t, 3>> tri;
	};

	void imageProcess(cv::Mat& image, bool detect = false) {
		auto size = image.size();
		auto h = size.height, w = size.width;

		if(detect) {
			auto max_size = std::max(h, w);
			if(max_size > 1000) {
				std::cout << "input image size is larger than 1000 pixels" << std::endl;
				size_t h_, w_;
				double factor = static_cast<double>(1000. / max_size * 1.0);
				if(max_size == h) {
					w_ = static_cast<size_t>(factor * w); 
					h_ = 1000;
				}
				else {
					h_ = static_cast<size_t>(factor * h);
					w_ = 1000;
				}
				cv::resize(image, image, cv::Size(h_, w_), 0, 0, cv::INTER_LINEAR);
			} else {
				face_detector fd(image);
				auto rectangles = fd.detect();
				if(rectangles.empty()) {
					std::cerr << "face not found in this picture" << std::endl;
				}
				
				cv::Rect face = fd.front();
				auto left = face.left();
				auto right = face.right();
				auto top = face.top();
				auto bottom = face.bottom();

				auto old_size = (right - left + bottom - top) / 2.0;
				cv::Point2f center(right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14);
				size_t size = static_cast<size_t>(old_size) * 1.58;
			} 

		} else {
			if(h == w) {
				//resize image to get resolution 256*256
				cv::resize(image, image, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
				//image /= 255;
			} else {

				//need to crop the image according to the bbox
				size_t top = 0, right = w-1, left = 0, bottom = h-1;
				double old_size = (right - left + bottom - top)/2.;
				cv::Vec2f center(right - (right - left) / 2.0, bottom - (bottom - top) / 2.0);
				size_t size = static_cast<size_t>(old_size * 1.6);
				cv::Point2f src_pos[3];
				cv::Point2f dst_pos[3];
				cv::Mat warp_mat;
				src_pos[0] = cv::Point2f(center[0]-size/2., center[1]-size/2.);
				src_pos[1] = cv::Point2f(center[0]-size/2., center[1]+size/2.);
				src_pos[2] = cv::Point2f(center[0]+size/2., center[1]-size/2.);

				dst_pos[0] = cv::Point2f(0.f, 0.f);
				dst_pos[1] = cv::Point2f(0, 255.f);
				dst_pos[2] = cv::Point2f(255.f, 0);

				warp_mat = cv::getAffineTransform(src_pos, dst_pos);
				cv::warpAffine(image, image, warp_mat, cv::Size(256, 256));
			}
		}
	}

	static bool save2OBJ(const std::string& filename, mesh& mesh_) {
		
		assert(mesh_.vertices.size() == mesh_.color.size());
		std::unique_ptr<std::ofstream> out_ = std::make_unique<std::ofstream>(filename);
		if(!out_) {
			std::cerr << "open file: " << filename << "failed!\n";
			return false;
		}
		for(size_t i = 0 ; i < mesh_.vertices.size(); ++i) {
			*out_ << "v " << mesh_.vertices[i][0] << ' ' << mesh_.vertices[i][1] << ' '
			<< mesh_.vertices[i][2] << ' ' << mesh_.color[i][0] << ' ' << 
			mesh_.color[i][1] << ' ' << mesh_.color[i][2] << "\r\n";
		}


		for(size_t i = 0; i < mesh_.tri.size(); ++i) {
			*out_ << "f " << mesh_.tri[i][0] << ' ' << mesh_.tri[i][1] << ' ' 
			<< mesh_.tri[i][2] << "\r\n";
		}
		return true;
	}


	void getAllImage(fs::path& path, std::vector<std::string>& im_list) {
	    if(!fs::exists(path) || !fs::is_regular_file(path)) {
	        return ;
	    }   
	
	    fs::directory_iterator it(path);
	    fs::directory_iterator endit;
	
	    while(it != endit) {
	        auto ext = it->path().extension();
	        if (fs::is_regular_file(*it) && (ext == ".png" || ext == ".jpg")) {
	            im_list.emplace_back(it->path().string());
	        }   
	        ++it;
	    }   
	}



	tensorflow::Tensor Mat2Tensor(cv::Mat &img, float normal = 1/255.0) {
	    tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT,
										 tensorflow::TensorShape({1, img.size().height,
										 img.size().width, img.channels()}));
	
	    float *tensor_data_ptr = image_input.flat<float>().data();
	    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
	    img.convertTo(fake_mat, CV_32FC(img.channels()));
	
	    fake_mat *= normal;
	
	    return image_input;
	
	}


	void extractVertices(const cv::Mat& pos_map, const face_index & idx, mesh& mesh_) {
		auto vert_mat = pos_map.reshape(256*256, 3);
		mesh_.index.reserve(idx.index.size());
		for(size_t i = 0; i < idx.index.size(); ++i) {
			auto x = vert_mat.at<float>(idx.index[i], 0);
			auto y = vert_mat.at<float>(idx.index[i], 1);
			auto z = vert_mat.at<float>(idx.index[i], 2);
			mesh_.vertices[i] = {x, y, z};
		}
	}

	void extractColor(const cv::Mat& pos_map, const cv::Mat& image, mesh& mesh_) {
		auto shape = image.size();
		auto h = shape.height, w = shape.width;
		float x = std::numeric_limits<float>::min(), y = std::numeric_limits<float>::min();
		mesh_.color.reserve(mesh_.vertices.size());
		for(size_t i = 0; i < mesh_.vertices.size(); ++i) {
			x = std::max(mesh_.vertices[i][0], 0.f);
			y = std::max(mesh_.vertices[i][1], 0.f);
		}

		x = std::min(x, static_cast<float>(h-1));
		y = std::min(y, static_cast<float>(w-1));
		for(size_t i = 0; i < mesh_.vertices.size(); ++i) {
			mesh_.vertices[i][0] = x;
			mesh_.vertices[i][1] = y;
		}

		for(size_t i = 0; i < mesh_.vertices.size(); ++i) {
			size_t ind_x = static_cast<size_t>(std::ceil(mesh_.vertices[i][1]));
			size_t ind_y = static_cast<size_t>(std::ceil(mesh_.vertices[i][0]));
			mesh_.color[i] = {image.at<float>(ind_x, ind_y, 0),
							  image.at<float>(ind_x, ind_y, 1),
							  image.at<float>(ind_x, ind_y, 2)};
		}

	}



	static bool loadFaceIndex(const std::string& filename, face_index& f_idx) {
		std::unique_ptr<std::ifstream> in = std::make_unique<std::ifstream>(filename);
		if(!in) {
			std::cerr << "bad file input:"  << filename << "\n";
			return false;
		}
		
		double val;
		while(*in >> val) {
			f_idx.index.push_back(static_cast<size_t>(val));
		}
	}

	static bool loadTriangle(const std::string& filename, triangle& tri) {
		std::unique_ptr<std::ifstream> in = std::make_unique<std::ifstream>(filename);
		if(!in) {
			std::cerr << "bad file input:"  << filename << "\n";
			return false;
		}

		std::string line;
		size_t a, b, c;
		while(*in >> line) {
			std::stringstream ss(line);
			ss >> a >> b >> c;
			tri.tri.emplace_back(std::array<size_t, 3>{a,b,c});
		}
	}



}
#endif
