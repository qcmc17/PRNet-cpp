// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// Copyright (c) 2018 Liu Xiao <liuxiao@foxmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "utils/mat2tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <unordered_map>

using namespace tensorflow;



int main(int argc, char* argv[]) {


	// Initialize a tensorflow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    if (argc < 2)
    {
        std::cerr << std::endl 
		<< "Usage: ./project " << std::endl;
		<< "--graph indicate the freeze graph with exstion .pb\n"
		<< "--input indicate the path to the image wanna predicted\n"
		<< "--output indicate the path to store the mesh file\n"
		<< "--detect_face [true/false] use OpenCV to detect face or not\n";
        return 1;
    }


	
	//parse console parameters
	std::unorderd_map<std::string, std::string> options;
	for(int i = 1; i < argc; i+=2) {
		std::string key(argv[i]);
		std::string value(argv[i+1]);
		options[key] = value;
	}

	bool detect_face = false;
	if(options["--detect_face"] == "true") {
		detect_face = true;
	} 


    // Load the protobuf graph
    GraphDef graph_def;
    std::string graph_path = argv[1];
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }

	fs::path image_path(option["--input"]);
	fs::path mesh_path(option["--output"]);
	std::vector<std::string> images;
	util::getAllImage(image_path, images);

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }


	//loop all the images in the path
	std::vector<std::pair<std::string, tensorflow::Tensor>> input;
	const std::string input_prefix = "Placeholder";
	const std::string output_prefix = "resfcn256/Conv2d_transpose_16/Sigmoid";
	std::vector<tensorflow::Tensor> outputs;

	for(size_t i = 0; i < images.size(); ++i) {
		auto image = cv::imread(images[i]);
		Tensor input_tensor = Mat2Tensor(image);
		input.empalce_back(std::make_pair{input_prority, input_tensor});
		status = session->Run(inputs, {output_prefix}, {}, &outputs);
		if(status.ok()) {
			std::cerr << status.ToString() << std::endl;
		} else {
			std::cout << "Run session successfully" << std::endl;
		}

		assert(outputs.empty());
        tensorflow::TTypes<float, 1>::Tensor tensor_ = outputs[0].tensor<float, 4>();
		int const out_shape[] = {1, outputs[0].dimension(1), outputs[0].dimension(2), outputs[0].dimension(3)};
		cv::Mat result(4, out_shape, CV_32F);
		for(auto i = 0; i < out_shape[1]; ++i) {
			for(auto j = 0; j < out_shape[2]; ++j) {
				result.at<cv::Vec<float, out_shape[3]>>(0, i, j) = tensor_(0, i, j);
			}
		}
	}
    


    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    // Print the results
    std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 30>

//    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = outputs[0].flat<float>();
    const tensorflow::TTypes<float, 1>::Tensor& prediction = outputs[0].flat_inner_dims<float, 1>();

    // Free any resources used by the session
    session->Close();

    return 0;
}

