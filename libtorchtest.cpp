#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>

int detect(const std::string& model_path, const std::string& image_path, const std::string& output_path) {
    std::cout << torch::cuda::is_available() << std::endl;
    std::cout << torch::cuda::cudnn_is_available() << std::endl;
    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.to(at::kCPU);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    }


    // Read the image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cout << "Failed to load the image." << std::endl;
        return -1;
    }

    // Preprocess the image
    cv::resize(image, image, cv::Size(512, 512));
    cv::Mat normalized_image;
    cv::normalize(image, normalized_image, 0, 255, cv::NORM_MINMAX, CV_32F);


    // Convert the image to a tensor
    //torch::Tensor tensor_image = torch::from_blob(normalized_image.data, { 1, 1, 180, 180 }, torch::kFloat);
    torch::Tensor tensor_image = torch::from_blob(normalized_image.data, { 1, 1, 512, 512 }, torch::kFloat);
    std::cout << "tensor_image.size" << tensor_image.sizes() << std::endl; //[ 1, 1, 750, 750 ]
    tensor_image.print();

    // Create a vector of IValue inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    // Perform the forward pass
    torch::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.sizes() << std::endl;
    output = output.to(at::kCPU);
    
  
    // Convert the tensor to a OpenCV Mat
    cv::Mat output_mat(output.size(2), output.size(3), CV_32F, output.data_ptr<float>());

    // Rescale the output to the 0-255 range
    cv::Mat output_rescaled;
    cv::normalize(output_mat, output_rescaled, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the output image
    cv::imwrite(output_path, output_rescaled);
    cv::imshow("Output Image", output_rescaled);
    cv::waitKey(0);
    return 0;
}
cv::Mat performHistogramEqualization(const cv::Mat& inputImage) {
    // Convert the input image to grayscale if it's a color image
    cv::Mat grayImage;
    if (inputImage.channels() > 1)
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    else
        grayImage = inputImage.clone();

    // Perform histogram equalization
    cv::Mat equalizedImage;
    cv::equalizeHist(grayImage, equalizedImage);

    return equalizedImage;
}
int main() {
    std::string model_path = "ns8.pt";
    std::string image_path = "spine2.jpg";
    std::string output_path = "cpu_spine12.png";

    // Measure the execution time of the detect function
    auto start_time = std::chrono::high_resolution_clock::now();
    detect(model_path, image_path, output_path);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print the execution time
    std::cout << "Execution time: " << duration_ms.count() << " ms" << std::endl;

    //// 直方图均衡化
    //cv::Mat originalImage = cv::imread("tooth_image/9_pred.png");

    //if (originalImage.empty()) {
      //  std::cout << "Error: Could not open or find the image." << std::endl;
     //   return -1;
    //}

    // Perform histogram equalization
    //cv::Mat equalizedImage = performHistogramEqualization(originalImage);

    //Display the images (optional)
    //cv::imshow("Original Image", originalImage);
    //cv::imshow("Equalized Image", equalizedImage);
    //cv::waitKey(0);

    //Save the equalized image (optional)
    //cv::imwrite("equalized_fad_spine12.png", equalizedImage);

    return 0;

}