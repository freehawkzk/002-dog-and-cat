/**
 * 猫狗分类推理程序
 * 使用 ONNX Runtime 进行模型推理
 * 使用 OpenCV 进行图像加载与结果显示
 * 
 * 编译方法 (使用 CMake):
 *   mkdir build && cd build
 *   cmake .. && cmake --build . --config Release
 * 
 * 运行方法:
 *   inference.exe <image_path> [model_path]
 *   示例: inference.exe ../data/test_set/cats/cat.4001.jpg
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <codecvt>
#include <locale>

class CatDogClassifier {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // 模型参数
    int input_size_;
    std::vector<float> mean_;
    std::vector<float> std_;
    
public:
    CatDogClassifier(const std::string& model_path, int input_size = 224)
        : env_(ORT_LOGGING_LEVEL_WARNING, "CatDogClassifier"),
          input_size_(input_size),
          mean_{0.485f, 0.456f, 0.406f},
          std_{0.229f, 0.224f, 0.225f}
    {
        // 设置会话选项
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 启用 CUDA (如果可用)
        // OrtCUDAProviderOptions cuda_options;
        // session_options_.AppendExecutionProvider_CUDA(cuda_options);
        
        // 创建会话 - 将路径转换为宽字符
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring model_path_wide = converter.from_bytes(model_path);
        session_ = std::make_unique<Ort::Session>(env_, model_path_wide.c_str(), session_options_);
        
        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());
        }
        
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());
        }
        
        std::cout << "模型加载成功: " << model_path << std::endl;
        std::cout << "输入名称: " << input_names_[0] << std::endl;
        std::cout << "输出名称: " << output_names_[0] << std::endl;
    }
    
    /**
     * 图像预处理
     * 1. 调整大小
     * 2. BGR -> RGB
     * 3. 归一化
     * 4. HWC -> CHW
     */
    std::vector<float> preprocess(const cv::Mat& image) {
        // 调整大小
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_size_, input_size_));
        
        // BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // 转换为浮点型并归一化
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
        
        // 创建输出张量 (CHW格式)
        std::vector<float> input_tensor_values(3 * input_size_ * input_size_);
        
        // HWC -> CHW 并进行标准化
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < input_size_; h++) {
                for (int w = 0; w < input_size_; w++) {
                    float pixel = rgb.at<cv::Vec3f>(h, w)[c];
                    // 标准化: (pixel - mean) / std
                    input_tensor_values[c * input_size_ * input_size_ + h * input_size_ + w] 
                        = (pixel - mean_[c]) / std_[c];
                }
            }
        }
        
        return input_tensor_values;
    }
    
    /**
     * 执行推理
     * @param image 输入图像
     * @return {预测类别, 置信度} 0=cat, 1=dog
     */
    std::pair<int, float> predict(const cv::Mat& image) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 预处理
        std::vector<float> input_tensor_values = preprocess(image);
        
        // 创建输入张量
        std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_values.size(), 
            input_shape.data(), 
            input_shape.size()
        );
        
        // 执行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size()
        );
        
        // 获取输出
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = output_info.GetElementCount();
        
        // Softmax
        std::vector<float> probs(output_size);
        float max_logit = *std::max_element(output_data, output_data + output_size);
        float sum_exp = 0.0f;
        
        for (size_t i = 0; i < output_size; i++) {
            probs[i] = std::exp(output_data[i] - max_logit);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < output_size; i++) {
            probs[i] /= sum_exp;
        }
        
        // 获取预测类别
        auto max_it = std::max_element(probs.begin(), probs.end());
        int predicted_class = static_cast<int>(std::distance(probs.begin(), max_it));
        float confidence = probs[predicted_class];
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "推理时间: " << duration.count() << " ms" << std::endl;
        
        return {predicted_class, confidence};
    }
    
    /**
     * 获取类别名称
     */
    std::string getClassName(int class_id) const {
        return class_id == 0 ? "cat" : "dog";
    }
};

/**
 * 在图像上绘制结果
 */
void drawResult(cv::Mat& image, int predicted_class, float confidence) {
    std::string class_name = (predicted_class == 0) ? "Cat" : "Dog";
    std::string text = class_name + " (" + std::to_string(int(confidence * 100)) + "%)";
    
    // 设置颜色 (猫=蓝色, 狗=绿色)
    cv::Scalar color = (predicted_class == 0) ? cv::Scalar(255, 100, 0) : cv::Scalar(0, 200, 100);
    
    // 绘制背景矩形
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
    cv::Point text_org(10, 40);
    cv::rectangle(image, 
                  cv::Point(text_org.x - 5, text_org.y - text_size.height - 5),
                  cv::Point(text_org.x + text_size.width + 5, text_org.y + 5),
                  color, -1);
    
    // 绘制文字
    cv::putText(image, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                cv::Scalar(255, 255, 255), 2);
}

void printUsage(const char* program_name) {
    std::cout << "使用方法: " << program_name << " <image_path> [model_path]" << std::endl;
    std::cout << std::endl;
    std::cout << "参数说明:" << std::endl;
    std::cout << "  image_path  : 要预测的图像路径" << std::endl;
    std::cout << "  model_path  : ONNX模型路径 (可选，默认: checkpoints/resnet18/best_model.onnx)" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " data/test_set/cats/cat.4001.jpg" << std::endl;
    std::cout << "  " << program_name << " data/test_set/dogs/dog.4001.jpg checkpoints/cnn/best_model.onnx" << std::endl;
}

int main(int argc, char* argv[]) {
    // 检查参数
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string image_path = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "checkpoints/resnet18/best_model.onnx";
    
    // 加载图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "错误: 无法加载图像: " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "图像加载成功: " << image_path << std::endl;
    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    
    // 创建分类器
    try {
        CatDogClassifier classifier(model_path, 224);
        
        // 执行推理
        auto [predicted_class, confidence] = classifier.predict(image);
        
        // 输出结果
        std::cout << "\n预测结果:" << std::endl;
        std::cout << "  类别: " << classifier.getClassName(predicted_class) << std::endl;
        std::cout << "  置信度: " << std::fixed << std::setprecision(2) << (confidence * 100) << "%" << std::endl;
        
        // 创建结果图像
        cv::Mat result_image = image.clone();
        drawResult(result_image, predicted_class, confidence);
        
        // 显示结果
        cv::namedWindow("Cat/Dog Classification", cv::WINDOW_AUTOSIZE);
        cv::imshow("Cat/Dog Classification", result_image);
        
        std::cout << "\n按任意键退出..." << std::endl;
        cv::waitKey(0);
        
        // 保存结果
        std::string output_path = "prediction_result.jpg";
        cv::imwrite(output_path, result_image);
        std::cout << "结果已保存: " << output_path << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime错误: " << e.what() << std::endl;
        return -1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV错误: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
