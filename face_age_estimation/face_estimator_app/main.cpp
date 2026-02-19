#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>

using namespace std;

void detectFace(cv::CascadeClassifier &faceClassifier, cv::dnn::Net &faceNet, cv::Mat &image, cv::Mat &imageGray);

string estimate(cv::dnn::Net &faceNet, const cv::Mat &image);

void drawFPS(double const fps, cv::Mat &image);

int main()
{
    auto capture = cv::VideoCapture(0);
    if (!capture.isOpened())
    {
        cerr << "Capture Didn't Open" << endl;
        return -1;
    }
    cout << "Capture Opened: " << capture.getBackendName() << endl;

    cout << "OpenCV Version " << cv::getVersionString() << endl;

    cv::CascadeClassifier faceClassifier;
    if (!faceClassifier.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"))
    {
        cerr << "Face Classifier not loaded" << endl;
        return -1;
    }
    cv::dnn::Net faceNet = cv::dnn::readNetFromONNX("face_estimator.onnx");
    if (faceNet.empty())
    {
        cerr << "ERROR: Could not load ONNX model. Check if face_model.onnx is in the same folder!" << endl;
        return -1;
    }
    faceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    faceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cout << "Face Classifier(s) Loaded!" << endl;

    cv::Mat frame, frameGray;
    using clock = chrono::high_resolution_clock;
    auto lastTime = clock::now();
    double fps = 0.0;

    const int flags = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO;
    cv::namedWindow("Face Detection", flags);
    cv::namedWindow("Face Detection Grayscale", flags);

    while (capture.isOpened())
    {
        if (!capture.read(frame))
        {
            cerr << "End of video, read error" << endl;
            break;
        }

        const auto currentTime = clock::now();
        auto elapsed = chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;
        fps = 1.0 / elapsed;

        // detect face
        detectFace(faceClassifier, faceNet, frame, frameGray);

        // draw fps
        drawFPS(fps, frame);

        cv::imshow("Face Detection", frame);
        cv::imshow("Face Detection Grayscale", frameGray);

        auto k = cv::waitKey(1);

        if (k == 27 || k == 'q')
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}

void drawFPS(double const fps, cv::Mat &image)
{
    const cv::String message = cv::format("FPS: %.1f", fps);
    cv::putText(image, message, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

string estimate(cv::dnn::Net &faceNet, const cv::Mat &image)
{
    if (image.empty())
        return "";

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    // Scale to [0, 1]
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // PyTorch Constants for ImageNet
    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std(0.229, 0.224, 0.225);

    // (image - mean) / std
    cv::Mat normalized = (rgb - mean) / std;

    // Convert HWC (Height, Width, Channel) to CHW (Channel, Height, Width) for the DNN
    auto processed_image = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false);

    faceNet.setInput(processed_image);

    vector<cv::Mat> outputs;
    faceNet.forward(outputs, faceNet.getUnconnectedOutLayersNames());

    if (outputs.size() >= 2)
    {
        const float age = outputs[0].at<float>(0, 0);
        const float genderProb = outputs[1].at<float>(0, 0);
        const string gender = (genderProb > 0.5) ? "Female" : "Male";

        return cv::format("%s, %.0f", gender.c_str(), age);
    }
    return "";
}

void detectFace(cv::CascadeClassifier &faceClassifier, cv::dnn::Net &faceNet, cv::Mat &image, cv::Mat &imageGray)
{
    cv::Mat faceImageGray;
    cv::cvtColor(image, faceImageGray, cv::COLOR_BGR2GRAY);
    const double scaleDown = 0.5;
    cv::resize(faceImageGray, faceImageGray, cv::Size(), scaleDown, scaleDown, cv::INTER_LINEAR);
    cv::equalizeHist(faceImageGray, faceImageGray);

    vector<cv::Rect> faces;
    faceClassifier.detectMultiScale(faceImageGray, faces, 1.2, 5, 0, cv::Size(30, 30));

    for (const auto &face : faces)
    {
        // Scale face coordinates back to original image size
        cv::Rect scaledFace = cv::Rect(
            face.x / scaleDown, face.y / scaleDown,
            face.width / scaleDown, face.height / scaleDown);

        // draw rect on face
        rectangle(image, scaledFace, cv::Scalar(255, 255, 255), 2);

        scaledFace &= cv::Rect(0, 0, image.cols, image.rows);
        const cv::Mat faceCrop = image(scaledFace);

        // estimation
        const string result = estimate(faceNet, faceCrop);

        cv::putText(image, result, cv::Point(scaledFace.x, scaledFace.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    faceImageGray.copyTo(imageGray);
}
