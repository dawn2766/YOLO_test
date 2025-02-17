#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

// 网络配置结构体
struct Net_config
{
  float confThreshold; // 置信度阈值
  float nmsThreshold; // 非极大值抑制阈值
  int inpWidth; // 输入图像宽度
  int inpHeight; // 输入图像高度
  string classesFile; // 类别文件路径
  string modelConfiguration; // 模型配置文件路径
  string modelWeights; // 模型权重文件路径
  string netname; // 网络名称
};

// YOLO类定义
class YOLO
{
  public:
    YOLO(Net_config config); // 构造函数
    void detect(Mat& frame); // 检测函数
    // void checkBackend(); // 检查后端设备
  private:
    float confThreshold; // 置信度阈值
    float nmsThreshold; // 非极大值抑制阈值
    int inpWidth; // 输入图像宽度
    int inpHeight; // 输入图像高度
    char netname[20]; // 网络名称
    vector<string> classes; // 类别名称
    dnn::Net net; // 网络模型
    void postprocess(Mat& frame, const vector<Mat>& outs); // 后处理函数
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame); // 绘制预测框
};

// 网络配置数组
Net_config yolo_nets[4] = {
  {0.5, 0.4, 416, 416,"coco.names", "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3"},
  {0.5, 0.4, 608, 608,"coco.names", "yolov4/yolov4.cfg", "yolov4/yolov4.weights", "yolov4"},
  {0.5, 0.4, 320, 320,"coco.names", "yolo-fastest/yolo-fastest-xl.cfg", "yolo-fastest/yolo-fastest-xl.weights", "yolo-fastest"},
  {0.5, 0.4, 320, 320,"coco.names", "yolobile/csdarknet53s-panet-spp.cfg", "yolobile/yolobile.weights", "yolobile"}
};

// YOLO类构造函数实现
YOLO::YOLO(Net_config config)
{
  cout << "Net use " << config.netname << endl;
  this->confThreshold = config.confThreshold;
  this->nmsThreshold = config.nmsThreshold;
  this->inpWidth = config.inpWidth;
  this->inpHeight = config.inpHeight;
  strncpy(this->netname, config.netname.c_str(), sizeof(this->netname) - 1);
  this->netname[sizeof(this->netname) - 1] = '\0'; // 确保字符串以空字符结尾

  // 读取类别名称
  ifstream ifs(config.classesFile.c_str());
  string line;
  while (getline(ifs, line)) this->classes.push_back(line);

  // 读取网络模型
  this->net = cv::dnn::readNetFromDarknet(config.modelConfiguration, config.modelWeights);
  this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  // this->checkBackend(); // 检查后端设备
}

// 检查后端设备函数实现
// void YOLO::checkBackend()
// {
//   int backend = this->net.getPreferableBackend();
//   int target = this->net.getPreferableTarget();
//   string backendStr = (backend == cv::dnn::DNN_BACKEND_OPENCV) ? "OpenCV" : "Unknown";
//   string targetStr = (target == cv::dnn::DNN_TARGET_CPU) ? "CPU" : 
//                      (target == cv::dnn::DNN_TARGET_OPENCL) ? "OpenCL" : 
//                      (target == cv::dnn::DNN_TARGET_CUDA) ? "CUDA" : "Unknown";
//   cout << "Backend: " << backendStr << ", Target: " << targetStr << endl;
// }

// 检测函数实现
void YOLO::detect(Mat& frame)
{
  // this->checkBackend(); // 检查后端设备
  Mat blob;
  // 创建输入blob
  cv::dnn::blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
  this->net.setInput(blob);
  vector<Mat> outs;
  // 前向传播
  this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
  // 后处理
  this->postprocess(frame, outs);

  vector<double> layersTimes;
  double freq = getTickFrequency() / 1000;
  double t = net.getPerfProfile(layersTimes) / freq;
  string label = format("%s Inference time : %.2f ms", this->netname, t);
  putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
  //imwrite(format("%s_out.jpg", this->netname), frame);
}

// 后处理函数实现
void YOLO::postprocess(Mat& frame, const vector<Mat>& outs)   // 移除低置信度的边界框，使用非极大值抑制
{
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  // 遍历所有输出层
  for (size_t i = 0; i < outs.size(); ++i)
  {
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
    {
      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;
      // 获取最大得分的值和位置
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > this->confThreshold)
      {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(Rect(left, top, width, height));
      }
    }
  }

  // 执行非极大值抑制，消除冗余的重叠框
  vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i)
  {
    int idx = indices[i];
    Rect box = boxes[idx];
    this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
      box.x + box.width, box.y + box.height, frame);
  }
}

// 绘制预测框函数实现
void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // 绘制预测边界框
{
  // 绘制边界框
  rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

  // 获取类别名称和置信度的标签
  string label = format("%.2f", conf);
  if (!this->classes.empty())
  {
    CV_Assert(classId < (int)this->classes.size());
    label = this->classes[classId] + ":" + label;
  }

  // 在边界框顶部显示标签
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(top, labelSize.height);
  //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
  putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

int main()
{
  YOLO yolo_model(yolo_nets[1]);
  string imgpath = "test_image.jpg";
  Mat srcimg = imread(imgpath);
  yolo_model.detect(srcimg);

  static const string kWinName = "Deep learning object detection in OpenCV";
  namedWindow(kWinName, WINDOW_NORMAL);
  imshow(kWinName, srcimg);
  waitKey(0);
  destroyAllWindows();
}
