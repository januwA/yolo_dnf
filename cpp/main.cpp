#include <iostream>
#include <ranges>
#include <fstream>
#include <format>
#include <sstream>
#include <windows.h>
#include <filesystem>
#include <locale.h>
#include <codecvt>
#include <regex>
#include <string>
#include <tuple>
#include <random>
#include <span>
#include <vector>

#include "json.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using json = nlohmann::json;
using namespace cv;
using namespace cv::dnn;
using predict_result_t = std::vector<std::tuple<int, int, int, int, float, int>>;

// https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
std::map<std::string, int> vkmap{
    {"lb", VK_LBUTTON},    // 鼠标左键
    {"rb", VK_RBUTTON},    // 鼠标右键
    {"cancel", VK_CANCEL}, // 控制中断处理
    {"mb", VK_MBUTTON},    // 鼠标中键
    {"xb1", VK_XBUTTON1},  // X1 鼠标按钮
    {"xb2", VK_XBUTTON2},  // X2 鼠标按钮
    {"back", VK_BACK},     // BACKSPACE 键
    {"tab", VK_TAB},       // Tab 键
    {"clear", VK_CLEAR},   // CLEAR 键
    {"enter", VK_RETURN},  // Enter 键
    {"shift", VK_SHIFT},   // SHIFT 键
    {"ctrl", VK_CONTROL},  // CTRL 键
    {"alt", VK_MENU},      // Alt 键
    {"pause", VK_PAUSE},   // PAUSE 键
    {"cl", VK_CAPITAL},    // CAPS LOCK 键
    //     VK_KANA	0x15	IME Kana 模式
    // VK_HANGUL	0x15	IME Hanguel 模式
    // VK_IME_ON	0x16	IME 打开
    // VK_JUNJA	0x17	IME Junja 模式
    // VK_FINAL	0x18	IME 最终模式
    // VK_HANJA	0x19	IME Hanja 模式
    // VK_KANJI	0x19	IME Kanji 模式
    // VK_IME_OFF	0x1A	IME 关闭
    {"esc", VK_ESCAPE}, // ESC 键
    //     VK_CONVERT	0x1C	IME 转换
    // VK_NONCONVERT	0x1D	IME 不转换
    // VK_ACCEPT	0x1E	IME 接受
    // VK_MODECHANGE	0x1F	IME 模式更改请求
    {"space", VK_SPACE},         // 空格键
    {"pu", VK_PRIOR},            // PAGE UP 键
    {"pd", VK_NEXT},             // PAGE DOWN 键
    {"end", VK_END},             // END 键
    {"home", VK_HOME},           // HOME 键
    {"left", VK_LEFT},           // LEFT ARROW 键
    {"up", VK_UP},               // UP ARROW 键
    {"right", VK_RIGHT},         // RIGHT ARROW 键
    {"down", VK_DOWN},           // DOWN ARROW 键
    {"select", VK_SELECT},       // SELECT 键
    {"print", VK_PRINT},         // PRINT 键
    {"execute", VK_EXECUTE},     // EXECUTE 键
    {"ps", VK_SNAPSHOT},         // PRINT SCREEN 键
    {"insert", VK_INSERT},       // INS 键
    {"delete", VK_DELETE},       // DEL 键
    {"help", VK_HELP},           // HELP 键
    {"lwin", VK_LWIN},           // 左 Windows 键
    {"rwin", VK_RWIN},           // 右侧 Windows 键
    {"apps", VK_APPS},           // 应用程序密钥
    {"sleep", VK_SLEEP},         // 计算机休眠键
    {"*", VK_MULTIPLY},          // 乘号键
    {"+", VK_ADD},               // 加号键
    {"separator", VK_SEPARATOR}, // 分隔符键
    {"-", VK_SUBTRACT},          // 减号键
    {".", VK_DECIMAL},           // 句点键
    {"/", VK_DIVIDE},            // 除号键
    {"numlock", VK_NUMLOCK},     // NUM LOCK 键
    {"scroll", VK_SCROLL},       // SCROLL LOCK 键
    {"lshift", VK_LSHIFT},       // 左 SHIFT 键
    {"rshift", VK_RSHIFT},       // 右 SHIFT 键
    {"lctrl", VK_LCONTROL},      // 左 Ctrl 键
    {"rctrl", VK_RCONTROL},      // 右 Ctrl 键
    {"lalt", VK_LMENU},          // 左 ALT 键
    {"ralt", VK_RMENU},          // 右 ALT 键

    //     VK_BROWSER_BACK	0xA6	浏览器后退键
    // VK_BROWSER_FORWARD	0xA7	浏览器前进键
    // VK_BROWSER_REFRESH	0xA8	浏览器刷新键
    // VK_BROWSER_STOP	0xA9	浏览器停止键
    // VK_BROWSER_SEARCH	0xAA	浏览器搜索键
    // VK_BROWSER_FAVORITES	0xAB	浏览器收藏键
    // VK_BROWSER_HOME	0xAC	浏览器“开始”和“主页”键
    // VK_VOLUME_MUTE	0xAD	静音键
    // VK_VOLUME_DOWN	0xAE	音量减小键
    // VK_VOLUME_UP	0xAF	音量增加键
    // VK_MEDIA_NEXT_TRACK	0xB0	下一曲目键
    // VK_MEDIA_PREV_TRACK	0xB1	上一曲目键
    // VK_MEDIA_STOP	0xB2	停止媒体键
    // VK_MEDIA_PLAY_PAUSE	0xB3	播放/暂停媒体键
    // VK_LAUNCH_MAIL	0xB4	启动邮件键
    // VK_LAUNCH_MEDIA_SELECT	0xB5	选择媒体键
    // VK_LAUNCH_APP1	0xB6	启动应用程序 1 键
    // VK_LAUNCH_APP2	0xB7	启动应用程序 2 键

    // VK_ATTN	0xF6	Attn 键
    // VK_CRSEL	0xF7	CrSel 键
    // VK_EXSEL	0xF8	ExSel 键
    // VK_EREOF	0xF9	Erase EOF 键
    // VK_PLAY	0xFA	Play 键
    // VK_ZOOM	0xFB	Zoom 键
    // VK_NONAME	0xFC	预留
    // VK_PA1	0xFD	PA1 键
    // VK_OEM_CLEAR	0xFE	Clear 键
};

static Net model;
std::vector<std::string_view> CLASSES = {"玩家", "fubenditu", "cailiao", "men", "diren", "jiangli"};

#define LOG1(v) \
    std::cout << v << "\n"

template <typename... Args>
void log2(Args &&...args)
{
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
}

static std::vector<cv::Scalar> randomcolor(std::span<std::string_view> from)
{
    std::random_device rd;                          // 获取一个随机数引擎
    std::mt19937 gen(rd());                         // 初始化一个 Mersenne Twister 引擎
    std::uniform_int_distribution<int> dis(0, 255); // 定义一个均匀分布

    std::vector<cv::Scalar> colors(from.size());
    for (size_t i = 0; i < from.size(); i++)
    {
        // BGR model
        colors[i] = cv::Scalar(dis(gen), dis(gen), dis(gen));
    }
    return colors;
}

static Rect points2rect(int x1, int y1, int x2, int y2)
{
    return Rect{x1, y1, x2 - x1, y2 - y1};
}

static std::wstring towstring(std::string_view str)
{
    // std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    // return converter.from_bytes(str.data());
    std::wstring wideStr;
    size_t size = MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.length()), NULL, NULL);
    if (size == NULL)
        return wideStr;

    wideStr.resize(size);
    MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.length()), (LPWSTR)wideStr.data(), static_cast<int>(wideStr.length()));
    return wideStr;
}

static bool is_chinese(std::string_view str)
{
    static std::regex chinese_regex("[\u4e00-\u9fa5]");
    return std::regex_search(str.data(), chinese_regex);
}

static json loadjson5file(std::string_view p)
{
    std::ifstream ifs;

    if (is_chinese(p))
    {
        ifs.open(towstring(p));
    }
    else
    {
        ifs.open(p);
    }
    if (!ifs.is_open())
    {
        std::cerr << "无法打开文件！" << std::endl;
        exit(1);
    }
    size_t i = 0;
    while (true)
    {
        if (ifs.get() == '{')
        {
            break;
        }
        if (i >= 100)
        {
            i = 0;
            break;
        }
        i++;
    }
    ifs.seekg(i, std::ios::beg);
    try
    {
        constexpr auto allow_exceptions = true;
        constexpr auto ignore_comments = true;
        auto d = nlohmann::json::parse(ifs, nullptr, allow_exceptions, ignore_comments);
        ifs.close();
        return d;
    }
    catch (json::parse_error &e)
    {
        std::cerr << e.what();
        exit(1);
    }
    // std::stringstream buffer;
    // buffer << ifs.rdbuf();
    // ifs.close();
    // return buffer.str();
}

static Mat imread2(std::string_view p)
{
    if (is_chinese(p))
    {
        std::ifstream ifs(towstring(p), std::ios::binary);
        if (!ifs.is_open())
        {
            std::cerr << "无法打开文件: " << std::endl;
            exit(1);
        }
        std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
        Mat img = imdecode(Mat(buffer), IMREAD_COLOR);
        ifs.close();
        return img;
    }
    else
    {
        return imread(p.data());
    }
}

static predict_result_t predict(Mat &original_image, double conf = 0.8, Size size = Size(640, 640))
{
    auto height = original_image.rows;
    auto width = original_image.cols;
    auto length = std::max(height, width);
    // log2("length", length);

    float scale = length / 640.0f;
    // log2("scale", scale);

    Mat image = Mat::zeros(length, length, CV_8UC3); // 创建一个 3 通道 uint8 型的零值图像
    // 将原图像嵌入方形图像
    original_image.copyTo(image(Rect(0, 0, width, height)));

    // imshow("original_image", original_image);
    // imshow("image", image);

    // 预处理图像
    Mat blob = blobFromImage(image, 1.0 / 255.0, size, Scalar(), true, false);
    model.setInput(blob);

    // 执行推理
    std::vector<cv::Mat> outputs;
    model.forward(outputs, model.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];
    // LOG1(std::format("rows:{}, step:{}", rows, dimensions));

    // 1 x 10 x 8400
    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    // 进程检测（可选，使用 NMS 进行过滤）
    std::vector<cv::Rect> boxes; // 矩形
    std::vector<float> scores;   // 得分
    std::vector<int> class_ids;  // labels

    auto modelScoreThreshold = conf;
    auto modelNMSThreshold = 0.5f;

    // 迭代输出以收集边界框、置信度分数和类别 ID
    float *data = (float *)outputs[0].data;
    // log2("size:", outputs[0].size);

    for (size_t i = 0; i < rows; i++)
    {
        float *classes_scores = data + 4;
        // cv::Mat scores(1, CLASSES.size(), CV_32FC1, classes_scores);
        cv::Point maxClassIndex;
        double maxScore;
        minMaxLoc(Mat(1, static_cast<int>(CLASSES.size()), CV_32FC1, classes_scores), 0, &maxScore, 0, &maxClassIndex);

        if (maxScore >= modelScoreThreshold)
        {
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int x1 = int((x - 0.5 * w) * scale);
            int y1 = int((y - 0.5 * h) * scale);

            // to python yolov8 out format
            int x2 = x1 + int(w * scale);
            int y2 = y1 + int(h * scale);

            // int x2 = int(w * scale);
            // int y2 = int(h * scale);

            boxes.push_back(Rect(x1, y1, x2, y2));
            scores.push_back(static_cast<float>(maxScore));
            class_ids.push_back(maxClassIndex.x);
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(modelScoreThreshold), modelNMSThreshold, nms_result);

    //[x1, y1, x2, y2, 概率, 分类坐标]
    std::vector<std::tuple<int, int, int, int, float, int>> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        auto class_id = class_ids[idx];
        auto confidence = scores[idx];
        Rect box = boxes[idx];
        auto className = CLASSES[class_id];
        auto x1 = box.x;
        auto y1 = box.y;
        auto x2 = box.width;
        auto y2 = box.height;

        //[x1, y1, x2, y2, 概率, 分类坐标]
        detections.push_back(std::make_tuple(x1, y1, x2, y2, confidence, class_id));
    }
    return detections;
}

static void spot(Mat &original_image, predict_result_t &detections)
{
    auto colors = randomcolor(CLASSES);
    for (auto &e : detections)
    {
        auto &[x1, y1, x2, y2, conf, id] = e;
        // log2(x1, y1, x2, y2, conf, id, CLASSES[id], id);
        auto &color = colors[id];
        cv::rectangle(original_image, points2rect(x1, y1, x2, y2), color, 2);
        cv::putText(original_image, std::format("{} {:.2f}", id, conf), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, .5, color, 2);
    }
    imshow("Image", original_image);
    waitKey(0);
}

int main(int argc, char **argv)
{
    SetConsoleOutputCP(CP_UTF8);
    // 0-9
    for (int i = 0; i <= 9; i++)
    {
        vkmap[std::to_string(i)] = 0x30 + i;
    }
    // a-z
    for (char i = 0x41; i <= 0x5A; i++)
    {
        vkmap[std::string(1, std::tolower(i))] = i;
    }
    // 数字键盘 0-9
    for (int i = 0; i <= 9; i++)
    {
        vkmap["num" + std::to_string(i)] = 0x60 + i;
    }
    // f1-f24
    for (int i = 0; i < 24; i++)
    {
        vkmap["f" + std::to_string(i + 1)] = 0x70 + i;
    }

    // for (auto it = vkmap.begin(); it != vkmap.end(); ++it)
    // {
    //     log2(std::format("{} {:#02X}", it->first, it->second));
    // }

    if (argc <= 1)
    {
        LOG1("启动参数不够");
        return 1;
    }
    // 加载配置文件
    auto configData = loadjson5file(argv[1]);
    std::filesystem::path 资源目录 = configData["资源目录"];
    auto model_path = 资源目录 / std::filesystem::path("best.onnx");
    if (configData["技能表"].is_string())
    {
        auto skillData = loadjson5file(configData["技能表"]);
    }

    model = readNetFromONNX(model_path.string());
    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    while (!GetAsyncKeyState(vkmap["delete"]))
    {
        LOG1("按下delete, 退出循环");
        Sleep(1000);
    }

    // 加载图片
    Mat img = imread2(R"(C:\Users\16418\Desktop\FenBaoYouChen\segment_merge4\1.png)");

    //[x1, y1, x2, y2, 概率, 分类坐标]
    auto detections = predict(img, 0.8);
    spot(img, detections);
    auto colors = randomcolor(CLASSES);

    destroyAllWindows();
    return 0;
}
