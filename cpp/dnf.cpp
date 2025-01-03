#include "dnf.h"

GameStatus gs;
SI si;
SkillBar skill_bar;
/**
 * 虚拟密钥代码
 * https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
 */

map<wstring, int> vk_map{
    {L"cancel", VK_CANCEL},  // 控制中断处理
    {L"back", VK_BACK},      // BACKSPACE 键
    {L"tab", VK_TAB},        // Tab 键
    {L"clear", VK_CLEAR},    // CLEAR 键
    {L"enter", VK_RETURN},   // Enter 键
    {L"shift", VK_SHIFT},    // SHIFT 键
    {L"ctrl", VK_CONTROL},   // CTRL 键
    {L"alt", VK_MENU},       // Alt 键
    {L"cl", VK_CAPITAL},     // CAPS LOCK 键
    {L"esc", VK_ESCAPE},     // ESC 键
    {L"pu", VK_PRIOR},       // PAGE UP 键
    {L"pd", VK_NEXT},        // PAGE DOWN 键
    {L"end", VK_END},        // END 键
    {L"home", VK_HOME},      // HOME 键

    {L"left", VK_LEFT},    // LEFT ARROW 键
    {L"up", VK_UP},        // UP ARROW 键
    {L"right", VK_RIGHT},  // RIGHT ARROW 键
    {L"down", VK_DOWN},    // DOWN ARROW 键
    {L"space", VK_SPACE},  // 空格键

    {L"左", VK_LEFT},   // LEFT ARROW 键
    {L"上", VK_UP},     // UP ARROW 键
    {L"右", VK_RIGHT},  // RIGHT ARROW 键
    {L"下", VK_DOWN},   // DOWN ARROW 键
    {L"跳", VK_SPACE},  // 空格键
    {L"空", VK_SPACE},  // 空格键

    {L"ps", VK_SNAPSHOT},      // PRINT SCREEN 键
    {L"insert", VK_INSERT},    // INS 键
    {L"delete", VK_DELETE},    // DEL 键
    {L"help", VK_HELP},        // HELP 键
    {L"lwin", VK_LWIN},        // 左 Windows 键
    {L"rwin", VK_RWIN},        // 右侧 Windows 键
    {L"apps", VK_APPS},        // 应用程序密钥
    {L"sleep", VK_SLEEP},      // 计算机休眠键
    {L"mul", VK_MULTIPLY},     // 乘号键
    {L"add", VK_ADD},          // 加号键
    {L"sub", VK_SUBTRACT},     // 减号键
    {L"decimal", VK_DECIMAL},  // 句点键
    {L"div", VK_DIVIDE},       // 除号键
    {L"numlock", VK_NUMLOCK},  // NUM LOCK 键
    {L"scroll", VK_SCROLL},    // SCROLL LOCK 键
    {L"lshift", VK_LSHIFT},    // 左 SHIFT 键
    {L"rshift", VK_RSHIFT},    // 右 SHIFT 键
    {L"lctrl", VK_LCONTROL},   // 左 Ctrl 键
    {L"rctrl", VK_RCONTROL},   // 右 Ctrl 键
    {L"lalt", VK_LMENU},       // 左 ALT 键
    {L"ralt", VK_RMENU},       // 右 ALT 键

    {L"[", VK_OEM_4},       // 对于美国标准键盘，键[{
    {L"]", VK_OEM_6},       // 对于美国标准键盘，键]}
    {L"\\", VK_OEM_5},      // 对于美国标准键盘，键\|
    {L"'", VK_OEM_7},       // 对于美国标准键盘，键'"
    {L"`", VK_OEM_3},       // 对于美国标准键盘，键`~
    {L"/", VK_OEM_2},       // 对于美国标准键盘，键/?
    {L";", VK_OEM_1},       // 对于美国标准键盘，键;:
    {L"=", VK_OEM_PLUS},    // 对于任何国家/地区，键+
    {L",", VK_OEM_COMMA},   // 对于任何国家/地区，键,
    {L"-", VK_OEM_MINUS},   // 对于任何国家/地区，键-
    {L".", VK_OEM_PERIOD},  // 对于任何国家/地区，键.

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
};

map<wstring, int> vsc_map;

cv::dnn::Net model;
vector<wstring> CLASSES{L"玩家", L"副本地图", L"材料", L"门", L"敌人", L"奖励"};
;
std::vector<cv::Scalar> colors;

wstring get_ssr_pos_zh(tuple_xy point) {
  auto [x, y] = point;
  wstring pos_zh;
  auto [x1, y1, x2, y2] = gs.ss_rect;

  if (x <= x1) pos_zh += L"左";
  if (x >= x2) pos_zh += L"右";
  if (y <= y1) pos_zh += L"上";
  if (y >= y2) pos_zh += L"下";
  return pos_zh;
}
tuple_xy eval_tuple(wstring str) {
  auto p1 = str.find(L'(');
  auto p2 = str.find(L',');
  auto p3 = str.find(L')');
  auto a = str.substr(p1 + 1, p2 - p1 - 1);
  auto b = str.substr(p2 + 1, p3 - p2 - 1);
  return make_tuple(stoi(a.data()), stoi(b.data()));
}
void sleep(double seconds) {
  if (seconds <= 0) return;
  Sleep(static_cast<int>(seconds * 1000));
}
tuple<wstring, tuple_xy> get_smap_move_zh_pos(tuple_xy current_node,
                                              tuple_xy next_node) {
  auto &[m1, c1] = current_node;
  auto &[m2, c2] = next_node;

  auto [x1, y1, x2, y2] = gs.ss_rect;
  auto sw = gs.gw_info.w;
  auto sh = gs.gw_info.h;

  auto xc = int((x1 + x2) / 2);
  auto yc = int((y1 + y2) / 2);

  xc = random_number(xc - 50, xc + 50);
  yc = random_number(yc - 50, yc + 50);

  if (m2 > m1) {
    auto pad = random_number(int(sw / 2), sw);
    return make_tuple(L"下", make_tuple(xc, y2 + pad));
  } else if (m2 < m1) {
    auto pad = random_number(int(sw / 2), sw);
    return make_tuple(L"上", make_tuple(xc, y1 + pad));
  } else if (c2 > c1) {
    auto pad = random_number(int(sh / 2), sh);
    return make_tuple(L"右", make_tuple(x2 + pad, yc));
  } else {
    auto pad = random_number(int(sh / 2), sh);
    return make_tuple(L"左", make_tuple(x1 - pad, yc));
  }
}

cv::Mat CaptureAnImage(HWND hWnd, bool &err) {
  HDC hdcWindow;
  HDC hdcMemDC = NULL;
  HBITMAP hbmScreen = NULL;

  cv::Mat img;
  int left, top, right, bottom, w, h;
  int scale = 2;  // TODO: 定义动态的缩放

  // 检索窗口客户区的显示设备上下文的句柄。7
  hdcWindow = GetDC(hWnd);

  int real_w = GetDeviceCaps(hdcWindow, DESKTOPHORZRES);
  int apparent_w = GetSystemMetrics(0);
  int d_scale = int(real_w / apparent_w);
  // log1(format("d_scale: {}", d_scale));

  // 创建一个兼容的DC，该DC可用于来自窗口DC的BitBlt。
  hdcMemDC = CreateCompatibleDC(hdcWindow);

  if (!hdcMemDC) {
    println("CreateCompatibleDC has failed");
    err = true;
    goto done;
  }

  // 获取客户区域以进行尺寸计算。
  RECT rect;
  GetWindowRect(hWnd, &rect);
  left = rect.left * scale;
  top = rect.top * scale;
  right = rect.right * scale;
  bottom = rect.bottom * scale;
  w = right - left;
  h = bottom - top;
  // log1(format("left={} top={} right={} bottom={} w={} h={}", left, top,
  // right, bottom, w, h));

  // 这是最好的拉伸模式。
  // SetStretchBltMode(hdcWindow, HALFTONE);

  // 从 Window DC 创建兼容的位图。
  hbmScreen = CreateCompatibleBitmap(hdcWindow, w, h);

  if (!hbmScreen) {
    println("CreateCompatibleBitmap failed");
    err = true;
    goto done;
  }

  // 将兼容的位图选择到兼容的内存DC中。
  SelectObject(hdcMemDC, hbmScreen);

  // 位块传输到我们兼容的内存DC中。
  if (!BitBlt(hdcMemDC, 0, 0, w, h, hdcWindow, 0, 0, SRCCOPY)) {
    println("BitBlt has failed");
    err = true;
    goto done;
  }

  BITMAPINFOHEADER bi;
  bi.biSize = sizeof(BITMAPINFOHEADER);
  bi.biWidth = w;
  bi.biHeight = -h;  // 对于位图图像，高度为负表示自顶向下
  bi.biPlanes = 1;
  bi.biBitCount = 24;  // 32 -> CV_8UC4, 24 -> CV_8UC3
  bi.biCompression = BI_RGB;
  bi.biSizeImage = 0;
  bi.biXPelsPerMeter = 0;
  bi.biYPelsPerMeter = 0;
  bi.biClrUsed = 0;
  bi.biClrImportant = 0;

  // 从位图中获取“位”，并将它们复制到指向的缓冲区中。
  // https://stackoverflow.com/questions/57200003/how-to-turn-a-screenshot-bitmap-to-a-cvmat
  img = cv::Mat(h, w, CV_8UC3);
  GetDIBits(hdcWindow, hbmScreen, 0, (UINT)h, img.data, (BITMAPINFO *)&bi,
            DIB_RGB_COLORS);

  // imshow("1", img);
  // waitKey(0);

  // Clean up.
done:
  DeleteObject(hbmScreen);
  DeleteObject(hdcMemDC);
  // ReleaseDC(NULL, hdcScreen);
  ReleaseDC(NULL, hdcWindow);

  return img;
}

void predict_game() {
  auto title = "1";
  // cv::namedWindow(title, cv::WINDOW_NORMAL);
  // cv::resizeWindow(title, 800, 600);

  while (true) {
    if (GetAsyncKeyState(vk_map[L"delete"]) & KS_NOW_DOWN) break;
    bool err = false;
    auto game_img = CaptureAnImage(gs.hwnd, err);
    if (err) {
      sleep(1);
      continue;
    }

    auto detections = predict(game_img, 0.8);
    map<wstring, vector<predict_result_t>> box_map;
    for (auto &el : detections) {
      auto name = CLASSES[get<5>(el)];
      box_map[name].push_back(el);
    }
    // println("{}", box_map.size());

    // spot(game_img, detections);
    // imshow(title, game_img);
    // cv::setWindowProperty(title, cv::WND_PROP_TOPMOST, 1);
    // cv::waitKey(1);

    sleep(0.01);
  }

  cv::destroyAllWindows();
}

int bootstrap(int argc, char *argv[]) {
  if (argc <= 1) {
    printf("参数不够\n");
    return 1;
  }
  colors = randomcolor(CLASSES);
  init_vkmap();
  si = SI();
  gs.init(strToWstr(argv[1]));
  gs.load_image_temps();
  gs.change_player();

  auto model_path = gs.src_dir / "best.onnx";
  model = cv::dnn::readNetFromONNX(model_path.string());
  model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  while (true) {
    if (GetAsyncKeyState(vk_map[L"delete"]) & KS_BEFORE_DOWN) break;

    println("选中游戏窗口，按delete启动");
    sleep(1);
  }

  gs.hwnd = GetForegroundWindow();
  auto title = getWindowCaption(gs.hwnd);
  if (title.find(L"地下城与勇士") != wstring::npos) {
    predict_game();
  }
  si.keyUpMany();
  return 0;
}

void init_vkmap() {
  // 0-9
  for (int i = 0; i <= 9; i++) {
    vk_map[to_wstring(i)] = 0x30 + i;
  }
  // a-z
  for (char i = 0x41; i <= 0x5A; i++) {
    vk_map[wstring(1, tolower(i))] = i;
  }
  // 数字键盘 0-9
  for (int i = 0; i <= 9; i++) {
    vk_map[L"num" + to_wstring(i)] = 0x60 + i;
  }
  // f1-f24
  for (int i = 0; i < 24; i++) {
    vk_map[L"f" + to_wstring(i + 1)] = 0x70 + i;
  }

  // 虚拟密钥代码转换为扫描代码
  // 如果没有转换，则返回值为零。
  for (auto it = vk_map.begin(); it != vk_map.end(); ++it) {
    // log2(format("{} {:#02X}", it->first, it->second));
    auto code = MapVirtualKeyA(it->second, MAPVK_VK_TO_VSC);
    if (code) {
      vsc_map[it->first] = code;
      // log2(format("{} {:#04X}->{:#04x}", it->first, it->second, code));
    } else {
      // log2(format("MapVirtualKeyA err: {} {:#04X}", it->first, it->second));
    }
  }
}

void spot(cv::Mat &original_image, vector<predict_result_t> &detections) {
  for (auto &e : detections) {
    auto &[x1, y1, x2, y2, conf, id] = e;
    // log2(x1, y1, x2, y2, conf, id, CLASSES[id], id);
    auto &color = colors[id];
    cv::rectangle(original_image, points2rect(x1, y1, x2, y2), color, 2);
    cv::putText(original_image, format("{} {:.2f}", id, conf),
                cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, .5, color, 2);
  }
  // imshow("Image", original_image);
  // waitKey(0);
}

vector<cv::Scalar> randomcolor(span<wstring> from) {
  random_device rd;   // 获取一个随机数引擎
  mt19937 gen(rd());  // 初始化一个 Mersenne Twister 引擎
  uniform_int_distribution<int> dis(0, 255);  // 定义一个均匀分布

  vector<cv::Scalar> colors(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    // BGR model
    colors[i] = cv::Scalar(dis(gen), dis(gen), dis(gen));
  }
  return colors;
}

cv::Rect points2rect(int x1, int y1, int x2, int y2) {
  return cv::Rect{x1, y1, x2 - x1, y2 - y1};
}

static wstring strToWstr(string_view str) {
  // wstring_convert<codecvt_utf8<wchar_t>> converter;
  // return converter.from_bytes(str.data());
  wstring wideStr;
  size_t size = MultiByteToWideChar(CP_UTF8, 0, str.data(),
                                    static_cast<int>(str.length()), NULL, NULL);
  if (size == NULL) return wideStr;

  wideStr.resize(size);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.length()),
                      (LPWSTR)wideStr.data(),
                      static_cast<int>(wideStr.length()));
  return wideStr;
}

u16string strToU16(string_view str) {
  u16string wideStr;
  size_t size = MultiByteToWideChar(CP_UTF8, 0, str.data(),
                                    static_cast<int>(str.length()), NULL, NULL);
  if (size == NULL) return wideStr;

  wideStr.resize(size);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.length()),
                      (LPWSTR)wideStr.data(),
                      static_cast<int>(wideStr.length()));
  return wideStr;
}

string u16ToStr(u16string_view ustr) {
  string str;
  size_t size =
      WideCharToMultiByte(CP_UTF8, 0, (LPCWCH)ustr.data(),
                          static_cast<int>(ustr.length()), NULL, NULL, 0, 0);
  if (size == NULL) return str;

  str.resize(size);
  WideCharToMultiByte(CP_UTF8, 0, (LPCWCH)ustr.data(),
                      static_cast<int>(ustr.length()), (LPSTR)str.data(),
                      static_cast<int>(str.length()), 0, 0);
  return str;
}

string wstrToStr(wstring_view wstr) {
  string str;
  size_t size =
      WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                          static_cast<int>(wstr.length()), NULL, NULL, 0, 0);
  if (size == NULL) return str;

  str.resize(size);
  WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.length()),
                      (LPSTR)str.data(), static_cast<int>(str.length()), 0, 0);
  return str;
}

wstring getWindowCaption(HWND hwnd) {
  wstring title;
  title.resize(MAX_CLASS_NAME);
  auto len = GetWindowTextW(hwnd, (LPWSTR)title.data(), MAX_CLASS_NAME);
  if (!len) {
    println("空标题!!");
    return L"";
  }
  return title;
}

static bool is_chinese(wstring str) {
  static wregex chinese_regex(L"[\u4e00-\u9fa5]");
  return regex_search(str.data(), chinese_regex);
}

static json loadjson5file(wstring p) {
  ifstream ifs(p.data());
  if (!ifs.is_open()) {
    cerr << "无法打开文件！" << endl;
    exit(1);
  }
  size_t i = 0;
  while (true) {
    if (ifs.get() == '{') {
      break;
    }
    if (i >= 100) {
      i = 0;
      break;
    }
    i++;
  }
  ifs.seekg(i, ios::beg);
  try {
    constexpr auto allow_exceptions = true;
    constexpr auto ignore_comments = true;
    auto d =
        nlohmann::json::parse(ifs, nullptr, allow_exceptions, ignore_comments);
    ifs.close();
    return d;
  } catch (json::parse_error &e) {
    cerr << e.what();
    exit(1);
  }
  // stringstream buffer;
  // buffer << ifs.rdbuf();
  // ifs.close();
  // return buffer.str();
}

static cv::Mat imread2(wstring p) {
  if (is_chinese(p)) {
    ifstream ifs(p.data(), ios::binary);
    if (!ifs.is_open()) {
      cerr << "无法打开文件: " << endl;
      exit(1);
    }
    vector<char> buffer((istreambuf_iterator<char>(ifs)),
                        istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(cv::Mat(buffer), cv::IMREAD_COLOR);
    ifs.close();
    return img;
  } else {
    return cv::imread(wstrToStr(p));
  }
}

static vector<predict_result_t> predict(cv::Mat &original_image, double conf,
                                        cv::Size size) {
  auto width = original_image.cols;
  auto height = original_image.rows;
  // auto length = max(height, width);
  double x_scale = width / static_cast<double>(size.width);
  double y_scale = height / static_cast<double>(size.height);
  // log2(x_scale, y_scale);

  // Mat image = Mat::zeros(length, length, original_image.type());
  // original_image.copyTo(image(Rect(0, 0, width, height)));

  // imshow("original_image", original_image);
  // imshow("image", image);

  // 预处理图像
  cv::Mat blob = cv::dnn::blobFromImage(original_image, 1.0 / 255.0, size,
                                        cv::Scalar(), true, false);
  model.setInput(blob);

  // 执行推理
  vector<cv::Mat> outputs;
  model.forward(outputs, model.getUnconnectedOutLayersNames());

  int rows = outputs[0].size[2];
  int dimensions = outputs[0].size[1];
  // log1(format("rows:{}, step:{}", rows, dimensions));

  // 1 x 10 x 8400
  outputs[0] = outputs[0].reshape(1, dimensions);
  cv::transpose(outputs[0], outputs[0]);

  // 进程检测（可选，使用 NMS 进行过滤）
  vector<cv::Rect> boxes;  // 矩形
  vector<float> scores;    // 得分
  vector<int> class_ids;   // labels

  auto modelScoreThreshold = conf;
  auto modelNMSThreshold = 0.5f;

  // 迭代输出以收集边界框、置信度分数和类别 ID
  float *data = (float *)outputs[0].data;
  // log2("size:", outputs[0].size);

  for (size_t i = 0; i < rows; i++) {
    float *classes_scores = data + 4;
    // cv::Mat scores(1, CLASSES.size(), CV_32FC1, classes_scores);
    cv::Point maxClassIndex;
    double maxScore;
    minMaxLoc(
        cv::Mat(1, static_cast<int>(CLASSES.size()), CV_32FC1, classes_scores),
        0, &maxScore, 0, &maxClassIndex);

    if (maxScore >= modelScoreThreshold) {
      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      int x1 = int((x - 0.5 * w) * x_scale);
      int y1 = int((y - 0.5 * h) * y_scale);

      // to python yolov8 out format
      int x2 = x1 + int(w * x_scale);
      int y2 = y1 + int(h * y_scale);

      // int x2 = int(w * scale);
      // int y2 = int(h * scale);

      boxes.push_back(cv::Rect(x1, y1, x2, y2));
      scores.push_back(static_cast<float>(maxScore));
      class_ids.push_back(maxClassIndex.x);
    }

    data += dimensions;
  }

  vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(modelScoreThreshold),
                    modelNMSThreshold, nms_result);

  //[x1, y1, x2, y2, 概率, 分类坐标]
  vector<tuple<int, int, int, int, float, int>> detections{};
  for (unsigned long i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    auto class_id = class_ids[idx];
    auto confidence = scores[idx];
    cv::Rect box = boxes[idx];
    auto className = CLASSES[class_id];
    auto x1 = box.x;
    auto y1 = box.y;
    auto x2 = box.width;
    auto y2 = box.height;

    //[x1, y1, x2, y2, 概率, 分类坐标]
    detections.push_back(make_tuple(x1, y1, x2, y2, confidence, class_id));
  }
  return detections;
}

SI::SI(/* args */) {
  ip_key.type = INPUT_KEYBOARD;
  ip_key.ki.wVk = 0;
  ip_key.ki.wScan = 0;
  ip_key.ki.time = 0;
  ip_key.ki.dwExtraInfo = 0;
}
void SI::press(wstring k, double pause_seconds) {
  keyDown(k);
  sleep(pause_seconds);
  keyUp(k);
}
void SI::keyUpMany(span<wstring> keys, double seconds) {
  if (keys.empty()) {
    for (auto k : _key_down_list) {
      keyUp(k);
      sleep(seconds);
    }
    _key_down_list.clear();
  } else {
    for (auto k : keys) {
      keyUp(k);
      sleep(seconds);
    }
  }
}
void SI::keyDownMany(span<wstring> keys, bool run, double seconds) {
  if (run && keys.size() >= 1) {
    press(keys[0]);
  }

  for (auto k : keys) {
    keyDown(k);
    sleep(seconds);
  }
}
void SI::pressMany(span<wstring> keys, double seconds) {
  for (auto k : keys) {
    press(k);
    sleep(seconds);
  }
}
void SI::keyUpMany(wstring keys, double seconds) {
  vector<wstring> v_keys;
  if (!keys.empty()) {
    if (is_chinese(keys)) {
      v_keys = zhStr2zhList(keys);
    } else {
      v_keys.push_back(keys.data());
    }
  }
  keyUpMany(v_keys, seconds);
}
void SI::keyDownMany(wstring keys, bool run, double seconds) {
  vector<wstring> v_keys;
  if (!keys.empty()) {
    if (is_chinese(keys)) {
      v_keys = zhStr2zhList(keys);
    } else {
      v_keys.push_back(keys);
    }
  }
  keyDownMany(v_keys, seconds);
}
void SI::pressMany(wstring keys, double seconds) {
  vector<wstring> v_keys;
  if (!keys.empty()) {
    if (is_chinese(keys)) {
      v_keys = zhStr2zhList(keys);
    } else {
      v_keys.push_back(keys);
    }
  }
  pressMany(v_keys, seconds);
}
void SI::keyUp(wstring k) {
  if (k.empty()) return;

  // println("keyUp: {}", k);

  ip_key.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;

  auto is_arrow_keys = ranges::find(_arrow_keys, k) != _arrow_keys.end();
  if (is_arrow_keys) ip_key.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;

  ip_key.ki.wScan = vsc_map[k];
  SendInput(1, &ip_key, sizeof(INPUT));

  if (is_arrow_keys && GetKeyState(vk_map[L"numlock"])) {
    ip_key.ki.wScan = 0xE0;
    ip_key.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
    SendInput(1, &ip_key, sizeof(INPUT));
  }
  _key_down_list.erase(remove(_key_down_list.begin(), _key_down_list.end(), k),
                       _key_down_list.end());
}
void SI::mouseDown(int x = -1, int y = -1, int button = MOUSE_LEFT) {
  if (x >= 0 && y >= 0) {
    moveTo(x, y);
  }

  INPUT ip_mouse;
  ip_mouse.type = INPUT_MOUSE;
  ip_mouse.mi.dx = 0;
  ip_mouse.mi.dy = 0;
  ip_mouse.mi.mouseData = 0;
  ip_mouse.mi.dwFlags = _mouse_downs[button];
  ip_mouse.mi.time = 0;
  ip_mouse.mi.dwExtraInfo = 0;
  SendInput(1, &ip_mouse, sizeof(INPUT));
}
void SI::mouseUp(int x = -1, int y = -1, int button = MOUSE_LEFT) {
  if (x >= 0 && y >= 0) {
    moveTo(x, y);
  }

  INPUT ip_mouse;
  ip_mouse.type = INPUT_MOUSE;
  ip_mouse.mi.dx = 0;
  ip_mouse.mi.dy = 0;
  ip_mouse.mi.mouseData = 0;
  ip_mouse.mi.dwFlags = _mouse_ups[button];
  ip_mouse.mi.time = 0;
  ip_mouse.mi.dwExtraInfo = 0;
  SendInput(1, &ip_mouse, sizeof(INPUT));
}
void SI::mouseClick(int x = -1, int y = -1, int button = MOUSE_LEFT) {
  mouseDown(x, y, button);
  sleep(0.1);
  mouseUp(x, y, button);
}
void SI::keyDown(wstring k) {
  if (k.empty()) return;

  auto is_down = find(_key_down_list.begin(), _key_down_list.end(), k) !=
                 _key_down_list.end();
  if (is_down) return;

  ip_key.ki.dwFlags = 0;
  ip_key.ki.dwFlags = KEYEVENTF_SCANCODE;

  if (ranges::find(_arrow_keys, k) != _arrow_keys.end()) {
    ip_key.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
    if (GetKeyState(vk_map[L"numlock"])) {
      auto expectedEvents = 2;
      ip_key.ki.wScan = 0xE0;
      SendInput(1, &ip_key, sizeof(INPUT));
    }
  }

  ip_key.ki.wScan = vsc_map[k];
  println("keyDown: {},{}", k, ip_key.ki.wScan);

  SendInput(1, &ip_key, sizeof(INPUT));
  _key_down_list.push_back(k);
}

void SI::moveTo(int x = -1, int y = -1) {
  tie(x, y) = to_windows_coordinates(position(x, y));
  INPUT ip_mouse;
  ip_mouse.type = INPUT_MOUSE;
  ip_mouse.mi.dx = x;
  ip_mouse.mi.dy = y;
  ip_mouse.mi.mouseData = 0;
  ip_mouse.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
  ip_mouse.mi.time = 0;
  ip_mouse.mi.dwExtraInfo = 0;
  SendInput(1, &ip_mouse, sizeof(INPUT));
}
tuple_xy SI::to_windows_coordinates(tuple_xy p) {
  // auto dw = GetSystemMetrics(0);
  // auto dh = GetSystemMetrics(1);

  // 实际分辨率
  HDC hDC = GetDC(NULL);
  int dw = GetDeviceCaps(hDC, DESKTOPHORZRES);
  int dh = GetDeviceCaps(hDC, DESKTOPVERTRES);
  ReleaseDC(NULL, hDC);

  // RECT desktop;
  // GetWindowRect(GetDesktopWindow(), &desktop);
  // int screenWidth2 = desktop.right - desktop.left;
  // int screenHeight2 = desktop.bottom - desktop.top;

  auto wx = (get<0>(p) * 65536) / dw + 1;
  auto wy = (get<1>(p) * 65536) / dh + 1;
  return tuple_xy(wx, wy);
}
tuple_xy SI::position(int x = -1, int y = -1) {
  POINT p;
  GetCursorPos(&p);
  if (x < 0) x = p.x;
  if (y < 0) y = p.y;
  return tuple_xy(x, y);
}

wstring GameStatus::player_on_screen_pos_zh() {
  if (player_point == nullopt) {
    return L"";
  }

  return get_ssr_pos_zh(*player_point);
}
bool GameStatus::next_room_is_boss() {
  if (boss_room_point == nullopt) return false;

  auto next_room_i = room_i + 1;
  if (room_path.empty() or next_room_i >= room_path.size()) return false;
  auto next_room = room_path[next_room_i];
  return next_room == *boss_room_point;
}
void GameStatus::add_room_block(wstring logname) {
  if (player_room_point == nullopt) return;

  auto cur_room = *player_room_point;

  if (room_road_block.find(cur_room) == room_road_block.end())
    room_road_block[cur_room] = {};

  auto next_room_point = room_path[room_i + 1];
  // println("此路不通: {} {}到{}", logname, cur_room, next_room_point);
  room_road_block[cur_room].push_back(next_room_point);

  // 重置路径
  room_path.clear();
  room_i = 0;
}
void GameStatus::reset_fb_status() {
  challenge_again = true;

  room_path.clear();
  room_i = 0;
  boss_room_point = nullopt;
  player_room_point = nullopt;
  player_point = nullopt;
  player_point_before = nullopt;
  player_point_time = 0;
  player_in_boss_room = false;
  smap_grid.clear();
  is_move_to_next_room = false;
  default_move_room = false;

  // 每次挑战副本的地图可能不一样，所以需要重新设置，若是固定的地图则不需要
  load_room_road_block();

  // TOOD: 清空技能冷却
  // skill_bar.reset_ct_all()
}
void GameStatus::reset_path_status() {
  room_i = 0;
  room_path.clear();
  player_room_point = nullopt;
}
void load_img_and_gray(filesystem::path p, cv::Mat *temp, cv::Mat *temp_gray,
                       int code) {
  *temp = imread2(p.wstring());
  if (temp_gray != nullptr) {
    cv::cvtColor(*temp, *temp_gray, code);
  }
}
void GameStatus::load_image_temps() {
  load_img_and_gray(src_dir / "地区传送.jpg", &area_temp, &area_temp_gray);
  load_img_and_gray(src_dir / "副本.jpg", &fb_temp, &fb_temp_gray);
  load_img_and_gray(src_dir / "选择角色.jpg", &select_role_temp,
                    &select_role_temp_gray);
  load_img_and_gray(src_dir / "领主特征.jpg", &boss_feature_temp,
                    &boss_feature_temp_gray);
  load_img_and_gray(src_dir / "关闭.jpg", &close_button_temp,
                    &close_button_temp_gray);
  load_img_and_gray(src_dir / "是否继续.jpg", &is_continue_temp,
                    &is_continue_temp_gray);
  load_img_and_gray(src_dir / "领主图标.jpg", &boss_icon_temp, nullptr);
  load_img_and_gray(src_dir / "玩家图标.jpg", &player_icon_temp, nullptr);
}
void GameStatus::change_player() {
  skill_bar.clear();
  skill_bar.from_json(gs.config["技能表"][gs.pinfo.技能]);
}
optional<tuple<wstring, tuple_xy>> GameStatus::next_room_info() {
  auto next_room_i = room_i + 1;
  if (room_path.empty() || next_room_i >= room_path.size() or
      player_room_point == nullopt) {
    return nullopt;
  }

  auto next_room = room_path[next_room_i];
  return optional{get_smap_move_zh_pos(*player_room_point, next_room)};
}

void GameStatus::init(wstring p) {
  wcout << p << endl;
  // 加载配置文件
  config = loadjson5file(p);
  if (config["技能表"].is_string())
    config["技能表"] = loadjson5file(strToWstr(config["技能表"]));

  // print1(config.dump());

  config["资源目录"].get_to(src_dir);
  // 游戏开始，城镇，副本
  config["刷图模式"]["场景"].get_to(scenario);

  // 一些固定的属性
  config["刷图模式"]["boss_bgr"].get_to(_boss_bgr);
  config["刷图模式"]["boss_tolerance"].get_to(_boss_tolerance);
  config["刷图模式"]["player_bgr"].get_to(_player_bgr);
  config["刷图模式"]["player_tolerance"].get_to(_player_tolerance);

  config["刷图模式"]["使用角色"].get_to(pi);
  config["角色列表"][pi].get_to(pinfo);

  load_room_road_block();
  load_match_path();
}

void GameStatus::load_match_path() {
  match_path.clear();
  config["刷图模式"]["匹配路线"].get_to(match_path);
}

void GameStatus::load_room_road_block() {
  room_road_block.clear();
  auto 路线阻碍 = config["刷图模式"]["路线阻碍"];
  for (json::iterator it = 路线阻碍.begin(); it != 路线阻碍.end(); ++it) {
    room_road_block[eval_tuple(strToWstr(it.key()))] = it.value();
  }
}

bool Skill::from_json(json &sk) {
  if (sk["key"].is_null() || !sk["key"].is_string()) {
    return false;
  }

  key = sk["key"];
  ct = sk.value("ct", 1);
  boss = sk.value("boss", false);
  buff = sk.value("buff", false);
  dt = sk.value("dt", PRESS_SECONDS);
  ut = sk.value("ut", 0);

  if (sk.contains("name")) {
    name = sk.at("name");
  }

  if (is_chinese(key)) {
    lkey = zhStr2zhList(key);
  }

  // 如果位数字，则重复key
  if (sk["mk"].is_number()) {
    int count = sk["mk"];
    for (size_t i = 0; i < count; i++) {
      mk.push_back(key);
    }
  } else if (sk["mk"].is_array()) {
    sk["mk"].get_to(mk);
  }
  return true;
}
void Skill::reset_ct() { release_t = 0; }

bool Skill::can_release() {
  auto _after_ms = (now_ms() - release_t) / 1000;
  return _after_ms > ct;
}

bool Skill::press_key(bool right) {
  if (!lkey.empty()) {
    auto lks_len = lkey.size() - 1;
    for (size_t i = 0; i < lkey.size(); i++) {
      const auto &k = lkey[i];
      if (i == lks_len) {
        si.press(k, dt);
      } else {
        if (!right) {
          if (k == L"left")
            si.press(L"right");
          else if (k == L"right")
            si.press(L"left");
        } else {
          si.press(k);
        }
      }
      sleep(PRESS_SECONDS);
    }
    sleep(ut);
    return true;
  }

  if (!key.empty()) {
    si.press(key, dt);
    sleep(ut);
    return true;
  }

  return false;
}

bool Skill::release(bool right) {
  if (!can_release()) return false;

  if (!press_key(right)) return false;

  if (!mk.empty()) {
    sleep(0.1);
    for (const auto &c : mk) si.press(c);
  }

  release_t = now_ms();
  return true;
}
vector<wstring> zhStr2zhList(const wstring &zhKeyStr) {
  vector<wstring> out;
  for (auto it = zhKeyStr.begin(); it != zhKeyStr.end(); ++it) {
    out.emplace_back(it, it + 1);
  }
  return out;
}

// vector<wstring> zhKeyStr2keyList(wstring zhKeyStr) {
//   vector<wstring> res;
//   for (const auto &c : zhKeyStr) {
//     res.push_back(zhKey2key(c));
//   }
//   return res;
// }
// wstring zhKey2key(const wchar_t &zhpos) {
//   static const std::unordered_map<wchar_t, wstring> zh_to_en{
//       {L'上', L"up"},   {L'下', L"down"},  {L'左', L"left"}, {L'右',
//       L"right"},

//       {L'U', L"up"},    {L'B', L"down"},   {L'L', L"left"},  {L'R',
//       L"right"},

//       {L'u', L"up"},    {L'b', L"down"},   {L'l', L"left"},  {L'r',
//       L"right"},

//       {L'↑', L"up"},    {L'↓', L"down"},   {L'←', L"left"},  {L'→',
//       L"right"},

//       {L' ', L"space"}, {L'空', L"space"}, {L'跳', L"space"}};

//   auto it = zh_to_en.find(zhpos);
//   return it != zh_to_en.end() ? it->second : wstring(1, zhpos);
// }

double now_ms() {
  auto ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::system_clock::now().time_since_epoch())
                .count();
  return static_cast<double>(ms);
}

void SkillBar::reset_ct_all() {
  for (auto &el : list) el.reset_ct();
}

void SkillBar::release_buff_all() {
  for (auto &el : list) {
    if (el.buff && el.release()) {
      // 等待释放时间
      sleep(0.5);
    }
  }
}

int SkillBar::release_attack(int count, bool boss, bool right) {
  int c = 0;
  // 优先释放
  if (boss) {
    for (auto &el : list) {
      if (el.boss && el.release(right)) {
        c += 1;
        if (c >= count) return c;
        sleep(0.2);
      }
    }
  }

  for (auto &el : list) {
    if (el.boss || el.buff) continue;
    if (el.release(right)) {
      c += 1;
      if (c >= count) return c;
      sleep(0.2);
    }
  }
  return c;
}

void SkillBar::clear() {
  list.clear();
  list_key_map.clear();
}

void SkillBar::from_json(json &sk_list) {
  for (auto &el : sk_list) {
    Skill sk;
    if (!sk.from_json(el)) continue;
    // list_key_map[sk.key] = static_cast<int>(list.size());
    list.push_back(sk);
  }
}
