#pragma once

#ifndef DNF_H
#define DNF_H

#include <locale.h>
#include <windows.h>

#include <chrono>
#include <codecvt>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <print>
#include <random>
#include <ranges>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "json.hpp"

// 该键当前是向上还是向下。如果设置了最高有效位，则该键处于向下状态
constexpr auto KS_NOW_DOWN = 0xFF00;
// 设置了最低有效位，则该键是在上次调用GetAsyncKeyState之后按下的
constexpr auto KS_BEFORE_DOWN = 0x00FF;

constexpr auto MOUSE_LEFT = 0;
constexpr auto MOUSE_MIDDLE = 1;
constexpr auto MOUSE_RIGHT = 2;

/**按键按下几秒抬起 */
constexpr auto PRESS_SECONDS = 0.05;

using namespace std;
using json = nlohmann::json;
using predict_result_t = tuple<int, int, int, int, float, int>;
using tuple_xy = tuple<int, int>;

wstring strToWstr(string_view str);
string wstrToStr(wstring_view wstr);
u16string strToU16(string_view str);
string u16ToStr(u16string_view ustr);

// https://www.cppstories.com/2022/custom-stdformat-cpp20/
template <typename T>
struct std::formatter<std::optional<T>> : std::formatter<T> {
  auto format(const std::optional<T> &obj, std::format_context &ctx) const {
    if (!obj) return std::format_to(ctx.out(), "std::nullopt");

    auto &v = *obj;
    return std::formatter<T>::format(v, ctx);
    // return std::format_to(ctx.out(), "{}", v);
  }
};

template <typename T>
concept WStringLike =
    std::is_same_v<T, std::wstring>;

template <WStringLike T>
struct std::formatter<T> : std::formatter<string_view> {
  auto format(const wstring &obj, std::format_context &ctx) const {
    return std::formatter<string_view>::format(wstrToStr(obj), ctx);
  }
};

// std::string and std::wstring to_json
// https://github.com/nlohmann/json/issues/1592
namespace nlohmann {
template <>
struct adl_serializer<wstring> {
  static void to_json(json &j, const wstring &str) { j = wstrToStr(str); }

  static void from_json(const json &j, wstring &str) {
    str = strToWstr(j.get<string>());
  }
};
template <>
struct adl_serializer<wstring_view> {
  static void to_json(json &j, const wstring_view &str) { j = wstrToStr(str); }

  static void from_json(const json &j, wstring_view &str) {
    str = j.get<wstring>();
  }
};
}  // namespace nlohmann

template <typename... Args>
void ppp(Args &&...args) {
  ((wcout << args << L" "), ...);
  wcout << endl;
}

// std::string and std::wstring to_json
// https://github.com/nlohmann/json/issues/1592
template <typename T>
T random_number(T b, T e) {
  static random_device rd;   // 用于生成随机数引擎的种子
  static mt19937 gen(rd());  // 随机数引擎

  if constexpr (is_integral_v<T>) {
    uniform_int_distribution<T> dis(b, e);
    return dis(gen);
  } else {
    uniform_real_distribution<T> dis(b, e);
    return dis(gen);
  }
}

template <typename Container>
auto random_choice(const Container &container) {
  static random_device rd;
  static mt19937 gen(rd());
  uniform_int_distribution<> dis(0, static_cast<int>(container.size() - 1));

  return container[dis(gen)];
}

void sleep(double);

/**
 * 小地图房间坐标方向
 */
tuple<wstring, tuple_xy> get_smap_move_zh_pos(tuple_xy current_node,
                                                   tuple_xy next_node);

wstring get_ssr_pos_zh(tuple_xy point);

/**
 * "(1,1)" => (1,1)
 */
tuple_xy eval_tuple(wstring str);

class GameStatus {
  struct GwInfo {
    int left{};
    int top{};
    int right{};
    int bottom{};
    int w{};
    int h{};
    int scale{};
  };
  class RoleInfo {
   public:
    int 移速{};
    string 技能{};
    int 释放距离{};

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RoleInfo, 移速, 技能, 释放距离)
  };

  class MatchPathInfo {
   public:
    tuple_xy begin{};
    tuple_xy end{};
    vector<tuple_xy> path{};

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(MatchPathInfo, begin, end, path)
  };

 private:
  /* data */
 public:
  // 地区传送图片
  cv::Mat area_temp;
  cv::Mat area_temp_gray;

  // 选择地图时副本图片
  cv::Mat fb_temp;
  cv::Mat fb_temp_gray;

  // esc 选择角色按钮图片
  cv::Mat select_role_temp;
  cv::Mat select_role_temp_gray;

  // boss房间，boss特征图片
  cv::Mat boss_feature_temp;
  cv::Mat boss_feature_temp_gray;

  // 关闭按钮图片
  cv::Mat close_button_temp;
  cv::Mat close_button_temp_gray;

  // 刷完副本，领完奖励后，是否继续图片
  cv::Mat is_continue_temp;
  cv::Mat is_continue_temp_gray;
  // 副本小地图boss图片（一个红色牛头），宽度必须与每格子相等，高度无所谓
  cv::Mat boss_icon_temp;
  // 副本小地图玩家图片，一个蓝色光标
  cv::Mat player_icon_temp;

  json config;
  filesystem::path src_dir;

  HWND hwnd{0};
  bool pause = false;
  /**
   * 游戏窗口信息
   */
  GwInfo gw_info;

  /*当前场景*/
  wstring scenario{};

  /**
   * 颜色信息，用于寻找玩家和boss在小地图上的位置
   */
  vector<int> _boss_bgr{};
  int _boss_tolerance{};
  vector<int> _player_bgr{};
  int _player_tolerance{};

  /**
   * 角色列表 索引
   */
  int pi{0};
  /**
   * 使用角色信息
   */
  RoleInfo pinfo;

  /**
   * 房间之间的通道阻碍，比如房间a从右到b房间没有通道，提供这个可以更好计算出动态路线
   *
   * 在使用A*方法时使用
   *
   * { [0,0]: [ [0,1], ... ] }
   *
   * ! 每次通关重置
   */
  map<tuple_xy, vector<tuple_xy>> room_road_block{};
  /**
   * 直接提供匹配路线，用玩家房间位置和boss房间位置，得到设定好的前进路线
   *
   * ! 每次通关重置
   */
  vector<MatchPathInfo> match_path{};

  /**
   * 游戏窗口内的矩形信息 (x1, y1, x2,
   * y2)，这个矩形用来定位玩家在屏幕上的那个位置
   */
  tuple<int, int, int, int> ss_rect{};

  /**
   * 0可通行，1不可通行
   * 如果房间是不存在的可以提前设置为1，可以更好计算出动态路线
   */
  vector<vector<int>> smap_grid{};
  /**
   * 玩家当前在 [room_path] 中的索引
   */
  int room_i{0};

  /**
   * 前往boss房间的路线
   */
  vector<tuple_xy> room_path{};
  /**
   * boss房间位置
   */
  optional<tuple_xy> boss_room_point{};
  /**
   * 玩家房间位置
   */
  optional<tuple_xy> player_room_point{};

  /**
   * 玩家在屏幕上的位置
   */
  optional<tuple_xy> player_point{};
  /**
   * 玩家上次出现在屏幕上的位置
   */
  optional<tuple_xy> player_point_before{};
  int player_point_time{0};

  /**
   * 玩家正在boss房间
   */
  bool player_in_boss_room = false;
  /**
   * 过去了多少秒
   */
  int next_room_time{0};
  /**
   * 正在移动到下个房间
   */
  bool is_move_to_next_room = false;
  /**
   * 再次挑战地下城
   */
  bool challenge_again = false;
  /**
   * 断言以移动到下个房间。通常在没找到玩家房间位置时
   */
  bool default_move_room = false;
  GameStatus() = default;
  void init(wstring);
  /**
   * 从config中加载匹配路线
   */
  void load_match_path();
  /**
   * 从config中加载路线阻碍信息
   */
  void load_room_road_block();

  /**
   * 从当前房间到下个房间的方位信息
   */
  optional<tuple<wstring, tuple_xy>> next_room_info();

  /**
   * 玩家处于屏幕上的哪个位置, 上中下左？
   */
  wstring player_on_screen_pos_zh();

  /**
   * 下个房间是boss?
   */
  bool next_room_is_boss();

  /**
   * 路不通，意味着重新设计路线
   */
  void add_room_block(wstring logname);

  /**
   * 每次刷完副本要重置一些状态
   */
  void reset_fb_status();

  /**
   * 清空副本路线信息
   */
  void reset_path_status();

  /**
   * 加载cv图片，用以match
   */
  void load_image_temps();

  /**
   * 却换角色，重新加载角色信息
   */
  void change_player();
};

/**
 * https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-sendinput
 *
 * INPUT_MOUSE:
 * https://learn.microsoft.com/zh-cn/windows/win32/api/winuser/ns-winuser-mouseinput
 *
 * INPUT_KEYBOARD:
 * https://learn.microsoft.com/zh-cn/windows/win32/api/winuser/ns-winuser-keybdinput
 *
 * INPUT_HARDWARE:
 * https://learn.microsoft.com/zh-cn/windows/win32/api/winuser/ns-winuser-hardwareinput
 *
 * type: INPUT_MOUSE mi 结构, INPUT_KEYBOARD ki 结构, INPUT_HARDWARE hi 结构
 *
 * examples:
 * https://learn.microsoft.com/zh-cn/windows/win32/api/winuser/nf-winuser-sendinput
 *
 * 合成键击、鼠标动作和按钮单击。
 */
class SI {
 private:
  /* data */
  INPUT ip_key;
  vector<wstring> _arrow_keys{
      L"left", L"up", L"right", L"down", L"左", L"上", L"右", L"下",
  };
  vector<int> _mouse_downs{MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_MIDDLEDOWN,
                           MOUSEEVENTF_RIGHTDOWN};
  vector<int> _mouse_ups{MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEUP,
                         MOUSEEVENTF_RIGHTUP};

 public:
  vector<wstring> _key_down_list;
  SI(/* args */);
  void press(wstring k, double pause_seconds = PRESS_SECONDS);
  void keyDown(wstring k);
  void keyUp(wstring k);

  void keyUpMany(span<wstring> keys, double seconds = PRESS_SECONDS);
  void keyDownMany(span<wstring> keys, bool run = false, double seconds = PRESS_SECONDS);
  void pressMany(span<wstring> keys, double seconds = PRESS_SECONDS);

  void keyUpMany(wstring keys=L"", double seconds = PRESS_SECONDS);
  void keyDownMany(wstring keys=L"", bool run = false, double seconds = PRESS_SECONDS);
  void pressMany(wstring keys=L"", double seconds = PRESS_SECONDS);

  void mouseDown(int x, int y, int button);
  void mouseUp(int, int, int);
  void mouseClick(int, int, int);
  void moveTo(int, int);

  tuple_xy position(int, int);
  tuple_xy to_windows_coordinates(tuple_xy);
};

vector<cv::Scalar> randomcolor(span<wstring> from);
cv::Rect points2rect(int x1, int y1, int x2, int y2);

wstring getWindowCaption(HWND hwnd);

static bool is_chinese(wstring str);

static json loadjson5file(wstring p);

static cv::Mat imread2(wstring p);

/**
 * examples:
 * https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-CPP-Inference
 *
 * size: yolo 训练时imgsz参数
 */
static vector<predict_result_t> predict(cv::Mat &original_image, double conf = 0.8,
                                cv::Size size = cv::Size(640, 640));

void spot(cv::Mat &original_image, vector<predict_result_t> &detections);

void init_vkmap();
// https://learn.microsoft.com/zh-cn/windows/win32/gdi/capturing-an-image
cv::Mat CaptureAnImage(HWND hWnd, bool &err);
void predict_game();
/**
 * temp 是必须的
 *
 * temp_gray 为空，则不加载
 */
void load_img_and_gray(filesystem::path, cv::Mat *temp, cv::Mat *temp_gray,
                       int code = cv::COLOR_BGR2GRAY);
int bootstrap(int argc, char *argv[]);

class Skill {
 private:
  /* data */
 public:
  double release_t{};
  /**
   * 键盘快捷键 a,b ctrl
   */
  wstring key{};

  /**
   * 连续的key 上上下下z
   */
  vector<wstring> lkey;
  /**
   * 冷却时间
   */
  double ct{};
  /**
   * 这个技能只会在boss房间释放
   */
  bool boss{};
  /**
   * 按下后几秒抬起
   */
  double dt{PRESS_SECONDS};
  /**
   * 抬起后等待几秒
   */
  double ut{0};
  /**
   * 技能名称
   */
  wstring name{};
  /**
   * 状态技能
   */
  bool buff{};

  /**
   * 多段技能, int重复按key多少次, list接下来要按的键位，只有快捷键可以
   */
  vector<wstring> mk;

  Skill() = default;
  ~Skill() = default;

  /**
   * 从json中加载数据,成功返回true
   */
  bool from_json(json &sk);
  /**
   * 重置技能冷却时间
   */
  void reset_ct();

  bool can_release();

  bool press_key(bool right = true);
  bool release(bool right = true);
};

class SkillBar {
 public:
  /* data */
  vector<Skill> list{};

  /**
   * 技能key在list中的索引
   */
  map<wstring, int> list_key_map{};
  SkillBar() = default;
  ~SkillBar() = default;

  /**重置所有技能冷却 */
  void reset_ct_all();

  /**释放所有buff技能 */
  void release_buff_all();

  /**释放攻击技能 */
  int release_attack(int count = 1, bool boss = false, bool right = true);

  /**清空所有技能 */
  void clear();

  void from_json(json &);
};

// wstring zhKey2key(const wchar_t &zhpos);
// vector<wstring> zhKeyStr2keyList(wstring zhKeyStr);
vector<wstring> zhStr2zhList(const wstring & zhKeyStr);

double now_ms();

#endif
