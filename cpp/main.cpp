#include <iostream>
#include <ranges>
#include <fstream>
#include <format>
#include <sstream>
#include <windows.h>
#include "json.hpp"

using json = nlohmann::json;

json loadjson5file(std::string_view p)
{
    std::ifstream ifs(p.data());
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

int main(int, char **)
{
    SetConsoleOutputCP(CP_UTF8);
   // 加载json5文件
   /* 
   auto configData = loadjson5file(R"(C:\Users\16418\Desktop\dnf_py\config.json5)");
    if (configData["技能表"].is_string()) {
        auto skillData = loadjson5file(configData["技能表"]);
        std::cout << skillData << "\n";
    }
    */

    // std::stringstream buffer;
    // Json5::serialize(buffer, jdata);
    // std::cout << std::format("{}", buffer.str());
}
