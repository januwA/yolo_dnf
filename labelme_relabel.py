import os
import json

if __name__ == '__main__':
  json_dir = "C:\\Users\\16418\\Desktop\\shenDianWaiWei\\1_segment"
  json_files = os.listdir(json_dir)
  json_dict = dict()

  for json_file in json_files:
      json_file_ext = os.path.splitext(json_file)

      if json_file_ext[1] == ".json":
          jsonfile = os.path.join(json_dir, json_file)

          with open(jsonfile, "r", encoding="utf-8") as jf:
              info = json.load(jf)

              for i, label in enumerate(info["shapes"]):
                  label_name = info["shapes"][i]["label"]
                  match label_name:
                    case "wanJia":
                      label_name = "player"
                    case "diRen":
                      label_name = "enemy"
                    case "men":
                      label_name = "door"
                    case "zaiCiTiaoZhan":
                      label_name = "zctz"
                    case "fanHuiChenZhen":
                      label_name = "fhcz"
                    case "jiangLi":
                      label_name = "JL"
                    case "caiLiao":
                      label_name = "CL"

                  label_name_new = label_name
                  info["shapes"][i]["label"] = label_name_new

              json_dict = info

          # 将替换后的内容写入原文件
          with open(jsonfile, "w") as new_jf:
              json.dump(json_dict, new_jf)
