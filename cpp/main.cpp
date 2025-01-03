#include "dnf.h"

int main(int argc, char* argv[]) {
  std::locale::global(std::locale("zh_CN.UTF-8"));
  // SetConsoleOutputCP(CP_UTF8);
  return bootstrap(argc, argv);
}
