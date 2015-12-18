#include "environment.h"
#include <ale_interface.hpp>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <iostream>
#include <vector>

namespace fast_dqn {

class ALEEnvironment : public Environment {

public:  
  ALEEnvironment(bool gui, const std::string rom_path) : ale_(gui) {
    ale_.setBool("display_screen", gui);
    ale_.loadROM(rom_path);

    ActionVect av = ale_.getMinimalActionSet();
    for (int i=0; i < av.size(); i++)
      legal_actions_.push_back(static_cast<ActionCode>(av[i]));
  }

  FrameDataSp PreprocessScreen() {
    ALEScreen raw_screen = ale_.getScreen();
    size_t rawFrameWidth = raw_screen.width();
    size_t rawFrameHeight = raw_screen.height();
    std::vector<pixel_t> raw_pixels(rawFrameWidth*rawFrameHeight);
    ale_.getScreenGrayscale(raw_pixels);

    auto screen = std::make_shared<FrameData>();
    assert(rawFrameHeight > rawFrameWidth);
    const auto x_ratio = rawFrameWidth / static_cast<double>(kCroppedFrameSize);
    const auto y_ratio = rawFrameHeight / static_cast<double>(kCroppedFrameSize);
    for (auto i = 0; i < kCroppedFrameSize; ++i) {
      for (auto j = 0; j < kCroppedFrameSize; ++j) {
        const auto first_x = static_cast<int>(std::floor(j * x_ratio));
        const auto last_x = static_cast<int>(std::floor((j + 1) * x_ratio));
        const auto first_y = static_cast<int>(std::floor(i * y_ratio));
        auto last_y = static_cast<int>(std::floor((i + 1) * y_ratio));
        if (last_y >= rawFrameHeight) {
          last_y = rawFrameHeight-1;
        }
        auto x_sum = 0.0;
        auto y_sum = 0.0;
        uint8_t resulting_color = 0.0;
        for (auto x = first_x; x <= last_x; ++x) {
          double x_ratio_in_resulting_pixel = 1.0;
          if (x == first_x) {
            x_ratio_in_resulting_pixel = x + 1 - j * x_ratio;
          } else if (x == last_x) {
            x_ratio_in_resulting_pixel = x_ratio * (j + 1) - x;
          }
          assert(
              x_ratio_in_resulting_pixel >= 0.0 &&
              x_ratio_in_resulting_pixel <= 1.0);
          for (auto y = first_y; y <= last_y; ++y) {
            double y_ratio_in_resulting_pixel = 1.0;
            if (y == first_y) {
              y_ratio_in_resulting_pixel = y + 1 - i * y_ratio;
            } else if (y == last_y) {
              y_ratio_in_resulting_pixel = y_ratio * (i + 1) - y;
            }
            assert(
                y_ratio_in_resulting_pixel >= 0.0 &&
                y_ratio_in_resulting_pixel <= 1.0);
            const auto grayscale =
              raw_pixels[static_cast<int>(y * rawFrameWidth + x)];
            resulting_color +=
                (x_ratio_in_resulting_pixel / x_ratio) *
                (y_ratio_in_resulting_pixel / y_ratio) * grayscale;
          }
        }
        (*screen)[i * kCroppedFrameSize + j] = resulting_color;
      }
    }
    return screen;
  }

  double ActNoop() {
    double reward = 0;
      for (auto i = 0; i < kInputFrameCount && !ale_.game_over(); ++i) {
        reward += ale_.act(PLAYER_A_NOOP);
      }
    return reward;
  }

  double Act(int action) {
    double reward = 0;
      for (auto i = 0; i < kInputFrameCount && !ale_.game_over(); ++i) {
        reward += ale_.act((Action)action);
      }
    return reward;
  }

  void Reset() { 
    ale_.reset_game(); 
  }

  bool EpisodeOver() { 
    return ale_.game_over(); 
  }

  std::string action_to_string(Environment::ActionCode a) { 
    return action_to_string(static_cast<Action>(a)); 
  }

  const ActionVec& GetMinimalActionSet() {
    return legal_actions_;
  }

 private:

  ALEInterface ale_;
  ActionVec legal_actions_;
  
};

EnvironmentSp CreateEnvironment(
    bool gui, const std::string rom_path) {
  return std::make_shared<ALEEnvironment>(gui, rom_path);
}

}  // namespace fast_dqn