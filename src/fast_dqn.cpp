#include "fast_dqn.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

namespace fast_dqn {


FrameDataSp PreprocessScreen(ALEInterface* ale) {
  ALEScreen raw_screen = ale->getScreen();
  size_t rawFrameWidth = raw_screen.width();
  size_t rawFrameHeight = raw_screen.height();
  std::vector<pixel_t> raw_pixels(rawFrameWidth*rawFrameHeight);
  ale->getScreenGrayscale(raw_pixels);

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


std::string DrawFrame(const FrameData& frame) {
  std::ostringstream o;
  for (auto row = 0; row < kCroppedFrameSize; ++row) {
    for (auto col = 0; col < kCroppedFrameSize; ++col) {
      o << std::hex <<
          static_cast<int>(frame[row * kCroppedFrameSize + col] / 16);
    }
    o << std::endl;
  }
  return o.str();
}

std::string PrintQValues(
    const std::vector<float>& q_values, const ActionVect& actions) {
  assert(!q_values.empty());
  assert(!actions.empty());
  assert(q_values.size() == actions.size());
  std::ostringstream actions_buf;
  std::ostringstream q_values_buf;
  for (auto i = 0; i < q_values.size(); ++i) {
    const auto a_str =
        boost::algorithm::replace_all_copy(
            action_to_string(actions[i]), "PLAYER_A_", "");
    const auto q_str = std::to_string(q_values[i]);
    const auto column_size = std::max(a_str.size(), q_str.size()) + 1;
    actions_buf.width(column_size);
    actions_buf << a_str;
    q_values_buf.width(column_size);
    q_values_buf << q_str;
  }
  actions_buf << std::endl;
  q_values_buf << std::endl;
  return actions_buf.str() + q_values_buf.str();
}

template <typename Dtype>
bool HasBlobSize(
    const caffe::Blob<Dtype>& blob,
    const int num,
    const int channels,
    const int height,
    const int width) {
  return blob.num() == num &&
      blob.channels() == channels &&
      blob.height() == height &&
      blob.width() == width;
}

void Fast_DQN::LoadTrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
}

void Fast_DQN::Initialize() {
  // Initialize net and solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
  solver_.reset(caffe::GetSolver<float>(solver_param));

  // New solver creation API.  Caution, caffe master current doesn't
  // work.  Something broke the training.
  // use commit:ff16f6e43dd718921e5203f640dd57c68f01cdb3 for now.  It's slower
  // though.  Let me know if you figure out the issue.
  // solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  net_ = solver_->net();

  // Cache pointers to blobs that hold Q values
  q_values_blob_ = net_->blob_by_name("q_values");

  // Initialize dummy input data with 0
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);

  // Cache pointers to input layers
  frames_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("frames_input_layer"));
  assert(frames_input_layer_);
  assert(HasBlobSize(
      *net_->blob_by_name("frames"),
      kMinibatchSize,
      kInputFrameCount,
      kCroppedFrameSize,
      kCroppedFrameSize));
  target_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("target_input_layer"));
  assert(target_input_layer_);
  assert(HasBlobSize(
      *net_->blob_by_name("target"), kMinibatchSize, kOutputCount, 1, 1));
  filter_input_layer_ =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net_->layer_by_name("filter_input_layer"));
  assert(filter_input_layer_);
  assert(HasBlobSize(
      *net_->blob_by_name("filter"), kMinibatchSize, kOutputCount, 1, 1));
}

Action Fast_DQN::SelectAction(
  const InputFrames& last_frames, const double epsilon) {
  assert(epsilon >= 0.0 && epsilon <= 1.0);
  auto action = SelectActionGreedily(last_frames).first;
  bool random = true;
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine_) < epsilon) {
    // Select randomly
    const auto random_idx =
        std::uniform_int_distribution<int>(
          0, legal_actions_.size() - 1)(random_engine_);
    action = legal_actions_[random_idx];
  } else {
    random = false;
  }
  if (verbose_) {
    if (random)
      LOG(INFO)  << action_to_string(action)
        << " (random)  epsilon:" << epsilon;
    else
      LOG(INFO)  << action_to_string(action)
        << " (greedy)  epsilon:" << epsilon;
  }

  return action;
}

std::pair<Action, float>Fast_DQN::SelectActionGreedily(
  const InputFrames& last_frames) {
  return SelectActionGreedily(std::vector<InputFrames>{{last_frames}}).front();
}

std::vector<std::pair<Action, float>> Fast_DQN::SelectActionGreedily(
    const std::vector<InputFrames>& last_frames_batch) {
  assert(last_frames_batch.size() <= kMinibatchSize);
  std::array<float, kMinibatchDataSize> frames_input;
  for (auto i = 0; i < last_frames_batch.size(); ++i) {
    // Input frames to the net and compute Q values for each legal actions
    for (auto j = 0; j < kInputFrameCount; ++j) {
      const auto& frame_data = last_frames_batch[i][j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize +
              j * kCroppedFrameDataSize);
    }
  }
  InputDataIntoLayers(frames_input, dummy_input_data_, dummy_input_data_);
  net_->ForwardPrefilled(nullptr);

  std::vector<std::pair<Action, float>> results;
  results.reserve(last_frames_batch.size());
  for (auto i = 0; i < last_frames_batch.size(); ++i) {
    // Get the Q values from the net
    const auto action_evaluator = [&](Action action) {
      const auto q = q_values_blob_->data_at(i, static_cast<int>(action), 0, 0);
      assert(!std::isnan(q));
      return q;
    };
    std::vector<float> q_values(legal_actions_.size());
    std::transform(
        legal_actions_.begin(),
        legal_actions_.end(),
        q_values.begin(),
        action_evaluator);
//     if (last_frames_batch.size() == 1) {
//       std::cout << PrintQValues(q_values, legal_actions_);
//     }

    // Select the action with the maximum Q value
    const auto max_idx =
        std::distance(
            q_values.begin(),
            std::max_element(q_values.begin(), q_values.end()));
    results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
  }
  return results;
}

void Fast_DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void Fast_DQN::Update() {
  if (verbose_)
    LOG(INFO) << "iteration: " << current_iteration() << std::endl;

  // Sample transitions from replay memory
  std::vector<int> transitions;
  transitions.reserve(kMinibatchSize);
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto random_transition_idx =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine_);
    transitions.push_back(random_transition_idx);
  }

  // Compute target values: max_a Q(s',a)
  std::vector<InputFrames> target_last_frames_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (!std::get<3>(transition)) {
      // This is a terminal state
      continue;
    }
    // Compute target value
    InputFrames target_last_frames;
    for (auto i = 0; i < kInputFrameCount - 1; ++i) {
      target_last_frames[i] = std::get<0>(transition)[i + 1];
    }
    target_last_frames[kInputFrameCount - 1] = std::get<3>(transition).get();
    target_last_frames_batch.push_back(target_last_frames);
  }
  const auto actions_and_values =
      SelectActionGreedily(target_last_frames_batch);

  FramesLayerInputData frames_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  auto target_value_idx = 0;
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    const auto action = std::get<1>(transition);
    assert(static_cast<int>(action) < kOutputCount);
    const auto reward = std::get<2>(transition);
    assert(reward >= -1.0 && reward <= 1.0);
    const auto target = std::get<3>(transition) ?
          reward + gamma_ * actions_and_values[target_value_idx++].second :
          reward;
    assert(!std::isnan(target));
    target_input[i * kOutputCount + static_cast<int>(action)] = target;
    filter_input[i * kOutputCount + static_cast<int>(action)] = 1;
    if (verbose_)
      VLOG(1) << "filter:" << action_to_string(action) << " target:" << target;
    for (auto j = 0; j < kInputFrameCount; ++j) {
      const auto& frame_data = std::get<0>(transition)[j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize +
              j * kCroppedFrameDataSize);
    }
  }
  InputDataIntoLayers(frames_input, target_input, filter_input);
  solver_->Step(1);
  // Log the first parameter of each hidden layer
//   VLOG(1) << "conv1:" <<
//     net_->layer_by_name("conv1_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "conv2:" <<
//     net_->layer_by_name("conv2_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "ip1:" <<
//     net_->layer_by_name("ip1_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "ip2:" <<
//     net_->layer_by_name("ip2_layer")->blobs().front()->data_at(1, 0, 0, 0);
}

void Fast_DQN::InputDataIntoLayers(
      const FramesLayerInputData& frames_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {
  frames_input_layer_->Reset(
      const_cast<float*>(frames_input.data()),
      dummy_input_data_.data(),
      kMinibatchSize);
  target_input_layer_->Reset(
      const_cast<float*>(target_input.data()),
      dummy_input_data_.data(),
      kMinibatchSize);
  filter_input_layer_->Reset(
      const_cast<float*>(filter_input.data()),
      dummy_input_data_.data(),
      kMinibatchSize);
}

}  // namespace fast_dqn

