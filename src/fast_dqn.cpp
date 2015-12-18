#include "fast_dqn.h"
#include "environment.h"
#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

namespace fast_dqn {


std::string PrintQValues(
    EnvironmentSp environmentSp,
    const std::vector<float>& q_values, const Environment::ActionVec& actions) {
  assert(!q_values.empty());
  assert(!actions.empty());
  assert(q_values.size() == actions.size());
  std::ostringstream actions_buf;
  std::ostringstream q_values_buf;
  for (auto i = 0; i < q_values.size(); ++i) {
    const auto a_str =
        boost::algorithm::replace_all_copy(
            environmentSp->action_to_string(actions[i]), "PLAYER_A_", "");
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


const State Transition::GetNextState() const {

  //  Create the s(t+1) states from the experience(t)'s

  if (next_frame_ == nullptr) {
    // Terminal state so no next_observation, just return current state
    return state_;
  } else {
    State state_clone;

    for (int i = 0; i < kInputFrameCount - 1; ++i)
      state_clone[i] = state_[i + 1];
    state_clone[kInputFrameCount - 1] = next_frame_;
    return state_clone;
  }

}

template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape) {
  net.has_blob(blob_name);
  const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
  const std::vector<int>& blob_shape = blob.shape();
  CHECK_EQ(blob_shape.size(), expected_shape.size());
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()));
}

void Fast_DQN::LoadTrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
}

void Fast_DQN::Initialize() {

  // Initialize dummy input data with 0
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);

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
  InitNet(net_);

  CloneTrainingNetToTargetNet();

  // Check the primary network
  HasBlobSize(*net_, train_frames_blob_name, {kMinibatchSize,
          kInputFrameCount, kCroppedFrameSize, kCroppedFrameSize});
  HasBlobSize(*net_, target_blob_name, {kMinibatchSize,kOutputCount,1,1});
  HasBlobSize(*net_, filter_blob_name, {kMinibatchSize,kOutputCount,1,1});


  LOG(INFO) << "Finished " << net_->name() << " Initialization";
}


Environment::ActionCode Fast_DQN::SelectAction(const State& frames, 
                                               const double epsilon) {
  return SelectActions(InputStateBatch{{frames}}, epsilon)[0];
}

Environment::ActionVec Fast_DQN::SelectActions(
                              const InputStateBatch& frames_batch,
                              const double epsilon) {
  CHECK(epsilon <= 1.0 && epsilon >= 0.0);
  CHECK_LE(frames_batch.size(), kMinibatchSize);
  Environment::ActionVec actions(frames_batch.size());
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine_) < epsilon) {
    // Select randomly
    for (int i = 0; i < actions.size(); ++i) {
      const auto random_idx = std::uniform_int_distribution<int>
          (0, legal_actions_.size() - 1)(random_engine_);
      actions[i] = legal_actions_[random_idx];
    }
  } else {
    // Select greedily
    std::vector<ActionValue> actions_and_values =
        SelectActionGreedily(target_net_, frames_batch);
    CHECK_EQ(actions_and_values.size(), actions.size());
    for (int i=0; i<actions_and_values.size(); ++i) {
      actions[i] = actions_and_values[i].action;
    }
  }
  return actions;
}


ActionValue Fast_DQN::SelectActionGreedily(
  NetSp net,
  const State& last_frames) {
  return SelectActionGreedily(net, InputStateBatch{{last_frames}}).front();
}

std::vector<ActionValue> Fast_DQN::SelectActionGreedily(
    NetSp net,
    const InputStateBatch& last_frames_batch) {
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
  InputDataIntoLayers(net, frames_input, dummy_input_data_, dummy_input_data_);
  net->ForwardPrefilled(nullptr);

  std::vector<ActionValue> results;
  results.reserve(last_frames_batch.size());
  CHECK(net->has_blob(q_values_blob_name));
  const auto q_values_blob = net->blob_by_name(q_values_blob_name);
  for (auto i = 0; i < last_frames_batch.size(); ++i) {
    // Get the Q values from the net
    const auto action_evaluator = [&](Environment::ActionCode action) {
      const auto q = q_values_blob->data_at(i, static_cast<int>(action), 0, 0);
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

  // Every clone_iters steps, update the clone_net_
  if (current_iteration() >= last_clone_iter_ + clone_frequency_) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneTrainingNetToTargetNet();
    last_clone_iter_ = current_iteration();
  }

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
  std::vector<State> target_last_frames_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (transition.is_terminal()) {
      continue;
    }

    target_last_frames_batch.push_back(transition.GetNextState());
  }

    // Get the next state QValues
  const auto actions_and_values =
      SelectActionGreedily(target_net_, target_last_frames_batch);

  FramesLayerInputData frames_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  auto target_value_idx = 0;
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    const auto action = transition.GetAction();
    const auto reward = transition.GetReward();
    assert(reward >= -1.0 && reward <= 1.0);
    const auto target = transition.is_terminal() ?
          reward :
          reward + gamma_ * actions_and_values[target_value_idx++].q_value;
    assert(!std::isnan(target));
    target_input[i * kOutputCount + static_cast<int>(action)] = target;
    filter_input[i * kOutputCount + static_cast<int>(action)] = 1;
    if (verbose_)
      VLOG(1) << "filter:" << environmentSp_->action_to_string(action) 
        << " target:" << target;
    for (auto j = 0; j < kInputFrameCount; ++j) {
      const State& state = transition.GetState();
      const auto& frame_data = state[j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize +
              j * kCroppedFrameDataSize);
    }
  }
  InputDataIntoLayers(net_, frames_input, target_input, filter_input);
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

void Fast_DQN::InitNet(NetSp net) {
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(target_layer_name));
    CHECK(target_input_layer);
    target_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
                              const_cast<float*>(dummy_input_data_.data()),
                              target_input_layer->batch_size());
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(filter_layer_name));
    CHECK(filter_input_layer);
    filter_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
                              const_cast<float*>(dummy_input_data_.data()),
                              filter_input_layer->batch_size());
}

void Fast_DQN::CloneNet(NetSp net) {
  caffe::NetParameter net_param;
  net->ToProto(&net_param);
  net_param.mutable_state()->set_phase(net->phase());
  if (target_net_ == nullptr) {
    target_net_.reset(new caffe::Net<float>(net_param));
  } else {
    target_net_->CopyTrainedLayersFrom(net_param);
  }
  InitNet(target_net_);
}


void Fast_DQN::InputDataIntoLayers(NetSp net,
      const FramesLayerInputData& frames_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {

  const auto frames_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net->layer_by_name(frames_layer_name));
  CHECK(frames_input_layer);

  frames_input_layer->Reset(const_cast<float*>(frames_input.data()),
                            const_cast<float*>(frames_input.data()),
                            frames_input_layer->batch_size());

  if (net == net_) { // training net?
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(target_layer_name));
    CHECK(target_input_layer);
    target_input_layer->Reset(const_cast<float*>(target_input.data()),
                              const_cast<float*>(target_input.data()),
                              target_input_layer->batch_size());
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(filter_layer_name));
    CHECK(filter_input_layer);
    filter_input_layer->Reset(const_cast<float*>(filter_input.data()),
                              const_cast<float*>(filter_input.data()),
                              filter_input_layer->batch_size());
  }

}

}  // namespace fast_dqn

