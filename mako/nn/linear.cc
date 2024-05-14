#include "mako/nn/linear.h"

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

class QKVParallelLinear{
private:
  int hidden_size;
  int head_size;
  int total_num_heads;
  std::optional<int> total_num_kv_heads;
  bool bias;
  bool skip_bias_add;
  std::optional<torch.dtype> params_dtype;
// TODO: implement QuantizationConfig
  std::optional<QuantizationConfig> quant_config;

public:
  void Init(
    int hidden_size;
    int head_size;
    int total_num_heads;
    std::optional<int> total_num_kv_heads;
    bool bias = true;
    bool skip_bias_add = false;
    std::optional<torch.dtype> params_dtype;
    std::optional<QuantizationConfig> quant_config;
  )

  void WeightLoader(
    Parameter param; // TODO: implement Parameter structure
    torch.Tensor loaded_weight;
    std::optional<str> loaded_shard_id;
  )
}

QKVParallelLinear::Init() {

}

QKVParallelLinear::WeightLoader() {

}


