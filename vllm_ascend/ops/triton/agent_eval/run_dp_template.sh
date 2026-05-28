
# 清除代理环境变量
unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY
source /usr/local/Ascend/driver/bin/setenv.bash
source /home/w00452989/CANN/0507/ascend-toolkit/set_env.sh

unset ftp_proxy
unset https_proxy
unset http_proxy

unset HCCL_INTRA_ROCE_ENABLE

source /home/liziyu/CONFIG

# 自动获取配置
nic_name="eth2"
local_ip="141.61.52.167"

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name


export HCCL_TOPO_FILE_PATH=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
export HCCL_ALGO=level0:fullmesh

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=1024

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TASK_QUEUE_ENABLE=1


dir="$PWD/$(date +%Y%m%d_%H%M%S)/plog" && mkdir -p "$dir"
export ASCEND_PROCESS_LOG_PATH="$dir"

export ASCEND_RT_VISIBLE_DEVICES=$1
#export DYNAMIC_EPLB="true"

vllm serve /home/weights/dsk_v3.1-T-w8a8c8_attn-0506-full \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --max_model_len 9000 \
  --max-num-batched-tokens 256 \
  --served-model-name dsv3 \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 108\
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp"}' \
  --quantization ascend \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/w00887678/TX/c8/wangyao/profiling", "torch_profiler_with_stack": false}' \
  --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30300",
    "engine_id": "3",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                }
        }
    }' \
  --additional_config '{"enable_cpu_binding": "True", "recompute_scheduler_enable":true, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp": false, "finegrained_tp_config": {"lmhead_tensor_parallel_size":8}}' 
