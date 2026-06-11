# 单机多卡
单机 2 DP，每 DP 1 卡
```shell
  python /vllm-workspace/shape_profiler.py \--serve \
    --model /home/weights/Qwen3.5-0.8B \
    --dp-size 2 \
    --dp-size-local 2 \
    --dp-address 127.0.0.1 \
    --dp-rpc-port 12321 \
    --vllm-start-port 8010 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --output-dir ./dp_output \
    --enforce-eager
```
  内部逻辑：自动为每个 DP rank 设置SHAPE_PROFILER_DP_RANK、SHAPE_PROFILER_DP_SIZE、ASCEND_RT_VISIBLE_DEVICES（按0,1,2... 顺序分配），启动子进程跑 vllm serve推理请求，Ctrl+C 后合并报告。
  
# 多机
设置环境变量
```shell
export SHAPE_PROFILER_OUTPUT_DIR=/path/to/output   # 必须，触发 profiler 
```
正常启动分布式DP命令（该命令根据业务场景变更）
```shell
python launch_online_dp.py     --dp-size 2     --tp-size 1     --dp-address 127.0.0.1     --dp-rpc-port 12325     --vllm-start-port 8300
```
以及proxy
```shell
python /vllm-workspace/dp_load_balance_proxy_server.py \
    --host 0.0.0.0 --port 9000 \
    --dp-hosts 127.0.0.1 127.0.0.1 \
    --dp-ports 8300 8301
```
正常发送推理请求后，Ctrl+C结束服务，然后进行合并报告：
```shell
python /vllm-workspace/shape_profiler.py --merge-only --output-dir /path/to/output
```
