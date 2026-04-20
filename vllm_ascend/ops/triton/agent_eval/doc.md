vllm-ascend\vllm_ascend\ops\__init__.py 最开头添加
import vllm.utils.torch_utils  # noqa
import vllm_ascend.ops.patch_shape_profiler  # noqa

patch_shape_profiler.py放到 vllm-ascend\vllm_ascend\ops\目录下

python shape_profiler.py --model /mnt/g30061903/Qwen3.5-0.8B/ --prompt "Hello" --tensor-parallel-size 4
python shape_profiler.py --model /mnt/g30061903/Qwen3.5-0.8B/ --serve --port 8010 --tensor-parallel-size 4
python roofline_analyzer.py --input output/shape_records_merged.jsonl  --output ./roofline_report
