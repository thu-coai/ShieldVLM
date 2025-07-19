model_path="thu-coai/ShieldVLM-7B-qwen"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --category "statement" \
    --model_path ${model_path} \
    --input_path examples/statement/example_input.json \
    --output_path examples/example_output.json \