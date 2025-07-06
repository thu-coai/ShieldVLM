model_path="/data3/public_checkpoints/cross-modal/qwen_vl2.5-7b/full/sft_lr1e-5_ep5_linear_warmup5_batchsize64_5/checkpoint-300"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --category "content" \
    --model_path ${model_path} \
    --input_path examples/statement/example_input.json \
    --output_path examples/example_output.json \