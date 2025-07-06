import json
import copy

# 输入文件路径
file1 = "/data3/zhangqinglin/project/MMRisks/analysis/wait/sft_lr5e-6_ep4_linear_warmup5_batchsize24_checkpoint-708_content_ood_system.json"
file2 = "/data3/zhangqinglin/project/MMRisks/analysis/wait/ours_sft_lr1e-5_ep5_linear_warmup5_batchsize64_5_checkpoint-300_content_ood_system.json"

def analyze_file(filepath):
    mismatch_items = {}  # id(str) -> 原始item副本 + error_type
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            id_str = str(item.get("id"))
            human = item.get("human_safety_label")
            pred = item.get("pred_safety_label")

            if human != pred:
                annotated = copy.deepcopy(item)  # 不修改原数据
                if pred == "unsafe" and human == "safe":
                    annotated["error_type"] = "FP"
                elif pred == "safe" and human == "unsafe":
                    annotated["error_type"] = "FN"
                else:
                    annotated["error_type"] = "OTHER"
                mismatch_items[id_str] = annotated
    return mismatch_items

# 分析两个文件
mismatch1 = analyze_file(file1)
mismatch2 = analyze_file(file2)

ids1 = set(mismatch1.keys())
ids2 = set(mismatch2.keys())

only1_ids = ids1 - ids2
only2_ids = ids2 - ids1

only1_items = [mismatch1[i] for i in only1_ids]
only2_items = [mismatch2[i] for i in only2_ids]

# 写入新 JSON 文件
with open("only_in_file1.json", "w", encoding="utf-8") as f:
    json.dump(only1_items, f, ensure_ascii=False, indent=2)

with open("only_in_file2.json", "w", encoding="utf-8") as f:
    json.dump(only2_items, f, ensure_ascii=False, indent=2)

# 输出提示信息
print("✅ 输出完成：")
print(f"  - 只在 file1 中不一致的条目数量: {len(only1_items)}，已保存为 only_in_file1.json")
print(f"  - 只在 file2 中不一致的条目数量: {len(only2_items)}，已保存为 only_in_file2.json")
