import json

# 输入文件路径
file1 = "/data3/zhangqinglin/project/MMRisks/analysis/wait/sft_lr5e-6_ep4_linear_warmup5_batchsize24_checkpoint-708_content_ood_system.json"
file2 = "/data3/zhangqinglin/project/MMRisks/analysis/wait/ours_sft_lr1e-5_ep5_linear_warmup5_batchsize64_5_checkpoint-300_content_ood_system.json"

# 提取不一致 ID 并统计准确率和误分类情况
def analyze_file(filepath):
    mismatch_ids = set()
    total = 0
    correct = 0
    false_positive = 0
    false_negative = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            hid = str(item.get("id"))
            human = item.get("human_safety_label")
            pred = item.get("pred_safety_label")
            if human != pred:
                mismatch_ids.add(hid)
                if pred == "unsafe" and human == "safe":
                    false_positive += 1
                elif pred == "safe" and human == "unsafe":
                    false_negative += 1
            else:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "mismatch_ids": mismatch_ids,
        "accuracy": accuracy,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "total": total,
        "correct": correct
    }

# 分析两个文件
result1 = analyze_file(file1)
result2 = analyze_file(file2)

# 比较不一致 ID 的差异
both = result1["mismatch_ids"] & result2["mismatch_ids"]
only1 = result1["mismatch_ids"] - result2["mismatch_ids"]
only2 = result2["mismatch_ids"] - result1["mismatch_ids"]

# 输出统计结果
print("📊 文件1")
print("  - 总样本数:", result1["total"])
print("  - 正确预测数:", result1["correct"])
print("  - 准确率: {:.2f}%".format(result1["accuracy"] * 100))
print("  - 假阳性 (FP):", result1["false_positive"])
print("  - 假阴性 (FN):", result1["false_negative"])
print("  - 不一致 ID 数量:", len(result1["mismatch_ids"]))

print("\n📊 文件2")
print("  - 总样本数:", result2["total"])
print("  - 正确预测数:", result2["correct"])
print("  - 准确率: {:.2f}%".format(result2["accuracy"] * 100))
print("  - 假阳性 (FP):", result2["false_positive"])
print("  - 假阴性 (FN):", result2["false_negative"])
print("  - 不一致 ID 数量:", len(result2["mismatch_ids"]))

print("\n📌 差异分析")
print("  - 两个文件中都存在的不一致 ID 数量：", len(both))
print("  - 只在第一个文件中不一致的 ID 数量：", len(only1))
print("  - 只在第二个文件中不一致的 ID 数量：", len(only2))
print("  ✅ 两个文件中都不一致的 ID：", sorted(both))
print("  📁 只在第一个文件中不一致的 ID：", sorted(only1))
print("  📁 只在第二个文件中不一致的 ID：", sorted(only2))
