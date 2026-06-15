import numpy as np
import glob
import os

directory = "results_metric"
txt_files = glob.glob(os.path.join(directory, "*.txt"))

results = []

for file_path in txt_files:

    file_name = os.path.basename(file_path)

    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split()
            dataset.append(line)
    dataset = np.array(dataset)

    if dataset.shape[0] < 2:
        print(f"跳过空文件: {file_name}")
        continue

    try:

        f1 = dataset[1:, 2].astype(float)
        pre = dataset[1:, 3].astype(float)
        rec = dataset[1:, 4].astype(float)
        auc = dataset[1:, -2].astype(float)
        ap = dataset[1:, -1].astype(float)
        train_time = dataset[1:, -6].astype(float)
        epoch_time = dataset[1:, -5].astype(float)
        test_time = dataset[1:, -4].astype(float)

        f1_mean = np.mean(f1)
        pre_mean = np.mean(pre)
        rec_mean = np.mean(rec)
        f1_star = 2 * (pre_mean * rec_mean) / (pre_mean + rec_mean) if (pre_mean + rec_mean) > 0 else 0
        auc_mean = np.mean(auc)
        ap_mean = np.mean(ap)
        train_time_mean = np.mean(train_time)
        epoch_time_mean = np.mean(epoch_time)
        test_time_mean = np.mean(test_time)

        results.append([
            file_name,
            f1_mean, pre_mean, rec_mean, f1_star,
            auc_mean, ap_mean,
            train_time_mean, epoch_time_mean, test_time_mean
        ])

    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {str(e)}")

if results:

    print("\n{:<60} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12} {:<12} {:<12}".format(
        "File Name", "F1", "Pre", "Rec", "F1*", "AUC", "AP",
        "Train Time", "Epoch Time", "Test Time"
    ))
    print("-" * 130)

    for row in results:
        print("{:<60} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(*row))

    output_file = os.path.join(directory, "summary_results.csv")
    with open(output_file, 'w') as f:

        f.write("File Name,F1,Precision,Recall,F1*,AUC,AP,Train Time,Epoch Time,Test Time\n")

        for row in results:
            f.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},"
                    f"{row[5]:.4f},{row[6]:.4f},{row[7]:.4f},{row[8]:.4f},{row[9]:.4f}\n")

    print(f"\n结果已保存至: {output_file}")
else:
    print("未找到有效结果")
