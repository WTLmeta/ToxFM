import os


def find_fold_files(base_dir):
    """Find all fold train and validation files"""
    print(f"\n在目录 {base_dir} 中查找 fold 文件...")

    if not os.path.exists(base_dir):
        print(f"目录不存在：{base_dir}")
        return []

    fold_files = []

    # Try multiple naming patterns
    patterns = [
        # fold_1/train.csv, fold_1/test.csv
        lambda i: (os.path.join(f"fold_{i}", "train.csv"), os.path.join(f"fold_{i}", "test.csv")),
        # fold_1/train.csv, fold_1/val.csv
        lambda i: (os.path.join(f"fold_{i}", "train.csv"), os.path.join(f"fold_{i}", "val.csv")),
        # fold1_train.csv, fold1_val.csv
        lambda i: (f"fold{i}_train.csv", f"fold{i}_val.csv"),
        # fold_1_train.csv, fold_1_val.csv
        lambda i: (f"fold_{i}_train.csv", f"fold_{i}_val.csv"),
        # train_fold_1.csv, val_fold_1.csv
        lambda i: (f"train_fold_{i}.csv", f"val_fold_{i}.csv"),
        # train_1.csv, val_1.csv
        lambda i: (f"train_{i}.csv", f"val_{i}.csv"),
    ]

    for i in range(1, 6):  # 5 folds
        found = False
        for pattern_func in patterns:
            train_pattern, val_pattern = pattern_func(i)

            train_file = os.path.join(base_dir, train_pattern)
            val_file = os.path.join(base_dir, val_pattern)

            if os.path.exists(train_file) and os.path.exists(val_file):
                fold_files.append((train_file, val_file))
                print(f"  ✓ Fold {i}: {train_pattern} + {val_pattern}")
                found = True
                break

        if not found:
            print(f"  ✗ Fold {i}: 未找到文件")

    return fold_files