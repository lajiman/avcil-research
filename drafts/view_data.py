import os
import numpy as np
import json
import matplotlib.pyplot as plt

data_root = './data_hard2easy/ordered'
save_dict = './data_hard2easy/ordered'
if not os.path.exists(save_dict):
    os.makedirs(save_dict)
    
all_classId_vid_dict_path = os.path.join(data_root, 'all_classId_vid_dict.npy')
all_id_category_dict_path = os.path.join(data_root, 'all_id_category_dict.npy')
category_encode_dict_path = os.path.join(data_root, 'category_encode_dict.npy')

all_classId_vid_dict = np.load(all_classId_vid_dict_path, allow_pickle=True).item()
all_id_category_dict = np.load(all_id_category_dict_path, allow_pickle=True).item()
category_encode_dict = np.load(category_encode_dict_path, allow_pickle=True).item()


json.dump(all_classId_vid_dict, open(os.path.join(save_dict, 'all_classId_vid_dict.json'), 'w'))
json.dump(all_id_category_dict, open(os.path.join(save_dict, 'all_id_category_dict.json'), 'w'))
json.dump(category_encode_dict, open(os.path.join(save_dict, 'category_encode_dict.json'), 'w'))

# # 打印 all_classId_vid_dict.json 的数据结构
# with open(os.path.join(save_dict, 'all_classId_vid_dict.json'), 'r') as f:
#     data = json.load(f)
#     print('类型:', type(data))
#     print('顶层键数量:', len(data))
#     print('顶层键示例:', list(data.keys())[:5])
#     for k in list(data.keys())[:1]:
#         print(f'键 {k} 的值类型:', type(data[k]))
#         if isinstance(data[k], dict):
#             print(f'  子键数量: {len(data[k])}')
#             print(f'  子键示例: {list(data[k].keys())[:5]}')
#             for subk in list(data[k].keys())[:1]:
#                 print(f'    子键 {subk} 的值类型:', type(data[k][subk]))
#                 print(f'    子键 {subk} 的值示例: {str(data[k][subk])[:100]}')
#         else:
#             print(f'  值示例: {str(data[k])[:100]}')

# # 统计每个set（train、val、test）中每一类的数量
# with open(os.path.join(save_dict, 'all_classId_vid_dict.json'), 'r') as f:
#     data = json.load(f)
#     for split in ['train', 'val', 'test']:
#         print(f'--- {split} ---')
#         if split in data:
#             for class_id, vid_list in data[split].items():
#                 print(f'class {class_id}: {len(vid_list)}')
#         else:
#             print(f'{split} 不存在于数据中')

# 统计并绘制每个set（train、val、test）中每一类的数量直方图
cid_to_label = {v: k for k, v in category_encode_dict.items()}

with open(os.path.join(save_dict, 'all_classId_vid_dict.json'), 'r') as f:
    data = json.load(f)

    for split in ['train', 'val', 'test']:
        if split in data:

            # 保证按 class_id 顺序
            class_ids = sorted([int(k) for k in data[split].keys()])
            class_counts = [len(data[split][str(cid)]) for cid in class_ids]
            class_names = [cid_to_label[cid] for cid in class_ids]

            plt.figure(figsize=(18, 6))
            plt.bar(range(len(class_counts)), class_counts)

            # 🔹 关键：用真实类名做 x 轴
            plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)

            plt.xlabel('Class Name')
            plt.ylabel('Count')
            plt.title(f'{split} set: Number of samples per class')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dict, f'{split}_class_counts.png'))
            plt.close()

            if split == 'train':
                # 排序版本（数量降序）
                sorted_pairs = sorted(
                    zip(class_counts, class_names),
                    key=lambda x: x[0],
                    reverse=True
                )

                sorted_counts = [x[0] for x in sorted_pairs]
                sorted_names = [x[1] for x in sorted_pairs]

                plt.figure(figsize=(18, 6))
                plt.bar(range(len(sorted_counts)), sorted_counts)
                plt.xticks(range(len(sorted_names)), sorted_names, rotation=90, fontsize=6)

                plt.xlabel('Class Name (sorted by count)')
                plt.ylabel('Count')
                plt.title('train set: Number of samples per class (sorted)')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dict, 'train_class_counts_sorted.png'))
                plt.close()

                with open(os.path.join(save_dict, 'stats.txt'), 'w') as f:
                    f.write(f'train set: {len(class_counts)} classes\n')
                    f.write(f'min samples per class: {min(class_counts)}\n')
                    f.write(f'max samples per class: {max(class_counts)}\n')
                    f.write(f'mean samples per class: {np.mean(class_counts):.2f}\n')

        else:
            print(f'{split} 不存在于数据中')