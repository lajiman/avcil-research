
import os

import h5py

visual_pretrained_feature_path = 'datasets_/VGGSound/visual_features.h5'


visual_pretrained_features = h5py.File(visual_pretrained_feature_path)

all_keys = list(visual_pretrained_features.keys())

print(len(all_keys))
print('QST1nskzA7g_000033' in all_keys)
print('zukjR7-Bflc_000009' in all_keys)

# print(visual_pretrained_features.keys())