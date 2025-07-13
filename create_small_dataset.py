import pickle
import random
import os

KEEP_RATIO = 0.1  # Keep ~10% of data
PKL_FILES = [
    'data/nuscenes/nuscenes_infos_temporal_train.pkl',
    'data/nuscenes/nuscenes_infos_temporal_val.pkl',
    'data/nuscenes/nuscenes_infos_temporal_test.pkl'
]

for path in PKL_FILES:
    if not os.path.exists(path):
        print(f"[⚠️] File not found: {path}")
        continue

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Downsize logic
    if isinstance(data, dict) and 'infos' in data:
        original_len = len(data['infos'])
        new_len = int(original_len * KEEP_RATIO)
        data['infos'] = random.sample(data['infos'], new_len)
        print(f"[{os.path.basename(path)}] Reduced infos from {original_len} to {new_len}")
    elif isinstance(data, list):
        original_len = len(data)
        new_len = int(original_len * KEEP_RATIO)
        data = random.sample(data, new_len)
        print(f"[{os.path.basename(path)}] Reduced list from {original_len} to {new_len}")
    else:
        print(f"[❌] Unsupported format in {path}")
        continue

    # Save new file
    new_path = path.replace('.pkl', '_small.pkl')
    with open(new_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[✅] Saved downsized file to {new_path}")