import tensorflow as tf
import os

# 关键修改点 1: 导入 scenario_pb2 而不是 dataset_pb2
from waymo_open_dataset.protos import scenario_pb2

def load_dataset(tfrecord_path):
    """
    读取并解析 Waymo E2E Dataset (Scenario 格式)
    """
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"找不到文件: {tfrecord_path}")

    # E2E 数据集通常是未压缩的，所以 compression_type=''
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for data in dataset:
        # 关键修改点 2: 使用 Scenario 原型来解析
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        yield scenario

if __name__ == "__main__":
    # 请确认这个路径是你 D 盘里的真实路径
    TEST_FILE = '/mnt/d/Datasets/WOD_E2E_Camera_v1/val/val_202504211843.tfrecord-00015-of-00093'
    
    print(f"🚀 正在尝试以 [Scenario] 格式读取: {os.path.basename(TEST_FILE)}")
    
    try:
        generator = load_dataset(TEST_FILE)
        first_scenario = next(generator)
        
        print("-" * 40)
        print(f"✅ 解析成功！")
        # Scenario 格式特有的字段
        print(f"🎬 Scenario ID: {first_scenario.scenario_id}")
        print(f"⏱️  时间步数量 (Timestamps): {len(first_scenario.timestamps_seconds)}")
        print(f"🚗 包含的轨迹 (Tracks): {len(first_scenario.tracks)}")
        print(f"🛣️  包含的地图特征 (Map Features): {len(first_scenario.map_features)}")
        
        # 检查有没有图片 ID (E2E 数据集的特征)
        # 注意: 具体的图片数据可能并不直接存在这里，而是通过 ID 关联，或者在特定字段中
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print("💡 提示: 如果依然报错，请检查是否需要安装 waymo-open-dataset-tf-2-11-0 或更高版本")