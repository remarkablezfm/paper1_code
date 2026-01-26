import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

def calculate_trajectory_safety(traj) -> float:
    """
    评估单条轨迹的安全分数 (0-1范围)
    参考: Failure Prediction at Runtime for Generative Robot Policies
    """
    safety_score = 1.0
    
    # 1. 位置平滑性分析 (曲率)
    if hasattr(traj, 'pos_x') and hasattr(traj, 'pos_y'):
        try:
            xs = list(traj.pos_x)
            ys = list(traj.pos_y)
            if len(xs) > 2:
                # 计算轨迹曲率
                curvatures = []
                for i in range(1, len(xs)-1):
                    # 计算三点形成的局部曲率
                    x1, y1 = xs[i-1], ys[i-1]
                    x2, y2 = xs[i], ys[i]
                    x3, y3 = xs[i+1], ys[i+1]
                    
                    # 向量计算
                    v1 = (x2-x1, y2-y1)
                    v2 = (x3-x2, y3-y2)
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if mag1 > 0.1 and mag2 > 0.1:
                        curvature = abs(cross) / (mag1 * mag2)
                        curvatures.append(curvature)
                
                if curvatures:
                    max_curvature = max(curvatures)
                    avg_curvature = sum(curvatures) / len(curvatures)
                    # 高曲率显著降低安全性
                    safety_score *= max(0.1, 1.0 - max_curvature * 15)
                    safety_score *= max(0.2, 1.0 - avg_curvature * 10)
        except Exception as e:
            print(f"  ⚠️ 曲率计算错误: {str(e)}")
    
    # 2. 速度平滑性
    if hasattr(traj, 'speed'):
        try:
            speeds = list(traj.speed)
            if len(speeds) > 1:
                # 计算加速度变化
                accels = [abs(speeds[i+1] - speeds[i]) for i in range(len(speeds)-1)]
                max_accel = max(accels) if accels else 0
                # 高加速度降低安全性
                safety_score *= max(0.2, 1.0 - max_accel * 0.5)
        except:
            pass
    
    return max(0.0, min(1.0, safety_score))

def calculate_trajectory_diversity(trajectories) -> float:
    """
    计算轨迹多样性分数 (0-1范围)
    参考: Failure Resilience in Learned Visual Navigation Control
    """
    if len(trajectories) < 2:
        return 0.0
    
    # 1. 计算终点分散度
    endpoints = []
    for traj in trajectories:
        if hasattr(traj, 'pos_x') and hasattr(traj, 'pos_y'):
            try:
                xs = list(traj.pos_x)
                ys = list(traj.pos_y)
                if xs and ys and len(xs) > 0:
                    endpoints.append((xs[-1], ys[-1]))
            except:
                continue
    
    if len(endpoints) < 2:
        return 0.0
    
    # 2. 计算平均成对距离
    total_dist = 0
    count = 0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            dx = endpoints[i][0] - endpoints[j][0]
            dy = endpoints[i][1] - endpoints[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            total_dist += dist
            count += 1
    
    avg_dist = total_dist / count if count > 0 else 0
    # 归一化 (假设100米为最大有意义距离)
    return min(1.0, avg_dist / 100.0)

def analyze_single_trajectory(traj, index: int = 0, prefix: str = "") -> Dict[str, Any]:
    """
    分析单条轨迹的关键特性
    """
    if traj is None:
        print(f"{prefix}⚠️ 轨迹为空")
        return {"error": "轨迹为空", "index": index}
    
    analysis = {
        "index": index,
        "safety_score": 0.0,
        "length": 0,
        "fields": {}
    }
    
    # 1. 检查基本字段
    fields_to_check = ['pos_x', 'pos_y', 'pos_z', 'speed', 'accel_x', 'accel_y', 'preference_score']
    for field in fields_to_check:
        field_info = {"exists": False}
        
        if hasattr(traj, field):
            field_info["exists"] = True
            try:
                value = getattr(traj, field)
                if hasattr(value, '__len__'):  # 重复字段
                    field_info["length"] = len(value)
                    if len(value) > 0:
                        field_info["sample"] = list(value)[:2]
                else:  # 标量值
                    field_info["value"] = float(value)
            except Exception as e:
                field_info["error"] = str(e)
        
        analysis["fields"][field] = field_info
    
    # 2. 轨迹长度
    if 'pos_x' in analysis["fields"] and "length" in analysis["fields"]['pos_x']:
        analysis["length"] = analysis["fields"]['pos_x']["length"]
    
    # 3. 安全分数
    analysis["safety_score"] = calculate_trajectory_safety(traj)
    print(f"{prefix}🛡️ 轨迹 {index} 安全分数: {analysis['safety_score']:.2f}")
    
    # 4. 可视化建议
    if analysis["length"] > 0:
        print(f"{prefix}📊 轨迹 {index} 长度: {analysis['length']} 点")
    
    return analysis

def analyze_preference_trajectories(e2e_record) -> Optional[Dict[str, Any]]:
    """
    全面分析 preference_trajectories 字段
    返回: 包含分析结果的字典，如果字段不存在则返回None
    """
    # 1. 获取 preference_trajectories 字段
    prefs = getattr(e2e_record, 'preference_trajectories', None)
    if prefs is None:
        print("  ⚠️ Preference trajectories 字段不存在")
        return None
    
    # 2. 获取轨迹列表
    trajectories = getattr(prefs, 'trajectories', None)
    if not trajectories or not hasattr(trajectories, '__len__'):
        print("  ⚠️ 无法访问轨迹列表")
        return None
    
    print(f"  📊 轨迹总数: {len(trajectories)}")
    result = {
        "num_trajectories": len(trajectories),
        "trajectories": [],
        "recovery_capability": "未知",
        "diversity_score": 0.0
    }
    
    # 3. 分析每条轨迹 (限制分析前5条)
    for i in range(min(5, len(trajectories))):
        traj = trajectories[i]
        print(f"    🛣️ 分析轨迹 {i}:")
        traj_analysis = analyze_single_trajectory(traj, i, "      ")
        result["trajectories"].append(traj_analysis)
    
    # 4. 评估故障恢复能力
    safety_scores = [t.get("safety_score", 0) for t in result["trajectories"] if "safety_score" in t]
    if safety_scores:
        # 计算首选轨迹(0号)与最佳备选轨迹的比较
        primary_safety = safety_scores[0] if len(safety_scores) > 0 else 0
        backup_safeties = safety_scores[1:] if len(safety_scores) > 1 else []
        
        backup_capability = "无备选轨迹"
        if backup_safeties:
            best_backup = max(backup_safeties)
            if best_backup > primary_safety * 0.7:
                backup_capability = "高 - 有高质量备选轨迹"
            elif best_backup > primary_safety * 0.4:
                backup_capability = "中 - 有中等质量备选轨迹"
            else:
                backup_capability = "低 - 备选轨迹质量不足"
        
        result["recovery_capability"] = backup_capability
        print(f"  🛡️ 故障恢复能力: {backup_capability}")
    
    # 5. 轨迹多样性分析
    diversity = calculate_trajectory_diversity(trajectories[:5])
    result["diversity_score"] = diversity
    print(f"  🌈 轨迹多样性分数: {diversity:.2f}/1.0")
    
    # 6. 生成恢复建议
    generate_recovery_suggestions(trajectories, safety_scores, diversity)
    
    return result

def generate_recovery_suggestions(trajectories, safety_scores, diversity):
    """
    生成基于轨迹分析的故障恢复建议
    参考: Robot Failure Recovery Using Vision-Language Models
    """
    print("\n  💡 故障恢复建议:")
    
    if not trajectories:
        print("    ❌ 无可用轨迹，建议请求人类操作员干预")
        return
    
    num_trajectories = len(trajectories)
    if num_trajectories == 0:
        print("    ❌ 无备选轨迹，单点故障风险极高")
        return
    
    # 1. 首选轨迹评估
    print("\n  🔍 首选轨迹评估:")
    if safety_scores and safety_scores[0] < 0.3:
        print("    🚨 首选轨迹安全性极低，建议主动切换到备选轨迹")
    elif safety_scores and safety_scores[0] < 0.6:
        print("    ⚠️ 首选轨迹安全性中等，建议监控并准备切换")
    else:
        print("    ✅ 首选轨迹安全性高，可继续执行")
    
    # 2. 备选轨迹质量
    print("\n  🔍 备选轨迹评估:")
    if num_trajectories > 1:
        best_backup_idx = 1
        best_backup_score = 0
        
        for i in range(1, min(num_trajectories, 5)):
            score = safety_scores[i] if i < len(safety_scores) else 0
            if score > best_backup_score:
                best_backup_score = score
                best_backup_idx = i
        
        print(f"    🔄 最佳备选轨迹: 轨迹 {best_backup_idx} (安全分: {best_backup_score:.2f})")
        
        if best_backup_score > 0.7:
            print("    ✅ 高质量备选轨迹，可作为无缝故障恢复方案")
        elif best_backup_score > 0.4:
            print("    🟡 中等质量备选轨迹，需要谨慎切换")
        else:
            print("    ⚠️ 低质量备选轨迹，建议结合人工干预")
    
    # 3. 多样性建议
    print("\n  🔍 轨迹多样性评估:")
    if diversity > 0.6:
        print("    🌈 轨迹多样性高，系统可适应多种场景变化")
    elif diversity > 0.3:
        print("    🟡 轨迹多样性中等，覆盖部分异常情况")
    else:
        print("    🔴 轨迹多样性低，系统弹性有限，建议增加轨迹生成策略")
    
    # 4. 具体恢复策略
    print("\n  🛠️ 具体恢复策略建议:")
    print("    • 实时监控首选轨迹的安全分数，低于0.4阈值时自动切换")
    print("    • 为高风险场景(如行人附近)预加载多条安全备选轨迹")
    print("    • 实现轨迹切换的平滑过渡机制，避免突然动作")
    print("    • 在轨迹多样性低的区域，增加人类操作员监督")

def visualize_preference_trajectories(trajectories_analysis, output_dir: str = "visualization_results"):
    """
    可视化 preference trajectories 分析结果
    """
    if not trajectories_analysis or not trajectories_analysis["trajectories"]:
        print("⚠️ 无轨迹数据可供可视化")
        return
    
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 收集轨迹数据
    trajectory_data = []
    for traj_analysis in trajectories_analysis["trajectories"]:
        if "fields" in traj_analysis and "pos_x" in traj_analysis["fields"] and "pos_y" in traj_analysis["fields"]:
            if traj_analysis["fields"]["pos_x"].get("exists") and traj_analysis["fields"]["pos_y"].get("exists"):
                try:
                    xs = traj_analysis["fields"]["pos_x"].get("sample", [])
                    ys = traj_analysis["fields"]["pos_y"].get("sample", [])
                    if xs and ys and len(xs) == len(ys):
                        trajectory_data.append({
                            "index": traj_analysis["index"],
                            "xs": xs,
                            "ys": ys,
                            "safety_score": traj_analysis["safety_score"]
                        })
                except Exception as e:
                    print(f"⚠️ 轨迹数据提取错误: {str(e)}")
    
    if not trajectory_data:
        print("⚠️ 无有效轨迹数据可供可视化")
        return
    
    # 3. 创建轨迹可视化
    plt.figure(figsize=(12, 10))
    
    # 绘制每条轨迹
    for data in trajectory_data:
        xs = data["xs"]
        ys = data["ys"]
        safety = data["safety_score"]
        index = data["index"]
        
        # 根据安全分数选择颜色
        if safety > 0.7:
            color = 'green'
        elif safety > 0.4:
            color = 'orange'
        else:
            color = 'red'
        
        # 根据安全分数设置透明度
        alpha = 0.3 + 0.7 * safety
        
        plt.plot(xs, ys, color=color, linewidth=2, alpha=alpha, 
                 label=f'轨迹 {index} (安全: {safety:.2f})')
        
        # 标记起点和终点
        plt.scatter([xs[0]], [ys[0]], color=color, s=100, marker='o')
        plt.scatter([xs[-1]], [ys[-1]], color=color, s=100, marker='x')
    
    plt.title('Preference Trajectories 分析')
    plt.xlabel('X 位置')
    plt.ylabel('Y 位置')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    
    # 4. 保存可视化结果
    viz_path = os.path.join(output_dir, "preference_trajectories.png")
    plt.savefig(viz_path)
    plt.close()
    
    print(f"✅ 轨迹可视化已保存到: {viz_path}")
    
    # 5. 创建安全分数分析图
    plt.figure(figsize=(10, 6))
    
    indices = [t["index"] for t in trajectory_data]
    safety_scores = [t["safety_score"] for t in trajectory_data]
    
    bars = plt.bar(indices, safety_scores, color=['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in safety_scores])
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.3, label='高质量阈值')
    plt.axhline(y=0.4, color='y', linestyle='--', alpha=0.3, label='中等质量阈值')
    
    plt.title('轨迹安全分数分析')
    plt.xlabel('轨迹索引')
    plt.ylabel('安全分数')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safety_path = os.path.join(output_dir, "trajectory_safety_scores.png")
    plt.savefig(safety_path)
    plt.close()
    
    print(f"✅ 安全分数可视化已保存到: {safety_path}")
    
    # 6. 保存分析摘要
    summary = {
        "total_trajectories": trajectories_analysis["num_trajectories"],
        "analyzed_trajectories": len(trajectory_data),
        "recovery_capability": trajectories_analysis["recovery_capability"],
        "diversity_score": trajectories_analysis["diversity_score"],
        "trajectory_safety_scores": {t["index"]: t["safety_score"] for t in trajectory_data}
    }
    
    with open(os.path.join(output_dir, "analysis_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ 分析摘要已保存到: {os.path.join(output_dir, 'analysis_summary.json')}")

def main_preference_analysis(e2e_record, output_dir: str = "preference_analysis"):
    """
    主函数：执行 preference_trajectories 分析并生成可视化
    """
    print("\n" + "="*60)
    print("🛣️ PREFERENCE TRAJECTORIES 深度分析")
    print("="*60)
    
    # 1. 执行分析
    prefs_analysis = analyze_preference_trajectories(e2e_record)
    
    if prefs_analysis:
        print("\n✅ Preference Trajectories 分析完成!")
        
        # 2. 生成可视化
        print("\n📈 生成可视化结果...")
        visualize_preference_trajectories(prefs_analysis, output_dir)
        
        print(f"\n💾 所有分析结果已保存到: {output_dir}")
        return prefs_analysis
    else:
        print("\n❌ 无法分析 Preference Trajectories")
        return None

# 使用示例:
"""
# 假设 e2e_record 是从WOD-E2E数据集中解析出的记录
prefs_result = main_preference_analysis(e2e_record)

if prefs_result:
    # 获取恢复能力评估
    recovery_capability = prefs_result["recovery_capability"]
    print(f"系统恢复能力: {recovery_capability}")
    
    # 获取所有轨迹的安全分数
    safety_scores = [t["safety_score"] for t in prefs_result["trajectories"]]
    print(f"轨迹安全分数: {safety_scores}")
"""