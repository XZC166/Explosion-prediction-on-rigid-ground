import os
import re

def batch_merge_max_pressure(input_folder, output_folder):
    """
    自动识别 input_folder 中的所有 i 值，
    并将对应的 valuei_1_p, valuei_2_p, valuei_3_p 合并取最大值。
    """
    # 1. 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    # 2. 扫描输入文件夹，找出所有的 i
    # 使用正则表达式匹配文件名中的数字 i
    files = os.listdir(input_folder)
    i_values = set()
    for f in files:
        match = re.match(r"value(\d+)_", f)
        if match:
            i_values.add(match.group(1))
    
    if not i_values:
        print("未在文件夹中提取到符合格式的文件。")
        return

    print(f"检测到需要处理的 i 值: {sorted([int(i) for i in i_values])}")

    # 3. 循环处理每一个 i
    for i in sorted(i_values):
        max_data = {}
        filenames = [f"value{i}_1_p", f"value{i}_2_p", f"value{i}_3_p"]
        found_any_file = False

        for fname in filenames:
            filepath = os.path.join(input_folder, fname)
            if os.path.exists(filepath):
                found_any_file = True
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        
                        # 提取坐标和压力值
                        try:
                            x, y, z, p = map(float, parts)
                            coord = (x, y, z)
                            # 取最大值逻辑
                            if coord not in max_data or p > max_data[coord]:
                                max_data[coord] = p
                        except ValueError:
                            continue # 跳过无法解析的行

        # 4. 写入合并后的文件
        if found_any_file:
            output_filename = f"value{i}"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'w') as f_out:
                for (x, y, z), p_max in max_data.items():
                    f_out.write(f"{x} {y} {z} {p_max}\n")
            print(f"成功合并: {output_filename}")
        else:
            print(f"跳过 i={i}: 未找到对应的子文件。")

# --- 调用脚本 ---
if __name__ == "__main__":
    # 在这里输入你的文件夹名称
    batch_merge_max_pressure("collect2", "collect_overpressure")