import cv2

# 读取图片
image_path = "006957.jpg"
image = cv2.imread(image_path)

# 读取标签信息
txt_path = "/home/lee/ws/src/yolov5_2/runs/detect/scd_new/lit_fus/labels/006957.txt"
with open(txt_path, "r") as f:
    lines = f.readlines()

# 类别到颜色的映射
category_colors = {
    0: (0, 0, 255),  # 红色（人）
    1: (192, 203, 255),
    2: (18, 153, 255),  # 绿色（车辆）
    6: (0, 255, 0)   # 蓝色（动物）
}

category_names = {
    0: 'Pedestrain',
    1: 'Cyclist',
    2: 'Car',
    6: 'Traffic_sign'
}

for line in lines:
    # 解析标签信息
    label = line.strip().split()
    category = int(label[0])
    x_center, y_center, width, height = map(float, label[1:5])
    confidence = float(label[-1])

    # 计算矩形框的坐标
    x1 = int((x_center - width / 2) * image.shape[1])
    y1 = int((y_center - height / 2) * image.shape[0])
    x2 = int((x_center + width / 2) * image.shape[1])
    y2 = int((y_center + height / 2) * image.shape[0])

    # 根据类别选择颜色
    color = category_colors.get(category, (255, 255, 255))  # 默认为白色

    # 绘制矩形框
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 添加文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    name = category_names.get(category, '')
    font_color = (255, 255, 255)  # 白色
    text_size = cv2.getTextSize(f"{confidence:.2f}", font, font_scale, thickness)[0]
    cv2.rectangle(image, (x1, y1 - text_size[1] - 6), (x1 + int(0.2*len(name)+3)*text_size[0], y1), color, -1)
    cv2.putText(image, name +':'+ f"{confidence:.2f}", (x1, y1 - 10), font, font_scale, font_color, thickness, cv2.LINE_AA)

# 保存图像
output_path = "1.jpg"
cv2.imwrite(output_path, image)
print(f"生成的带有标签的图像已保存到 {output_path}")
