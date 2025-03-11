import os
import numpy as np
import time
import codecs
import matplotlib.pyplot as plt
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

# 设置基本参数
target_size = [3, 512, 512]
mean_rgb = [127.5, 127.5, 127.5]
goal_dir = "../goal1"  # 存放待分类图片的文件夹
label_file = "../data/data1/label_list.txt"  # 标签文件路径
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "../code/freeze-model"  # 已训练模型路径
paddle.enable_static()
# 加载训练好的冻结模型
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
    dirname=save_freeze_dir, executor=exe)


# 加载label_list.txt并创建类别映射字典
def load_label_dict(label_file):
    label_dict = {}
    with codecs.open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            label_dict[int(parts[0])] = parts[1]  # 将类别编号和类别名称存入字典
    return label_dict


label_dict = load_label_dict(label_file)


# 图像预处理函数
def resize_img(img, target_size):
    ret = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return ret


def read_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img, target_size)
    img = np.array(img).astype('float32')
    img -= mean_rgb
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    img = img[np.newaxis, :]
    return img


# 推理函数
def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    return np.argmax(label)  # 返回预测的类别索引


# 显示多张图片并绘制预测结果
def show_images_with_results(img_paths, img_names, predicted_labels):
    # 创建绘图窗口
    n = len(img_paths)
    cols = 4  # 每行显示4张图像
    rows = (n // cols) + (1 if n % cols != 0 else 0)  # 计算需要的行数
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
    axes = axes.flatten()  # 将 axes 展平为一维数组，便于遍历

    for i, (img_path, img_name, predicted_label) in enumerate(zip(img_paths, img_names, predicted_labels)):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')  # 关闭坐标轴显示
        axes[i].set_title(f"{img_name} - Pred: {predicted_label}", fontsize=12)

    # 调整布局，使得图片不重叠
    fig.tight_layout()
    plt.show()


# 对目标文件夹中的每张图片进行分类
def classify_images_in_folder():
    total_files = os.listdir(goal_dir)
    total_count = 0
    right_count = 0
    img_paths = []  # 存储图片路径
    img_names = []  # 存储图片名称
    predicted_labels = []  # 存储预测标签

    print(f"Found {len(total_files)} images in '{goal_dir}' directory.\n")

    for img_name in total_files:
        img_path = os.path.join(goal_dir, img_name)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 推理
            result = infer(img_path)

            # 获取预测标签
            predicted_label = label_dict.get(result, "Unknown")

            # 假设文件名格式是：`真实标签_序号.jpg`，提取真实标签
            true_label = img_name.split('_')[0]

            # 输出预测信息
            print(f"Image: {img_name}, Predicted Class: {predicted_label}, True Label: {true_label}")
            total_count += 1

            # 保存图片路径、名称和预测标签
            img_paths.append(img_path)
            img_names.append(img_name)  # 保存图片名称
            predicted_labels.append(predicted_label)

            # 判断是否预测正确
            if true_label == predicted_label:
                right_count += 1

    # 显示所有图片，并在标题中加上图片名称和预测类别
    show_images_with_results(img_paths, img_names, predicted_labels)

    # 输出分类结果统计
    accuracy = (right_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nClassification completed. Correct: {right_count}/{total_count}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    classify_images_in_folder()
