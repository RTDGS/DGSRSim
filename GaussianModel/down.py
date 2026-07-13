import argparse
from PIL import Image
import os


def resize_image(input_path, output_path, target_size=(1500, 1000)):
    """调整单张图片尺寸"""
    try:
        with Image.open(input_path) as img:
            resized_img = img.resize(target_size, Image.LANCZOS)
            resized_img.save(output_path)
            print(f"已处理: {input_path} -> {output_path}")
    except Exception as e:
        print(f"处理失败 {input_path}: {str(e)}")


def batch_resize_images(input_dir, output_dir, target_size=(1500, 1000)):
    """批量处理目录下的所有图片"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件是否为图片
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_image(input_path, output_path, target_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory.")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output image directory")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入目录不存在 - {input_dir}")
    else:
        batch_resize_images(input_dir, output_dir, (args.width, args.height))
        print("批量处理完成！")
