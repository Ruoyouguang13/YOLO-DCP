import torch
from ultralytics import YOLOv10
from torchvision import transforms
from PIL import Image


def preprocess_image(img_path, img_size=640):
    """预处理图像：调整尺寸、转Tensor、增加批次维度"""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整为模型输入尺寸（YOLOv10默认640）
        transforms.ToTensor(),  # 转为浮点型Tensor（范围0-1）
    ])
    img_tensor = transform(img).unsqueeze(0)  # 增加批次维度（形状：[1, 3, H, W]）
    return img_tensor


def register_hooks(model):
    """注册前向钩子，记录每层的层名、模块类型、通道数、空间尺寸"""
    outputs = {}  # 存储结果：{层名: {"type": 模块类型, "channels": 通道数, "spatial_size": (H, W)}}

    def hook(module, input, output):
        # 获取层名（通过遍历named_modules匹配模块对象）
        layer_name = next((name for name, mod in model.named_modules() if mod is module), "Unknown")

        # 提取关键信息
        module_type = module.__class__.__name__  # 模块类型（如Conv2d、BatchNorm2d）
        output_shape = output.shape  # 输出形状：[batch, channels, H, W]

        # 分解形状信息（假设输出是4维张量，符合图像特征图格式）
        batch_size = output_shape[0]  # 批次大小（推理时通常为1）
        channels = output_shape[1]  # 通道数（关键特征维度）
        height, width = output_shape[2], output_shape[3]  # 空间尺寸（高×宽）

        # 存储结果（字典格式更易扩展）
        outputs[layer_name] = {
            "type": module_type,
            "channels": channels,
            "spatial_size": (height, width)
        }

        # 打印易读的信息（重点突出通道数和空间尺寸）
        print(f"层名: {layer_name}")
        print(f"  模块类型: {module_type}")
        print(f"  通道数: {channels}")
        print(f"  空间尺寸: {height}×{width}\n")  # 空格分隔更清晰

    # 为模型的每一层注册前向钩子
    for name, module in model.named_modules():
        module.register_forward_hook(hook)

    return outputs


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r"D:\yolov10\runs\detect\train189\weights\best.pt"  # 替换为你的模型路径
    img_path = r"E:\paper\orign\DJI_0040.JPG"  # 替换为你的图像路径

    # 加载模型并配置
    model = YOLOv10(model_path).model  # 直接访问YOLOv10的内部nn.Module结构
    model.to(device).eval()  # 移动到设备并设为推理模式

    # 预处理图像
    img_tensor = preprocess_image(img_path).to(device)

    # 注册钩子并获取各层信息
    layer_info = register_hooks(model)

    # 推理（触发前向传播，钩子自动记录信息）
    with torch.no_grad():  # 禁用梯度计算加速推理
        _ = model(img_tensor)

    # 可选：保存结果到文件（如需后续分析）
    # import json
    # with open("layer_info.json", "w") as f:
    #     json.dump(layer_info, f, indent=2)
    print("所有层信息已打印，结果存储在layer_info变量中（或layer_info.json文件）。")