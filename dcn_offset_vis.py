import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics.nn.modules.block import DCNv2

# 设置保存路径
SAVE_DIR = r"D:\yolov10\DCN_view"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 预处理图片函数
def preprocess_image(img_path, img_size=640):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # 如有归一化需求可添加 Normalize
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# 2. 偏移热力图可视化
def visualize_offset_heatmap(offset_tensor, kernel_size=3):
    offset_np = offset_tensor.cpu().numpy()[0]
    offset_x = offset_np[0::2, :, :]
    offset_y = offset_np[1::2, :, :]
    offset_magnitude = np.sqrt(offset_x**2 + offset_y**2)
    heatmap = np.mean(offset_magnitude, axis=0)
    plt.figure(figsize=(8,6))
    plt.title('DCNv2 Offset Magnitude Heatmap')
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    save_path = os.path.join(SAVE_DIR, "offset_heatmap.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

# 3. 单点采样偏移箭头可视化
def visualize_sampling_offsets(offset_tensor, kernel_size=3, pos=(10,10)):
    offset_np = offset_tensor.cpu().numpy()[0]
    h, w = pos
    grid_range = np.arange(-(kernel_size//2), kernel_size//2 + 1)
    grid_x, grid_y = np.meshgrid(grid_range, grid_range)
    base_points = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
    offset_x = offset_np[0::2, h, w]
    offset_y = offset_np[1::2, h, w]
    plt.figure(figsize=(6,6))
    plt.title(f'DCNv2 Sampling Offsets at position {pos}')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().invert_yaxis()
    plt.grid(True)
    for i in range(kernel_size*kernel_size):
        plt.arrow(base_points[i,0], base_points[i,1],
                  offset_x[i], offset_y[i],
                  head_width=0.05, head_length=0.1, fc='r', ec='r')
        plt.plot(base_points[i,0], base_points[i,1], 'bo')

    import os

    if not os.path.exists(SAVE_DIR):
        print(f"[INFO] SAVE_DIR 不存在，正在创建: {SAVE_DIR}")
        os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "single_offset_arrow.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

# 4. 全层所有采样点在输入图上的坐标获取
def get_all_sampling_points_on_input(offset_tensor, stride=8, kernel_size=3):
    offset_np = offset_tensor.cpu().numpy()[0]
    H, W = offset_tensor.shape[2], offset_tensor.shape[3]
    k = kernel_size

    grid_range = np.arange(-(k//2), k//2 + 1)
    grid_x, grid_y = np.meshgrid(grid_range, grid_range)
    base_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)  # (k*k, 2)

    orig_pts_list = []
    offset_pts_list = []

    for h in range(H):
        for w in range(W):
            orig_pts = base_points + np.array([w, h])
            offset_x = offset_np[0::2, h, w]
            offset_y = offset_np[1::2, h, w]
            offset_pts = orig_pts + np.stack([offset_x, offset_y], axis=1)

            orig_pts_img = orig_pts * stride
            offset_pts_img = offset_pts * stride

            orig_pts_list.append(orig_pts_img)
            offset_pts_list.append(offset_pts_img)

    orig_pts_all = np.concatenate(orig_pts_list, axis=0)
    offset_pts_all = np.concatenate(offset_pts_list, axis=0)

    return orig_pts_all, offset_pts_all

# 5. 抽样部分采样点可视化
def visualize_sampling_points_subset(orig_pts_all, offset_pts_all, input_img_size=640, sample_num=1000):
    total_points = orig_pts_all.shape[0]
    sample_num = min(sample_num, total_points)
    idx = np.random.choice(total_points, sample_num, replace=False)

    sampled_orig = orig_pts_all[idx]
    sampled_offset = offset_pts_all[idx]

    plt.figure(figsize=(10,10))
    plt.title(f'Sampled {sample_num} Sampling Points on Input Image')
    plt.xlim(0, input_img_size)
    plt.ylim(input_img_size, 0)
    plt.grid(True)

    plt.scatter(sampled_orig[:,0], sampled_orig[:,1], c='b', label='Original Sampling Points', s=1)
    plt.scatter(sampled_offset[:,0], sampled_offset[:,1], c='r', label='Offset Sampling Points', s=1)
    plt.legend()
    plt.show()
    plt.close()


#单一点数绘制
def visualize_sampling_points_only(orig_pts_all, color='blue', title='Original Sampling Points', save_path=None, input_img_size=640):
    """
    单独绘制一类采样点（原始或偏移）在输入图上。
    参数：
    - orig_pts_all: 所有点坐标 (N, 2)
    - color: 'blue' 或 'red'
    - title: 图标题
    - save_path: 保存路径（为None时显示，不保存）
    """
    plt.figure(figsize=(12,12))
    plt.title(title)
    plt.xlim(0, input_img_size)
    plt.ylim(input_img_size, 0)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.scatter(orig_pts_all[:,0], orig_pts_all[:,1], c=color, s=0.5)
    if save_path:
        save_path_full = os.path.join(SAVE_DIR, save_path)
        plt.savefig(save_path_full, dpi=600, bbox_inches='tight')
        print(f'Saved: {save_path_full}')
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

# 全部采样点可视化
def visualize_all_sampling_points(orig_pts_all, offset_pts_all, input_img_size=640, show_arrows=True):
    """
    显示全部原始采样点和偏移采样点在输入图片坐标系下的位置。

    - orig_pts_all: (N, 2) 所有原始采样点坐标
    - offset_pts_all: (N, 2) 所有偏移采样点坐标
    - show_arrows: 是否显示箭头（慎用，点数太多时建议为False）
    """
    plt.figure(figsize=(12, 12))
    plt.title(f'All DCNv2 Sampling Points on Input Image')
    plt.xlim(0, input_img_size)
    plt.ylim(input_img_size, 0)
    plt.axis('off')
    plt.gca().set_aspect('equal')

    # 散点显示所有原始/偏移采样点
    plt.scatter(orig_pts_all[:, 0], orig_pts_all[:, 1], c='blue', s=0.5, label='Original Sampling Points')
    plt.scatter(offset_pts_all[:, 0], offset_pts_all[:, 1], c='red', s=0.5, label='Offset Sampling Points')

    if show_arrows:
        for i in range(len(orig_pts_all)):
            plt.arrow(orig_pts_all[i, 0], orig_pts_all[i, 1],
                      offset_pts_all[i, 0] - orig_pts_all[i, 0],
                      offset_pts_all[i, 1] - orig_pts_all[i, 1],
                      head_width=0.5, head_length=1.0, fc='gray', ec='gray', alpha=0.3)

    plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "all_dcn_sampling_points.png")
    plt.savefig(save_path, dpi=600)
    print(f'Saved: {save_path}')
    plt.show()
    plt.close()

#绘制密度图
def visualize_sampling_density_fast(points, title='Sampling Density', cmap='Reds', img_size=640, bins=128, save_name=None):
    plt.figure(figsize=(10, 10))
    plt.hist2d(points[:, 0], points[:, 1], bins=bins, range=[[0, img_size], [0, img_size]], cmap=cmap)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    if save_name:
        save_path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(save_path, dpi=600)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def visualize_points_on_image(image_path, points, color='red', title='Offset Points on Image', save_path=SAVE_DIR, point_size=1):
    """
    在原始图像上绘制采样点（原始/偏移都可）。
    参数：
        image_path: 原图路径
        points: (N, 2) 点坐标（必须是输入图尺度上的）
        color: 点颜色，如 'red'
        title: 图标题
        save_path: 如需保存，提供路径
        point_size: 点大小
    """
    # 加载原图（确保和采样点对应）
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))  # 必须与输入图尺寸一致

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], c=color, s=point_size, alpha=0.6)
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved to: {save_path}")
        plt.show()
        plt.close()
    else:
        plt.imshow(img)
        plt.scatter(points[:, 0], points[:, 1], c='red', s=0.5)
        plt.gca().invert_yaxis()  # 反转 y 坐标，确保图像和点方向一致



#可视化一个区域内多个位置的 DCNv2 动态采样偏移箭头。
# def visualize_region_sampling_offsets(offset_tensor, kernel_size=3, region_top_left=(20, 20), region_size=(5, 5)):
#     """
#     可视化一个区域内多个位置的 DCNv2 动态采样偏移箭头。
#     每个位置显示 3x3 采样核的偏移。
#
#     参数：
#     - offset_tensor: DCNv2输出的 offset tensor，shape [1, 18, H, W]
#     - kernel_size: 卷积核大小，默认3
#     - region_top_left: 区域左上角坐标（h, w）
#     - region_size: 区域大小（高度, 宽度）
#     """
#     offset_np = offset_tensor.cpu().numpy()[0]
#     h0, w0 = region_top_left
#     region_h, region_w = region_size
#
#     grid_range = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
#     grid_x, grid_y = np.meshgrid(grid_range, grid_range)
#     base_points = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
#
#     plt.figure(figsize=(12, 12))
#     plt.title(f'DCNv2 Region Sampling Offsets: from {region_top_left}, size {region_size}')
#     plt.axis('equal')
#     plt.xlim(w0 - 2, w0 + region_w + 2)
#     plt.ylim(h0 + region_h + 2, h0 - 2)
#     plt.gca().invert_yaxis()
#     plt.grid(True)
#
#     for dh in range(region_h):
#         for dw in range(region_w):
#             h, w = h0 + dh, w0 + dw
#             offset_x = offset_np[0::2, h, w]
#             offset_y = offset_np[1::2, h, w]
#             for i in range(kernel_size * kernel_size):
#                 src_x, src_y = (base_points[i][0] + w) * stride, (base_points[i][1] + h) * stride
#                 dx, dy = offset_x[i] * stride, offset_y[i] * stride
#                 plt.arrow(src_x, src_y, dx, dy, head_width=0.1, head_length=0.15, fc='r', ec='r', alpha=0.8)
#                 plt.plot(src_x, src_y, 'bo', markersize=2)
#
#     plt.xlabel("W axis (feature map coord)")
#     plt.ylabel("H axis (feature map coord)")
#     plt.tight_layout()
#     save_path = os.path.join(SAVE_DIR, "DCNv2 Region Sampling Offsets.png")
#     plt.savefig(save_path, dpi=600)
#     plt.show()
#
# def visualize_region_sampling_offsets(offset_tensor, kernel_size=3, region_top_left=(20, 20), region_size=(5, 5), stride=8, input_img_size=640):
#     """
#     可视化一个区域内多个位置的 DCNv2 动态采样偏移箭头（输入图像坐标系下）。
#
#     参数：
#     - offset_tensor: DCNv2输出的 offset tensor，shape [1, 18, H, W]
#     - kernel_size: 卷积核大小，默认3
#     - region_top_left: 区域左上角坐标（h, w），特征图坐标系
#     - region_size: 区域大小（高，宽），特征图坐标系
#     - stride: 输入图像与特征图的缩放倍数
#     - input_img_size: 原始图像尺寸（假设为方形）
#     """
#     offset_np = offset_tensor.cpu().numpy()[0]
#     h0, w0 = region_top_left
#     region_h, region_w = region_size
#
#     grid_range = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
#     grid_x, grid_y = np.meshgrid(grid_range, grid_range)
#     base_points = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1) * stride
#
#     plt.figure(figsize=(12, 12))
#     plt.title('DCNv2 Region Sampling Offsets (on input image scale)')
#     plt.xlim(0, input_img_size)
#     plt.ylim(input_img_size, 0)
#     plt.grid(True)
#
#     for dh in range(region_h):
#         for dw in range(region_w):
#             h, w = h0 + dh, w0 + dw
#             offset_x = offset_np[0::2, h, w] * stride
#             offset_y = offset_np[1::2, h, w] * stride
#             center_x = w * stride
#             center_y = h * stride
#
#             for i in range(kernel_size * kernel_size):
#                 base_dx, base_dy = base_points[i]
#                 orig_x = center_x + base_dx
#                 orig_y = center_y + base_dy
#                 dx = offset_x[i]
#                 dy = offset_y[i]
#                 plt.arrow(orig_x, orig_y, dx, dy, head_width=1.5, head_length=2.5, fc='r', ec='r', linewidth=0.6)
#                 plt.plot(orig_x, orig_y, 'bo', markersize=1.0)
#                 plt.plot(orig_x + dx, orig_y + dy, 'ro', markersize=1.0)
#
#     plt.xlabel("W axis (input image coord)")
#     plt.ylabel("H axis (input image coord)")
#     plt.tight_layout()
#     save_path = os.path.join(SAVE_DIR, "DCNv2_Region_Sampling_Offsets_on_image_scale.png")
#     plt.savefig(save_path, dpi=600)
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_region_sampling_offsets_with_image(
    offset_tensor,
    input_image_path,  # 保留参数但不使用
    kernel_size=3,
    region_top_left=(20, 20),
    region_size=(5, 5),
    stride=8,
    save_prefix="DCN_view"
):
    """
    在白色背景 + 网格线的图上可视化 DCNv2 偏移采样点（不映射原图）。

    参数：
    - offset_tensor: DCNv2输出的 offset tensor，shape [1, 18, H, W]
    - input_image_path: 原图路径（保留但不使用）
    - kernel_size: 卷积核大小
    - region_top_left: 特征图坐标系下区域左上角 (h, w)
    - region_size: 区域大小（高，宽），特征图坐标系下
    - stride: 特征图到输入图的缩放因子
    - save_prefix: 保存目录前缀
    """
    offset_np = offset_tensor.cpu().numpy()[0]
    h0, w0 = region_top_left
    region_h, region_w = region_size

    grid_range = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    grid_x, grid_y = np.meshgrid(grid_range, grid_range)
    base_points = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1) * stride

    plt.figure(figsize=(10, 10), facecolor='white')  # 设置整个图为白色背景
    ax = plt.gca()
    ax.set_facecolor('white')  # 坐标区域背景为白
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # 添加网格线

    all_x, all_y = [], []

    for dh in range(region_h):
        for dw in range(region_w):
            h, w = h0 + dh, w0 + dw
            offset_x = offset_np[0::2, h, w] * stride
            offset_y = offset_np[1::2, h, w] * stride
            center_x = w * stride
            center_y = h * stride

            for i in range(kernel_size * kernel_size):
                base_dx, base_dy = base_points[i]
                orig_x = center_x + base_dx
                orig_y = center_y + base_dy
                dx = offset_x[i]
                dy = offset_y[i]
                ax.arrow(orig_x, orig_y, dx, dy,
                         head_width=1.5, head_length=2.5,
                         fc='r', ec='r', linewidth=0.6)
                ax.plot(orig_x, orig_y, 'bo', markersize=10.0)  # 原始点
                ax.plot(orig_x + dx, orig_y + dy, 'ro', markersize=10.0)  # 偏移后点
                all_x.extend([orig_x, orig_x + dx])
                all_y.extend([orig_y, orig_y + dy])

    # 自动缩放显示
    padding = 5
    x_min, x_max = int(min(all_x)) - padding, int(max(all_x)) + padding
    y_min, y_max = int(min(all_y)) - padding, int(max(all_y)) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # 注意反转Y轴方向

    ax.set_title("  ", fontsize=12)
    plt.axis('on')  # 开启坐标轴（可选）

    os.makedirs(save_prefix, exist_ok=True)
    save_path = os.path.join(save_prefix, f"region_{h0}_{w0}_white_grid.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    plt.show()
    plt.close()



#可视化 DCNv2 在指定区域的卷积采样点（原始点和偏移点）
# def visualize_region_sampling_points(offset_tensor, region_top_left=(20, 20), region_size=(10, 10),
#                                      stride=8, kernel_size=3, input_img_size=640,
#                                      save_prefix="region_sampling"):
#     """
#     可视化 DCNv2 在指定区域的卷积采样点（原始点和偏移点）。
#
#     参数：
#     - offset_tensor: DCNv2 偏移张量，形状 [1, 18, H, W]
#     - region_top_left: 区域左上角坐标 (h, w)，在 offset feature map 坐标系中
#     - region_size: 区域尺寸 (height, width)，单位是 feature map 点数
#     - stride: 相对于原图的下采样倍率
#     - kernel_size: 卷积核尺寸，默认 3
#     - input_img_size: 输入图大小（宽高）
#     - save_prefix: 保存文件名前缀
#     """
#     offset_np = offset_tensor.cpu().numpy()[0]
#     k = kernel_size
#
#     grid_range = np.arange(-(k // 2), k // 2 + 1)
#     grid_x, grid_y = np.meshgrid(grid_range, grid_range)
#     base_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)  # (9, 2)
#
#     orig_pts_list = []
#     offset_pts_list = []
#
#     h0, w0 = region_top_left
#     h_end = min(offset_tensor.shape[2], h0 + region_size[0])
#     w_end = min(offset_tensor.shape[3], w0 + region_size[1])
#
#     for h in range(h0, h_end):
#         for w in range(w0, w_end):
#             base = base_points + np.array([w, h])
#             offset_x = offset_np[0::2, h, w]
#             offset_y = offset_np[1::2, h, w]
#             offset = base + np.stack([offset_x, offset_y], axis=1)
#
#             orig_pts_img = base * stride
#             offset_pts_img = offset * stride
#
#             orig_pts_list.append(orig_pts_img)
#             offset_pts_list.append(offset_pts_img)
#
#     orig_pts_all = np.concatenate(orig_pts_list, axis=0)
#     offset_pts_all = np.concatenate(offset_pts_list, axis=0)
#
#     # 原始点图
#     plt.figure(figsize=(10, 10))
#     plt.title(f"Original Sampling Points in Region ({region_top_left})")
#     plt.scatter(orig_pts_all[:, 0], orig_pts_all[:, 1], c='blue', s=1)
#     plt.xlim(0, input_img_size)
#     plt.ylim(input_img_size, 0)
#     plt.axis('off')
#     plt.gca().set_aspect('equal')
#     plt.savefig(f"{save_prefix}_original.png", dpi=600, bbox_inches='tight')
#     plt.show()
#     plt.close()
#
#     # 偏移点图
#     plt.figure(figsize=(10, 10))
#     plt.title(f"Offset Sampling Points in Region ({region_top_left})")
#     plt.scatter(offset_pts_all[:, 0], offset_pts_all[:, 1], c='red', s=1)
#     plt.xlim(0, input_img_size)
#     plt.ylim(input_img_size, 0)
#     plt.axis('off')
#     plt.gca().set_aspect('equal')
#     plt.savefig(f"{save_prefix}_offset.png", dpi=600, bbox_inches='tight')
#     plt.show()
#     plt.close()
#
#     print(f"保存完成：\n - 原始点：{save_prefix}_original.png\n - 偏移点：{save_prefix}_offset.png")
def visualize_region_sampling_points(offset_tensor, region_top_left=(20, 20), region_size=(10, 10),
                                     stride=8, kernel_size=3, input_img_size=640,
                                     save_prefix="region_sampling"):
    """
    可视化 DCNv2 在指定区域的卷积采样点（原始点和偏移点），并与真实图像叠加。

    生成三张图：
    - 原始采样点（蓝）+原图
    - 偏移采样点（红）+原图
    - 原始+偏移点叠加 + 原图

    参数：
    - offset_tensor: DCNv2 偏移张量，形状 [1, 18, H, W]
    - region_top_left: 区域左上角坐标 (h, w)，在 offset feature map 坐标系中
    - region_size: 区域尺寸 (height, width)，单位是 feature map 点数
    - stride: 相对于原图的下采样倍率
    - kernel_size: 卷积核尺寸，默认 3
    - input_img_size: 输入图大小（宽高）
    - save_prefix: 保存文件名前缀，将尝试读取 {save_prefix}_background.jpg 作为背景图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import os

    offset_np = offset_tensor.cpu().numpy()[0]
    k = kernel_size
    grid_range = np.arange(-(k // 2), k // 2 + 1)
    grid_x, grid_y = np.meshgrid(grid_range, grid_range)
    base_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)  # (9, 2)

    orig_pts_list = []
    offset_pts_list = []

    h0, w0 = region_top_left
    h_end = min(offset_tensor.shape[2], h0 + region_size[0])
    w_end = min(offset_tensor.shape[3], w0 + region_size[1])

    for h in range(h0, h_end):
        for w in range(w0, w_end):
            base = base_points + np.array([w, h])
            offset_x = offset_np[0::2, h, w]
            offset_y = offset_np[1::2, h, w]
            offset = base + np.stack([offset_x, offset_y], axis=1)

            orig_pts_img = base * stride
            offset_pts_img = offset * stride

            orig_pts_list.append(orig_pts_img)
            offset_pts_list.append(offset_pts_img)

    orig_pts_all = np.concatenate(orig_pts_list, axis=0)
    offset_pts_all = np.concatenate(offset_pts_list, axis=0)

    # 尝试加载背景图
    # background_path = f"{save_prefix}_background.jpg"
    background_path = r"E:\paper\orign\DJI_0040.JPG"
    if os.path.exists(background_path):
        bg_image = Image.open(background_path).resize((input_img_size, input_img_size)).convert('RGB')
        bg_array = np.array(bg_image)
        print(f"[INFO] 使用背景图：{background_path}")
    else:
        bg_array = np.ones((input_img_size, input_img_size, 3), dtype=np.uint8) * 240  # 灰色背景
        print(f"[WARNING] 未找到背景图，使用灰色背景：{background_path}")

    # 蓝色点（原始采样点）叠加
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_array)
    ax.scatter(orig_pts_all[:, 0], orig_pts_all[:, 1], c='blue', s=2, label='Original', alpha=0.8)
    ax.set_xlim(0, input_img_size)
    ax.set_ylim(input_img_size, 0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_original_on_img.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 红色点（偏移采样点）叠加
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_array)
    ax.scatter(offset_pts_all[:, 0], offset_pts_all[:, 1], c='red', s=2, label='Offset', alpha=0.8)
    ax.set_xlim(0, input_img_size)
    ax.set_ylim(input_img_size, 0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_offset_on_img.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 红蓝点 + 连线叠加图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_array)
    ax.scatter(orig_pts_all[:, 0], orig_pts_all[:, 1], c='blue', s=2, label='Original', alpha=0.7)
    ax.scatter(offset_pts_all[:, 0], offset_pts_all[:, 1], c='red', s=2, label='Offset', alpha=0.7)
    for o, d in zip(orig_pts_all, offset_pts_all):
        ax.plot([o[0], d[0]], [o[1], d[1]], color='gray', linewidth=0.3, alpha=0.4)
    ax.set_xlim(0, input_img_size)
    ax.set_ylim(input_img_size, 0)
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_overlay_on_img.png", dpi=600, bbox_inches='tight')
    plt.close()

    print(f"保存完成：\n"
          f"- 原始点图：{save_prefix}_original_on_img.png\n"
          f"- 偏移点图：{save_prefix}_offset_on_img.png\n"
          f"- 红蓝叠加图：{save_prefix}_overlay_on_img.png")


if __name__ == '__main__':


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_path = r"E:\paper\orign\DJI_0040.JPG"
    model_path = r"D:\yolov10\runs\detect\train189\weights\best.pt"

    from ultralytics import YOLOv10
    model = YOLOv10(model_path).model
    model.to(device).eval()

# #输出所有的层数（offset）
#     for i, (name, module) in enumerate(model.named_modules()):
#         if isinstance(module, DCNv2):
#             print(f"[{i}] {name}")

    offsets = {}

    def hook_fn(module, input, output):
        if hasattr(module, '_last_offset') and module._last_offset is not None:
            offsets[module] = module._last_offset

    for name, module in model.named_modules():
        if isinstance(module, DCNv2):
            module.register_forward_hook(hook_fn)

    img_tensor = preprocess_image(img_path, img_size=640).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    if len(offsets) == 0:
        print("No offset captured, 请检查DCNv2 forward是否返回offset，以及hook注册是否正确。")
        exit(1)

    first_dcn_layer = list(offsets.keys())[18]          #17层c2fdcn,18层注意力
    offset_tensor = offsets[first_dcn_layer]
    print(f"Offset tensor shape: {offset_tensor.shape}")

    print(f'Visualizing offsets from layer: {first_dcn_layer}')

    # 1. 偏移热力图
    visualize_offset_heatmap(offset_tensor, kernel_size=3)

    # 2. 中心点采样偏移箭头
    H, W = offset_tensor.shape[2], offset_tensor.shape[3]
    center_pos = (H//2, W//2)
    visualize_sampling_offsets(offset_tensor, kernel_size=3, pos=center_pos)

    # 3. 全层所有采样点坐标获取
    stride = 8  # 根据你模型实际调整
    orig_pts_all, offset_pts_all = get_all_sampling_points_on_input(offset_tensor, stride=stride, kernel_size=3)

    # 4. 抽样可视化部分采样点，防止图像过密
    visualize_sampling_points_subset(orig_pts_all, offset_pts_all, input_img_size=640, sample_num=2000)

    # 5.获取所有采样点
    orig_pts_all, offset_pts_all = get_all_sampling_points_on_input(offset_tensor, stride=stride, kernel_size=3)

    # 6.可视化：显示全部采样点（不抽样）
    visualize_all_sampling_points(orig_pts_all, offset_pts_all, input_img_size=640, show_arrows=True)
    # 获取所有采样点
    orig_pts_all, offset_pts_all = get_all_sampling_points_on_input(offset_tensor, stride=stride, kernel_size=3)

    # 7.分别可视化原始采样点和偏移采样点（两张图）
    visualize_sampling_points_only(orig_pts_all, color='blue', title='Original Sampling Points on Input Image',
                                   save_path="original_points.png", input_img_size=640)
    visualize_sampling_points_only(offset_pts_all, color='red', title='Offset Sampling Points on Input Image',
                                   save_path="offset_points.png", input_img_size=640)
    # 8.分别可视化原始采样点和偏移采样点的密度图（两张图）
    visualize_sampling_density_fast(orig_pts_all, title=' ', cmap='Blues', save_name='Original Sampling Point Density')
    visualize_sampling_density_fast(offset_pts_all, title=' ', cmap='Reds', save_name='Offset Sampling Point Density')

    # np.savetxt("original_points.csv", orig_pts_all, delimiter=',')
    # np.savetxt("offset_points.csv", offset_pts_all, delimiter=',')

    # 9.可视化偏移采样点叠加在原图上
    visualize_points_on_image(
        img_path,
        offset_pts_all,
        color='red',
        title='Offset Sampling Points over Original Image',
        save_path="DCN_view/offset_on_image.png",
        point_size=0.8,
    )
    #10.原图的可视化区域带箭头
    #visualize_region_sampling_offsets(offset_tensor, kernel_size=3, region_top_left=(45, 50), region_size=(5, 5))

    visualize_region_sampling_offsets_with_image(
        offset_tensor,
        input_image_path="E:/paper/orign/DJI_0040.JPG",
        kernel_size=3,
        region_top_left=(45, 50),
        region_size=(5, 5),
        stride=8,
        save_prefix="DCN_view"
    )

    #11.可视化 DCNv2 在指定区域的卷积采样点（原始点和偏移点）。


    visualize_region_sampling_points(
        offset_tensor,
        region_top_left=(45,50),  # 可修改为你感兴趣的区域
        region_size=(5, 5),  # 区域大小，比如 10×10 个点
        stride=8,
        kernel_size=3,
        input_img_size=640,
        save_prefix="DCN_view/region_20_20")


    # visualize_region_sampling_points_with_overlay(offset_tensor, input_img,
    #                                               region_top_left=(20, 20),
    #                                               region_size=(10, 10),
    #                                               stride=8,
    #                                               kernel_size=3,
    #                                               save_path=save_path)

