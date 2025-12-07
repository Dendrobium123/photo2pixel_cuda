# Photo2Pixel (CUDA)

简短说明

这是一个基于CUDA加速的图像像素化（photo-to-pixel）实验工程，包含GPU（CUDA）实现与CPU串行实现两种版本。程序先对图像进行预处理（亮度/对比度增强 + 桑原滤波），再进行基于亮度分层的像素化（马赛克）处理，并输出结果同时统计 CPU / GPU 的运行时间以便对比加速比。

主要特性

- 预处理：亮度与对比度增强 + Kuwahara（桑原）滤波，保留边缘的同时去噪，产生油画般的涂抹感
- 像素化：按 `pixel_size` 划分马赛克块，基于亮度分层（4 个 bin）统计主导颜色填充块
- 两套实现：GPU（CUDA kernel）与串行 CPU 版本（验证正确性并作性能对比）
- 输出结果：`result_with_preprocess.png`（GPU）和 `result_cpu.png`（CPU）

算法概览

1. 预处理（Kuwahara + 增强）
   - 对每个像素取一个 `k_size × k_size` 窗口，按左上/右上/左下/右下 4 个重叠子区域分别计算每通道的均值与平方均值
   - 通过 `E[X^2] - (E[X])^2` 计算每个子区域的方差，选择方差最小的区域的均值作为该像素的输出（方差最小 -> 最平滑）
   - 在采样前对每个像素做亮度/对比度增强：`c = clamp(((p/255 * brightness) - 0.5) * contrast + 0.5, 0, 1)`

2. 像素化（分层统计）
   - 将图像划分为 `pixel_size × pixel_size` 的马赛克块。每个 GPU 线程负责一个块（CPU为循环处理）
   - 以块中心为基准，在 `kernel_size × kernel_size` 采样窗口内统计像素的亮度并分到 `NUM_BINS=4` 层
   - 找出像素数量最多的亮度层，计算该层的平均颜色并将整块填充为该颜色

CUDA 并行化要点

- 线程组织：使用二维 `blockSize(16,16)`（256 线程/Block）
  - 预处理核：每个线程处理一个像素，Grid = `(width+15)/16, (height+15)/16`
  - 像素化核：每个线程处理一个马赛克块，Grid = `(width/pixel_size+15)/16, (height/pixel_size+15)/16`
- 内存访问：输入指针加 `__restrict__` 避免别名，线程到像素的线性映射保证了全局内存的合并访问（coalesced access）
- 同步：在两个核之间使用 `cudaDeviceSynchronize()`，确保第一阶段写入的临时结果可被第二阶段读取

依赖

- CUDA Toolkit（支持 .cu 编译）
- OpenCV（用于图片读写与 Mat 操作）
- C++14

构建与运行（示例）

- Windows / Visual Studio：建议把工程导入带有 CUDA 支持的 Visual Studio 项目，链接 OpenCV 库（示例代码中有 `#pragma comment(lib, "opencv_world4120.lib")`）

- Linux / nvcc（示例）:

```bash
# 仅示例，具体OpenCV库名称取决于系统安装
nvcc -std=c++14 kernel.cu -o photo2pixel `pkg-config --cflags --libs opencv4`
./photo2pixel
```

使用说明

- 默认输入图片路径在代码中为 `input.jpg`（CPU部分）和 `input.jpg`（main 中 GPU 部分也使用同一路径），运行前请把图片放到可执行路径或修改代码中的路径
- 运行后会产生：
  - `result_cpu.png`：CPU串行结果
  - `result_with_preprocess.png`：GPU 结果
- 程序会在控制台输出 CPU 和 GPU 的运行时间并计算加速比

性能注意

- GPU 优势在于高度并行的场景（大分辨率图片），数据量越大并行化收益越明显
- 当前实现未使用共享内存或纹理内存做 tile 缓存，若需进一步加速可考虑：共享内存 tiling、纹理缓存、流（streams）并发、半精度计算等

文件结构

- `kernel.cu`：主程序与 CUDA kernel（预处理 + 像素化）与 CPU 实现

许可

请根据需要自行选择开源许可（如 MIT、Apache 2.0 等）。
