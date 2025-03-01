# 支持修复 FLUX 全景图接缝的扩展脚本

## 概述
这是一个针对 [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) 的扩展脚本，在原有功能基础上进行了修改，现在可以支持修复 FLUX 全景图接缝。它允许用户独立配置 X 和 Y 轴的无缝图像平铺，同时具备处理 FLUX 全景图接缝问题的能力。

## FLUX 全景图接缝修复功能
当脚本处于激活状态时，会自动启用 FLUX 全景图接缝修复功能。在图像生成过程中，脚本会对模型的卷积层进行调整，通过特殊的填充模式和处理逻辑，尽可能减少全景图中的接缝问题。
## 注意事项
- 与现有的平铺选项一样，此功能不能 100% 保证无缝平铺，但对于大多数提示词应该可以有效管理接缝问题。
- 可以使用在线工具如([https://www.pycheung.com/checker/](https://renderstuff.com/tools/360-panorama-web-viewer/)) 来检查图像是否有接缝

- ![ComfyUI_00040_](https://github.com/user-attachments/assets/5e0cbcea-bd1f-48b4-87ed-80f7e53dd0b6)

  
