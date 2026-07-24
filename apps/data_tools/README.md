# Data Tools

来源：

- 由原 `img_tools/core` 与 `img_tools/utils` 迁入

当前职责：

- 图像读写
- 批量裁剪
- 自动切块
- 关键词分类
- CABF 数据集 facade / 兼容桥接

说明：

- 当前已改为包内相对导入
- 这些模块主要供 `apps/labeling_ui` 调用
- `processing/` 对应原 `core/`
- `common/` 对应原 `utils/`
