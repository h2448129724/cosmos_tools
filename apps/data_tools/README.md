# Data Tools

批量处理逻辑，供 `apps/labeling_ui` 的页面调用：

- `processing/batch_crop.py`：按 ROI 批量裁剪（裁剪框规则来自 `img_tools.core.crop_plan`）
- `processing/auto_tile_crop.py`：自动切块
- `processing/keyword_split.py`：按文件名关键字分类
- `processing/image_io.py`：兼容转发，实际实现在 `img_tools.core.image_io`
