# Labeling UI

来源：

- 由原 `img_tools/gui` 与入口 `main.py` 迁入

当前职责：

- 点边一体标注
- 数据筛选
- 数据集校验与导出对话框
- 9 步 CABF 流程页面

启动方式：

```bash
python D:/project/tianwei/train_model/scripts/labeling_ui.py
python -m apps.labeling_ui
```

说明：

- 当前已经改为包内导入与 `apps.data_tools.*` 依赖
- 启动脚本只负责补上 `shared/cabf_common` 的导入根
- 可以直接作为 monorepo 内的应用包运行
- UI 主代码现在位于 `apps/labeling_ui/app`
