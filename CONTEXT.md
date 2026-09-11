# Cosmos Toolbox Context

## Domain language

- **项目工作台**：围绕同一个数据集、图片、标注、模型和产物运行的唯一主界面。
- **工作区**：项目工作台中的素材、标注或训练主页面，不创建新的顶层窗口。
- **业务页面**：CAB-F 流程、点边标注、筛选、导出、设置、命令或日志等可持续操作的页面。
- **系统对话框**：文件/目录选择、危险操作确认和错误提示；允许保持原生弹窗。
- **能力**：用户可以发现并打开的一项项目活动；声明所属流程、输入语义、产物语义和呈现方式。导航与定制功能都从能力目录生成。
- **活动宿主**：项目工作台中承载当前能力的唯一主内容区域；一次只突出一个业务目标。
- **项目会话**：在项目路径之外记录数据集、标注、模型、运行与产物关系的持久状态。
- **产物**：数据处理、训练、推理或导出产生的可追踪结果，保存来源能力、来源任务和输入关系。
- **任务中心**：统一呈现运行进度、日志、历史、取消状态和产物的业务页面；迁移期通过左侧“任务与日志”能力打开。
- **任务账本**：任务中心内部的确定性状态，记录任务快照、状态、进度、受限日志和取消资格；不保存 Qt signal、取消 callback 或持久化细节。
- **任务呈现**：把任务账本记录投影为统一中文状态、语义 tone、活动状态、进度、取消资格和摘要的不可变值；任务中心、业务页与常驻任务条只消费该值，不各自解释状态。
- **能力目录计划**：根据显式工作区与训练 feature facts 生成能力 specification、排序、分组、检索、默认 action、可见性和 shell adapter key；Qt factories 与磁盘扫描不属于该计划，扫描无结果时原生训练工作区必须保持可发现。
- **工作台状态**：当前能力、路由深度、检查器开关、viewport 和活动任务数的不可变状态；reducer 只产生 shell intents，不直接操作 Qt widget。
- **页面路由状态**：按 route id、来源工作区和 dialog/page 类型记录的不可变页面栈；Qt router 只执行 push、finish、close 与返回 intents。
- **UI foundation**：共享主题 token、语义 role、页面标题、内容 surface、路径字段、换行动作栏、状态反馈、指标、空状态、日志面板和自适应密度投影的唯一 UI interface。
- **兼容 adapter**：迁移期间把旧 `QMainWindow` 接入能力 interface 的实现；只负责兼容，不能成为新增功能的默认形态。
- **初始原图**：用户为 CAB-F TOP/BOTTOM 手动选择的现场彩色原图；它只作为两种标准产物的共同输入。
- **校准基准图**：初始原图应用与模板相同的质心平移后得到的全尺寸彩色图；YAML ROI 坐标唯一基于该图。
- **匹配模板**：由初始原图经过半尺寸缩放、胶体分割和质心居中产生的二值图；用于生产标定匹配，不承载 ROI。
- **完整检测结局**：一次 Cosmos complete-inspection 运行的终态业务含义；分为业务 OK、业务 NG、技术失败和已停止，不等同于进程退出码。
- **完整检测运行会话**：一次 complete-inspection 请求在模型加载前捕获的产品配置与已解析 backend 事实；runner 全程显式传递，父 Cosmos 全局配置只作为兼容 adapter 的镜像。
- **完整检测请求计划**：由产品配置、输入清单和 engine 选择确定的项目、输入槽、case 顺序与执行方式；不解析 YAML、不访问路径，也不加载模型。
- **CAB-F ROI 文档**：承载 TOP/BOTTOM ROI 发现、归一化、校验与更新规则的领域文档；坐标始终使用全尺寸校准基准图语义，YAML 只是它的持久化载体。
- **运行计划**：一次训练、推理或导出动作的确定性描述，包含参数向量、工作目录、解释器和输出目标；计划本身不创建目录、不读取环境，也不启动进程。
- **CAB-F 工作流计划**：CAB-F 点预测、边预测、校验、导出与训练步骤的有序、带类型描述；自动流水线和人工复核流程显式绑定各自的标注输入，CLI 与项目工作台只负责执行。
- **CAB-F 标注源决策**：根据 requested、master、prediction 三类目录的显式 JSON facts，按 requested → master → prediction 选择可用点标注；`sew` 与 `keypoint` 属于同一 canonical 点标签，目录扫描和 JSON 解码由 shell 完成。
- **CAB-F 校准计划**：根据初始原图统计、前景统计、half-size 模板尺寸与 matcher offset，决定原图/前景有效性并生成 half→full offset 和全尺寸 transform contract；OpenCV、分割模型与图像数组留在 shell。
- **连边推理策略**：把模型分数、候选边、点坐标与后处理参数确定性映射为稳定 edge schema 的规则；不负责 checkpoint、Torch device、图构造或文件读写。
- **连边评估**：统一无向 edge normalization、TP/FP/FN、零分母指标、点的一对一几何匹配与 graph edge lifting；批处理、可视化和 pipeline 评估只消费同一个比较结果。
- **连边运行时**：一次单图、批次、评估或可视化任务显式持有的 checkpoint、模型和 device；由 shell 管理生命周期，同一任务复用且不进入进程全局 cache。
- **模型构造计划**：根据任务、模式、候选优先级和显式可用性事实选择 factory，并补齐模型键专属默认参数；动态 import、可用性探测和实例化由训练 shell 执行。
- **CAB-F 数据集判定**：根据显式的图片、标注、规范化和配对事实，生成单样本评估、数据集验证汇总与导出决定；扫描、解码、复制和写入不属于判定。
- **标注编辑迁移**：把不可变的点边标注状态与编辑命令映射为新状态、移除事实和动作说明；Qt canvas 只翻译输入事件并呈现结果。
- **缝纫点推理策略**：根据 checkpoint payload、TTA 路径、heatmap 和阈值事实确定 metadata、逆变换、合并与峰值；Torch/ONNX runtime 负责张量执行和设备生命周期。
- **YOLO 动作计划**：把训练、预测或导出参数确定为模型来源、kwargs、CLI argv 与产物搬运 intent；Ultralytics import、模型构造、运行和文件移动由 shell 执行。
- **图像裁剪计划**：把参考 ROI、图片尺寸和 adapter 策略映射为半开像素框、拒绝原因与输出名称；图片解码、数组切片、重名探测和写入由 shell 执行。
- **数据审阅会话**：保存待审样本顺序、当前选择和保留/移除计数的不可变状态；审阅决定先生成文件移动 intents，shell 执行成功后才提交状态迁移。

## Architecture invariants

1. Functional core 只能根据显式输入计算领域事实、状态或计划；不得访问 Qt、文件系统、进程/线程、Conda、GPU/模型、全局配置、时钟/UUID 或数据库。
2. Imperative shell 负责收集外部事实、执行计划和持久化结果，并把执行结果翻译回 functional core 定义的领域事实。
3. 业务规则只能在 core module 的 interface 后定义一次；CLI、项目工作台、旧工作区和测试 adapter 不得各自复制同一规则。
4. 兼容 adapter 保持既有 caller interface 和行为，但新增规则必须先进入 functional core。
5. 一次请求或任务使用的配置、backend、模型和 device 必须由 shell 显式捕获并持有；core 不得回读进程全局状态，任务级资源不得藏入进程全局 cache。
6. 计划中的副作用按声明顺序执行；依赖副作用成功的状态迁移只能在执行成功后提交，失败时保留原 core 状态并由 shell 报告。
7. 每个新增 functional core 必须在 source-effect gate 与 inert-import gate 中各登记一次，防止后来重新引入 Qt、文件系统、模型 runtime 或 shell 初始化。
8. 工作台、页面路由、任务账本与任务呈现的状态迁移必须保持纯函数；Qt shell 只翻译事件、执行 intents 并投影 widget。
9. 后台操作的业务结果 signal 与线程完成 signal 必须分离；shell 在真实线程完成前持有 worker，并在页面关闭时完成 teardown。
10. 能力 specification、排序、分组、检索、默认 action 与 workspace fallback 必须由能力目录计划统一投影；扫描 shell 的异常不得令 Training 入口静默消失。
11. CAB-F 标注源 precedence、点标签 aliases、模板校准 offset 与全尺寸 transform 只能在相应 core module 的 interface 后定义一次。
12. Sew Point Connect 的 edge 比较、指标、点匹配和 graph lifting 只能由连边评估 core 提供；数据集、批处理和可视化 shell 只能保留兼容 wrappers。

## UI invariants

1. 项目工作台运行时只能有一个业务主窗口。
2. 业务页面通过主界面页面栈打开，并提供返回路径；不得调用模态 `exec()`。
3. 嵌套业务页面按后进先出返回，CAB-F 流程状态不能因打开编辑器而丢失。
4. `QFileDialog`、确认框和错误提示不属于业务页面，可以保留。
5. 独立启动旧工作区时允许使用原有模态行为，作为兼容 adapter。
6. 顶层导航必须由能力目录投影，新增定制能力不得直接修改主窗口导航。
7. 新增功能必须提供原生业务页面并进入活动宿主；不得新增完整业务 `QMainWindow`。
8. 长时间运行通过任务中心的 interface 汇总进度、日志、历史和产物。
9. 旧工作区只能通过兼容 adapter 逐步迁移，不得继续扩展其顶层导航职责。
10. CAB-F ROI 始终保存全尺寸校准基准图坐标；初始原图、画布缩略图和匹配模板不得改变这一坐标语义。
11. CAB-F 模板生成模型从生产 `backend_config.yaml` 延迟解析，长时间推理必须进入任务中心。
12. 默认视觉采用朴素桌面工程工具风格：浅灰工作区、细边框、小圆角和紧凑表单；蓝色只用于当前选择与主要操作，不使用深色品牌侧栏、胶囊徽章或卡片套卡片。
13. 默认 Image、Labeling、Training 工作区必须是原生 `QWidget`，不得伪造 `menuBar()`；`QMainWindow` 只保留为独立启动或显式 legacy adapter。
14. 页面视觉必须通过 UI foundation 的 token、语义 role 和 primitives 表达；静态颜色、状态词汇与日志样式不得在业务页重复定义。
15. 页面在 960×640 起必须可操作：内容可滚动，动作可换行，画布/侧栏可收缩；固定尺寸只允许表达真实画布或紧凑控件约束。
16. 页面标题只能由活动宿主或页面自身拥有一次；拥有 `PageScaffold`/`PageHeader` 的页面必须声明 header ownership，避免重复标题。
17. 长任务摘要在切换能力后仍保持可见，并能直接进入任务中心；业务页不能把唯一运行状态藏在离开即不可见的局部 label 中。
18. 纵向空间不足时，路径字段和检查面板必须保持各行最小高度并进入滚动容器，不得压缩到标签、输入框或操作互相重叠。
