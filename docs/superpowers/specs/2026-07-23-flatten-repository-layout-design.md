# PhyRC-GZSL 仓库目录扁平化设计

## 目标

将 `phyrc_gzsl/` 下的全部项目内容上移到 `PhyRC-GZSL` 仓库根目录，移除 `phyrc_gzsl` Python 包命名空间，并保证源码、测试、脚本、文档和结果清单继续引用有效路径。

## 目录结构

迁移后，`baseline/`、`checkpoints/`、`configs/`、`data/`、`models/`、`phyrc/`、`tests/` 以及各入口脚本直接位于仓库根目录。原 `phyrc_gzsl/__init__.py` 不保留，因为仓库根目录不再作为名为 `phyrc_gzsl` 的 Python 包使用。

## 代码与路径调整

- 将 `phyrc_gzsl.<module>` 导入改为根目录下的 `<module>` 导入，例如 `phyrc_gzsl.models.backbone` 改为 `models.backbone`。
- 将模块运行方式从 `python -m phyrc_gzsl.<module>` 改为 `python -m <module>`。
- 将项目内相对路径的 `phyrc_gzsl/` 或 `phyrc_gzsl\` 前缀移除，包括源码默认值、测试、README、PowerShell 脚本和文本格式结果清单。
- 根据文件上移后的层级修正 `Path(__file__).resolve().parents[...]` 和 `sys.path` 处理，使项目根目录仍指向 `PhyRC-GZSL`。
- 不增加旧 `phyrc_gzsl` 命名空间兼容层。

## 数据与历史结果

数据文件、checkpoint JSON 和上游 baseline 目录只做位置移动。文本结果清单内记录的仓库相对路径同步去掉 `phyrc_gzsl/` 前缀；数值结果和模型数据不修改。

## 验证

- 更新发布布局检查，断言根目录结构存在且 `phyrc_gzsl/` 已不存在。
- 先运行更新后的布局检查并确认它在旧结构上失败，再实施迁移。
- 运行 Python 源码编译检查和关键模块导入检查。
- 运行现有 pytest 测试集；需要外部原始数据或未发布权重的训练、评估任务不作为本次验证项。
- 扫描受支持的文本文件，确认没有残留的 `phyrc_gzsl` 导入、模块命令或仓库相对路径。

## 边界

本次只调整仓库布局及其直接引用，不重构模型、训练逻辑、实验协议或第三方 baseline 实现。
