# PhyRC-GZSL 仓库目录扁平化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `phyrc_gzsl/` 的全部内容上移到仓库根目录，并让源码、脚本、文档和结果清单在新布局下保持一致。

**架构：** 删除外层 Python 包边界，原子包改为根目录包（如 `models`、`utils`、`phyrc`），入口脚本改为根模块。通过发布布局测试约束目录形态，并用机械替换统一移除旧命名空间和路径前缀。

**技术栈：** Python、pytest、PowerShell、Git

---

## 文件结构

- 修改：`test_release_layout.py`，约束新根目录结构和禁止旧前缀。
- 移动：`phyrc_gzsl/*` 到仓库根目录。
- 删除：`phyrc_gzsl/__init__.py`，旧包不再存在。
- 修改：移动后的 Python、Markdown、YAML、JSON、TXT、PowerShell 文本文件，移除旧 import、模块名和相对路径前缀。
- 修改：入口脚本、baseline 脚本与测试中的 `Path(__file__)` 层级，使其仍解析到仓库根目录。

### 任务 1：建立新布局回归检查

**文件：**
- 修改：`test_release_layout.py`

- [ ] **步骤 1：编写失败的布局测试**

将包存在断言替换为：

```python
ROOT_ENTRIES = {"baseline", "checkpoints", "configs", "data", "models", "phyrc", "tests"}


def test_release_layout():
    assert not (ROOT / "phyrc_gzsl").exists()
    assert ROOT_ENTRIES <= {path.name for path in ROOT.iterdir()}
```

并在文本扫描中将 `phyrc_gzsl` 作为禁止残留的仓库前缀。

- [ ] **步骤 2：运行测试验证失败**

运行：`python test_release_layout.py`

预期：FAIL，因为 `phyrc_gzsl/` 仍存在且根目录项尚未上移。

### 任务 2：上移项目内容并修正引用

**文件：**
- 移动：`phyrc_gzsl/baseline`、`phyrc_gzsl/checkpoints`、`phyrc_gzsl/configs`、`phyrc_gzsl/data`、`phyrc_gzsl/models`、`phyrc_gzsl/phyrc`、`phyrc_gzsl/tests` 和 `phyrc_gzsl/*.py`
- 删除：`phyrc_gzsl/__init__.py`
- 修改：所有受版本控制的 `.py`、`.md`、`.yaml`、`.yml`、`.json`、`.txt`、`.ps1`

- [ ] **步骤 1：验证移动目标**

运行 PowerShell 检查每个 `phyrc_gzsl` 直属项在根目录均无同名冲突；如有冲突则停止。

- [ ] **步骤 2：机械上移**

使用 `git mv` 将各目录和入口脚本移动到根目录，使用 `git rm phyrc_gzsl/__init__.py` 删除失效包标记。

- [ ] **步骤 3：机械修正命名空间和相对路径**

对受支持文本文件执行以下替换：

```text
phyrc_gzsl.  -> （空）
phyrc_gzsl/  -> （空）
phyrc_gzsl\  -> （空）
```

随后逐项修正上移造成的根目录层级变化：根入口脚本的 `parents[1]` 改为 `parent`，`baseline/` 中原 `parents[2]` 改为 `parents[1]`，`tests/` 中原 `parents[2]` 改为 `parents[1]`。`models/backbone.py` 与 `utils/config.py` 已处于与仓库资源相对一致的层级，不做无依据改动。

- [ ] **步骤 4：运行布局测试验证通过**

运行：`python test_release_layout.py`

预期：PASS，无输出，退出码 0。

### 任务 3：验证运行入口和测试

**文件：**
- 验证：全部迁移后的 Python 文件与测试

- [ ] **步骤 1：编译源码**

运行：`python -m compileall -q -x "baseline/(CADA-VAE-PyTorch|CE-GZSL|f-clswgan_pytorch|FREE|GenZSL)" .`

预期：退出码 0。

- [ ] **步骤 2：检查关键导入**

运行：`python -c "import models.backbone, utils.config, phyrc.decoder, diagnose_rstd_ot"`

预期：退出码 0。

- [ ] **步骤 3：运行现有测试**

运行：`python -m pytest tests -q`

预期：全部通过；若环境缺少 requirements 中的依赖，准确记录依赖错误而不改业务代码掩盖问题。

- [ ] **步骤 4：扫描旧引用**

运行 `rg` 扫描受支持文本文件中的 `phyrc_gzsl`，预期只有设计/计划文档中对旧结构的说明；源码、README、配置、结果清单和脚本中为 0。

- [ ] **步骤 5：检查最终差异**

运行：`git status --short`、`git diff --stat HEAD^` 和 `git diff --check HEAD^`

预期：迁移文件完整、无意外文件、无空白错误。

- [ ] **步骤 6：提交布局迁移**

验证全部通过后运行：

```text
git add -A
git commit -m "refactor: flatten repository layout"
```
