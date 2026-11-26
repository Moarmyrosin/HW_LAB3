# CIFAR 图像分类作业项目

本仓库包含代码、实验配置、实测结果和报告。

- [五页 PDF 报告（含参考资料）](output/pdf/CIFAR10_HW3_report_5pages.pdf)
- [精简报告](report.md) · [完整实验记录](report_full.md)
- [三模型结果对比](reports/comparison.md) · [逐项验收清单](reports/checklist.md)

|模型|最佳验证准确率|测试准确率|训练轮数|
|---|---:|---:|---:|
|CNN|81.42%|80.49%|30|
|ResNet18|92.76%|92.53%|30|
|SE-ResNet18|93.96%|93.29%|30|

正式测试集为完整 10,000 张 CIFAR-10 图像，最终模型按验证集选定为 SE-ResNet18。原始数据和模型权重保留本地，仓库保留划分索引、训练日志、预测结果及图表。

作业指定 CIFAR-10；用户另外授权使用已下载的 CIFAR-100 做真实数据流程校验。两类数据、分类头、划分索引和输出目录分开，CIFAR-100 smoke 成绩不能代替 CIFAR-10 正式实验。

## 当前状态

- 三个模型的合成数据前向、反向、保存加载和中断恢复已验证。
- CIFAR-100 包的 MD5 已通过，已完成三个模型的真实 smoke 训练、恢复和测试；准确结果以 `reports/verification_real_cifar100.json` 为准。
- 替换后的 CIFAR-10 数据已通过校验与划分审计，三个模型真实 smoke 检查均通过；三个正式实验各完成 30 轮并评估完整测试集。CNN/ResNet18/SE 测试准确率分别为 80.49%/92.53%/93.29%，结果见 report.md。最终模型按验证集选为 SE，已生成 3 个正确和 3 个错误 Grad-CAM 案例。
- 原始代码和报告备份保留在本地 `backups/`，未纳入仓库。

## 安装和环境检查

本次检查的环境为 Python 3.13.2、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118、RTX 4060 Laptop（8 GiB）。可继续使用现有环境；新建隔离环境时运行：

```powershell
git clone https://github.com/Moarmyrosin/HW_LAB3.git
cd HW_LAB3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果 PowerShell 限制激活，可直接使用 `.\.venv\Scripts\python.exe` 代替下文的 `python`。CPU 可安装对应 CPU 构建，并在运行时使用 `--device cpu --no-amp`；CPU 配置不承诺与 CUDA 逐位一致或相同耗时。

## 数据下载与校验

从 [官方数据页面](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 **CIFAR-10 python version**，保存为：

`data/cifar-10-python.tar.gz`

正确 MD5：`c58f30108f718f92721af3b95e74349a`。文件就绪后执行：

```powershell
Get-FileHash .\data\cifar-10-python.tar.gz -Algorithm MD5
python prepare_data.py --dataset cifar10
```

程序先校验 MD5，成功才解压，并核验 torchvision 官方批次文件。数据缺失或错误时停止，保留文件，不会自动重新下载。

CIFAR-100 Python 包 MD5 为 `eb9058c3a382ffc7106e4002c42a8d85`，建议保存为 `data/cifar-100-python.tar.gz`。显式选择 cifar100 时，也兼容用户先前误命名的文件，但仍须匹配 CIFAR-100 的正确校验值。

## 小规模与恢复训练验证

离线合成数据验证：

```powershell
python verify.py --device cuda
python verify.py --early-stop-only
```

使用官方真实数据，逐个验证三个模型连续两轮训练与“先训练一轮、加载后继续第二轮”是否一致，并生成独立测试图表：

```powershell
python verify.py --real --dataset cifar10 --device cuda --out runs/smoke_real_cifar10
```

当前已执行的 CIFAR-100 校验命令为：

```powershell
python prepare_data.py --dataset cifar100
python verify.py --real --dataset cifar100 --device cuda --out runs/smoke_real_cifar100
```

每类取 8 张训练、4 张验证、4 张测试。CIFAR-10 为 80/40/40，CIFAR-100 为 800/400/400；均来自固定分层索引，不是正式训练。输出目录已有记录时不会覆盖；重复校验可用新的 `--out`。

单次烟雾测试入口也可使用：

```powershell
python main.py --dataset cifar10 --model cnn --smoke --epochs 2 --amp --out runs/manual_smoke
python main.py --dataset cifar10 --model cnn --smoke --test --out runs/manual_smoke
```

## 克隆后重新实验

仓库已经包含已完成的轻量结果，训练程序会保护已有目录。若要按下面的默认路径完整重跑，先在仓库根目录归档已发布结果，再执行真实数据验证与正式实验：

```powershell
Rename-Item -LiteralPath .\runs -NewName published_runs
Rename-Item -LiteralPath .\reports -NewName published_reports
python prepare_data.py --dataset cifar10
python verify.py --real --dataset cifar10 --device cuda --out runs/smoke_real_cifar10
python run_experiments.py --phase all --epochs 30 --batch_size 128 --amp
```

上述归档保留发布时的结果，新的记录写入 runs/ 与 reports/。若只运行单个模型，可直接为 --out 指定新目录。仓库不包含 checkpoint，独立测试、恢复训练和重算 Grad-CAM 需要先训练产生相应权重。

查看现成报告无需训练。仅利用已保存的紧凑图表重新排版，可运行 `python build_report.py --reuse-figures`；该命令会按生成器更新 report.md，手工修改的正文请先保留副本。

## 三个正式实验

数据校验和 CIFAR-10 真实 smoke 验证通过后，可运行完整顺序流程：

```powershell
python run_experiments.py --phase all --epochs 30 --batch_size 128 --amp
```

此命令不会下载数据；依次训练 CNN、ResNet18、SE-ResNet18，全部完成后按最佳验证准确率固定最终模型，再统一评估测试集。中断后执行同一命令，会从各实验 last.pt 恢复。

逐个训练的等价命令：

```powershell
python main.py --config configs/cnn.json --amp --out runs/cnn
python main.py --config configs/resnet18.json --amp --out runs/resnet18
python main.py --config configs/se.json --amp --out runs/se
```

默认 seed=42、batch=128、30 epoch、SGD、lr=0.1、momentum=0.9、weight_decay=0.0005、余弦退火；早停默认关闭。显存不足时，先统一决定三个实验的 batch size 后运行，例如均改为 64，记录实际配置。CPU 建议从 batch=32 开始，并使用 `--device cpu --no-amp`。不要用 smoke 准确率预测正式性能或承诺准确率。

实验配置 JSON 可直接通过 `--config` 加载，显式命令行参数优先。支持可选 `--patience N`，但三组固定预算比较默认关闭。

## 恢复训练与独立测试

例如恢复 ResNet18：

```powershell
python main.py --config configs/resnet18.json --amp --out runs/resnet18 --resume runs/resnet18/last.pt
```

恢复时保持 dataset、model、seed、epochs、batch、AMP、优化器参数、worker 数和数据划分一致。epochs 是总预算，不是追加轮数；不要把两轮 smoke 权重改成 30 轮正式训练。默认 num_workers=0，便于精确检验数据增强和随机状态恢复。

训练完成后，推荐统一测试以先记录模型选择：

```powershell
python run_experiments.py --phase evaluate
```

单模型测试入口：

```powershell
python main.py --model cnn --test --out runs/cnn
python main.py --model resnet18 --test --out runs/resnet18
python main.py --model se --test --out runs/se
```

默认加载 best.pt，核验数据集、模型、种子、smoke 标记、划分指纹以及与 last.pt 的实验身份。已有测试结果会拒绝覆盖；统一流程会保留同一实验的有效测试记录。不要根据测试集成绩调参或重新选择模型。

## 图表、Grad-CAM 与报告

```powershell
python summarize.py
python gradcam.py --device cuda
python build_report.py
```

Grad-CAM 默认读取 `reports/model_selection.json` 中按验证集选定的模型，也可显式指定最佳 checkpoint：

```powershell
python gradcam.py --checkpoint runs/se/best.pt --device cuda --out runs/se/gradcam.png
```

当前 CIFAR-100 smoke 的 Grad-CAM 功能校验：

```powershell
python gradcam.py --checkpoint runs/smoke_real_cifar100/se/continuous/best.pt --smoke --device cuda --out reports/cifar100_smoke_gradcam.png
```

报告生成器从既有正式结果生成精简 report.md 和固定 5 页 PDF（包含参考资料）：`output/pdf/CIFAR10_HW3_report_5pages.pdf`。正文结合课程课件，合并训练曲线和每类准确率，保留最终模型的两种混淆矩阵、代表性错误与 Grad-CAM。所有原始实验图表仍在 runs 中。

原详细报告保留为 `report_full.md`，可运行 `python build_report_full.py` 更新，不会覆盖五页提交版。PDF 排版使用 Windows 宋体与黑体，Codex 环境会自动选择已捆绑的 ReportLab；外部环境缺少该依赖时先执行 `python -m pip install -r requirements-report.txt`。修改正文时编辑 `build_report.py` 中相应段落后重新生成，单独编辑 Markdown 不会自动同步 PDF。

## 文件结构与指标口径

- `data.py`：官方文件检查、索引划分、独立数据变换；`prepare_data.py`：解压前 MD5 校验。
- `models.py`：自行实现 CNN、ResNet18、SE。
- `main.py`、`engine.py`：训练、恢复、早停与独立测试。
- `plots.py`、`gradcam.py`：图表与梯度解释。
- `verify.py`：合成/真实数据流程和断点恢复验证。
- `run_experiments.py`：三组正式实验与测试前模型选择。
- `configs/`：可执行 JSON 配置。
- `runs/<模型>/`：CIFAR-10 正式权重、训练环境、history.json/csv、curves.png、独立测试结果。
- `runs/smoke_real_cifar100/`：CIFAR-100 校验，continuous 和 resumed 分别保留。
- `reports/`：数据审计、验证记录、对比表和解释图；`report.md`：中文报告。

测试文件包括 metrics.json、classification_report.json/txt、predictions.npz、confusion_counts.csv/png、confusion_normalized.csv/png、errors.png。混淆矩阵行是真实类别、列是预测类别；每类准确率=该真实类别预测正确数/该真实类别样本数，即 recall，不是 precision。

保存完整 epoch 历史；总训练时间统计所有完整轮次的训练与验证，不止最佳 checkpoint 前的轮次。训练环境和测试环境分别记录。固定随机种子与 cuDNN 设置仍不保证不同硬件、库版本、AMP 和 worker 配置之间逐位一致。数据归一化使用预设常数，未从测试集拟合。
