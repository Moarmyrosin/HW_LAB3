# CIFAR-10 作业三
实际目录：D:\学习资料 信息\课内\当代人工智能\LAB3\CIFAR10_HW3。指定 D:\Projects 不在本次写入权限内。
## PowerShell 命令
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --smoke --epochs 1 --out runs/smoke
python main.py --smoke --test --out runs/smoke
python main.py --model cnn --epochs 30 --out runs/cnn
python main.py --model resnet18 --epochs 30 --out runs/resnet18
python main.py --model se --epochs 30 --out runs/se
python main.py --model cnn --epochs 30 --out runs/cnn --resume runs/cnn/last.pt
python main.py --model cnn --test --out runs/cnn
python main.py --model resnet18 --test --out runs/resnet18
python main.py --model se --test --out runs/se
```
训练自动下载官方数据；三个实验固定 seed=42 和 30 epoch。先根据验证集确定方案，再统一测试，禁止根据测试结果调参。CPU 使用 --device cpu --batch_size 32；GPU 可用 --amp。曲线和混淆矩阵随训练、测试自动生成。不要覆盖已有输出目录；smoke 仅检查流程，不代表正式性能。
实现采用 Python、NumPy、PyTorch 固定随机种子和确定性 cuDNN；不同硬件、库版本、混合精度仍可能产生差异。恢复需保持原模型、seed、训练预算和 smoke 设置。训练集指标含随机增强与训练模式，验证指标为评估模式，不能直接等同比较。
类别顺序：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck。每类准确率=该真实类别预测正确数/该真实类别样本数，即 recall，不是 precision。
模型均自行实现；框架提供张量、卷积、优化器与 CIFAR10 下载接口。ResNet18 使用 3×3 stem，无初始 maxpool，四组各两个残差块；SE 在残差支路末端使用全局池化与通道门控。正式训练及 Grad-CAM 完成状态见报告。

Grad-CAM：`python gradcam.py --checkpoint runs/se/best.pt --out runs/se/gradcam.png`。最终模型应先按验证集选定，再生成解释图。

离线合成流程验证：`python verify.py`，输出 verification.json。对比汇总：`python summarize.py`，输出 reports/comparison.md。configs 中记录建议配置，运行参数通过命令行指定。
若本机 SSLKEYLOGFILE 指向不可写位置，先在当前 PowerShell 执行 `Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue` 再下载。

当前状态：官方数据下载中断并未通过完整性检查。真实数据 smoke 和全部正式实验待运行；请先重试 smoke。不可将 verification.json 中合成数据 loss 当作实验性能。
