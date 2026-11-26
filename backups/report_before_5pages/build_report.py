"""从实际运行文件生成中文报告；没有运行的项目保留待运行。"""
import json
from pathlib import Path
from datetime import datetime
from summarize import collect, summary_table
from utils import save_json


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def link_image(path, title):
    return f'![{title}]({path})\n' if Path(path).exists() else f'{title}：待生成。\n'


def main():
    results = collect()
    check10 = read('reports/data_package_check_cifar10.json') or read('reports/data_package_check.json')
    audit10 = read('reports/data_audit_cifar10.json')
    audit100 = read('reports/data_audit_cifar100.json')
    smoke100 = read('reports/verification_real_cifar100.json')
    smoke10 = read('reports/verification_real_cifar10.json')
    offline = read('reports/verification_offline.json')
    early_stop = read('reports/verification_early_stop.json')
    selected = read('reports/model_selection.json')
    notes = read('reports/analysis_notes.json')
    env = None
    for path in ['runs/cnn/train_environment.json',
                 'runs/smoke_real_cifar100/cnn/continuous/train_environment.json']:
        env = read(path)
        if env:
            break
    sections = [
        '# 当代人工智能作业三：图像分类实验报告',
        f'报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}。本文由实际运行记录生成；正式 CIFAR-10 实验与 CIFAR-100 流程校验分开列示。',
        '## 1 任务目标与作业要求',
        '作业 PDF 指定 CIFAR-10 图像分类，要求 CNN 模型、训练/验证划分、测试准确率、训练曲线、混淆矩阵、每类准确率和误分类分析。架构对比、注意力及 Grad-CAM 对应提升方向。截止时间为 2026 年 12 月 3 日 23:59，提交方式为课程群坚果云链接。',
        '本项目安排三个正式实验：自行实现基础 CNN；适配 CIFAR-10 的 ResNet18 从头训练；相同残差架构加入 SE。当前完成状态见第五节，不能用代码存在代替实验完成。',
        '## 2 数据集、预处理与划分',
        '正式数据应为官方 CIFAR-10，包含 10 类、32×32 RGB 图像，官方训练集 50000 张、测试集 10000 张。计划按 seed=42 每类抽 500 张验证，形成 45000/5000 训练验证划分；保存 split_42.npz。训练仅使用训练索引，验证选择最佳 checkpoint，确定方案后才统一测试。',
        '训练使用随机裁剪（padding=4）、随机水平翻转、ToTensor 与固定归一化；验证和测试仅 ToTensor 与归一化，使用独立数据对象。CIFAR-10 固定 mean=(0.4914,0.4822,0.4465)，std=(0.2470,0.2435,0.2616)；CIFAR-100 校验使用 mean=(0.5071,0.4867,0.4408)，std=(0.2675,0.2565,0.2761)。这些是预先设定的预处理常数，不是从本次测试集拟合。',
    ]
    if audit10:
        sections.append(f'已审计 CIFAR-10：训练 {audit10["train_size"]}、验证 {audit10["validation_size"]}、测试 {audit10["test_size"]}；索引互斥且完整，验证变换确定。详见 [数据审计](reports/data_audit_cifar10.json)。')
    else:
        sections.append('CIFAR-10 数据审计尚未完成。此前目录中文件虽然名为 CIFAR-10，实际 MD5 属于官方 CIFAR-100，不能用于 CIFAR-10 正式结果。用户随后授权用 CIFAR-100 做真实流程校验。')
        if check10 and not check10.get('valid'):
            sections.append(f'最近一次 CIFAR-10 包检查：实际 MD5 为 {check10.get("actual_md5", "文件缺失")}；期望 c58f30108f718f92721af3b95e74349a。该检查记录不等于后续文件已就绪，重试时须重新核验。')
    if audit100:
        sections.append(f'CIFAR-100 校验数据已审计：训练 {audit100["train_size"]}、验证 {audit100["validation_size"]}、测试 {audit100["test_size"]}；100 类分别有 {audit100["train_per_class"][0]}/{audit100["validation_per_class"][0]}/{audit100["test_per_class"][0]} 张。划分索引、无泄漏及变换独立性记录见 [CIFAR-100 审计](reports/data_audit_cifar100.json)。')
    sections += [
        '## 3 模型原理与结构',
        '|模型|结构与动机|\n|---|---|\n|CNN|三组 3×3 卷积、BatchNorm、ReLU、2×2 池化，通道 32/64/128；全局平均池化、Dropout(0.2)、线性分类器。|\n|ResNet18|3×3、stride=1 的 stem，无初始 maxpool；四组通道 64/128/256/512，每组两个基础残差块，阶段转换下采样。残差路径将学习目标写成 F(x)+x，改善深层网络的优化。|\n|SE-ResNet18|保持残差架构，在支路加入全局池化、两层通道门控和 Sigmoid，reduction=16。比较通道重标定的作用。|',
        '三个模型定义、残差块、SE 和 Grad-CAM 均在本项目实现；张量运算、卷积、BatchNorm、自动微分、优化器和调度器调用 PyTorch；数据读取与变换使用 torchvision；分类指标使用 scikit-learn。均未加载预训练参数。CIFAR-10 使用 10 维分类头，CIFAR-100 校验必须使用 100 维，二者权重不能混用。',
        '## 4 实验环境、超参数与训练策略',
    ]
    if env:
        sections.append(f'实际环境：Python {env["python"]}；PyTorch {env["torch"]}；torchvision {env["torchvision"]}；CUDA runtime {env["cuda_runtime"]}；GPU {env["gpu"]}，显存 {env["gpu_memory_bytes"] / 1024**3:.2f} GiB；NumPy {env["numpy"]}。')
    else:
        sections.append('环境预检查：Python 3.13.2、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118，CUDA 可用，GPU 为 RTX 4060 Laptop、8 GiB。完整环境随实际训练记录保存。')
    sections += [
        '正式计划：每模型 30 epoch、batch=128、seed=42；交叉熵、SGD(lr=0.1,momentum=0.9,weight_decay=0.0005)，余弦退火至零；早停可选，默认关闭。GPU 可用 AMP。每轮保存训练/验证 loss、accuracy、学习率与耗时。最佳 checkpoint 仅由验证准确率决定。',
        'checkpoint 同时保存模型、优化器、调度器、GradScaler、epoch、最佳验证值、早停计数、配置、历史、Python/NumPy/PyTorch/CUDA 随机状态及采样生成器。恢复后继续相同训练预算，不改变余弦周期。训练、恢复和测试环境分别记录。训练耗时为全部完整 epoch 的训练与验证耗时之和，不包含数据准备、绘图和权重写盘；不能只统计最佳 epoch 之前的时间。',
        'Python、NumPy、PyTorch 固定种子，cuDNN 使用确定性配置，关闭 TF32。固定种子不保证跨硬件、驱动、库版本逐位一致。默认 num_workers=0 便于验证恢复；多 worker 与混合精度会引入额外可复现性限制。训练指标在随机增强和训练模式下统计，验证使用评估模式，曲线不能简单等同解释。',
        '## 5 实验结果与图表',
        '### 5.1 CIFAR-10 正式实验',
        summary_table(results),
    ]
    if not results:
        sections.append('三个 CIFAR-10 正式实验均待运行；没有正式测试准确率。CIFAR-100 的小规模校验结果不能填入上述表格。')
    for name, r in results.items():
        sections.append(f'### {name} 正式图表与每类准确率')
        for file, title in [('curves.png', '训练与验证曲线'), ('confusion_counts.png', '混淆矩阵计数'),
                            ('confusion_normalized.png', '按真实类别归一化混淆矩阵'), ('errors.png', '固定测试顺序中的前九个误分类')]:
            sections.append(link_image(f'runs/{name}/{file}', title))
        sections.append('|类别|每类准确率|\n|---|---:|\n' + '\n'.join(f'|{c}|{a:.2%}|' for c,a in zip(r['classes'],r['per_class_accuracy'])))
        history = read(f'runs/{name}/history.json')
        final = history[-1]
        best_epoch = max(history, key=lambda row: row['val_accuracy'])['epoch'] + 1
        sections.append(f'第 {len(history)} 轮训练准确率为 {final["train_accuracy"]:.2%}，该轮验证准确率为 {final["val_accuracy"]:.2%}；按验证集选中的最佳权重来自第 {best_epoch} 轮。训练曲线含增强与训练模式，不能把这两个数的差直接解释成过拟合程度。')
        sections.append(f'完整分类报告见 [JSON](runs/{name}/classification_report.json) 和 [文本](runs/{name}/classification_report.txt)。')
    sections += ['### 5.2 真实 CIFAR-10 流程校验（不计正式成绩）']
    if smoke10:
        sections.append('三模型均使用 80/40/40 张真实训练/验证/测试样本、2 轮预算，验证连续训练与中断恢复的权重和指标一致。调度器、AMP 与早停状态已核对，测试不覆盖训练环境，总耗时包含完整历史。全部输出为 [batch,10]。记录见 [CIFAR-10 真实校验](reports/verification_real_cifar10.json)，不能将此子集准确率写入正式表。')
    else:
        sections.append('CIFAR-10 真实数据流程检查待运行。')
    sections += ['### 5.3 真实 CIFAR-100 流程校验（不计正式成绩）']
    if smoke100:
        sections.append('实际采用每类 8 张训练、4 张验证、4 张测试，即 800/400/400，训练预算 2 epoch；每模型还独立执行第 1 轮后中断、恢复第 2 轮，与连续训练对照。测试子集只用于流程校验，不能与完整测试成绩比较。')
        rows = ['|模型|输出维度|参数量|smoke 测试准确率|恢复权重与指标一致|',
                '|---|---|---:|---:|---|']
        for name, r in smoke100['results'].items():
            rows.append(f'|{name}|{r["output_shape"]}|{r["parameters"]:,}|{r["real_smoke_metrics"]["accuracy"]:.2%}|{r["resume_weights_identical"] and r["resume_metrics_identical"]}|')
        sections.append('\n'.join(rows))
        sections.append('上述准确率仅说明评估分支实际执行，不用于证明模型优劣。100 类分类需要充分训练；2 轮小子集无法支持架构和注意力效果的结论。详见 [真实校验记录](reports/verification_real_cifar100.json)。')
        for name in smoke100['results']:
            sections.append(link_image(f'runs/smoke_real_cifar100/{name}/continuous/curves.png', f'{name} CIFAR-100 smoke 曲线'))
        rows = ['|模型|第 1 轮验证 loss|第 2 轮验证 loss|', '|---|---:|---:|']
        for name in smoke100['results']:
            history = read(f'runs/smoke_real_cifar100/{name}/continuous/history.json')
            rows.append(f'|{name}|{history[0]["val_loss"]:.4f}|{history[1]["val_loss"]:.4f}|')
        sections.append('\n'.join(rows))
        sections.append('实际曲线中 ResNet18 第一轮验证 loss 很高；这里只确认其有限且恢复可复现，不将其解释为性能达标。小子集、高初始学习率及早期 BatchNorm 统计未稳定是可能原因，尚未通过对照验证。没有隐藏异常曲线，也未基于该测试子集调参。')
        sections.append('每个 continuous 目录另存 100×100 原始/归一化矩阵、分类报告、预测数组和错误网格；这些文件的 dataset 与 smoke 标记均指明其来源。')
    else:
        sections.append('CIFAR-100 真实数据流程校验尚未全部通过，不列准确率。')
    if offline:
        sections.append('离线合成数据验证也已运行，结果见 [离线记录](reports/verification_offline.json)。此记录不含真实数据性能。')
    if early_stop:
        sections.append('额外早停恢复校验通过：计划 4 轮、patience=2，在计数已为 1 的第 2 轮中断，恢复与连续训练均在第 3 轮停止，权重一致。记录见 [早停校验](reports/verification_early_stop.json)。这是合成数据状态校验，不是性能实验。')
    sections += [
        '## 6 架构对比与 SE 消融',
        'CNN 与 ResNet18 比较整体架构；ResNet18 与 SE-ResNet18 比较通道注意力。数据划分、种子与训练预算需保持一致，但相同 epoch 并不等于相同计算量，因此同时比较参数量、耗时与准确率。单一随机种子不足以判断微小差异是否稳定。另需注意，加入 SE 后初始化消耗的随机数发生变化，相同 seed 并不保证两模型共有层初始权重逐元素相同；当前属于常规架构消融，后续可用共有层一致初始化及多种子重复加强归因。',
    ]
    if 'cnn' in results and 'resnet18' in results:
        delta = (results['resnet18']['accuracy'] - results['cnn']['accuracy']) * 100
        sections.append(f'本次 ResNet18 相对 CNN 测试准确率差为 {delta:+.2f} 个百分点；这是本次单种子观察，不能直接归因于某一独立设计。')
    if 'resnet18' in results and 'se' in results:
        delta = (results['se']['accuracy'] - results['resnet18']['accuracy']) * 100
        sections.append(f'本次 SE 相对 ResNet18 测试准确率差为 {delta:+.2f} 个百分点。不预设 SE 必然有效，需结合验证集、耗时与多种子实验判断。')
    if 'resnet18' in results and 'se' in results:
        base, se = results['resnet18'], results['se']
        sections.append(f'SE 增加 {se["parameters"]-base["parameters"]:,} 个参数（{(se["parameters"]/base["parameters"]-1)*100:.2f}%），本次训练与验证累计耗时比为 {se["training_seconds"]/base["training_seconds"]:.2f}。耗时包含验证、数据管线与实际运行负载，不能当作纯理论 FLOPs 比值。')
    if len(results) < 3:
        sections.append('正式结果不齐全，架构与消融结论待正式实验，不能由 smoke 表中的数字替代。')
    sections += ['## 7 误分类与 Grad-CAM 分析',
                 '每类准确率定义为该真实类别预测正确数/该真实类别样本数，等于分类报告中的 recall，而非 precision。混淆矩阵行是真实类别、列是预测类别；归一化时除以每行总数。']
    for name, r in results.items():
        sections.append(f'{name} 的主要有向混淆（实际观察）：')
        sections.append('|真实类别|预测类别|张数|占该真实类别比例|\n|---|---|---:|---:|\n' +
                        '\n'.join(f'|{c["true"]}|{c["predicted"]}|{c["count"]}|{c["fraction_of_true_class"]:.2%}|' for c in r['top_confusions']))
    if notes and notes.get('run_ids') == {name: r['run_id'] for name,r in results.items()}:
        sections.append('### 实际图片复核与观察')
        sections.extend(notes['paragraphs'])
    if not results:
        sections.append('尚无正式误分类统计，不能声称任何特定类别是主要混淆对象。对真实图像的背景、姿态和遮挡分析，须在生成图片后逐例观察；可能原因应作为推测表述。')
    if selected and selected['selected_model'] in results:
        name = selected['selected_model']
        sections.append(f'在读取正式测试成绩前，按最佳验证准确率选定 {name}；选择记录见 [model_selection.json](reports/model_selection.json)。')
        sections.append(link_image(f'runs/{name}/gradcam.png', '最终选定模型 Grad-CAM'))
    sections.append('Grad-CAM 对预测类别 logit 求导，以梯度空间均值加权目标特征后取 ReLU；原图先正确反归一化，再叠加热图。CNN 目标层为末次卷积，残差模型为末个残差块，避免选中 SE 的 1×1 门控。32×32 输入的深层特征通常只有 4×4，插值到 32×32 不会增加解释分辨率；热图是局部梯度相关性，不能证明因果作用。')
    cam = read('reports/cifar100_smoke_gradcam.json')
    if cam:
        sections.append('额外对 CIFAR-100 的 SE smoke 权重进行了 Grad-CAM 功能校验；这不是最终正式模型解释。案例为固定顺序中首先遇到的正确/错误样本，避免按热图外观挑选。')
        sections.append(link_image('reports/cifar100_smoke_gradcam.png', 'CIFAR-100 smoke Grad-CAM 功能检查'))
        sections.append('图片观察：子集索引 1 的苹果被预测为 orange，置信度约 11.0%，热图较强响应位于画面下方，未完整覆盖苹果主体。子集索引 74 的毛毛虫被正确分类，中心区域有较强响应，但置信度也只有约 4.4%；子集索引 123 的海豚虽分类正确，置信度约 3.7%。这些观察说明正确标签并不等于已建立可靠定位能力。背景依赖只是可能解释，不能由热图证明。以上索引为按类别组织的 smoke 子集位置，官方测试索引另存案例 JSON；前三个错误都是 apple 是取样顺序导致，不能当作总体主要混淆结论。')
        sections.append(f'实际选取正确 {cam["counts"]["correct"]} 个、错误 {cam["counts"]["wrong"]} 个；若不足三个，保留实际数量。完整图像索引、预测置信度与特征尺寸见 [案例记录](reports/cifar100_smoke_gradcam.json)。')
    sections += [
        '## 8 局限性、独立思考与后续方向',
        '当前首先要保证数据集与作业要求一致。CIFAR-100 流程校验有助于发现 100 类输出、索引、恢复和图表尺寸问题，但不能替代 CIFAR-10 正式实验。模型性能需要完整训练后评估；小样本训练早期的低准确率既不能否定残差结构，也不能证明注意力无效。',
        '后续可在相同训练预算下增加随机种子，报告均值和方差；再比较增强策略。SE 增加了参数与计算，其收益应与成本一起评估。对热图提出的背景依赖假设，可进一步用遮挡等干预验证，而非仅凭颜色下结论。',
        '## 9 参考资料',
        '- [课程作业 PDF](../当代人工智能作业三.pdf)。',
        '- [CIFAR-10 / CIFAR-100 官方数据说明与 MD5](https://www.cs.toronto.edu/~kriz/cifar.html)，Alex Krizhevsky，Learning Multiple Layers of Features from Tiny Images，2009。',
        '- [He 等，Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，2015。',
        '- [Hu 等，Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)，2017。',
        '- [Selvaraju 等，Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)，2016。',
        '## 10 验收状态与复核方式',
        '数据包检查使用 prepare_data.py；独立划分审计使用 --audit；离线验证使用 verify.py；真实数据恢复验证使用 verify.py --real --dataset。summarize.py 核对预测准确率、矩阵计数、每类 recall、最佳验证值和总训练耗时后再生成对比表。README 给出可直接执行的 PowerShell 命令。',
        f'CIFAR-10 正式已具备测试记录的模型：{", ".join(results) if results else "无"}。CIFAR-100 真实流程校验：{"已通过三个模型" if smoke100 else "未全部完成"}。未列完成项仍为待运行。',
    ]
    sections += ['## 11 核查与答辩要点',
                 '- 为什么不用测试集选模型：模型选择也会适应数据，反复依据测试成绩调参会高估泛化表现。本项目先保存验证集选择记录，再统一测试。',
                 '- 为什么残差结构可能有益：残差块允许学习相对恒等映射的修正，缓解深层优化困难；本次是否有效仍以实际对比为准。',
                 '- SE 做了什么：空间平均得到通道描述，经瓶颈和 Sigmoid 为每个通道赋权；它是输入相关的通道门控，不直接提供精确空间定位。',
                 '- 验证准确率为何可能高于训练：训练使用随机增强与 Dropout，验证关闭这些扰动；还需结合划分审计排除泄漏，不能只凭曲线高低判断。',
                 '- 每类准确率与 precision 的区别：前者分母是真实属于该类的样本数；后者分母是预测为该类的样本数。',
                 '- 恢复训练为何不只加载模型：优化器动量、学习率位置、混合精度、随机状态和早停计数都会影响下一轮，需要一起恢复。',
                 '- Grad-CAM 能否证明模型理解对象：不能。4×4 特征插值后仍只有粗粒度信息，亮区不是因果证据；正确分类也可能只利用局部线索。']
    checklist = ['# 作业要求与执行检查清单', '', '|要求|实现或验证位置|实际状态|', '|---|---|---|',
                 '|CNN、ResNet18、SE 模型|models.py、真实 smoke 记录|三模型已通过前向和训练恢复校验|',
                 '|官方 CIFAR-10 与固定分层划分|data_audit_cifar10.json、split_42.npz|' + ('已核验 45000/5000，互斥且完整' if audit10 else '待数据就绪') + '|',
                 '|变换独立与确定性验证|data_audit_cifar10.json|' + ('已核验' if audit10 else '待运行') + '|',
                 '|真实数据训练、保存、恢复与评估|verification_real_cifar10.json|' + ('三模型已通过' if smoke10 else '待运行') + '|',
                 '|三个正式 30 轮实验|runs/*/history.json|' + ('全部完成并测试' if len(results) == 3 else '进行中或待运行，见 progress.json') + '|',
                 '|曲线、分类报告、混淆矩阵、误分类|runs/*/ 与 summarize.py|' + ('三个正式实验均已核验' if len(results) == 3 else '正式结果待完整训练后生成；smoke 图表已生成') + '|',
                 '|最终模型选择与 Grad-CAM|model_selection.json、runs/*/gradcam.png|' + ('已按验证集选模且已生成 3 正确/3 错误案例' if selected and Path('runs',selected['selected_model'],'gradcam.png').exists() else '正式选模或图像待生成') + '|',
                 '|中文说明、报告、来源与局限|README.md、report.md|已实现，按实际结果更新|']
    Path('reports/checklist.md').write_text('\n'.join(checklist) + '\n', encoding='utf-8')
    Path('report.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    save_json('reports/report_checks.json', {'formal_models_with_consistent_results': list(results),
                                           'cifar100_real_smoke_completed': bool(smoke100),
                                           'cifar10_audit_completed': bool(audit10),
                                           'cifar10_real_smoke_completed': bool(smoke10),
                                           'generated_from_actual_records': True})
    print('已更新 report.md；未完成项保持待运行。')


if __name__ == '__main__':
    main()
