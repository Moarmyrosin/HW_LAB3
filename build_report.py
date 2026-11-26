"""从实测记录生成精简 report.md 与固定 5 页 PDF；原完整版另存 report_full.md。"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from summarize import collect
DISPLAY={'cnn':'CNN','resnet18':'ResNet18','se':'SE-ResNet18'}


def text(value): return {'type':'p','text':value}
def heading(value): return {'type':'h','text':value}
def caption(value): return {'type':'caption','text':value}
def table(headers,rows,widths): return {'type':'table','headers':headers,'rows':rows,'widths':widths}
def image(name): return {'type':'image','path':f'reports/compact_figures/{name}.png'}
def reference(label,url=None): return {'type':'ref','text':label,'url':url}


def content():
    records=collect()
    if set(records)!={'cnn','resnet18','se'}:
        raise RuntimeError('提交版需要三个完整正式实验，缺失结果时停止，避免填入推测值')
    audit=json.loads((ROOT/'reports/data_audit_cifar10.json').read_text(encoding='utf-8'))
    env=json.loads((ROOT/'runs/se/train_environment.json').read_text(encoding='utf-8'))
    selected=json.loads((ROOT/'reports/model_selection.json').read_text(encoding='utf-8'))['selected_model']
    if selected!='se':
        raise RuntimeError('当前精简版案例针对本次 SE 最终模型；若重新训练改变选模，请同步复核案例')
    cnn,res,se=[records[n] for n in DISPLAY]
    arch_gain=(res['accuracy']-cnn['accuracy'])*100
    se_gain=(se['accuracy']-res['accuracy'])*100
    added=se['parameters']-res['parameters']
    extra_time=(se['training_seconds']/res['training_seconds']-1)*100
    matrix=__import__('numpy').loadtxt(ROOT/'runs/se/confusion_counts.csv',delimiter=',')
    pages=[
      {'label':'任务、数据与模型','blocks':[
       {'type':'title','text':'CIFAR-10 图像分类实验报告'},
       {'type':'subtitle','text':'基础 CNN、ResNet18 与 SE 通道注意力的对比实验'},
       text(f'摘要：在同一分层划分与 30 轮预算下，从头训练三种模型。按验证集选择的 SE-ResNet18 在完整测试集上取得 {se["accuracy"]:.2%} 准确率，较 ResNet18 提高 {se_gain:.2f} 个百分点。下文结合课程原理、误分类和 Grad-CAM 分析结果及局限。'),
       heading('1 任务目标与课程内容'),
       text('本实验对应课程“从图像获得语义”的目标[1]，属于图像级分类：输入一张 32×32 RGB 图像，输出 10 个类别分数。目标识别课件对分类、定位、检测与分割的区分[3]帮助明确了评价范围，本实验使用 top-1 准确率。'),
       text('CNN 课件介绍卷积、非线性、池化与反向传播[2]。卷积通过局部计算和参数共享提取模式，ReLU 提供非线性，池化减小空间尺寸并增强对小幅变换的适应性；多层叠加形成由局部到更抽象的表示。'),
       heading('2 数据集与实验规范'),
       text(f'使用官方 CIFAR-10[4]：训练集 50,000 张、测试集 10,000 张。以 seed=42 在每类中抽取 500 张验证，得到 {audit["train_size"]:,}/{audit["validation_size"]:,} 的训练/验证划分。保存 split_42.npz，并核验两者互斥、并集覆盖官方训练集；测试集每类 1,000 张。'),
       text('训练使用随机裁剪（padding=4）、水平翻转、ToTensor 和归一化；验证/测试仅使用确定性转换，采用独立数据对象。固定 mean=(0.4914,0.4822,0.4465)，std=(0.2470,0.2435,0.2616)，未从测试集拟合。仅训练集更新参数，验证集选最佳权重，全部方案确定后统一测试。'),
       heading('3 模型设计与改进动机'),
       table(['模型','核心结构','设计目的'],[
        ['基础 CNN','3 组 3×3 卷积、BN、ReLU、2×2 最大池化，通道 32/64/128；全局平均池化、Dropout(0.2)、10 维分类头。','作为参数量较小、易理解的基线。'],
        ['ResNet18','3×3、步长 1 的 stem，无初始最大池化；通道 64/128/256/512，四阶段各 2 个残差块；输出 10 类。','用 F(x)+x 的捷径改善深层优化；尺寸变化时使用投影。[5]'],
        ['SE-ResNet18','保持残差架构，在支路末端加入全局池化、瓶颈映射（reduction=16）与 Sigmoid 通道门控。','按输入重标定通道响应，与 ResNet18 形成架构消融。[6]']
       ],[90,260,155]),
       text('模型结构、残差块、SE 和 Grad-CAM 在项目中实现；卷积、自动微分和优化器调用 PyTorch，数据接口与变换使用 torchvision，指标使用 scikit-learn。三种模型均未加载预训练权重。')
      ]},
      {'label':'训练设置与总体结果','blocks':[
       heading('4 实验环境与训练策略'),
       text(f'实测环境：Python {env["python"]}，PyTorch {env["torch"]}，torchvision {env["torchvision"]}，CUDA {env["cuda_runtime"]}；RTX 4060 Laptop GPU（8 GiB）。三模型固定 seed=42、batch=128、30 epoch，使用 AMP，num_workers=0。'),
       text('损失为交叉熵 L=-log p(y|x)，直接将 logits 输入 CrossEntropyLoss；softmax 仅用于预测置信度。优化器为 SGD：初始学习率 0.1、momentum=0.9、weight_decay=0.0005，余弦退火；早停作为可选项，本次正式对比关闭。正则化与数据增强对应课件的训练实践[2]。'),
       text('固定 Python/NumPy/PyTorch 随机种子并设置确定性 cuDNN。checkpoint 保存权重、优化器、调度器、AMP、早停计数与随机状态。真实小子集验证了连续训练与中断恢复一致，另核验非零早停计数恢复；CIFAR-100 仅用于流程校验，未纳入正式成绩。'),
       heading('5 总体结果与训练过程'),
       table(['模型','参数量','最佳验证 / %','测试 / %','耗时 / min'],[
        [DISPLAY[n],f'{records[n]["parameters"]:,}',f'{records[n]["best_validation"]*100:.2f}',f'{records[n]["accuracy"]*100:.2f}',f'{records[n]["training_seconds"]/60:.2f}'] for n in DISPLAY
       ],[110,110,100,90,95]),
       caption('表 1  三组均完成 30 轮。耗时统计全部完整轮次的训练与验证，不含数据准备、绘图和权重写盘。最佳权重分别来自第 30、29、28 轮；正式测试均为完整 10,000 张。'),
       image('learning_curves'),
       caption('图 1  三模型训练/验证 loss 与 accuracy 的合并曲线，来自逐轮 history.json。所有图线均保留原始轮次和实际波动。'),
       text('CNN 前期验证波动明显，学习率降低后改善；两种残差模型后期训练准确率接近 99%，验证改善趋缓。训练指标包含增强和训练模式，验证关闭 Dropout 等扰动，因此曲线差距不能直接等同于过拟合程度。')
      ]},
      {'label':'类别表现与混淆','blocks':[
       heading('6 每类准确率与混淆分析'),
       text('每类准确率=该真实类别预测正确数/该真实类别样本数，即 recall；precision 的分母则是预测为该类的样本数。下表三模型均按相同测试集统计，总准确率为全部预测正确数/10,000。'),
       table(['类别','CNN / %','ResNet18 / %','SE / %'],[
        [label]+[f'{records[n]["per_class_accuracy"][i]*100:.1f}' for n in DISPLAY]
        for i,label in enumerate(['飞机 airplane','汽车 automobile','鸟 bird','猫 cat','鹿 deer','狗 dog','青蛙 frog','马 horse','船 ship','卡车 truck'])
       ],[175,110,110,110]),
       caption('表 2  合并的每类准确率。完整 precision、recall、F1 与 support 另存于各实验 classification_report.json/txt。'),
       image('confusion_selected'),
       caption('图 2  最终模型 SE-ResNet18 的原始计数与按真实类别归一化矩阵。行是真实类别、列是预测类别，每行样本数为 1,000；右图以百分比取整标注，精确比例由左图计数/1,000 得到。plane/auto 为 airplane/automobile 的缩写。'),
       text(f'实际主要混淆为 dog→cat：{int(matrix[5,3])} 张（{matrix[5,3]/10:.1f}%），cat→dog：{int(matrix[3,5])} 张（{matrix[3,5]/10:.1f}%）；bird→deer 与 truck→automobile 各 26 张（2.6%）。这些结论来自完整矩阵，不是根据少量展示案例推断。'),
       text(f'SE 对各类的影响不一致：cat 的 recall 从 {res["per_class_accuracy"][3]:.1%} 升至 {se["per_class_accuracy"][3]:.1%}，dog 则从 {res["per_class_accuracy"][5]:.1%} 降至 {se["per_class_accuracy"][5]:.1%}。总体准确率上升不代表所有类别都改善。')
      ]},
      {'label':'架构消融与误分类','blocks':[
       heading('7 架构对比、SE 消融与误分类'),
       text(f'架构对比：ResNet18 相对 CNN 的测试准确率提高 {arch_gain:.2f} 个百分点，显示本次预算下残差架构有更强的拟合与泛化表现。但参数量、深度及初始化都发生变化，不能把差异完全归因于单个设计。'),
       text(f'SE 消融：相对 ResNet18，测试准确率提高 {se_gain:.2f} 个百分点，增加 {added:,} 个参数（{added/res["parameters"]*100:.2f}%），训练与验证耗时增加约 {extra_time:.1f}%。这是单个 seed 的结果；加入 SE 会改变随机数消耗，共有层也未逐层对齐初值，因此仍需多种子与共有层一致初始化来加强归因。'),
       image('errors_selected'),
       caption('图 3  SE 模型的 6 个代表性误分类：# 为官方测试索引，左侧为真实类，右侧为预测类，p 为原 AMP 测试置信度。案例由固定顺序的错误中选取，覆盖不同错误类型，不用于估计整体错误分布。'),
       text('案例观察：#37 是灰白车辆近景，automobile 被预测为 truck；#58 是 deer 头部特写，ResNet18 与 SE 都预测为 dog；#35 为 bird 正面头部，CNN 与 SE 都预测为 cat。低分辨率下局部轮廓和纹理相似是可能解释，尚未通过遮挡等干预验证。'),
       text('置信度并不保证正确：#58 的错误预测在 ResNet18 中达到 98.45%，SE 为 92.04%；#118 的 bird 被 SE 预测为 frog，置信度为 90.64%（见完整误分类网格）。单例不能替代总体校准指标，但提醒应结合错误样本与类别指标分析模型。'),
       text('独立思考：SE 的收益需同时考虑精度和成本。课程提出的增强、正则化与学习率实践值得进一步比较，但本次没有在看到正式测试结果后继续调参；后续应先制定验证方案，再做新实验。')
      ]},
      {'label':'可解释性、局限与参考','blocks':[
       heading('8 Grad-CAM、局限与改进方向'),
       text('对按验证集选定的第 28 轮 SE 权重，取最后残差块的 4×4 空间特征，对预测类别 logit 求导，以梯度空间均值为通道权重，加权求和后 ReLU，再插值到 32×32[7]。避开 SE 的 1×1 门控层；原图使用正确反归一化显示。'),
       image('gradcam_selected'),
       caption('图 4  固定测试顺序中最先出现的 3 个正确与 3 个错误案例，每组左为原图、右为 Grad-CAM。解释采用 float32 单图重算，置信度与 AMP 测试存在细小数值差异；图中 100.0% 为四舍五入显示。'),
       text('观察：#0 的 cat 热区位于头部与身体上方，#1/#2 的 ship 热区覆盖部分船体或上层结构。错误案例 #15 的热区在船体局部且周围大面积绿色，#25 的响应偏画面下方，#35 集中于正面头部。背景依赖或纹理误认只是推测，热图不能证明因果。'),
       text('局限与后续：4×4 特征上采样不会产生精细定位信息；单种子结果尚无方差与跨硬件复现证据。后续可用多种子重复、相同共有层初始化、验证集内的增强/早停对比，并用遮挡实验检验背景依赖。本文所有性能数字来自实际正式运行，未增加推测结果。'),
       heading('9 参考资料与核查位置'),
       reference('[1] 课程课件 vision_01_intro.pptx，第 3 张：图像语义理解。'),
       reference('[2] 课程课件 vision_07_cnns.pptx，第 42-43、83-90、122-124、130-131 张：交叉熵、CNN、ResNet 与训练实践。'),
       reference('[3] 课程课件 vision_08_objrec.pptx，第 2 张：目标识别任务类型。以上均按文件幻灯片序号计。'),
       reference('[4] Krizhevsky. CIFAR-10 / CIFAR-100 官方数据说明，2009。','https://www.cs.toronto.edu/~kriz/cifar.html'),
       reference('[5] He 等. Deep Residual Learning for Image Recognition，2015。','https://arxiv.org/abs/1512.03385'),
       reference('[6] Hu 等. Squeeze-and-Excitation Networks，2017。','https://arxiv.org/abs/1709.01507'),
       reference('[7] Selvaraju 等. Grad-CAM，2016。','https://arxiv.org/abs/1610.02391'),
       caption('核查：完整图表与数值见 runs/{cnn,resnet18,se}/；数据和恢复审计见 reports/；完整记录见 report_full.md；运行命令见 README.md。提交正文共 5 页，含本页参考资料。')
      ]}
    ]
    return {'title':'CIFAR-10 图像分类实验报告','pages':pages,
            'run_ids':{n:r['run_id'] for n,r in records.items()},
            'selected_model':selected,'page_limit':5}


def markdown(data):
    lines=[]
    for number,page in enumerate(data['pages'],1):
        lines.append(f'<!-- PDF 第 {number}/5 页：{page["label"]} -->')
        for block in page['blocks']:
            kind=block['type']
            if kind=='title': lines.append('# '+block['text'])
            elif kind=='h': lines.append('## '+block['text'])
            elif kind=='subtitle': lines.append('*'+block['text']+'*')
            elif kind=='image': lines.append(f'![紧凑实验图]({block["path"]})')
            elif kind=='table':
                lines.append('| '+' | '.join(block['headers'])+' |')
                lines.append('| '+' | '.join(['---']*len(block['headers']))+' |')
                lines.extend('| '+' | '.join(row)+' |' for row in block['rows'])
            elif kind=='ref' and block.get('url'):
                lines.append(block['text']+' [原始资料]('+block['url']+')')
            else: lines.append(block['text'])
            lines.append('')
    return '\n'.join(lines)


def renderer_python():
    if importlib.util.find_spec('reportlab'):
        return sys.executable
    poppler=shutil.which('pdftoppm')
    if poppler:
        path=Path(poppler).resolve().parents[5]/'dependencies/python/python.exe'
        if path.exists(): return str(path)
    raise RuntimeError('缺少 PDF 排版依赖。请先执行 python -m pip install -r requirements-report.txt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-figures',action='store_true')
    args=parser.parse_args()
    if not args.reuse_figures:
        subprocess.run([sys.executable,str(ROOT/'report_figures.py')],cwd=ROOT,check=True)
    data=content()
    (ROOT/'reports/compact_report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'report.md').write_text(markdown(data),encoding='utf-8')
    subprocess.run([renderer_python(),str(ROOT/'render_report.py')],cwd=ROOT,check=True)
    print('已生成精简 report.md 与 5 页 PDF；完整实验记录保留为 report_full.md。')


if __name__=='__main__':
    main()
