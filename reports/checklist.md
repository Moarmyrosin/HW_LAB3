# 作业要求与执行检查清单

|要求|实现或验证位置|实际状态|
|---|---|---|
|CNN、ResNet18、SE 模型|models.py、真实 smoke 记录|三模型已通过前向和训练恢复校验|
|官方 CIFAR-10 与固定分层划分|data_audit_cifar10.json、split_42.npz|已核验 45000/5000，互斥且完整|
|变换独立与确定性验证|data_audit_cifar10.json|已核验|
|真实数据训练、保存、恢复与评估|verification_real_cifar10.json|三模型已通过|
|三个正式 30 轮实验|runs/*/history.json|全部完成并测试|
|曲线、分类报告、混淆矩阵、误分类|runs/*/ 与 summarize.py|三个正式实验均已核验|
|最终模型选择与 Grad-CAM|model_selection.json、runs/*/gradcam.png|已按验证集选模且已生成 3 正确/3 错误案例|
|中文说明、报告、来源与局限|README.md、report.md|已实现，按实际结果更新|
