import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================

class EZPC(torch.nn.Module):
    def __init__(
        self, 
        # concept_matrix：论文 Φ 的转置 Φ^T，形状 (m, d)
        # m 个概念 × d 维 CLIP 文本嵌入，每一行 是一个概念
        # 
        # 论文 Φ 是 (d, m)、每一列 一个概念
        # 代码按行存，故 concept_matrix = Φ^T，下面 .T 转回 Φ
        concept_matrix, 
        # weight：论文式(5) 的平衡系数 λ
        # 总损失 L_total = L_match + λ·L_recon
        # forward 里就是 matching + weight * reconstruction
        weight
    ):
        super().__init__()

        # 两个入参原样存成成员
        # 
        # 注意：concept_matrix（= Φ^T）存成普通属性，不是 Parameter 也不是 buffer
        # 它是「固定不动的锚」，转置后即论文的 Φ，训练全程不更新
        # （不进 optimizer，即便 mse 给它梯度也不会被更新）
        # 只在匹配损失 L_match（式(3)）里当目标、把 A 往 Φ 上拽
        # 
        self.concept_matrix = concept_matrix
        self.weight = weight

        # self.A：论文里的可学习投影矩阵 A，形状 (d, m)（§3.1）
        # 初值 A^(0) = Φ（式(10)）
        # concept_matrix = Φ^T，故 concept_matrix.T = Φ
        # A 初值就是 Φ（不是 Φ 的转置）
        # 
        # A 的每一列 = 一个概念在 CLIP 空间里的方向向量
        # 图像 v_x 点乘 A 得概念激活 c_x = v_x·A（式(2)）
        # 例：RN50 的 d=1024
        #     m=概念数（由 概念表 去重 排序 后决定，非定值，如 ImageNet-1k m=4751）
        #     → 论文 Φ 与 A 都是 (d, m)、concept_matrix 是 (m, d)
        #
        # .clone()  复制独立内存：
        #           concept_matrix.T(=Φ) 只是转置视图、与 concept_matrix 共享底层数据，
        #           不复制的话训练时原地更新 A 会连带改坏 concept_matrix（它要当固定锚 Φ）
        # .detach() 摘出计算图，得到干净的叶子张量，梯度不回流到 Φ，A 从全新起点开始训
        self.A = nn.Parameter(
            data=concept_matrix.T.clone().detach(), 
            # 显式声明可训练（nn.Parameter 默认即 True，写出来更醒目）
            requires_grad=True
        )

    # @torch.no_grad()：原地改参数、不记入计算图（这步不参与反传）
    @torch.no_grad()
    def normalize_weights(self):
        # 可学习投影矩阵 A，形状 (d, m)
        # 
        # 把 A 的每一列投回单位球（对应论文式(11) 的列归一化 A_:,j ← A_:,j / ‖A_:,j‖）
        # 
        # norm(dim=0) 沿 d 维求范数 = 逐「列」
        # +1e-8 防除零
        # 
        # train.py 在每次参数更新后调用一次（默认整批训练 → 约等于每个 epoch 一次）
        # 
        # 作用：把各概念方向重新约束成单位范数，稳定概念几何、防训练漂移
        #       （A 初值各列本就是单位范数 —— Φ 各列已 L2 归一化 —— 此步在训练中持续维持）
        self.A.copy_( self.A / (self.A.norm(dim=0, keepdim=True) + 1e-8) )

    def forward(
        self,
        vx,
        ty,
        # y：本批样本的真实标签，形状 (B,)，交给 F.cross_entropy 当目标
        # 本方案只在 seen 类列上做 CE，故 y 是「seen 局部下标」0..K_seen-1（不是全局类 id）
        # 例：CIFAR-100 共 100 类、其中 80 个 seen → ty 只传这 80 个 seen 类名嵌入，y∈0..79
        #     全局→局部的重映射在 train.py 里用 seen_map 做好后才传进来（与 test.py 受限口径一致）
        y
    ):
        # 匹配损失 L_match（论文式(3)）= MSE(A, Φ)，其中 Φ = concept_matrix.T（固定锚）
        # 把 A 往 Φ 上拽，保证各列始终贴近已知概念方向、维持可解释性
        # 
        # L_match = A - Φ
        matching_loss = F.mse_loss(self.A, self.concept_matrix.T)

        # EZPC（概念瓶颈）分布的 logits = v_x A Aᵀ Tᵀ，形状 (B, K_seen)
        # 等价于 (v_x A)·(t_k A) = ⟨c_x, c_k⟩ = Σ_j c_x[j]·c_k[j]（论文式(8)，推理取 argmax 即式(6)）
        # 几何上：A Aᵀ 是秩≤m 的半正定度量，把打分瓶颈到 m 个单位范数概念方向上
        # （c_x=v_x A、c_k=t_k A 是图像/类名的概念激活；逐概念乘积 c_x⊙c_k 即式(7) 的 s_{x,k}）
        # vx = v_x 图像嵌入 (B, d)、ty = seen 类名嵌入 (K_seen, d)，二者上游均已 L2 归一化
        # 注意：本方案 ty 只含 seen 类（train.py 已切片），故列空间是 seen 子集、不含 unseen 列
        ezpc_logits = (vx @ self.A @ self.A.T @ ty.T)

        # —————————— 改动点：重构损失(KL) → 监督损失(交叉熵) ——————————
        # 原 L_recon = KL(EZPC ‖ CLIP)：把 EZPC 分布往 CLIP 零样本分布上拽
        #   目标是 CLIP 自己 → EZPC 的天花板就是 CLIP，准确率追不过原始 CLIP 零样本
        # 现 L_ce   = CE(EZPC_logits, y)：直接把 EZPC 分布往真实标签上拽
        #   监督信号来自真标签 → 摆脱 CLIP 天花板，用标签把精度往上抬
        #
        # F.cross_entropy 内部自带 log_softmax，传「原始 logits」即可，别先 softmax
        # ezpc_logits 是 (B, K_seen)、y 是 seen 局部下标 0..K_seen-1，二者列对齐
        # 例：某图 seen 局部下标是 3 → 期望第 3 列 logit 最大；CE = −log softmax(ezpc_logits)[行,3]
        #
        # 为什么只在 seen 列上做（本方案的选择）：本仓库是广义零样本设定，训练只覆盖 seen 类。
        #   若把全 100 类一起喂 CE，unseen 列永远是「错答案」、会被一路压低 → seen 精度↑但 unseen 泛化↓。
        #   这里 ty 只给 seen 列，CE 只在 seen 类之间做区分、根本不碰 unseen 列 → 护住 unseen 的零样本能力。
        ce_loss = F.cross_entropy(ezpc_logits, y)

        # 总损失 L_total = L_match + λ·L_ce（λ = self.weight）
        # 注意：λ 现在平衡的是「匹配锚 A→Φ」与「真标签监督」，语义已不同于原文的「匹配 vs 重构」
        total_loss = matching_loss + self.weight * ce_loss

        # 返回三个损失，位置与原来一致：train.py 按 (matching, ce, total) 解包记录
        # 中间项现在是 CE 损失（接线后的 train.py 已把变量/打印改成 ce）
        return matching_loss, ce_loss, total_loss