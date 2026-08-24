# Counterfactual Conditional Flow Preference Optimization for Context-Aware TTS

> 暂定简称：**CC-FPO**（Counterfactual Conditional Flow Preference Optimization）  
> 暂定中文名：**反事实条件流偏好优化**

## 1. 研究问题

当前 Ming-Omni-TTS 的 Context TTS 输入为：

```text
<role>HUMAN</role>Please generate speech based on the following description.

speaker_1:<spk>PROMPT_AUDIO_PATCH×1</spk>

Conversation History:<|conv_start|>user:{context_text}<|conv_end|>

Text input:

{content}<role>ASSISTANT</role><audio>TARGET_AUDIO_PATCH×N
```

模型仅在 `<audio>` 后的目标语音 patch 区域计算：

\[
L_{\mathrm{base}}
=
L_{\mathrm{flow}}+0.01L_{\mathrm{stop}}.
\]

`context_text`、`content`、参考音频和 `<audio>` 起始位置本身不计算文本生成损失。对于一条目标音频，Flow Matching 构造：

\[
x_t=(1-t)x_0+t x_1,
\qquad
u_t=x_1-x_0,
\]

并优化：

\[
L_{\mathrm{flow}}
=
\left\|
v_\theta(x_t,t\mid C,Y,S)-u_t
\right\|_2^2,
\]

其中：

- \(C\)：复杂对话语境 `context_text`；
- \(Y\)：要合成的文本 `content`；
- \(S\)：说话人条件或参考音频；
- \(x_1\)：目标音频 latent；
- \(x_0\)：采样噪声；
- \(v_\theta\)：模型预测的 Flow velocity。

### 1.1 Condition neglect 问题

普通 Flow MSE 只要求模型在训练样本上预测正确的 \(u_t\)，并不显式要求模型区分：

```text
这条目标音频与当前context是否匹配。
```

当 `content`、说话人、目标音频 teacher forcing 等条件已经足够强时，大模型可能弱化甚至忽略复杂 `context_text`，形成 **context condition neglect**：

- 音频自然度、WER、SIM 正常；
- 换一个语境后，生成结果变化很小；
- 模型没有真正学习“语境 → 说话策略 → 声学表现”的条件关系。

## 2. 核心思想

对于每条真实样本，保持以下变量不变：

- `content`；
- speaker / prompt audio；
- target audio；
- Flow 时间步 \(t\)；
- 噪声 \(x_0\)；
- 混合状态 \(x_t\)；
- 目标速度 \(u_t\)。

只替换 `context_text`，构造：

1. **正确语境** \(C^+\)：原始目标音频对应的真实 context；
2. **等价语境** \(C^\approx\)：表达不同但说话策略基本相同的 context；
3. **反事实语境** \(C^-\)：要求不同甚至相反说话策略的 context。

模型应当满足：

\[
v_\theta(x_t,t\mid C^+,Y,S)
\approx
v_\theta(x_t,t\mid C^\approx,Y,S),
\]

同时：

\[
v_\theta(x_t,t\mid C^+,Y,S)
\text{ 比 }
v_\theta(x_t,t\mid C^-,Y,S)
\text{ 更匹配目标音频。}
\]

该方法不生成 `doubao_context_text`，不引入 CoT，不增加推理步骤。`doubao_context_text` 仅作为训练期的语义标注，用于构造高质量等价/反事实语境关系。

## 3. 数据定义

一条原始记录包含：

```json
{
  "context_text": "康复师，我父亲听力不太好……能麻烦您凑近一点，用比较轻的声音再重复一遍吗？",
  "doubao_context_text": "以普通人日常闲聊的语气说话，不要生硬朗读，小声音量说话",
  "content": "好的。第一步，双手扶住栏杆……",
  "audio": "target.wav"
}
```

记为：

\[
(C^+,D^+,Y,S,A),
\]

其中 \(D^+\) 是 `doubao_context_text`。

### 3.1 说话策略属性

可从 \(D\) 中离线抽取或计算以下属性：

| 属性 | 示例取值 |
|---|---|
| 情绪 | 平静、高兴、悲伤、愤怒、恐惧、厌恶、安慰 |
| 音量 | 小声、正常、大声 |
| 语速 | 慢、正常、快 |
| 距离 | 靠近、正常距离、远距离 |
| 交流方式 | 日常交流、耐心解释、正式播报、公开提醒 |
| 表达自然度 | 自然闲聊、生硬朗读、戏剧化表达 |
| 方言/口音 | 普通话、粤语、川渝话等 |

这些属性只用于数据采样和分析，不要求在推理时提供。

### 3.2 等价语境 \(C^\approx\)

选择与 \(D^+\) 语义接近的样本，或对 \(C^+\) 进行语义保持的改写：

```text
C+：对方听力不好，请凑近后轻声重复。
C≈：听者听觉较弱，麻烦靠近一些，降低音量并耐心复述。
```

要求：

- 说话策略相同；
- 具体措辞和场景表述不同；
- 不改变决定目标韵律的关键信息。

### 3.3 反事实语境 \(C^-\)

选择一个在关键说话策略上与 \(D^+\) 不同的 context：

```text
C+：靠近听力不好的老人，小声耐心重复。
C-：在嘈杂大厅向远处的人大声、清晰地提醒。
```

优先使用 hard negative：

- 场景主题相似，但音量/情绪/语速不同；
- content 长度、说话人、语言尽可能相近；
- 避免仅靠明显的领域词汇判断正负；
- 避免与目标 `content` 逻辑上完全不可组合的 context。

## 4. 同步 Flow 状态构造

正负条件必须共享完全相同的 Flow 随机变量：

```python
t = sample_t(batch_size)
x0 = torch.randn_like(x1)
xt = (1 - t) * x0 + t * x1
ut = x1 - x0
```

然后分别预测：

\[
v^+
=
v_\theta(x_t,t\mid C^+,Y,S),
\]

\[
v^\approx
=
v_\theta(x_t,t\mid C^\approx,Y,S),
\]

\[
v^-
=
v_\theta(x_t,t\mid C^-,Y,S).
\]

如果正负分支使用不同 \(x_0\) 或不同 \(t\)，误差差异将同时包含随机轨迹差异，无法可靠表示条件偏好。

## 5. 基础反事实 Margin Loss

定义正确和错误 context 下的 per-sample Flow error：

\[
e^+
=
\frac{1}{|M|}
\sum_{i\in M}
w(t)
\left\|v_i^+-u_{t,i}\right\|_2^2,
\]

\[
e^-
=
\frac{1}{|M|}
\sum_{i\in M}
w(t)
\left\|v_i^--u_{t,i}\right\|_2^2,
\]

其中 \(M\) 是 `<audio>` 后目标 audio patch 的有效 mask。

要求：

\[
e^+ + m < e^-.
\]

对应 margin loss：

\[
L_{\mathrm{margin}}
=
\max(0,m+e^+-e^-).
\]

基础总损失：

\[
L
=
L_{\mathrm{flow}}^+
+0.01L_{\mathrm{stop}}
+\lambda_{\mathrm{margin}}L_{\mathrm{margin}}.
\]

该版本实现简单，可作为第一阶段验证 context preference 是否有效的 baseline。

## 6. Counterfactual Conditional Flow Preference Optimization

### 6.1 Flow 条件兼容性分数

将负加权 Flow error 定义为条件与目标音频的兼容性分数：

\[
s_\theta(C,A,t)
=
-\frac{1}{2\sigma_t^2}
\frac{1}{|M|}
\sum_{i\in M}
\left\|
v_{\theta,i}(x_t,t\mid C,Y,S)-u_{t,i}
\right\|_2^2.
\]

分数越高，表示给定 context 下的 velocity field 越匹配目标音频轨迹。

需要说明：该分数是基于 Flow Matching regression objective 的 likelihood surrogate，不应在论文中未经推导直接宣称为精确 log probability。

### 6.2 条件偏好差

\[
\Delta_\theta
=
s_\theta(C^+,A,t)
-
s_\theta(C^-,A,t).
\]

如果 \(\Delta_\theta>0\)，模型认为目标音频与正确 context 更匹配。

### 6.3 Reference anchor

冻结 SFT 初始模型 \(\theta_{\mathrm{ref}}\)，计算：

\[
\Delta_{\mathrm{ref}}
=
s_{\mathrm{ref}}(C^+,A,t)
-
s_{\mathrm{ref}}(C^-,A,t).
\]

最终偏好目标：

\[
L_{\mathrm{CCFPO}}
=
-\log\sigma
\left(
\beta
[\Delta_\theta-\Delta_{\mathrm{ref}}]
\right).
\]

Reference anchor 的作用包括：

- 防止模型仅通过无限恶化负 context 分支获得较大 margin；
- 限制模型偏离原有 TTS 能力；
- 保持 WER、speaker similarity 和自然度。

### 6.4 正样本 anchor

偏好目标必须与原始正样本 Flow loss 联合训练：

\[
L_{\mathrm{anchor}}
=
e^+.
\]

总损失：

\[
\boxed{
L
=
L_{\mathrm{flow}}^+
+0.01L_{\mathrm{stop}}
+\lambda_{\mathrm{pref}}L_{\mathrm{CCFPO}}
}
\]

## 7. 等价语境不变性

对于语义不同表述但说话策略相同的 \(C^+\) 与 \(C^\approx\)，定义：

\[
L_{\mathrm{inv}}
=
\frac{1}{|M|}
\sum_{i\in M}
\left\|
v_{\theta,i}(x_t,t\mid C^+,Y,S)
-
v_{\theta,i}(x_t,t\mid C^\approx,Y,S)
\right\|_2^2.
\]

完整目标：

\[
\boxed{
L
=
L_{\mathrm{flow}}^+
+0.01L_{\mathrm{stop}}
+\lambda_{\mathrm{pref}}L_{\mathrm{CCFPO}}
+\lambda_{\mathrm{inv}}L_{\mathrm{inv}}
}
\]

它同时要求模型：

- 对等价语境的表面措辞保持不变；
- 对真正改变说话策略的语境保持敏感。

## 8. 与 Ming-Omni-TTS 的输入和 Loss Mask 对接

### 8.1 正分支输入

```text
<role>HUMAN</role>Please generate speech based on the following description.

speaker_1:<spk>PROMPT_AUDIO_PATCH×1</spk>

Conversation History:<|conv_start|>user:{C+}<|conv_end|>

Text input:

{Y}<role>ASSISTANT</role><audio>TARGET_AUDIO_PATCH×N
```

### 8.2 负分支输入

```text
<role>HUMAN</role>Please generate speech based on the following description.

speaker_1:<spk>同一个PROMPT_AUDIO_PATCH×1</spk>

Conversation History:<|conv_start|>user:{C-}<|conv_end|>

Text input:

{同一个Y}<role>ASSISTANT</role><audio>同一个TARGET_AUDIO_PATCH×N
```

### 8.3 Loss mask

正负分支均保持现有区域规则：

```text
prompt/context/content/<audio>     TARGET_AUDIO_PATCH×N
│            loss=0             ││  flow/stop有效区域  │
```

区别只是：

- 正分支同时承担原始 `flow_loss` 和 `stop_loss`；
- 正负分支的 per-sample Flow error 用于 `CCFPO loss`；
- 默认不对负分支计算 `stop_loss`；
- `doubao_context_text` 不进入模型序列，不设置 token CE loss。

## 9. 训练伪代码

```python
def masked_flow_error(v_pred, flow_target, audio_mask, time_weight=None):
    error = (v_pred - flow_target).pow(2)

    # 对latent维度求平均，保留batch和patch维度
    error = error.mean(dim=-1)

    if time_weight is not None:
        error = error * time_weight

    error = error * audio_mask
    return error.sum(dim=-1) / audio_mask.sum(dim=-1).clamp_min(1)


def training_step(batch, model, ref_model):
    context_pos = batch["context_text"]
    context_neg = batch["negative_context_text"]
    context_equiv = batch.get("equivalent_context_text")

    content = batch["content"]
    prompt_audio = batch["prompt_audio"]
    target_audio = batch["target_audio"]

    # 只构造一次Flow状态，所有条件分支共享
    x1 = audio_encoder(target_audio)
    x0 = torch.randn_like(x1)
    t = sample_t(x1.shape[0], device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    ut = x1 - x0

    out_pos = model(
        context=context_pos,
        content=content,
        prompt_audio=prompt_audio,
        flow_state=xt,
        flow_t=t,
        target_audio_latent=x1,
    )

    out_neg = model(
        context=context_neg,
        content=content,
        prompt_audio=prompt_audio,
        flow_state=xt,
        flow_t=t,
        target_audio_latent=x1,
    )

    e_pos = masked_flow_error(
        out_pos.velocity,
        ut,
        out_pos.audio_target_mask,
    )

    e_neg = masked_flow_error(
        out_neg.velocity,
        ut,
        out_neg.audio_target_mask,
    )

    with torch.no_grad():
        ref_pos = ref_model(
            context=context_pos,
            content=content,
            prompt_audio=prompt_audio,
            flow_state=xt,
            flow_t=t,
            target_audio_latent=x1,
        )

        ref_neg = ref_model(
            context=context_neg,
            content=content,
            prompt_audio=prompt_audio,
            flow_state=xt,
            flow_t=t,
            target_audio_latent=x1,
        )

        e_ref_pos = masked_flow_error(
            ref_pos.velocity,
            ut,
            ref_pos.audio_target_mask,
        )

        e_ref_neg = masked_flow_error(
            ref_neg.velocity,
            ut,
            ref_neg.audio_target_mask,
        )

    # score = negative flow error
    delta = (-e_pos) - (-e_neg)
    delta_ref = (-e_ref_pos) - (-e_ref_neg)

    preference_loss = -F.logsigmoid(
        beta * (delta - delta_ref)
    ).mean()

    invariance_loss = 0.0
    if context_equiv is not None:
        out_equiv = model(
            context=context_equiv,
            content=content,
            prompt_audio=prompt_audio,
            flow_state=xt,
            flow_t=t,
            target_audio_latent=x1,
        )

        invariance_loss = masked_pair_distance(
            out_pos.velocity,
            out_equiv.velocity,
            out_pos.audio_target_mask,
        )

    total_loss = (
        out_pos.flow_loss
        + 0.01 * out_pos.stop_loss
        + lambda_pref * preference_loss
        + lambda_inv * invariance_loss
    )

    return {
        "loss": total_loss,
        "flow_loss": out_pos.flow_loss,
        "stop_loss": out_pos.stop_loss,
        "preference_loss": preference_loss,
        "invariance_loss": invariance_loss,
        "positive_error": e_pos.mean(),
        "negative_error": e_neg.mean(),
        "context_margin": (e_neg - e_pos).mean(),
    }
```

实际接入时，应复用当前 `diffusion_loss` 内部已经构造的 \(x_t,t,u_t\)，不要在模型外重复生成一套不一致的 Flow 状态。

## 10. 训练配置建议

第一阶段建议先不用 reference model，只验证 margin objective：

```yaml
context_preference:
  enabled: true
  objective: margin
  margin: 0.1
  preference_weight: 0.05
  invariance_weight: 0.01
  negative_ratio: 0.5
  equivalent_ratio: 0.25
```

确认 context margin、主观语境匹配度有效后，再切换完整 CCFPO：

```yaml
context_preference:
  enabled: true
  objective: ccfpo
  beta: 0.1
  preference_weight: 0.05
  invariance_weight: 0.01
  use_reference_model: true
  reference_model_path: ${sft_checkpoint}
  shared_flow_noise: true
  shared_flow_timestep: true
```

初始建议：

- `preference_weight`: 0.01～0.1；
- `invariance_weight`: 0.005～0.05；
- `beta`: 0.05～0.5；
- 先冻结或降低 audio encoder 学习率，避免负偏好通过改变 latent 空间获得捷径；
- 记录各 loss 对 LLM、aggregator、flowmodel 的 gradient norm。

## 11. 计算与显存优化

完整训练最多需要：

- 正分支 forward；
- 负分支 forward；
- reference 正/负分支 forward；
- 可选等价语境 forward。

直接实现成本较高，可按以下顺序优化：

1. **正负样本拼 batch**：将 \(C^+\) 和 \(C^-\) 沿 batch 维拼接，一次 model forward；
2. **共享音频特征**：target audio encoder、speaker embedding、\(x_t,t,u_t\) 只计算一次；
3. **Reference 离线/EMA**：先做无 reference 的 margin 版本；
4. **稀疏 pair step**：不是每个 step 都启用 preference loss；
5. **只在部分 Flow 时间段优化偏好**：根据实验选择 context 最敏感的 \(t\) 区间；
6. **先不启用等价分支**：把它作为后续增强与消融实验。

## 12. 防止退化和捷径

### 12.1 负分支无限恶化

风险：模型通过任意增大 \(e^-\) 降低偏好 loss，而非提高 \(C^+\) 的 context 使用。

措施：

- 保留正样本 Flow anchor；
- 使用 reference-relative objective；
- 对 margin 设置上限；
- 监控 \(e^+\)、\(e^-\) 的绝对值，而不只看差值。

### 12.2 利用主题词区分正负

风险：负 context 来自完全不同领域，模型只需识别主题差异，不需理解说话策略。

措施：

- 选择相似场景下的 style hard negative；
- 尽量匹配语言、长度、角色与主题；
- 仅在音量、情绪、距离、语速等关键维度形成冲突。

### 12.3 损害 WER/SIM/自然度

措施：

- 偏好权重从小到大 warm-up；
- 混合 zero-shot、采买、distillation 等基础 TTS 数据；
- 基础数据只计算原 Flow loss，context 数据才计算 CCFPO；
- 使用 reference anchor；
- 同时监控 WER、SIM、DNSMOS 和 context adherence。

### 12.4 Context 与 target audio 本身不一致

如果原始 `context_text`、`doubao_context_text` 和目标音频之间存在弱对应或标注噪声，偏好目标会放大错误监督。

建议先做数据过滤：

- 目标音频的 F0、energy、speech rate 与 \(D^+\) 是否基本一致；
- 音频情绪识别结果是否与描述冲突；
- 对低置信度样本只计算普通 Flow loss。

## 13. 评测设计

### 13.1 常规指标

- WER / CER；
- speaker similarity；
- DNSMOS / UTMOS；
- 音频时长和 stop accuracy；
- RTF / 首包延迟。

### 13.2 Context adherence

构造同一个 `content`、同一个 speaker、不同 context 的测试组：

```text
Y：明天下午两点，我们在会议室见面讨论这个项目。

C1：以高兴、轻快的语气宣布好消息。
C2：以严肃、克制的语气通知重要事项。
C3：以悲伤、低沉的语气说明遗憾结果。
```

测量：

- 人工 context-audio 匹配准确率；
- 音频描述模型对 emotion/volume/rate 的识别；
- F0、energy、duration、pause 等声学差异；
- 同一 context 不同改写的一致性；
- context swap 后的生成敏感度。

### 13.3 Context Utilization Margin

在真实目标音频上定义：

\[
\mathrm{CUM}
=
\mathbb{E}[e^- - e^+].
\]

CUM 越大，说明模型在 Flow trajectory 层面越能区分正确与错误 context。但该指标必须与生成音频的主观评测联合报告，避免模型只恶化负分支。

### 13.4 反事实生成测试

固定：

- content；
- speaker；
- sampling seed；
- Flow solver 配置。

只改变 context，观察生成音频在以下维度是否按预期改变：

- 情绪；
- 音量；
- 语速；
- 停顿；
- 重音；
- 交流距离感；
- 自然闲聊/正式朗读风格。

## 14. 消融实验

建议至少包含：

1. Base Context TTS：仅 Flow + stop loss；
2. Base + 随机 context negative；
3. Base + hard negative margin；
4. Base + CCFPO；
5. Base + CCFPO + equivalence invariance；
6. 去掉 reference anchor；
7. 正负分支使用不同 \(t/x_0\)；
8. 不同 negative mining 策略；
9. 不同 Flow timestep 权重；
10. 不同 context 数据比例；
11. 对 LLM、aggregator、flowmodel 分别启用/冻结 preference gradient；
12. 显式 `doubao_context_text` instruction 上限实验。

## 15. 论文定位

可能的论文标题：

> **Does Your TTS Really Listen to Context? Counterfactual Conditional Preference Optimization for Flow-Based Speech Synthesis**

建议贡献点：

1. 揭示 free-form context-aware Flow-TTS 中的 condition neglect 问题；
2. 提出保持 content、speaker、audio 和 Flow state 不变，仅干预 context 的反事实条件偏好目标；
3. 提出正确/反事实语境敏感性与等价语境不变性的联合训练；
4. 构造同文本、同说话人、多语境的 Context Utilization benchmark；
5. 在不增加任何推理模块和延迟的情况下提升 context adherence。

## 16. 与相关方向的区别

- **不是 CoT**：不生成或压缩 `doubao_context_text`；
- **不是 Teacher–Student OPD**：没有 privileged teacher 输出供 student 拟合；
- **不是 prompt tuning**：不增加固定 query token；
- **不是普通 Flow-GRPO**：训练信号针对 context 与目标音频的条件对应关系；
- **不是普通情绪分类**：支持自由文本复杂语境，并通过反事实条件对直接作用于 Flow velocity field。

相邻方向可参考：

- [Linear-DPO: Linear Direct Preference Optimization for Diffusion and Flow-Matching Generative Models](https://arxiv.org/abs/2605.21123)
- [RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching](https://arxiv.org/abs/2605.22083)
- [Causal Prosody Mediation for Text-to-Speech](https://arxiv.org/abs/2603.11683)
- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
- [F5R-TTS: Improving Flow Matching based TTS with GRPO](https://arxiv.org/abs/2504.02407)

## 17. 推荐实施顺序

### Phase 0：诊断

- 固定 audio/content/speaker，随机 swap context；
- 测量 base model 的 \(e^- - e^+\)；
- 验证模型是否存在 context neglect。

### Phase 1：最小版本

- 只构造一个 hard negative；
- 共享 \(x_t,t,u_t\)；
- 使用 `flow_loss + margin_loss + stop_loss`；
- 不加载 reference model，不做等价语境分支。

### Phase 2：完整 CCFPO

- 加入 reference-relative preference objective；
- 增加正样本 anchor 和 margin clipping；
- 完善 hard-negative mining。

### Phase 3：等价语境与时间步分析

- 加入 \(C^\approx\) invariance；
- 分析不同 Flow 时间步的 context sensitivity；
- 尝试 timestep-adaptive preference weight。

### Phase 4：Benchmark 与论文实验

- 构造同文本、同说话人、多 context 测试集；
- 完成人工 ABX/context matching；
- 完成常规质量、context adherence、延迟和消融实验。

---

## 18. 当前最需要先验证的假设

在进行大规模实现前，应先验证：

\[
\mathbb{E}[e^- - e^+]
\]

在当前 SFT 模型上是否显著大于零。

- 如果接近零：说明模型确实忽略 context，CCFPO 有明确优化空间；
- 如果已经很大：说明模型在 teacher-forced Flow trajectory 上能识别 context，需要进一步检查推理生成时为何情绪仍弱；
- 如果正负关系经常反转：优先检查数据标注、negative mining 和目标音频与 context 的一致性。

该诊断实验成本最低，也是后续方法是否成立的关键证据。
