# OpenWAM 运行指南

本接入参考 OpenWAM `main` 的 `ac8aab4f6ba16b1739c94df823cc444478e9dab0`，在运行时替换函数及其调用别名，模型源码不需要打补丁。使用原生 Wan 视频骨干和 ActionDiT，主要面向 `dual_system/joint_self_attn` 训练。

优化接入基线为 `1204dde1b9e9c232131fc99a3f2c6b78330e7f64`。请使用该上游基线，不要叠加已经内置同类优化的 OpenWAM 分支。随包配置记录了基线函数的 AST 身份；目前完成的是必要静态检查，尚未在训练环境验证多卡正确性、数值、收敛或性能。

## 环境准备

以下命令在 Linux HCU 训练容器内、同一个 Bash 会话中依次执行。先激活已有的 OpenWAM 训练环境：DTK、HCU PyTorch、FlashAttention、DeepSpeed 和 Accelerate 应已匹配；Turbo 源码构建还需要 hipDNN、Ninja 及编译器，详见[安装指南](../../docs/zh/get_started/installation.md#从源码安装)。在线优化配方还需要匹配当前 PyTorch/DTK 的 Lightop。不要直接套用 Cosmos 分支的镜像版本替换现有 OpenWAM 训练环境。

### 准备代码并切换版本

当前接入尚在 Draft PR 分支，Turbo 从 fork 的 `feat/openwam` 获取；OpenWAM 从来源仓库获取，创建固定在**未内置优化的基线提交**上的本地分支。`ac8aab4` 是优化的参考来源，运行时 OpenWAM 应切换到 `1204dde1`。

```bash
export WORKSPACE=/path/to/openwam-turbo-workspace
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

git clone --branch feat/openwam --single-branch \
  https://github.com/iMountTai/TurboPhysAI.git TurboPhysAI
git clone https://github.com/iMountTai/OpenWAM.git OpenWAM

export TURBO_ROOT="$WORKSPACE/TurboPhysAI"
export OPENWAM_ROOT="$WORKSPACE/OpenWAM"
export OPENWAM_BASELINE=1204dde1b9e9c232131fc99a3f2c6b78330e7f64
export RUN_ROOT="$WORKSPACE/openwam-runs"
mkdir -p "$RUN_ROOT/configs"

cd "$OPENWAM_ROOT"
git switch -c turbo-baseline "$OPENWAM_BASELINE"
test "$(git rev-parse HEAD)" = "$OPENWAM_BASELINE"
git status --short --branch
```

`WORKSPACE` 请设为实际绝对路径；两个仓库为相邻目录，配置和输出写入仓库之外的 `RUN_ROOT`。本接入通过 Turbo 在运行时安装 monkey-patch，准备好基线后直接安装和启动，无需执行 `git apply`。当前 Wan 场景无需初始化 Cosmos 子模块。

### 安装代码

在上面已准备好训练依赖的同一 Python 环境中安装两个仓库。`--no-deps` 保留现有 HCU 版本的 PyTorch/DeepSpeed 等依赖；它适用于依赖已经齐备的环境，不会自动补全训练依赖。全新环境需先按固定基线的 `pyproject.toml` 准备 Python 依赖，平台包使用 HCU 软件源提供的匹配版本。

```bash
cd "$OPENWAM_ROOT"
python -m pip install -e . --no-build-isolation --no-deps

cd "$TURBO_ROOT"
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python -m pip install -e . --no-build-isolation --no-deps

cd "$OPENWAM_ROOT"
python -c "import openwam, turbo_physai; print(openwam.__file__); print(turbo_physai.__file__)"
turbo-physai --help
```

打印路径应分别位于刚克隆的 OpenWAM 和 TurboPhysAI 目录；若落到其他 checkout，应先修正当前 Python 环境中的安装。Turbo 的原生扩展在 HCU 环境构建，本地 Windows 不执行此安装流程。

## 数据与模型权重

准备 RoboTwin 2.0 数据集和自包含的 OpenWAM-Alpha foundation checkpoint。下载方法见固定基线的 [Assets Preparation](https://github.com/iMountTai/OpenWAM/blob/1204dde1b9e9c232131fc99a3f2c6b78330e7f64/README.md#assets-preparation)。微调 checkpoint 目录应包含 `config.yaml`、`checkpoint_step_*.safetensors` 权重和配套 tokenizer；仅提供单独的 Wan 视频骨干目录不能替代它。

```bash
export DATASET_ROOT=/path/to/robotwin2.0/dataset
export FOUNDATION_CKPT=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model
test -d "$DATASET_ROOT"
test -f "$FOUNDATION_CKPT/config.yaml"
```

数据、权重路径替换为实际绝对路径。本例固定 `dual_system/joint_self_attn`、Wan22-TI2V、ActionDiT 和 RoboTwin 的 `blocks_ranking_size` 任务。自包含 checkpoint 提供模型重建信息和冻结编码器权重，微调路径不要求原始 Wan 下载目录仍存在。

## 生成优化配置并启动训练

### 生成在线优化配方

以下使用在线训练优化组合，包括本次迁入的预取调度和 workspace 默认限制。先复制 recipe 和 RuntimeConfig，再针对干净 OpenWAM 基线生成可运行的 OptimizationConfig；recipe 文件不能直接传给 `--optimization-config`。

```bash
cp "$TURBO_ROOT/turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml" \
  "$RUN_ROOT/configs/recipe.online.yaml"
cp "$TURBO_ROOT/turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml" \
  "$RUN_ROOT/configs/runtime.optimized.yaml"

cd "$OPENWAM_ROOT"
turbo-physai optimization generate \
  --recipe "$RUN_ROOT/configs/recipe.online.yaml" \
  --repo "$OPENWAM_ROOT" --commit "$OPENWAM_BASELINE" \
  --output "$RUN_ROOT/configs/optimization.online.yaml"
```

生成过程会在配置旁写入 `.optimization.online.yaml.generation.json`，运行时会校验它，移动配置时也应保留同目录下的凭据和 recipe。修改 recipe 后需重新生成；重复生成已有输出时，在生成命令后添加 `--force`。数据下载器可能修改仓库内 YAML，请使用独立资产目录和下文命令行路径覆盖，确保生成时 `git status --porcelain` 为空。

### 设置训练参数并运行

所有训练和缓存命令均从 `OPENWAM_ROOT` 执行。下面将公共参数保存为 Bash 数组，后续在线、缓存生成和缓存训练复用同一份数据/模型设置：

```bash
cd "$OPENWAM_ROOT"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

TRAIN_ARGS=(
  model=dual_system
  model/video_backbone=wan22_ti2v_5b
  model/action_backbone=separate_action_dit
  model.architecture.variant=joint_self_attn
  dataloader=robotwin
  "dataloader.dataset_dir=$DATASET_ROOT"
  dataloader.variant=clean_50
  '+dataloader.tasks=[blocks_ranking_size]'
  "training.finetune_ckpt_path=$FOUNDATION_CKPT"
  training.num_epochs=null training.max_steps=30 training.batch_size=16
  training.zero_stage=2 training.offload_optimizer_device=none
  training.gradient_accumulation_steps=1 training.dataset_num_workers=8
  training.use_gradient_checkpointing=true
  training.use_gradient_checkpointing_offload=false
  training.save_steps=100 training.save_full_states_for_resume=false
  project.wandb.project=null
)

turbo-physai run --model openwam --log-report \
  --optimization-config "$RUN_ROOT/configs/optimization.online.yaml" \
  --runtime-config "$RUN_ROOT/configs/runtime.optimized.yaml" \
  torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train.py \
    "${TRAIN_ARGS[@]}" "training.output_path=$RUN_ROOT/online"
```

按实际卡数同步调整 `HIP_VISIBLE_DEVICES` 和 `NPROC_PER_NODE`。`training.batch_size=16` 是每卡批量，需按目标显存调整；`max_steps=30` 是流程检查示例，评估稳定吞吐时应增加步数并排除编译预热。`save_steps=100` 仅改变周期保存间隔，正常结束仍会保存最终 checkpoint，需预留空间。

若只运行默认基础优化，将前面复制的 recipe 换成随包 `recipe.yaml` 并重新生成配置；默认组合不会开启在线重叠等全部优化。后文说明各组选项及缓存组合。

## 优化分组

模型优化通过 Turbo Group 和 `options` 控制，启动环境由 RuntimeConfig 配置，不读取 `OPENWAM_OPT_*`。

| Group | 内容 | 主要 options |
| --- | --- | --- |
| `openwam.data` | uint8 视频上传、RoboTwin 任务筛选和离线 latent 读取 | `uint8_preprocess`, `vae_latent_cache`, `vae_latent_cache_dir` |
| `openwam.compute` | Wan/ActionDiT pointwise、实数 RoPE、Norm、投影、SAC 算子边界 | `rope_real`, `pointwise_compile`, `lightop_norm`, `linear_bias2d` |
| `openwam.architecture` | 文本/本体感知 mask 接入、训练常量、latent 输入和连续 checkpoint 张量 | `constant_cache` |
| `openwam.mot` | 分段 Attention、block compile、尾层取消重算和选择性重算 | `mot_split_attn`, `compile_mot`, `checkpoint_skip_layers`, `sac_attention`, `sac_ffn_full`, `sac_gemm`, `sac_layers`, `sac_extra_ops` |
| `openwam.text` | 冻结 T5 的前缀裁剪、LRU/离线缓存、文本长度与 FA2 padding plan | `text_cache`, `text_cache_size`, `text_cache_dir`, `t5_prefix`, `text_trim`, `padding_plan_cpu`, `fa2_padding`, `fa2_mask_fused` |
| `openwam.vae` | channels-last、因果缓存 layout、VAE pointwise 和 encoder compile | `vae_channels_last`, `vae_pointwise_compile`, `compile_vae` |
| `openwam.training` | ZeRO 参数、ReduceScatter、在线编码/deferred gather、fused AdamW、Host 异步性能路径 | `zero_overlap`, `zero_reduce_scatter`, `zero_scaled_pack`, `zero_norm_overlap`, `online_encode`, `deferred_gather`, `fused_adamw`, `host_async`, `hw_queues` |

默认配置启用数据/文本/layout/实数 RoPE 和 ZeRO overlap 设置；compile、SAC、通信调度改动、Lightop、fused AdamW、磁盘缓存默认关闭。这是待目标环境验证的接入配置，不代表新的最优性能组合。

`openwam.mot` 依赖 `openwam.compute`；`openwam.training` 依赖其余六组，以保证所需调用入口一起安装。Group 内各 options 可独立配置。

## 最新 main 性能增量

参考 `main@d1ea554ba44057b766fff94c0d999fb11a217ffb`，新增四阶段 MoT 编译与分离 Q/输出的结构化注意力路径；保持拼接 K/V 的单份存储，避免扩大反向激活驻留。SAC 仅在实际 checkpoint 区间保留 FFN 边界，非重算层允许融合。

ZeRO-2 增加有界梯度批量入桶、owner 范围计划缓存、分区复制合并、批量范数及参数 view 复用。默认 ZeRO overlap 的 reduction bucket 调整为 100M 元素，保留显式配置。

`openwam.training` 另提供两个默认关闭的可选优化：`zero_scaled_pack` 将稀疏 owner 的 BF16 缩放与打包融合；`zero_norm_overlap` 在最终梯度累积完成后，于侧流计算分区范数。两者均要求 `zero_reduce_scatter=true`；使用 Triton 的目标环境需安装匹配版本。数值裁剪边界及多卡收益仍需目标环境验证。

在线编码在梯度累积大于 1 或无 DataLoader worker 时保留普通训练与 Deferred Gather 路径。Host 异步复用 ZeRO 已计算的设备端梯度范数，避免 getter 的标量同步。缓存支持目录迁移后按元数据匹配旧 reader key，生成与读取共用相同 key。

后续增量已对齐 `main@ac8aab4`：仅在 AllGather 或 Deferred Gather 已启动后的下一批预处理窗口中先执行 VAE、再执行 T5，以重叠编码和通信；初始批、普通调用和离线 latent 输入保持原有 T5 优先顺序。临时调度状态在预处理结束或异常时恢复。上游已撤回的 worker 像素打包和全局 VAE 优先实验未纳入。

启用 `zero_overlap`、ZeRO-2 且使用 HIP BLASLt（`TORCH_BLAS_PREFER_HIPBLASLT=1`）时，若用户未设置 `CUBLASLT_WORKSPACE_SIZE` 或 `HIPBLASLT_WORKSPACE_SIZE`，接入会在 Accelerate 的 DeepSpeed 配置边界将后者默认为 `32768` KiB（32 MiB）。任一显式 workspace 配置均保留。应从 Turbo 启动训练，使配置在第一次 BLASLt 调用前生效；已有进程中初始化过的库不适合作为该默认值的验证环境。

## 自定义 options

编辑随包 [recipe.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.yaml) 的副本，然后生成自定义 OptimizationConfig。不要直接修改带生成凭据的 `optimization.yaml`。

例如在原 recipe 中调整：

```yaml
- id: openwam.mot
  options:
    compile_mot: true
    checkpoint_skip_layers: 12
    sac_ffn_full: true
    sac_layers: null
- id: openwam.vae
  options:
    vae_channels_last: true
    compile_vae: true
```

原生 30 层模型中，`checkpoint_skip_layers: 12` 表示末尾 12 层不 checkpoint，前 18 层仍 checkpoint。`sac_layers: N` 选择剩余 checkpoint 层中末尾 N 层；`null` 表示全部剩余 checkpoint 层。完整 SAC 需要联合视频/动作训练、启用 gradient checkpointing、关闭 checkpoint offload，且不使用 VACE。

如需自定义，先复制 recipe 到前文的配置目录，编辑后生成配置并复用 `TRAIN_ARGS` 启动：

```bash
cp "$TURBO_ROOT/turbo_physai/optimizations/models/openwam/configs/recipe.yaml" \
  "$RUN_ROOT/configs/recipe.custom.yaml"
# 在此编辑 "$RUN_ROOT/configs/recipe.custom.yaml" 中的 options。

cd "$OPENWAM_ROOT"
turbo-physai optimization generate \
  --recipe "$RUN_ROOT/configs/recipe.custom.yaml" \
  --repo "$OPENWAM_ROOT" --commit "$OPENWAM_BASELINE" \
  --output "$RUN_ROOT/configs/optimization.custom.yaml"

turbo-physai run --model openwam \
  --optimization-config "$RUN_ROOT/configs/optimization.custom.yaml" \
  --runtime-config "$RUN_ROOT/configs/runtime.optimized.yaml" \
  torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train.py \
    "${TRAIN_ARGS[@]}" "training.output_path=$RUN_ROOT/custom"
```

随包配置使用 [generate_openwam_config.py](../../tools/generate_openwam_config.py) 从同一基线源定义生成 AST 证据及标准生成凭据，生成过程不导入模型依赖、不运行模型。需要只做静态配置生成时也可使用此工具；它仅接受上述干净基线。

`zero_reduce_scatter`、`online_encode` 和 `deferred_gather` 需要原生 DeepSpeed ZeRO-2 且关闭 CPU offload/MoE。在线编码需要梯度累积为 1、DataLoader workers 大于 0，以及冻结的原生 Wan22Ti2v VAE38/T5。`host_async` 会将训练指标延后一拍记录。`hw_queues` 必须在 HIP 初始化之前设置，已有 `GPU_MAX_HW_QUEUES` 优先；默认 `0` 保持环境设置。Turbo 不迁移模型日志实现、profiler 采集、GC trace 或 SAC 统计逻辑。

## 报告优化组合

除默认接入配置外，随包提供三份可生成的 recipe。它们复用以上七个 Group，不增加模型侧环境开关。

| 配方 | 用途 | 关键配置 |
| --- | --- | --- |
| [recipe.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml) | 在线编码训练 | 编译、Lightop、融合 mask、FFN SAC、ReduceScatter、在线编码重叠、Fused AdamW、Host 异步、Deferred Gather；尾部 12 层不重算；不读写磁盘缓存，关闭在线文本 LRU。 |
| [recipe.cache-build.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cache-build.yaml) | 生成 VAE latent 与 T5 特征 | 与缓存训练保持编码设置；关闭缓存读取、MoT 编译、SAC 与训练通信调度，缓存入口直接编码。 |
| [recipe.cached.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cached.yaml) | 读取离线缓存训练 | 在在线组合上启用 VAE/T5 磁盘缓存，尾部 24 层不重算，其余 checkpoint 层使用 FFN SAC。 |

这些配方移植报告中的选项组合，尚未完成 Turbo 路径的 HCU 正确性与性能验证。它们要求原生 Wan22Ti2v VAE38/T5、`dual_system/joint_self_attn`、ZeRO-2、无 CPU optimizer offload、gradient accumulation=1、DataLoader workers>0、gradient checkpointing 开启且 checkpoint offload 关闭；Lightop 需要在目标环境预先安装。缓存训练还需准备完整且签名一致的缓存，并确认尾部 24 层不重算所需显存。

搭配 [runtime.optimized.yaml](../../turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml) 使用。该 RuntimeConfig 在启动 `torchrun` 前设置坐标下降调优、单线程编译、16 个硬件队列、hipBLASLt 偏好及报告中的运行环境参数。显式选择该文件会覆盖同名环境变量；需要自定义时先复制并编辑。Python `apply()` 不加载 RuntimeConfig，需要调用方在导入 Torch 前设置对应环境。

OPT-01 的 hipBLASLt 库包与配套数据目录需要由目标容器提供。`TORCH_BLAS_PREFER_HIPBLASLT=1` 只选择 backend，不会安装或切换库版本；目标环境应预先配置匹配的动态库搜索路径。

在线组合的完整生成和启动命令见前文“生成优化配置并启动训练”；缓存组合见下文“离线缓存”。两者都使用干净 OpenWAM 基线和同一套公共训练参数。

| 报告项 | Turbo 对应位置 |
| --- | --- |
| OPT-01 hipBLASLt 迁移 | 外部库包及动态库路径；RuntimeConfig 设置 backend 偏好。 |
| OPT-02 VAE layout 与局部编译 | `openwam.vae`。 |
| OPT-03 分块注意力、FA2 varlen、CPU padding、融合 mask | `openwam.mot`、`openwam.text`、`openwam.compute`。 |
| OPT-04 全图编译、实数 RoPE、Lightop Norm | `openwam.compute`、`openwam.mot`、`openwam.vae`。 |
| OPT-05 尾部 12 层不重算与 FFN SAC | 在线配方的 `openwam.mot`。 |
| OPT-06 常量、前缀、文本裁剪、uint8 | `openwam.architecture`、`openwam.text`、`openwam.data`。 |
| OPT-07 ReduceScatter、硬件队列、在线编码重叠 | `openwam.training` 和优化 RuntimeConfig。 |
| OPT-08 离线 VAE 缓存 | `openwam.data`、`openwam.architecture`、`openwam.training` 与缓存生成入口。 |
| OPT-09 离线 T5 缓存 | `openwam.text`、`openwam.training` 与缓存生成入口。 |
| OPT-10 Fused AdamW | `openwam.training.fused_adamw`。 |
| OPT-11 坐标下降调优与 Host 异步 | 优化 RuntimeConfig 与 `openwam.training.host_async`。 |
| OPT-12 缓存模式尾部 24 层不重算 | 缓存训练配方的 `openwam.mot.checkpoint_skip_layers=24`。 |
| OPT-13 Deferred Gather | `openwam.training` 提交与清理、`openwam.mot` / `openwam.architecture` 等待边界。 |

## 离线缓存

沿用前面的环境变量和 `TRAIN_ARGS`，为缓存生成与缓存训练分别生成配置。下面将两份 recipe 的 `vae_latent_cache_dir` 写成同一绝对路径；生成配方保持 `vae_latent_cache: false`，训练配方为 `true`。随包缓存配方已设置 `text_cache: true`，T5 缓存默认写入 latent 目录的 `text` 子目录。

```bash
export CACHE_ROOT="$RUN_ROOT/cache"
cp "$TURBO_ROOT/turbo_physai/optimizations/models/openwam/configs/recipe.cache-build.yaml" \
  "$RUN_ROOT/configs/recipe.cache-build.yaml"
cp "$TURBO_ROOT/turbo_physai/optimizations/models/openwam/configs/recipe.cached.yaml" \
  "$RUN_ROOT/configs/recipe.cached.yaml"

python - <<'PY'
import os
from pathlib import Path
import yaml

for name in ("cache-build", "cached"):
    path = Path(os.environ["RUN_ROOT"]) / "configs" / f"recipe.{name}.yaml"
    original = path.read_text()
    recipe = yaml.safe_load(original)
    for group in recipe["optimization_groups"]:
        if group["id"] == "openwam.data":
            group["options"]["vae_latent_cache_dir"] = os.environ["CACHE_ROOT"]
    header = original.split("schema_version:", 1)[0]
    path.write_text(header + yaml.safe_dump(recipe, sort_keys=False))
PY

cd "$OPENWAM_ROOT"
for name in cache-build cached; do
  turbo-physai optimization generate \
    --recipe "$RUN_ROOT/configs/recipe.$name.yaml" \
    --repo "$OPENWAM_ROOT" --commit "$OPENWAM_BASELINE" \
    --output "$RUN_ROOT/configs/optimization.$name.yaml" || break
done
```

确认两份生成命令均成功后，先生成缓存：

```bash
cd "$OPENWAM_ROOT"
turbo-physai run --model openwam \
  --optimization-config "$RUN_ROOT/configs/optimization.cache-build.yaml" \
  --runtime-config "$RUN_ROOT/configs/runtime.optimized.yaml" \
  torchrun --nproc_per_node="$NPROC_PER_NODE" --module \
    turbo_physai.optimizations.models.openwam.cache_vae_latents \
    "${TRAIN_ARGS[@]}" "training.output_path=$RUN_ROOT/cache-build"
```

缓存入口会遍历所选任务的训练窗口，`TRAIN_ARGS` 中的 `training.max_steps=30` 不会将缓存生成限制为 30 步。缓存生成完成后，再运行缓存训练：

```bash
cd "$OPENWAM_ROOT"
turbo-physai run --model openwam --log-report \
  --optimization-config "$RUN_ROOT/configs/optimization.cached.yaml" \
  --runtime-config "$RUN_ROOT/configs/runtime.optimized.yaml" \
  torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train.py \
    "${TRAIN_ARGS[@]}" "training.output_path=$RUN_ROOT/cached"
```

缓存签名校验数据窗口、相机、尺寸、权重和文本编码设置；使用训练 split 且关闭随机视频增强。生成和训练必须使用相同配置与 checkpoint。数据 worker 采用原生 Linux fork 模式以继承运行时补丁。

## 来源与许可证

[provenance.json](../../turbo_physai/optimizations/models/openwam/configs/provenance.json) 记录来源提交与函数映射。OpenWAM 及其原生 Wan 派生函数保持 Apache-2.0；新增 Turbo 接入代码为 BSD-3-Clause。详见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
