# MindIE 相关 CVE 影响分析报告

## 说明

本报告基于当前目录下 `MindIE-SD`、`MindIE-Motor`、`MindIE-LLM` 的依赖声明和静态代码触发点排查，对扫描命中的 CVE 逐项分析。结论区分“依赖版本命中”和“当前仓库代码路径可触发”：依赖版本命中表示组件版本落入漏洞影响范围；当前代码未发现直接触发点不代表运行环境完全无风险，因为用户脚本、模型文件、包安装流程或外部运行参数仍可能触发漏洞。

## 本地依赖命中基线

- `MindIE-SD`：`diffusers==0.29.0`，见 `MindIE-SD/requirements.txt:3`。
- `MindIE-SD`：`transformers==4.44.2`，见 `MindIE-SD/requirements.txt:6`。
- `MindIE-Motor`：`transformers==4.38.2`，见 `MindIE-Motor/tests/requirements.txt:12`。
- `MindIE-Motor`：`setuptools==57.5.0`，见 `MindIE-Motor/tests/requirements.txt:27`。
- `MindIE-LLM`：`transformers==4.30.2`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:1`。
- `MindIE-LLM`：`sentencepiece==0.2.0`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:2`。
- `MindIE-LLM`：`protobuf==4.23.3`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:7`。
- `MindIE-LLM`：`setuptools==68.0.0`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:10`。
- `MindIE-LLM`：`wheel==0.43.0`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:11`。
- `MindIE-LLM`：模型依赖中存在 `onnx==1.16.0`，代表位置 `MindIE-LLM/examples/atb_models/requirements/models/requirements_qwen1.5.txt:28`。

## 总体整改建议

- `transformers`：建议升级到 `>=4.49.0`，覆盖 `CVE-2023-7018`、`CVE-2023-6730`、`CVE-2024-11392`、`CVE-2024-11393`、`CVE-2024-11394`、`CVE-2025-2099`。
- `diffusers`：建议升级到 `>=0.38.0`，覆盖 `CVE-2026-45804`、`CVE-2026-44513`。
- `setuptools`：建议升级到 `>=78.1.1`，覆盖 `CVE-2022-40897`、`CVE-2024-6345`、`CVE-2025-47273`。
- `wheel`：建议升级到 `>=0.46.2`，覆盖 `CVE-2026-24049`。
- `sentencepiece`：建议升级到 `>=0.2.1`，覆盖 `CVE-2026-1260`。
- `onnx`：建议升级到 `>=1.21.0`，覆盖本报告中大部分 ONNX 相关漏洞。
- `protobuf`：`CVE-2026-0994` 页面未给出明确修复版本，建议跟随上游或发行版安全更新，并在业务入口限制 JSON 深度和大小。

---

## CVE-2026-45804（diffusers）

### 影响范围和风险

`diffusers==0.29.0` 受影响，`MindIE-SD` 命中。该漏洞风险为通过 HuggingFace Hub 远端 diffusion pipeline 绕过 `trust_remote_code` 防护并静默执行远端自定义代码，影响等级高。漏洞说明中明确归属组件为 `diffusers`，归属版本为 `0.29.0`，修复版本为 `0.38.0`，见 `CVE-2026-4504.md:4`、`CVE-2026-4504.md:6`、`CVE-2026-4504.md:7`；本地命中依赖见 `MindIE-SD/requirements.txt:3`。

### 触发条件

触发需要调用 `DiffusionPipeline.from_pretrained` 从 HuggingFace Hub 加载远端仓库，且 `revision` 未固定到具体 commit hash。漏洞利用还依赖首次或强制下载场景，使攻击者可在 `hf_hub_download` 与 `snapshot_download` 两次请求之间切换仓库提交，见 `CVE-2026-4504.md:17`、`CVE-2026-4504.md:108`、`CVE-2026-4504.md:163`。

### 故障现象

通常不会出现明显报错。被利用时，模型加载会成功返回 pipeline，但加载过程中可能执行远端仓库中的恶意 `pipeline.py`，表现为异常文件写入、异常子进程、异常网络连接、模型加载阶段执行非预期代码等。

### 根本原因

`hf_hub_download` 和 `snapshot_download` 是两次独立 HTTP 请求，默认分支在两次请求时可能解析到不同提交。信任检查使用第一次请求得到的配置，实际导入使用第二次 snapshot 中的配置和代码，导致信任检查与最终加载对象不一致，见 `CVE-2026-4504.md:108`、`CVE-2026-4504.md:147`。

### 排查/判断方法

检查依赖是否为 `diffusers<0.38.0`。检查仓库代码、运行脚本、用户自定义脚本或服务参数中是否存在 `DiffusionPipeline.from_pretrained`，以及是否从远端 Hub 加载模型且未固定 `revision` 到 commit hash。

### 规避方案及剩余风险

优先升级到 `diffusers>=0.38.0`。如果暂不能升级，应只加载本地模型目录，或远端加载时固定 commit hash 形式的 `revision`。剩余风险在于仓库外部脚本或用户运行环境仍可能直接使用旧版 `diffusers` 加载远端 diffusion pipeline。

### 无攻击场景排查和举证

当前静态排查未发现 `DiffusionPipeline.from_pretrained` 直接调用。可补充举证：生产运行配置只加载本地模型目录；远端模型加载全部固定 commit hash；模型缓存目录只读；无 HuggingFace Hub 动态下载日志；无异常文件写入、异常进程和异常网络连接记录。

---

## CVE-2026-44513（diffusers）

### 影响范围和风险

`diffusers<0.38.0` 受影响，`MindIE-SD` 的 `diffusers==0.29.0` 命中，见 `MindIE-SD/requirements.txt:3`。风险为 `trust_remote_code` 检查绕过后加载自定义 pipeline 或组件，导致远程代码执行。

### 触发条件

通过 `DiffusionPipeline.from_pretrained` 加载包含自定义 pipeline 或 custom components 的远端仓库或本地快照，且触发了未被正确 `trust_remote_code` 检查覆盖的代码路径。

### 故障现象

模型加载成功但执行了自定义 Python 代码。异常表现包括模型加载阶段出现非预期文件创建、进程启动、网络访问，或加载非预期模块。

### 根本原因

`diffusers` 对自定义 pipeline 的 `trust_remote_code` 防护没有覆盖所有解析路径，导致默认未显式信任远端代码时仍可能执行自定义代码。

### 排查/判断方法

检查 `diffusers` 版本是否低于 `0.38.0`。检查是否存在 `DiffusionPipeline.from_pretrained`，是否允许用户传入模型 repo/path，是否加载包含 `pipeline.py` 或自定义组件的模型目录。

### 规避方案及剩余风险

升级到 `diffusers>=0.38.0`。暂不能升级时，只允许可信本地模型，远端模型必须固定 revision 并执行文件白名单检查。剩余风险是业务外脚本绕过仓库代码直接调用旧依赖。

### 无攻击场景排查和举证

当前仓库未发现 `DiffusionPipeline.from_pretrained` 直接调用。可举证生产环境无远端 diffusion repo 加载记录、无自定义 `pipeline.py`、无用户可控模型路径、模型目录来自可信发布流程且文件 hash 可验证。

---

## CVE-2024-11392（transformers）

### 影响范围和风险

`transformers<4.48.0` 受影响。`MindIE-SD` 的 `4.44.2`、`MindIE-Motor` 的 `4.38.2`、`MindIE-LLM` 的 `4.30.2` 均命中，见 `MindIE-SD/requirements.txt:6`、`MindIE-Motor/tests/requirements.txt:12`、`MindIE-LLM/examples/atb_models/requirements/requirements.txt:1`。风险为加载恶意模型或配置时发生不可信数据反序列化，可能导致代码执行。

### 触发条件

触发条件是使用受影响版本的 Transformers 加载恶意模型文件或配置，通常发生在 `from_pretrained` 加载不可信本地目录或远端 Hub repo 的场景。

### 故障现象

模型加载阶段可能执行非预期代码。正常使用可信模型时通常无故障现象；被利用时可能出现异常文件写入、异常网络连接、异常进程或加载失败。

### 根本原因

Transformers 特定模型配置或模型文件处理路径对不可信数据缺少安全校验，存在反序列化执行风险。

### 排查/判断方法

检查 `transformers` 版本是否低于 `4.48.0`。排查 `AutoConfig.from_pretrained`、`AutoTokenizer.from_pretrained`、`AutoModel.from_pretrained` 等调用是否加载用户可控路径或远端 repo。

### 规避方案及剩余风险

升级到 `transformers>=4.49.0`。无法升级时，只加载可信模型目录，固定远端模型 revision，禁止用户控制模型路径。剩余风险是示例、测试或开发脚本仍可能使用旧依赖加载不可信模型。

### 无攻击场景排查和举证

`MindIE-LLM` 存在 `AutoTokenizer.from_pretrained`，见 `MindIE-LLM/mindie_llm/connector/request_router/router_impl.py:271`。若可证明模型路径固定、来源可信、运行时使用 `local_files_only=True`、无用户可控模型输入、无异常加载日志，可作为无攻击场景举证。

---

## CVE-2024-11393（transformers）

### 影响范围和风险

`transformers<4.48.0` 受影响。当前三个仓库中的 `transformers==4.44.2`、`4.38.2`、`4.30.2` 均落入影响范围。风险为加载恶意模型文件或配置时触发远程代码执行。

### 触发条件

用户或服务通过 Transformers 的 `from_pretrained` 系列接口加载恶意模型、恶意配置文件或不可信 Hub repo。

### 故障现象

模型加载可能成功但执行了攻击者代码，也可能出现异常模块导入、异常文件操作、异常网络请求或进程异常退出。

### 根本原因

Transformers 模型文件解析过程中存在不安全反序列化路径，旧版本未充分限制不可信配置触发的对象构造或代码路径。

### 排查/判断方法

检查所有 requirements 中的 `transformers` 版本。排查模型加载路径是否允许用户输入，是否存在未固定 revision 的远端模型加载，是否开启或变相触发 `trust_remote_code`。

### 规避方案及剩余风险

升级到 `transformers>=4.49.0`。禁止加载未经审核的远端模型，远端模型必须固定 commit hash。剩余风险来自开发者手工运行示例或测试脚本时加载第三方模型。

### 无攻击场景排查和举证

可提供模型来源白名单、模型文件 hash、运行环境无外网拉取模型记录、无用户可写模型目录、无异常进程和文件变更记录作为无攻击场景证据。

---

## CVE-2024-11394（transformers）

### 影响范围和风险

`transformers<4.48.0` 受影响。`MindIE-SD`、`MindIE-Motor`、`MindIE-LLM` 均命中对应版本。风险为通过恶意模型内容触发不可信数据反序列化并执行代码。

### 触发条件

加载恶意模型文件或配置，尤其是触发相关模型 loader 的 `from_pretrained` 场景，模型路径或远端 repo 由攻击者控制时风险最高。

### 故障现象

模型加载阶段可能无报错但执行恶意代码。可观测现象包括异常文件创建、配置被篡改、异常网络访问、异常 Python 模块导入或进程行为异常。

### 根本原因

Transformers 特定模型加载路径对模型文件中的不可信内容处理不安全，导致反序列化执行风险。

### 排查/判断方法

确认 `transformers` 版本是否低于 `4.48.0`。检查运行入口是否允许用户指定模型目录、远端 repo、revision 或模型缓存路径。

### 规避方案及剩余风险

升级到 `transformers>=4.49.0`。限制远端模型加载，固定 revision，模型目录使用只读权限。剩余风险是外部脚本或人工操作仍可能使用旧依赖加载不可信模型。

### 无攻击场景排查和举证

`MindIE-SD` 示例中存在 `trust_remote_code=True` 的加载路径，见 `MindIE-SD/examples/dummy_run/model/hunyuan_image3_model.py:101`、`MindIE-SD/examples/dummy_run/model/hunyuan_image3_model.py:107`。若该示例不进入生产、模型目录可信且 hash 固定，可作为风险不落地证据。

---

## CVE-2025-2099（transformers）

### 影响范围和风险

该漏洞影响旧版 `transformers`，公开信息显示影响至 `4.48.3` 附近。当前 `MindIE-SD 4.44.2`、`MindIE-Motor 4.38.2`、`MindIE-LLM 4.30.2` 均命中。风险主要是正则回溯导致 CPU 消耗型拒绝服务。

### 触发条件

调用 `transformers.testing_utils.preprocess_string()` 处理恶意构造的大量换行或 docstring 文本时触发。

### 故障现象

CPU 使用率异常升高，测试或文本处理流程长时间卡住、超时，服务线程被阻塞。

### 根本原因

`preprocess_string()` 中处理 docstring code block 的正则表达式存在灾难性回溯，恶意输入可造成指数级处理开销。

### 排查/判断方法

检查生产代码、测试代码和工具脚本是否调用 `transformers.testing_utils` 或 `preprocess_string()`。确认是否存在外部可控文本进入该函数。

### 规避方案及剩余风险

升级到包含修复的 Transformers 版本，建议统一升级到 `>=4.49.0`。避免生产路径调用 testing utils。剩余风险主要存在于测试工具链、CI 或开发脚本处理不可信文本时。

### 无攻击场景排查和举证

当前触发点排查中发现的是模型加载相关调用，未发现该 testing utility 的生产触发点。可用 CI 日志无超时、生产代码无 `testing_utils` 调用、无外部文本输入进入该函数作为举证。

---

## CVE-2023-7018（transformers）

### 影响范围和风险

`transformers<4.36.0` 受影响。`MindIE-LLM` 的 `transformers==4.30.2` 命中，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:1`。风险为加载恶意模型或配置时触发不可信数据反序列化，导致代码执行。

### 触发条件

通过受影响版本的 Transformers 加载恶意模型文件、恶意配置或不可信远端 repo。

### 故障现象

模型加载阶段执行非预期代码，或出现异常文件写入、异常模块导入、异常进程、加载失败等现象。

### 根本原因

旧版 Transformers 对模型数据中的不可信内容缺少足够安全约束，存在反序列化风险。

### 排查/判断方法

定位所有 `transformers<4.36.0` 的 requirements，重点关注 `MindIE-LLM` 模型专用依赖。检查是否允许用户指定模型路径或使用第三方模型。

### 规避方案及剩余风险

至少升级到 `transformers>=4.36.0`，建议统一升级到 `>=4.49.0`。剩余风险是部分模型专用 requirements 继续固定旧版本，开发者手动安装后仍可能受影响。

### 无攻击场景排查和举证

若对应 requirements 仅用于离线可信模型测试，且模型来源、文件 hash、模型目录权限和安装日志均可证明可信，可作为无攻击场景证据。

---

## CVE-2023-6730（transformers）

### 影响范围和风险

`transformers<4.36.0` 受影响。`MindIE-LLM` 的 `transformers==4.30.2` 命中。风险为加载恶意 Transformers 模型文件或配置时造成代码执行。

### 触发条件

加载攻击者控制的模型目录、配置文件或远端 Hub repo，触发旧版 Transformers 的不安全模型加载逻辑。

### 故障现象

加载阶段出现远程代码执行、异常文件或网络行为，或者模型加载过程异常退出。

### 根本原因

旧版 Transformers 存在不可信数据反序列化问题，对部分模型配置和文件处理缺少安全边界。

### 排查/判断方法

检查 `transformers==4.30.2` 及其他 `<4.36.0` 固定版本。核对模型来源是否可信、是否允许用户上传或指定模型目录。

### 规避方案及剩余风险

升级到 `transformers>=4.36.0`，更建议统一升级到 `>=4.49.0`。禁止用户控制模型路径，远端模型固定 revision。剩余风险来自旧模型 requirements 被单独安装使用。

### 无攻击场景排查和举证

可提供离线模型白名单、模型文件 hash、无远端下载日志、无 `trust_remote_code` 放开记录、模型目录只读权限作为无攻击场景证据。

---

## CVE-2025-47273（setuptools）

### 影响范围和风险

`setuptools<78.1.1` 受影响。`MindIE-Motor` 的 `setuptools==57.5.0` 和 `MindIE-LLM` 的 `setuptools==68.0.0` 均命中，见 `MindIE-Motor/tests/requirements.txt:27`、`MindIE-LLM/examples/atb_models/requirements/requirements.txt:10`。风险为路径穿越导致任意文件写入，严重时可进一步导致代码执行。

### 触发条件

使用 `setuptools.package_index.PackageIndex.download` 或相关旧下载逻辑处理攻击者控制的包 URL、包索引或下载文件名。

### 故障现象

安装或构建过程中出现非预期文件写入，文件被写到目标目录之外，后续可能造成启动脚本、配置或代码被覆盖。

### 根本原因

`setuptools` 包下载逻辑对下载路径和文件名处理不充分，未正确防止路径穿越。

### 排查/判断方法

检查 `setuptools` 版本是否低于 `78.1.1`。检查构建流程是否调用 `easy_install`、`PackageIndex` 或从不可信包源下载依赖。

### 规避方案及剩余风险

升级到 `setuptools>=78.1.1`。构建环境固定可信包源并启用 hash 校验。剩余风险是测试 requirements 被开发者手动安装，或离线包源被污染。

### 无攻击场景排查和举证

当前未发现 `setuptools.PackageIndex`、`easy_install` 直接调用。可举证 CI 使用可信 PyPI 镜像、无不可信包 URL、安装日志中无异常下载路径、工作目录无异常文件覆盖。

---

## CVE-2024-6345（setuptools）

### 影响范围和风险

旧版 `setuptools` 受影响，公开修复版本为 `70.0`。`MindIE-Motor setuptools==57.5.0` 和 `MindIE-LLM setuptools==68.0.0` 均命中。风险为通过 `package_index` 下载攻击者控制的包链接导致代码执行或恶意文件落盘。

### 触发条件

构建或安装流程使用旧版 `setuptools.package_index` 处理不可信 package URL 或包索引页面。

### 故障现象

安装阶段可能执行非预期代码，或出现异常下载文件、异常落盘文件、构建环境被污染。

### 根本原因

`setuptools.package_index` 下载处理对 URL、文件名或下载内容信任过高，缺少必要的安全校验。

### 排查/判断方法

检查 `setuptools` 版本是否低于 `70.0`。检查构建脚本、安装脚本、CI 日志是否使用 `easy_install` 或访问不可信索引。

### 规避方案及剩余风险

至少升级到 `setuptools>=70.0`，建议直接升级到 `>=78.1.1`。剩余风险是离线镜像或缓存中已有恶意包。

### 无攻击场景排查和举证

可举证无 `PackageIndex` 直接调用、CI 只安装锁定包、包源为可信内网镜像、依赖包 hash 与白名单一致、无异常构建产物。

---

## CVE-2022-40897（setuptools）

### 影响范围和风险

`setuptools<65.5.1` 受影响。`MindIE-Motor` 的 `setuptools==57.5.0` 命中，见 `MindIE-Motor/tests/requirements.txt:27`。风险为正则 ReDoS，导致安装或构建流程 CPU 消耗异常。

### 触发条件

旧版 `package_index.py` 解析恶意 HTML 或恶意包索引页面时触发。

### 故障现象

依赖安装或构建流程卡住、CPU 占满、CI 超时、安装命令长时间无响应。

### 根本原因

`setuptools.package_index` HTML 解析正则存在灾难性回溯，恶意页面可触发高复杂度匹配。

### 排查/判断方法

确认 `setuptools` 是否低于 `65.5.1`。检查安装流程是否访问不可信 package index，是否存在安装超时或 CPU 异常日志。

### 规避方案及剩余风险

升级到 `setuptools>=65.5.1`，建议统一升级到 `>=78.1.1`。剩余风险是开发测试环境继续安装旧测试依赖并访问不可信索引。

### 无攻击场景排查和举证

构建日志中只访问可信镜像、无安装超时、无 CPU 异常、无外部包索引访问，可作为无攻击场景证据。

---

## CVE-2026-24049（wheel）

### 影响范围和风险

`wheel 0.40.0` 到 `0.46.1` 受影响。`MindIE-LLM` 的 `wheel==0.43.0` 命中，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:11`。风险为解包恶意 wheel 时通过路径穿越修改任意文件权限，可能进一步导致提权或代码执行。

### 触发条件

使用受影响版本的 `wheel` 解包攻击者构造的 wheel 文件，且归档头中包含恶意路径。

### 故障现象

敏感文件权限被异常修改，安装后出现可执行权限异常、配置文件权限异常、用户目录或系统目录文件权限变化。

### 根本原因

wheel 解包阶段对路径做了部分清理，但后续 `chmod` 使用归档头中的原始路径，导致路径穿越后权限修改。

### 排查/判断方法

检查 `wheel` 版本是否处于 `0.40.0` 到 `0.46.1`。检查是否安装不可信 wheel 包、本地 wheelhouse 是否可被非可信用户写入。

### 规避方案及剩余风险

升级到 `wheel>=0.46.2`。只安装可信来源 wheel，并启用 hash 校验。剩余风险是本地缓存或离线 wheelhouse 中已有恶意 wheel。

### 无攻击场景排查和举证

安装日志显示仅来自可信包源，wheel 文件 hash 与白名单一致，本地 wheelhouse 权限受控，敏感文件权限无异常变化，可作为无攻击场景证据。

---

## CVE-2026-1260（sentencepiece）

### 影响范围和风险

`sentencepiece<0.2.1` 受影响。`MindIE-LLM` 的 `sentencepiece==0.2.0` 命中，部分模型依赖中还存在 `0.1.99`，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:2`。风险为加载恶意 SentencePiece 模型文件时发生非法内存访问，可能造成崩溃或未定义行为。

### 触发条件

加载攻击者构造的 SentencePiece 模型文件，例如用户可上传或指定 tokenizer/model 文件时触发。

### 故障现象

进程崩溃、非法内存访问、服务异常退出、tokenizer 初始化失败。

### 根本原因

SentencePiece 对模型文件内容校验不足，恶意模型文件可触发底层内存访问错误。

### 排查/判断方法

检查 `sentencepiece` 版本是否低于 `0.2.1`。检查业务是否允许用户上传 tokenizer 文件，或是否加载来源不明的模型包。

### 规避方案及剩余风险

升级到 `sentencepiece>=0.2.1`。只加载可信 tokenizer 文件，模型包启用 hash 校验。剩余风险是已有模型包中的 tokenizer 文件未验证来源。

### 无攻击场景排查和举证

可提供 tokenizer 文件来源、hash、只读权限、无用户上传入口、无 tokenizer 初始化崩溃日志作为无攻击场景证据。

---

## CVE-2026-0994（protobuf）

### 影响范围和风险

NVD 标记 Python Protobuf `<=v33.4` 受影响。`MindIE-LLM` 的 `protobuf==4.23.3` 属于旧版本，按扫描命中处理，见 `MindIE-LLM/examples/atb_models/requirements/requirements.txt:7`。风险为递归耗尽导致拒绝服务。

### 触发条件

调用 `google.protobuf.json_format.ParseDict()` 解析包含深层嵌套 `google.protobuf.Any` 的恶意输入。

### 故障现象

解析时出现 `RecursionError`、请求超时、服务线程异常退出、CPU 或调用栈异常。

### 根本原因

嵌套 `Any` 解析未正确计入递归深度，导致攻击者可绕过 `max_recursion_depth` 并触发 Python 递归耗尽。

### 排查/判断方法

搜索是否调用 `google.protobuf.json_format.ParseDict` 或 `json_format.Parse`，并确认输入是否来自外部请求。检查 `protobuf` 版本是否落入 NVD 标记范围。

### 规避方案及剩余风险

跟随上游或发行版安全更新升级 protobuf。修复版本不明确时，在业务入口限制 JSON 深度、大小和 `Any` 嵌套层级。剩余风险是第三方库内部间接调用 `ParseDict` 并处理外部输入。

### 无攻击场景排查和举证

当前未发现 `google.protobuf.json_format.ParseDict` 或 `json_format.Parse` 直接调用。可用接口 schema、请求体大小限制、无 `RecursionError` 日志、无外部 protobuf JSON 解析入口作为无攻击场景证据。

---

## CVE-2026-34445（onnx）

### 影响范围和风险

`onnx<1.21.0` 受影响。`MindIE-LLM` 模型依赖存在 `onnx==1.16.0`，代表位置 `MindIE-LLM/examples/atb_models/requirements/models/requirements_qwen1.5.txt:28`。风险为加载恶意 ONNX 模型时内部属性被覆盖，可能造成拒绝服务、模型行为异常或进一步安全影响。

### 触发条件

加载恶意 ONNX 模型，其 external data metadata 包含攻击者控制的字段名，使 `ExternalDataInfo` 处理 metadata 时触发属性覆盖。

### 故障现象

模型加载异常、推理失败、进程报错、内部状态异常，严重时可能造成服务不可用。

### 根本原因

`ExternalDataInfo` 对模型提供的 metadata 使用 `setattr()`，未校验字段名是否合法，导致模型数据可覆盖内部属性。

### 排查/判断方法

检查 `onnx` 版本是否低于 `1.21.0`。检查业务是否加载用户提供的 ONNX 文件，模型目录是否可被用户写入。

### 规避方案及剩余风险

升级到 `onnx>=1.21.0`。只加载可信 ONNX 模型，校验 hash，模型目录使用只读权限。剩余风险是历史缓存模型或第三方模型包未清理。

### 无攻击场景排查和举证

若业务不接收外部 ONNX 模型，且模型目录来自可信发布流程、文件 hash 与发布件一致、目录只读、无模型加载异常日志，可作为无攻击场景证据。

---

## CVE-2026-28500（onnx）

### 影响范围和风险

`onnx<=1.20.1` 受影响。`MindIE-LLM` 的 `onnx==1.16.0` 命中。风险为 `onnx.hub.load()` 的仓库信任告警被静默抑制，可能导致用户在无确认情况下加载不可信模型。

### 触发条件

调用 `onnx.hub.load()` 从远端仓库加载模型，尤其是在 `silent=True` 时加载不可信仓库。

### 故障现象

信任提示不出现或被静默抑制，远端模型被直接加载；后续可能出现模型文件污染、异常下载或安全告警缺失。

### 根本原因

`silent=True` 可绕过或静默掉 ONNX Hub 的信任提示流程，削弱用户确认机制。

### 排查/判断方法

搜索 `onnx.hub.load` 和 `silent=True`。检查是否存在 ONNX Hub 远端加载、用户可控 repo 参数或自动下载模型逻辑。

### 规避方案及剩余风险

禁用 ONNX Hub 远端加载，或只允许白名单仓库；不要使用 `silent=True` 加载不可信来源。建议升级 ONNX。剩余风险是第三方库内部间接调用 ONNX Hub。

### 无攻击场景排查和举证

当前未发现 `onnx.hub.load` 直接调用。可举证无 ONNX Hub 网络访问日志、无用户可控 repo 参数、无远端 ONNX 自动下载配置。

---

## CVE-2026-27489（onnx）

### 影响范围和风险

`onnx<1.21.0` 受影响。`MindIE-LLM` 的 `onnx==1.16.0` 命中。风险为符号链接路径穿越，可能暴露模型目录或用户指定目录之外的文件。

### 触发条件

处理包含恶意 symlink 的模型目录或用户提供目录，ONNX 在访问相关文件时跟随符号链接并越界访问。

### 故障现象

读取到预期目录外文件，敏感信息泄露，或因访问异常路径导致模型加载失败。

### 根本原因

路径处理未正确防止 symlink 路径穿越，对模型目录边界校验不足。

### 排查/判断方法

检查 `onnx` 版本是否低于 `1.21.0`。检查模型目录是否来自用户输入，是否存在 symlink，是否从压缩包解压得到。

### 规避方案及剩余风险

升级到 `onnx>=1.21.0`。禁止模型目录包含 symlink，加载前做路径规范化和目录边界校验。剩余风险是压缩包解压阶段引入 symlink 后未被清理。

### 无攻击场景排查和举证

模型目录归属可信发布流程、无 symlink、权限只读、无目录外文件访问日志、模型包 hash 固定，可作为无攻击场景证据。

---

## CVE-2024-7776（onnx）

### 影响范围和风险

公开记录显示 `onnx<=1.16.1` 受影响。`MindIE-LLM` 的 `onnx==1.16.0` 命中。风险为下载并解压恶意模型包时路径穿越导致任意文件覆盖。

### 触发条件

调用 ONNX 模型下载相关函数处理恶意 tar 或模型包，归档中包含路径穿越、绝对路径或 symlink 条目。

### 故障现象

任意文件被覆盖，可能覆盖启动脚本、配置文件、凭据文件或用户目录文件，后续造成代码执行或数据破坏。

### 根本原因

模型下载后的归档解压未充分校验成员路径，导致 tar 成员可写出目标目录。

### 排查/判断方法

搜索 `download_model` 相关 API。检查是否从不可信来源下载 ONNX 模型包，是否自动解压远端模型归档。

### 规避方案及剩余风险

升级 ONNX。禁用不可信模型下载；解压前检查绝对路径、`..` 和 symlink。剩余风险是第三方工具链内部调用下载函数。

### 无攻击场景排查和举证

当前未发现 `download_model` 直接调用。可举证无不可信 ONNX tar 下载、模型包 hash 固定、无敏感文件异常修改、无自动解压远端 ONNX 包逻辑。

---

## CVE-2024-5187（onnx）

### 影响范围和风险

NVD 明确指出 `onnx 1.16.0` 受影响。`MindIE-LLM` 的 `onnx==1.16.0` 命中。风险为 `download_model_with_test_data` 解压恶意 tar 时路径穿越，导致任意文件覆盖。

### 触发条件

调用 `download_model_with_test_data` 处理恶意 tar 文件，tar 中包含穿越路径或绝对路径成员。

### 故障现象

任意文件被覆盖，例如 `.ssh/authorized_keys`、配置文件、脚本文件等被篡改。可能导致后续登录、任务执行或服务启动被攻击者控制。

### 根本原因

`download_model_with_test_data` 解压 tar 时缺少路径穿越防护，未限制归档成员只能写入目标目录内。

### 排查/判断方法

搜索 `download_model_with_test_data`。检查测试、示例或工具脚本是否自动下载 ONNX test data，是否处理不可信 tar。

### 规避方案及剩余风险

避免使用该函数处理不可信 tar，升级 ONNX，测试数据包固定来源和 hash。剩余风险是测试脚本在开发环境被手动运行并处理不可信测试数据。

### 无攻击场景排查和举证

当前未发现 `download_model_with_test_data` 直接调用。可举证无 ONNX test data 远端下载日志、无敏感文件异常修改、无自动解压不可信 tar 逻辑。

---

## 参考来源

- `CVE-2026-4504.md:3`：本地 `CVE-2026-45804` 说明文件。
- `https://nvd.nist.gov/vuln/detail/CVE-2026-44513`
- `https://nvd.nist.gov/vuln/detail/CVE-2026-34445`
- `https://nvd.nist.gov/vuln/detail/CVE-2026-27489`
- `https://nvd.nist.gov/vuln/detail/CVE-2026-0994`
- `https://nvd.nist.gov/vuln/detail/CVE-2024-5187`
- `https://api.osv.dev/v1/vulns/CVE-2024-11392`
- `https://api.osv.dev/v1/vulns/CVE-2024-11393`
- `https://api.osv.dev/v1/vulns/CVE-2024-11394`
- `https://api.osv.dev/v1/vulns/CVE-2025-2099`
- `https://api.osv.dev/v1/vulns/CVE-2025-47273`
- `https://api.osv.dev/v1/vulns/CVE-2026-24049`
