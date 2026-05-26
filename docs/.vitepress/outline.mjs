export const repoBaseUrl = 'https://github.com/Weihong-Liu/hello-ai-infra/blob/main'

export const parts = [
  {
    prefix: '/part0-preface/',
    navText: '前言',
    title: '前言与学习路线',
    readmeTitle: '第 0 篇：前言与学习路线',
    chapters: [
      {
        title: '写给读者的话',
        summary: 'AI Infra 的重要性、教程特色、学习路线',
        status: '🚧',
        lead: '本章是整个教程的入口，先说明这本书为什么存在、适合谁读，以及你最终会完成什么项目。读完后，你应该能判断自己是否适合继续学下去，并理解后续章节为什么总是围绕硬件、profiling、优化和 Agent 展开。',
        sections: [
          ['为什么 AI Infra 很重要', '从大模型和多模态模型的系统瓶颈出发，解释为什么只会调用框架还不够。'],
          ['AI Infra 到底包括什么', '梳理训练框架、推理引擎、算子优化、AI 编译器、通信库和调度系统等核心模块。'],
          ['为什么选择 AMD GPU / ROCm 作为主线', '说明本教程选择 AMD GPU / ROCm 的原因，以及它与 CUDA 体系的差异和互补价值。'],
          ['本教程和普通 AI Infra 教程有什么不同', '强调 Profiling-Driven、Hardware-Aware、Agent-Driven 三条主线。'],
          ['你将完成的毕业项目：AutoInfra Agent', '提前预告最终项目如何自动完成性能分析、优化建议、代码修改、benchmark 和报告生成。'],
          ['学习路线图', '展示从硬件认知到算子优化、推理优化、编译器理解和 Agent 自动化的学习路径。'],
          ['环境与前置知识说明', '说明需要的 Python、Linux、深度学习基础，以及后续实验默认基线。']
        ]
      },
      {
        title: '环境准备与验证',
        summary: 'uv sync、AI MAX 395 + ROCm 7.12.0 基线、最小环境验证',
        status: '🚧',
        lead: '本章先不深入讲 ROCm 软件栈原理，而是帮你用最短路径确认实验环境能不能继续往后跑。读完后，你应该能通过 uv sync 复现本篇环境，确认 ROCm 能看到 GPU，并完成 PyTorch ROCm 与最小 HIP 路径的基础验证。',
        sections: [
          ['本教程的实验基线', '明确所有实验默认在 AI MAX 395 + ROCm 7.12.0 上验证，其他设备只参考方法。'],
          ['同步本篇 uv 环境', '进入 code/part0-preface 后运行 uv sync，并使用 activate-rocm.sh 激活 ROCm wheel 环境。'],
          ['验证 GPU 可见性', '用 rocminfo 和 rocm-smi 检查 GPU、驱动、显存和运行状态。'],
          ['验证 PyTorch ROCm', '运行最小 PyTorch ROCm smoke test，确认框架能看到 GPU。'],
          ['验证最小 HIP 程序', '直接用 hipcc 编译并运行最小 vector add 程序。'],
          ['环境不通时先收集什么', '列出报错、版本、命令输出、硬件信息和日志，避免盲目排错。'],
          ['附录 A：环境安装细节与常见坑', '解释 pyproject、AMD wheel 源、rocm-sdk init 和 ROCM_PATH 的关系。']
        ]
      }
    ]
  },
  {
    prefix: '/part1-hardware-rocm/',
    navText: 'GPU 体系结构',
    title: 'GPU 体系结构与 ROCm 软件栈',
    readmeTitle: '第 1 篇：GPU 体系结构与 ROCm 软件栈',
    chapters: [
      {
        title: 'GPU 在 AI 中的位置',
        summary: '从一次推理请求出发，把模型、框架、算子、kernel、runtime、硬件串成单卡视角的链路图',
        status: '🚧',
        lead: '本章把"GPU 性能问题"放回完整链路里看。读完后，你应该能区分 kernel 慢、调度慢、搬运慢、测量错这几类性能现象，并知道本书后续每一篇分别盯住哪一层。本篇后续四章会逐步把硬件、内存、软件栈、第一个程序串起来。',
        sections: [
          ['从模型到硬件：一次推理请求经历了什么', '沿着请求路径观察框架、算子、kernel、runtime、硬件的关系。'],
          ['AI 计算栈中的 GPU 角色', '说明 GPU 既是计算单元又是访存单元，性能问题的两类来源。'],
          ['算子优化、推理优化、编译器优化分别解决什么', '区分三类优化的边界，避免把所有性能问题都归因于 kernel。'],
          ['拿到性能问题时先怎么分诊', '用一张决策图把常见性能现象导向算子、运行时或测量层。'],
          ['本书的学习闭环：理解 → 测量 → 优化 → 复盘', '说明后续每一篇如何围绕同一个闭环推进。'],
          ['GPU 工程师的核心能力模型', '总结硬件理解、性能分析、工程实现和复现表达四类能力。']
        ]
      },
      {
        title: 'AMD GPU 体系结构',
        summary: 'CU/SIMD/Wavefront/VGPR/SGPR/LDS、RDNA vs CDNA、gfx1151 定位、MFMA/WMMA Tensor 单元',
        status: '🚧',
        lead: '本章建立后续优化会反复用到的 AMD GPU 硬件最小模型。读完后，你应该能用自己的话讲清楚一个 kernel 在 AMD GPU 上从 launch 到执行经过哪些硬件单元，能区分 RDNA 与 CDNA 在 AI 工作负载上的差异，并知道 gfx1151（AI MAX 395）在两条产品线中的位置。',
        sections: [
          ['Compute Unit（CU）的内部结构', '逐层拆解 CU：SIMD / VALU / SALU / Branch / Scalar / Vector 寄存器堆。'],
          ['Wavefront 与 SIMT 执行', '理解 64 线程一组（RDNA 上 32 或 64）的执行方式以及分支收敛代价。'],
          ['VGPR、SGPR 与 LDS 资源', '解释片上寄存器与共享内存为什么是 kernel 优化的核心资源。'],
          ['AMD GPU 演进简史：从 GCN 到 RDNA / CDNA', '用一张时间线串起 TeraScale → GCN → RDNA 1-4 与 CDNA 1-4 的演进，定位 gfx1151 在哪一支。'],
          ['RDNA vs CDNA 的关键差异', '从面向场景（图形 vs 数据中心）到指令集差异、wavefront 宽度、Tensor 单元的对比。'],
          ['gfx1151 / AI MAX 395 定位', '说明本书硬件基线 gfx1151 在 RDNA 体系下的位置、能力与差异说明（不替 MI 系列编数据）。'],
          ['MFMA 与 WMMA：Tensor 加速单元', '介绍 CDNA 上的 MFMA 与 RDNA3+ 上的 WMMA，以及它们在 AI 算子里的角色。'],
          ['Roofline 的硬件来源', '把硬件参数（峰值算力、带宽）翻译成 Roofline 上的两条线。']
        ]
      },
      {
        title: '内存层次与访存模式',
        summary: 'HBM/GDDR/Infinity Cache、L1/L2、LDS bank 冲突、合并访存、atomics 与 fence',
        status: '🚧',
        lead: '本章把单卡内存层次拆开。读完后，你应该能解释一个 GPU 内存请求从指令到 HBM 到底走了几跳、哪一步代价最高，并知道为什么"合并访存""LDS bank 冲突"这些词决定了多数 AI 算子的性能上限。所有数字以 AI MAX 395 + ROCm 7.12.0 为基线。',
        sections: [
          ['内存层次总览', '从寄存器 → LDS → L1/L2 → Infinity Cache → HBM/GDDR 的容量、带宽、延迟分层。'],
          ['HBM vs GDDR vs Infinity Cache', '解释三类显存与片上缓存的取舍，以及 gfx1151 上的实际形态。'],
          ['L1 / L2 Cache 的工作方式', '理解 cacheline 大小、命中策略与共享粒度对算子带宽的影响。'],
          ['LDS 详解与 bank 冲突', '解释 LDS 的 bank 结构、为什么 stride 选错会让性能腰斩。'],
          ['全局内存合并访存（Coalescing）', '观察连续线程访问连续地址为什么是带宽利用率的关键。'],
          ['原子操作与内存一致性', '解释 atomics、fence、memory order 在 reduction、histogram 等算子里的代价。'],
          ['用 micro-benchmark 测量带宽与延迟', '给出一组可复现的小程序，验证 HBM、L2、LDS 的实测带宽与延迟。']
        ]
      },
      {
        title: 'ROCm 软件栈与工具链',
        summary: '驱动、HSA、HIP runtime、算子库、上层框架的分层关系，以及如何用命令检查每一层',
        status: '🚧',
        lead: '本章用一张分层图把 ROCm 拆开，让你知道 PyTorch、Triton、MIGraphX 这些上层工具最终怎么走到硬件。读完后，你应该能看懂 rocminfo / rocm-smi / hipcc 的输出在描述哪一层，并能在出问题时迅速定位是哪一层失联。',
        sections: [
          ['ROCm 是什么', '用分层图理解驱动、运行时、编译器、库和工具链之间的关系。'],
          ['AMDGPU Driver、HSA Runtime、HIP Runtime', '厘清三层 runtime 的职责和接口边界。'],
          ['算子库：rocBLAS / MIOpen / Composable Kernel', '介绍核心库的定位，方便后续 kernel 章节做对比。'],
          ['编译器侧：HIPCC / LLVM-AMDGPU', '理解 HIP 代码到 ISA 的编译路径。'],
          ['上层框架与 ROCm 的关系', '说明 PyTorch / Triton / MIGraphX / vLLM 各自依赖哪些 ROCm 能力。'],
          ['如何检查一台机器的 AMD GPU 环境', '把环境检查命令放回软件栈语境里，解释每个命令在检查什么。']
        ]
      },
      {
        title: '第一个 AMD GPU 程序与 baseline',
        summary: 'PyTorch ROCm + 最小 HIP kernel 双路径，建立可复用的 benchmark 习惯',
        status: '🚧',
        lead: '本章在已经验证环境可用、且对硬件与软件栈有基本心智模型的基础上，带你跑通第一个真正的 GPU 程序。重点不是安装百科，而是建立后续所有实验都会复用的代码、计时、日志习惯——所有数字都来自 AI MAX 395 + ROCm 7.12.0 实测。',
        sections: [
          ['从已经验证的环境开始', '复用前面的环境验证结果，直接进入当前章节代码目录。'],
          ['过一遍框架侧 smoke test', '运行一个 PyTorch ROCm tensor 运算，作为进入手写 HIP 程序前的热身。'],
          ['跑通第一个 HIP kernel', '编写、编译并运行一个最小 HIP kernel。'],
          ['建立 baseline benchmark', '用固定输入、热身、重复运行和 GPU event 建立可复查的计时 baseline。'],
          ['留下实验底稿', '说明源码、命令输出、benchmark 配置和 EXPERIMENT.md 应该如何对应。']
        ]
      }
    ]
  },
  {
    prefix: '/part2-profiling/',
    navText: 'Profiling',
    title: '性能分析与瓶颈定位',
    readmeTitle: '第 2 篇：性能分析与瓶颈定位',
    chapters: [
      {
        title: '性能优化的基本方法论',
        summary: 'Latency、Throughput、Bandwidth、FLOPS、Roofline、可信 benchmark',
        status: '🚧',
        lead: '本章先不急着打开 profiling 工具，而是建立判断性能问题的基本语言。读完后，你应该能区分延迟、吞吐、带宽、FLOPS，以及为什么不能凭感觉优化。',
        sections: [
          ['为什么不能凭感觉优化', '用常见误区说明没有数据的优化为什么容易走偏。'],
          ['Latency、Throughput、Bandwidth、FLOPS', '定义最常用的性能指标，并说明它们分别回答什么问题。'],
          ['Memory-bound 与 Compute-bound', '理解访存瓶颈和计算瓶颈的差异。'],
          ['Roofline 思想入门', '用简单图示理解理论上限、实际性能和优化方向之间的关系。'],
          ['如何设计一个可信的 benchmark', '说明热身、重复次数、同步、输入规模和日志记录的基本要求。'],
          ['如何避免伪优化', '识别缓存偶然命中、测量范围错误、数据拷贝遗漏等伪提升。']
        ]
      },
      {
        title: '用一个慢算子跑通 Profiling 闭环',
        summary: '同一案例贯穿 benchmark、rocprof、PyTorch Profiler、瓶颈判断',
        status: '🚧',
        lead: '本章用一个固定的慢算子做主线，不把 profiling 工具当清单介绍，而是让每个工具服务同一个问题：它到底慢在哪里。读完后，你应该能完成一次从 benchmark 到优化假设的最小闭环。',
        sections: [
          ['选择一个可控的慢算子', '确定输入规模、baseline 实现和预期瓶颈，避免一开始就分析复杂模型。'],
          ['运行 baseline benchmark', '用统一脚本记录延迟、吞吐和硬件上下文。'],
          ['用 rocprof 看 kernel 时间', '采集 kernel 级耗时，找出主要开销来自哪里。'],
          ['用 PyTorch Profiler 关联框架调用', '把 Python / framework 层的调用和 GPU kernel 时间线对应起来。'],
          ['分析 H2D / D2H / Kernel Launch 开销', '判断问题来自数据搬运、启动开销还是 kernel 本身。'],
          ['从 profiling 结果反推优化方向', '把观测结果转成下一步实验假设，而不是直接拍脑袋改代码。']
        ]
      },
      {
        title: '建立你的第一个性能分析报告',
        summary: '采集数据、判断瓶颈、提出假设、生成 Markdown 报告',
        status: '🚧',
        lead: '本章把上一章的 profiling 过程整理成一份可复查的报告。读完后，你应该能把命令、日志、关键数字、瓶颈判断和下一步计划写成别人能复现的 Markdown。',
        sections: [
          ['选择一个慢算子', '明确报告对象、输入规模和 baseline 版本。'],
          ['运行 baseline', '记录原始运行命令、日志路径和关键指标。'],
          ['收集 profiling 数据', '保存 timeline、kernel 表格和必要的原始输出。'],
          ['判断瓶颈类型', '结合方法论判断当前更像访存瓶颈、计算瓶颈还是调度/搬运开销。'],
          ['提出优化假设', '把瓶颈判断转成可验证的下一步改动。'],
          ['生成 Markdown 性能报告', '形成一份包含硬件上下文、命令、结果和结论的报告。']
        ]
      },
      {
        title: 'Omniperf 与硬件计数器进阶',
        summary: '用进阶计数器解释访存、Occupancy、波前行为和 Roofline 证据',
        status: '🚧',
        lead: '本章是 profiling 的进阶篇，目标不是堆更多工具名，而是在已经有慢算子案例的基础上，用硬件计数器解释为什么它慢。读完后，你应该知道什么时候需要 Omniperf，以及哪些计数器能支撑优化判断。',
        sections: [
          ['什么时候需要硬件计数器', '说明 kernel 时间不够回答问题时，为什么需要更底层的硬件证据。'],
          ['Omniperf 基本采集流程', '用同一个案例采集硬件计数器，并保存可复查输出。'],
          ['访存相关指标怎么看', '观察带宽、缓存、访存合并和内存等待相关信号。'],
          ['Occupancy 与 Wavefront 行为', '理解占用率、寄存器压力和 wavefront 调度对性能的影响。'],
          ['把计数器放回 Roofline', '用硬件指标解释当前点为什么离理论上限有差距。'],
          ['进阶报告模板', '扩展上一章报告结构，加入计数器证据和风险说明。']
        ]
      }
    ]
  },
  {
    prefix: '/part3-hip-kernels/',
    navText: 'HIP 算子',
    title: 'HIP 算子优化实战',
    readmeTitle: '第 3 篇：HIP 算子优化实战',
    chapters: [
      {
        title: 'HIP 编程基础',
        summary: 'Kernel、Thread、Block、Grid、Host / Device、内存管理',
        status: '🚧',
        lead: '本章建立 HIP 编程的最小语法和执行模型，为后续手写算子做准备。读完后，你应该能看懂一个 HIP kernel 如何从 Host 端启动并在 Device 上执行。',
        sections: [
          ['HIP 和 CUDA 的关系', '用迁移视角理解 HIP 的定位，但不把本教程写成 CUDA API 对照表。'],
          ['Kernel、Thread、Block、Grid', '理解 GPU 并行程序的基本层级。'],
          ['Host 与 Device', '区分 CPU 侧控制逻辑和 GPU 侧执行逻辑。'],
          ['Device Memory 管理', '介绍分配、拷贝和释放 device memory 的基本流程。'],
          ['Kernel Launch', '理解启动参数如何影响并行度和数据映射。'],
          ['错误检查与调试', '建立最小错误检查习惯，避免失败时只看到空输出。'],
          ['思考题', '通过小问题确认你是否理解基本执行模型。']
        ]
      },
      {
        title: '从 Vector Add 理解 GPU 并行',
        summary: 'CPU baseline、Naive HIP、线程映射、访存合并、benchmark',
        status: '🚧',
        lead: '本章用最简单的 Vector Add 连接 CPU 思维和 GPU 并行思维。读完后，你应该能解释线程如何映射到数据，以及为什么看似简单的向量加法也需要严谨 benchmark。',
        sections: [
          ['CPU 版本', '先写一个清晰的 CPU baseline，明确要搬到 GPU 上的计算是什么。'],
          ['Naive HIP 版本', '写出第一个一线程处理一个元素的 HIP kernel。'],
          ['线程映射', '理解 blockIdx、threadIdx 和全局元素下标的关系。'],
          ['访存合并', '观察连续线程访问连续地址为什么重要。'],
          ['Benchmark 与 profiling', '对比 CPU、GPU 和不同输入规模下的表现。'],
          ['优化报告', '把实验结果整理成一份最小优化报告。'],
          ['思考题', '通过修改输入规模和 block size 理解并行度变化。']
        ]
      },
      {
        title: 'Reduction 优化',
        summary: 'Naive Reduction、LDS、Wavefront、多阶段 Reduction、性能对比',
        status: '🚧',
        lead: '本章进入第一个真正体现 GPU 层级协作的算子：Reduction。读完后，你应该能理解为什么跨线程汇总需要 LDS、同步和多阶段设计。',
        sections: [
          ['Reduction 为什么重要', '说明求和、归约、归一化和注意力中为什么经常出现 reduction。'],
          ['Naive Reduction', '从简单但低效的实现开始，观察瓶颈。'],
          ['Shared Memory / LDS 优化', '使用 LDS 减少全局内存访问并组织 block 内归约。'],
          ['Warp / Wavefront 级优化思路', '理解 wavefront 内协作和分支收敛对 reduction 的影响。'],
          ['多阶段 Reduction', '把大规模输入拆成 block 内归约和跨 block 合并。'],
          ['性能对比', '用 benchmark 和 profiling 对比不同版本。'],
          ['思考题', '分析不同输入长度和 block size 对性能的影响。']
        ]
      },
      {
        title: 'Softmax 优化',
        summary: '数值稳定性、访存优化、Block 级并行、PyTorch 对齐',
        status: '🚧',
        lead: '本章用 Softmax 把 reduction、数值稳定性和访存优化串起来。读完后，你应该能写出一个结果正确、能被 benchmark 和 profiling 验证的教学版 Softmax。',
        sections: [
          ['Softmax 在 Transformer 中的位置', '说明为什么 Softmax 是理解注意力性能的重要入口。'],
          ['Naive Softmax', '从直接实现开始，观察重复访存和数值问题。'],
          ['数值稳定性', '使用减去最大值的形式避免指数溢出。'],
          ['访存优化', '减少多次读取和写回，理解中间结果如何组织。'],
          ['Block 级并行', '用 block 内协作处理一行或一段数据。'],
          ['与 PyTorch 结果对齐', '确认数值误差、输入范围和边界条件。'],
          ['思考题', '分析不同 hidden size 下实现策略的变化。']
        ]
      },
      {
        title: 'LayerNorm 优化',
        summary: '均值方差、Reduction + Normalize 融合、向量化读写、性能分析',
        status: '🚧',
        lead: '本章用 LayerNorm 继续练习 reduction，并引入融合与向量化读写的思路。读完后，你应该能解释为什么把多步归一化合在一个 kernel 里通常更高效。',
        sections: [
          ['LayerNorm 原理', '回顾均值、方差、缩放和平移的计算流程。'],
          ['均值与方差计算', '把两个 reduction 放到 GPU 执行模型里分析。'],
          ['Reduction + Normalize 融合', '减少 kernel launch 和中间数据写回。'],
          ['向量化读写', '观察数据对齐和一次处理多个元素对性能的影响。'],
          ['性能分析', '用 profiling 判断瓶颈是否仍然来自访存。'],
          ['思考题', '比较不同 hidden size 和 batch size 下的实现选择。']
        ]
      },
      {
        title: 'Matmul 入门优化',
        summary: 'Naive GEMM、Tiling、LDS 缓存、Register Blocking、rocBLAS 差距观察',
        status: '🚧',
        lead: '本章用教学版 GEMM 理解矩阵乘为什么是 AI 计算的核心。目标不是追平 rocBLAS，而是通过 tiling、LDS 和寄存器复用看懂高性能 GEMM 的基本方向。',
        sections: [
          ['GEMM 为什么是核心算子', '说明矩阵乘在神经网络和注意力计算中的地位。'],
          ['Naive Matmul', '写出最直接的一线程计算一个输出元素的实现。'],
          ['Tiling', '把矩阵拆块，理解数据复用的第一步。'],
          ['LDS 缓存', '用 LDS 缓存 tile，减少全局内存重复读取。'],
          ['Register Blocking', '观察每个线程计算多个输出时的寄存器复用。'],
          ['简化版高性能 GEMM', '组合前面的优化，形成一个教学版优化实现。'],
          ['与 rocBLAS 对比', '只观察差距和方向，不承诺达到库级性能。'],
          ['思考题', '分析 tile size、数据类型和矩阵形状对性能的影响。']
        ]
      }
    ]
  },
  {
    prefix: '/part4-triton/',
    navText: 'Triton',
    title: 'Triton on AMD 与自动调参',
    readmeTitle: '第 4 篇：Triton on AMD 与自动调参',
    chapters: [
      {
        title: 'Triton 编程模型',
        summary: 'Triton vs HIP、program model、block 级张量、AMD 环境验证',
        status: '🚧',
        lead: '本章从 HIP 切换到 Triton，重点理解 Triton 为什么让算子开发更接近张量块级编程。读完后，你应该能解释 program、block 和 mask 如何对应到数据。',
        sections: [
          ['为什么需要 Triton', '说明在手写 HIP 和调用库之间，Triton 提供了什么折中。'],
          ['Triton 和 HIP 的区别', '对比线程级编程和 block 级张量编程的思维差异。'],
          ['Triton on AMD 环境配置', '验证当前 AMD ROCm 环境下 Triton 是否可用。'],
          ['第一个 Triton kernel', '用最小示例理解 program id、block 指针和 mask。'],
          ['Triton 的 block / program model', '把 Triton 抽象映射回 GPU 执行模型。']
        ]
      },
      {
        title: 'Triton Matmul 优化',
        summary: 'Triton GEMM、tile 设计、数据复用、benchmark、HIP / rocBLAS 对比',
        status: '🚧',
        lead: '本章用 Matmul 作为第一个 Triton 主案例，理解 block 级矩阵乘如何表达 tiling 和数据复用。读完后，你应该能写出一个教学版 Triton GEMM，并用 benchmark 观察它和 HIP / rocBLAS 的差距。',
        sections: [
          ['Matmul 的计算形状', '明确 M、N、K 维度、输入布局和输出 tile 的关系。'],
          ['PyTorch 与 rocBLAS baseline', '建立可对比的库函数基线。'],
          ['Naive Triton Matmul', '用 Triton 写出第一个可运行矩阵乘 kernel。'],
          ['Tile 大小如何影响性能', '观察 BLOCK_M、BLOCK_N、BLOCK_K 对并行度和复用的影响。'],
          ['数据加载、mask 与边界处理', '处理非整除形状和越界访问。'],
          ['Benchmark 与 profiling', '用相同实验流程比较不同 config。'],
          ['与 HIP 实现对比', '从代码复杂度和性能证据两方面比较 Triton 与 HIP。']
        ]
      },
      {
        title: 'Triton Softmax 优化',
        summary: '行级 Softmax、数值稳定、block reduction、访存优化、PyTorch 对齐',
        status: '🚧',
        lead: '本章用 Softmax 作为第二个 Triton 主案例，重点练习 block 内 reduction、数值稳定性和访存控制。读完后，你应该能把 HIP Softmax 的思路迁移到 Triton 表达。',
        sections: [
          ['Softmax 的输入形状与 baseline', '确定按行计算的输入规模、输出校验和 PyTorch baseline。'],
          ['Naive Triton Softmax', '写出一行对应一个 program 的基础实现。'],
          ['数值稳定性', '实现减最大值、指数、求和和归一化。'],
          ['Block reduction 怎么表达', '理解 Triton 张量操作如何表达一行内归约。'],
          ['访存与中间结果优化', '减少重复加载和多余写回。'],
          ['Benchmark 与 profiling', '比较不同 block size 和输入形状。'],
          ['与 HIP Softmax 对比', '从实现复杂度和性能瓶颈角度复盘差异。']
        ]
      },
      {
        title: 'Triton Attention 优化',
        summary: 'QK^T、Softmax、PV、分块注意力、显存访问、可复现实验边界',
        status: '🚧',
        lead: '本章用 Attention 作为第三个 Triton 主案例，把 Matmul 和 Softmax 串成一个更接近真实模型的算子。读完后，你应该能理解注意力优化的主要难点来自分块、数值稳定和显存访问，而不是简单把三个算子拼起来。',
        sections: [
          ['Attention 计算流程', '拆解 QK^T、Softmax 和乘 V 三个阶段。'],
          ['PyTorch baseline 与输入约束', '先固定 batch、head、sequence 和 hidden size，保证实验可复现。'],
          ['Naive Triton Attention', '写出便于理解的基础实现，不一开始追求 FlashAttention 级复杂度。'],
          ['分块计算与显存压力', '理解为什么不能直接物化所有中间矩阵。'],
          ['数值稳定与 mask', '处理 causal mask、padding mask 和 softmax 稳定性。'],
          ['Benchmark 与 profiling', '观察序列长度、head dim 和 batch 对性能的影响。'],
          ['优化边界与后续方向', '说明哪些优化需要更多实测证据，避免未验证承诺。']
        ]
      },
      {
        title: 'Triton 自动调参',
        summary: '搜索空间、autotune、自动 benchmark、选择最优 kernel config',
        status: '🚧',
        lead: '本章把前面 Matmul、Softmax 和 Attention 中反复出现的 config 选择系统化。读完后，你应该知道如何定义搜索空间、运行自动 benchmark，并用证据选择当前硬件上的最优配置。',
        sections: [
          ['BLOCK_SIZE 怎么影响性能', '从前面算子的实验结果回顾 block 参数为什么重要。'],
          ['num_warps / num_stages 的意义', '理解并行度、流水和资源占用之间的权衡。'],
          ['搜索空间设计', '避免盲目枚举过大的配置集合。'],
          ['Triton autotune', '使用 Triton 的 autotune 机制管理候选配置。'],
          ['自动 benchmark', '记录每个候选的命令、输入、硬件和结果。'],
          ['自动选择最优 kernel config', '把性能结果转成可复用的配置选择。']
        ]
      }
    ]
  },
  {
    prefix: '/part5-inference/',
    navText: '单卡推理',
    title: '单卡推理与毕业项目',
    readmeTitle: '第 5 篇：单卡推理与毕业项目',
    chapters: [
      {
        title: '单卡推理性能全景',
        summary: '延迟、吞吐、精度、batch、单卡端到端 pipeline；明确单卡能解 vs 不能解的问题',
        status: '🚧',
        lead: '本章从单卡端到端视角理解推理性能，不只盯着单个 kernel。读完后，你应该能区分模型计算、预处理、后处理、数据传输各自的开销，并知道哪些问题在单卡上可解、哪些必须等 hello-mlsys（多副本/动态 batching/多卡通信）和 hello-ai-infra（服务化/集群部署）。',
        sections: [
          ['训练和推理的差异', '说明推理为什么更关注延迟、吞吐、稳定性和资源利用率。'],
          ['Latency 与 Throughput', '定义端到端延迟、单请求延迟和单卡吞吐。'],
          ['Batch 与吞吐（单卡视角）', '理解 batch size 对单卡性能的影响（不展开多请求调度）。'],
          ['FP32 / FP16 / BF16 / INT8', '介绍精度选择和量化对推理性能的影响边界。'],
          ['模型加载、预处理、后处理开销', '把非模型计算也纳入性能分析范围。'],
          ['端到端推理 pipeline', '画出从输入到输出的完整路径。'],
          ['单卡能解 vs 不能解的问题', '明确多副本、动态 batching、多卡通信交给后续两本书。']
        ]
      },
      {
        title: 'ONNX Runtime 与 MIGraphX 实战',
        summary: 'ONNX 导出、ROCm 推理、MIGraphX 运行、工具层性能对比',
        status: '🚧',
        lead: '本章只站在工具使用者角度，带你把模型导出、加载、运行并做性能对比。图优化为什么有效会留到编译器篇讲，这里重点是怎么正确使用和记录结果。',
        sections: [
          ['ONNX 模型导出', '从 PyTorch 模型导出 ONNX，并检查输入输出签名。'],
          ['ONNX Runtime on ROCm', '在 ROCm 环境下运行 ONNX Runtime，并记录基线性能。'],
          ['MIGraphX 基础运行', '使用 MIGraphX 加载和执行模型，观察工具链差异。'],
          ['开启工具层优化选项', '只介绍如何使用优化选项，并把效果与编译器篇原理对应。'],
          ['模型推理性能对比', '用相同输入和指标比较 PyTorch、ONNX Runtime、MIGraphX。'],
          ['结果记录与风险说明', '标注硬件、版本、模型和已验证边界。']
        ]
      },
      {
        title: '视觉模型推理案例：YOLO',
        summary: '图像预处理、NMS、单卡 batch 推理、pipeline profiling、性能报告',
        status: '🚧',
        lead: '本章用 YOLO 作为单卡视觉推理案例，练习从模型外部开销到 GPU kernel 的完整分析。读完后，你应该能确认一个推理 pipeline 慢，不一定是模型本身慢，所有数字均在 AI MAX 395 + ROCm 7.12.0 上实测。',
        sections: [
          ['YOLO 模型部署', '准备模型、输入图片和最小推理脚本。'],
          ['图像预处理优化', '分析 resize、normalize、layout transform 等 CPU/GPU 开销。'],
          ['后处理 NMS 优化', '观察 NMS 是否成为端到端瓶颈。'],
          ['Batch 推理（单卡范围）', '测试单卡上 batch size 对吞吐和延迟的影响。'],
          ['Pipeline profiling', '把预处理、模型、后处理开销拆开记录。'],
          ['性能报告', '输出一份包含瓶颈判断和下一步计划的案例报告。']
        ]
      },
      {
        title: 'LLM 单卡推理性能分析入门',
        summary: 'Prefill、Decode、TTFT、TPOT、KV Cache、显存观测；多卡/多请求留给 hello-mlsys',
        status: '🚧',
        lead: '本章建立 LLM 推理性能分析的基本框架，只覆盖单卡视角。读完后，你应该能在 AI MAX 395 + ROCm 7.12.0 上观察 TTFT、TPOT、KV Cache 和单卡显存占用，并理解 vLLM 多请求调度、PagedAttention、多卡 TP 推理等内容为什么留给 hello-mlsys。',
        sections: [
          ['LLM 推理流程', '拆解 tokenizer、prefill、decode、采样、输出后处理。'],
          ['Prefill 与 Decode', '理解两个阶段的计算形态和性能指标差异。'],
          ['TTFT、TPOT 与单卡吞吐', '定义 LLM 推理中常用的延迟和吞吐指标。'],
          ['KV Cache 与显存占用', '观察上下文长度、batch、cache 对单卡显存的影响。'],
          ['小模型推理 benchmark', '选择能在 AI MAX 395 上稳定运行的模型做可复现实验。'],
          ['单卡边界与下一步', '明确多副本、并发调度、动态 batching、多卡 TP 推理留给 hello-mlsys / hello-ai-infra。']
        ]
      },
      {
        title: '毕业项目：单卡 GPU 性能诊断报告',
        summary: '结合前 4 篇能力，对一个真实模型出一份单卡性能诊断与优化报告',
        status: '🚧',
        lead: '本章是全书的毕业项目。读完后，你应该能挑选一个真实模型，从硬件特征采集、baseline benchmark、profiling、瓶颈判断、优化尝试、对比验证、报告输出走完一个完整闭环——所有结果都在 AI MAX 395 + ROCm 7.12.0 上实测。',
        sections: [
          ['项目目标与验收标准', '定义毕业项目要解决的问题和最终输出。'],
          ['硬件画像与环境记录', '采集 GPU、ROCm、显存等硬件上下文。'],
          ['选择模型与定义 baseline', '在小 LLM 或 YOLO 中挑选一个，固定输入与指标。'],
          ['分层 profiling', '分别从 framework、kernel、硬件计数器三层收集证据。'],
          ['提出并验证优化', '至少尝试一个 kernel 替换或图优化，对比验证。'],
          ['输出诊断报告', '形成包含命令、日志、对比表格、风险说明的最终报告。'],
          ['延伸阅读：通向 hello-mlsys / hello-ai-infra', '说明哪些问题超出单卡范围，需要进入后续两本书。']
        ]
      }
    ]
  },
  {
    prefix: '/part6-compiler/',
    navText: '编译器',
    title: 'AI 编译器与自动调优',
    readmeTitle: '第 6 篇：AI 编译器与自动调优',
    chapters: [
      {
        title: 'AI 编译器到底在优化什么',
        summary: '模型图、计算图、算子、kernel、ISA、手写优化关系',
        status: '🚧',
        lead: '本章把编译器放回 AI Infra 优化链路中，解释它连接模型表达和硬件执行的方式。读完后，你应该能理解编译器优化和手写 kernel 优化不是互斥关系。',
        sections: [
          ['从模型图到计算图', '理解模型结构如何被表示成可优化的计算图。'],
          ['从计算图到算子', '观察图中节点如何落到具体算子或子图。'],
          ['从算子到 kernel', '理解算子实现如何选择库函数、生成代码或调用自定义 kernel。'],
          ['从 kernel 到 ISA', '说明最终执行仍然受硬件指令和资源约束。'],
          ['编译器优化和手写优化的关系', '解释两者如何互补，而不是互相替代。']
        ]
      },
      {
        title: '图优化原理基础',
        summary: '算子融合、常量折叠、死代码消除、布局优化、Memory Planning 原理',
        status: '🚧',
        lead: '本章只讲图优化背后的原理，不重复第 17 章的工具命令。读完后，你应该能解释为什么工具打开某些优化选项后，模型可能更快或更省显存。',
        sections: [
          ['Operator Fusion', '理解算子融合如何减少中间写回、kernel launch 和内存带宽压力。'],
          ['Constant Folding', '说明常量计算为什么可以提前完成。'],
          ['Dead Code Elimination', '理解无用节点如何被删除。'],
          ['Common Subexpression Elimination', '观察重复计算如何被复用。'],
          ['Layout Transform', '说明数据布局为什么影响后端 kernel 的访问效率。'],
          ['Memory Planning', '理解内存复用和生命周期分析如何降低峰值显存。']
        ]
      },
      {
        title: 'Kernel 生成与调度搜索',
        summary: 'Schedule 原语、搜索空间、Cost Model、AutoScheduler、硬件反馈',
        status: '🚧',
        lead: '本章从图优化进入 kernel 生成和调度搜索，解释自动调优为什么需要硬件反馈。读完后，你应该能把 Triton autotune 和编译器 schedule 搜索放到同一张图里理解。',
        sections: [
          ['Schedule 是什么', '理解同一个计算可以有不同执行计划。'],
          ['Tile / Split / Reorder / Vectorize / Unroll', '介绍常见 schedule 原语对执行方式的影响。'],
          ['搜索空间设计', '说明为什么可选 schedule 太多，需要约束搜索范围。'],
          ['Cost Model', '理解代价模型如何预测候选实现。'],
          ['AutoScheduler / MetaSchedule 思想', '认识自动搜索 schedule 的基本流程。'],
          ['为什么自动调优需要硬件反馈', '回到 AI MAX 395 实测，说明真实硬件数据不可替代。']
        ]
      },
      {
        title: 'TVM / Triton / MIGraphX 对比',
        summary: '三个工具的定位、适用问题和选择指南',
        status: '🚧',
        lead: '本章用前面学过的工具和原理做一次定位对比。读完后，你应该能根据问题类型判断是该写 Triton kernel、用 MIGraphX 跑推理，还是研究 TVM 类编译器。',
        sections: [
          ['TVM 的定位', '理解 TVM 更偏通用编译器和 schedule 搜索平台。'],
          ['Triton 的定位', '理解 Triton 更适合手写和自动调参 GPU kernel。'],
          ['MIGraphX 的定位', '理解 MIGraphX 在 AMD 推理优化链路中的角色。'],
          ['三者适合解决的问题', '用问题类型对比各工具的使用边界。'],
          ['如何选择工具', '给出基于目标、成本和可复现性的选择建议。']
        ]
      }
    ]
  }
]

export function numberedChapters() {
  let number = 0
  return parts.flatMap((part) =>
    part.chapters.map((chapter) => {
      const chapterNumber = number++
      const partSlug = part.prefix.replace(/^\/|\/$/g, '')
      const chapterDir = `chapter${chapterNumber}`

      return {
        ...chapter,
        part,
        number: chapterNumber,
        path: `${part.prefix}${chapterDir}/`,
        source: `docs/${partSlug}/${chapterDir}/index.md`,
        code: `code/${partSlug}/${chapterDir}`,
      }
    }),
  )
}

export const chapters = numberedChapters()
export const chapterCount = chapters.length
export const bodyPartCount = parts.length - 1

export const navItems = [
  { text: '首页', link: '/' },
  { text: '全书目录', link: '/part0-preface/chapter0/' },
  { text: '学习路线', link: '/part0-preface/chapter0/#_0-6-学习路线图' },
  { text: '实验环境', link: '/part0-preface/chapter1/' },
  { text: 'GitHub', link: 'https://github.com/Weihong-Liu/hello-ai-infra' },
]

export const sidebar = parts.map((part) => ({
  text: part.readmeTitle,
  collapsed: false,
  items: chapters
    .filter((chapter) => chapter.part.prefix === part.prefix)
    .map((chapter) => ({
      text: `第 ${chapter.number} 章 ${chapter.title}`,
      link: chapter.path,
    })),
}))
