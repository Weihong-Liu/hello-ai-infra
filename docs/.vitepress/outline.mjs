export const repoBaseUrl = 'https://github.com/Weihong-Liu/hello-ai-infra/blob/main'

export const parts = [
  {
    prefix: '/part0-preface/',
    navText: '前言',
    title: '前言与学习路线',
    readmeTitle: '第 0 篇：前言与学习路线',
    chapters: [
      {
        path: '/part0-preface/chapter0/',
        source: 'docs/part0-preface/chapter0/index.md',
        code: 'code/part0-preface/chapter0',
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
        path: '/part0-preface/chapter1/',
        source: 'docs/part0-preface/chapter1/index.md',
        code: 'code/part0-preface/chapter1',
        title: '环境准备与验证',
        summary: '本地文档预览、AI MAX 395 + ROCm 7.12.0 基线、最小环境验证',
        status: '🚧',
        lead: '本章先不深入讲 ROCm 软件栈原理，而是帮你用最短路径确认两件事：本地文档站点能打开，实验机上的 AMD GPU 环境能跑。读完后，你应该能判断自己的环境是否适合继续跟着后续章节做实验。',
        sections: [
          ['本教程的实验基线', '明确所有实验默认在 AI MAX 395 + ROCm 7.12.0 上验证，其他设备只参考方法。'],
          ['本地文档环境', '用 Node、npm 和 VitePress 在本地启动教程站点，方便边读边改。'],
          ['实验机环境分工', '说明本地 Mac 负责 git 和写作，AMD-AIMAX395 只负责运行 ROCm / HIP / profiling 实验。'],
          ['准备本篇 uv 环境', '使用项目 bootstrap 脚本创建独立 Python 环境，并理解每篇一个 venv 的约定。'],
          ['验证 GPU 可见性', '用 rocminfo 和 rocm-smi 检查 GPU、驱动、显存和运行状态。'],
          ['验证 PyTorch ROCm', '运行最小 PyTorch ROCm smoke test，确认框架能看到 GPU。'],
          ['验证最小 HIP 程序', '跑通一个最小 GPU 程序，为后续 HIP kernel 实验做准备。'],
          ['环境不通时先收集什么', '列出报错、版本、命令输出、硬件信息和日志，避免盲目排错。']
        ]
      }
    ]
  },
  {
    prefix: '/part1-hardware-rocm/',
    navText: '硬件基础',
    title: 'AI Infra 全景与 AMD GPU 基础',
    readmeTitle: '第 1 篇：AI Infra 全景与 AMD GPU 基础',
    chapters: [
      {
        path: '/part1-hardware-rocm/chapter1/',
        source: 'docs/part1-hardware-rocm/chapter1/index.md',
        code: 'code/part1-hardware-rocm/chapter1',
        title: 'AI Infra 全景图',
        summary: '从模型到硬件的完整链路、HPOA 方法论',
        status: '🚧',
        lead: '本章从一次模型请求出发，把模型、框架、算子、编译器、运行时和硬件串成一张图。读完后，你应该能说清楚 AI Infra 优化到底在优化哪一层。',
        sections: [
          ['从模型到硬件：一次推理请求经历了什么', '沿着请求路径观察模型服务、框架调度、算子执行和硬件运行的关系。'],
          ['AI Infra 的核心模块', '理解训练框架、推理引擎、算子库、编译器、profiling 和调度系统各自的位置。'],
          ['算子优化、推理优化、编译器优化分别解决什么问题', '区分三类优化的边界，避免把所有性能问题都归因于 kernel。'],
          ['AI Infra 工程师的核心能力模型', '总结硬件理解、性能分析、工程实现和实验复现四类能力。'],
          ['本教程的学习闭环：理解 -> 测量 -> 优化 -> 自动化', '说明后续每一篇如何围绕同一个优化闭环推进。']
        ]
      },
      {
        path: '/part1-hardware-rocm/chapter2/',
        source: 'docs/part1-hardware-rocm/chapter2/index.md',
        code: 'code/part1-hardware-rocm/chapter2',
        title: 'AMD GPU 与 ROCm 软件栈',
        summary: 'AMD GPU 执行模型、ROCm 层次、上层框架关系',
        status: '🚧',
        lead: '本章建立后续优化会反复用到的 AMD GPU 最小心智模型，同时解释 ROCm 软件栈如何把上层框架连接到硬件。读完后，你应该能看懂后续章节里 CU、Wavefront、LDS、HIP、MIGraphX 等词的位置。',
        sections: [
          ['AMD GPU 基本架构', '介绍 CU、Wavefront 和 SIMD 执行模型，建立硬件执行的第一层直觉。'],
          ['CU、Wavefront、SIMD、LDS、VGPR、SGPR', '解释关键硬件单元，并用后续算子优化会遇到的问题来理解它们。'],
          ['HBM、Cache 与访存层次', '理解显存带宽、缓存层次和数据复用为什么影响 AI 算子性能。'],
          ['ROCm 是什么', '用分层图理解驱动、运行时、编译器、库和工具链之间的关系。'],
          ['HIP、HSA、AMDGPU Driver 的关系', '厘清 HIP 程序如何经由运行时和驱动落到设备执行。'],
          ['PyTorch / Triton / MIGraphX / vLLM 与 ROCm 的关系', '说明上层框架如何依赖 ROCm 能力，但不在本章展开使用细节。'],
          ['如何检查一台机器的 AMD GPU 环境', '把环境检查命令放回软件栈语境中，解释每个命令在检查什么。']
        ]
      },
      {
        path: '/part1-hardware-rocm/chapter3/',
        source: 'docs/part1-hardware-rocm/chapter3/index.md',
        code: 'code/part1-hardware-rocm/chapter3',
        title: '第一个 AMD GPU 程序',
        summary: 'PyTorch ROCm、最小 HIP kernel、baseline benchmark',
        status: '🚧',
        lead: '本章在已经验证环境可用的基础上，带你跑通第一个真正的 AMD GPU 程序。重点不是安装百科，而是建立后续实验都会复用的代码、计时和日志习惯。',
        sections: [
          ['确认当前章节环境', '复用 Part 0 的环境基线，确认当前章节代码目录和运行环境已经就绪。'],
          ['读取硬件信息', '用 rocminfo 和 rocm-smi 记录实验机器的关键上下文。'],
          ['跑通 PyTorch ROCm', '运行最小 tensor 运算，确认 PyTorch 能把计算放到 AMD GPU 上。'],
          ['跑通第一个 HIP kernel', '编写、编译并运行一个最小 HIP kernel。'],
          ['建立 baseline benchmark', '用固定输入、热身、重复运行和日志文件建立可复现的计时 baseline。'],
          ['留下实验底稿', '说明代码、日志和 EXPERIMENT.md 应该如何对应。']
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
        path: '/part2-profiling/chapter4/',
        source: 'docs/part2-profiling/chapter4/index.md',
        code: 'code/part2-profiling/chapter4',
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
        path: '/part2-profiling/chapter5/',
        source: 'docs/part2-profiling/chapter5/index.md',
        code: 'code/part2-profiling/chapter5',
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
        path: '/part2-profiling/chapter6/',
        source: 'docs/part2-profiling/chapter6/index.md',
        code: 'code/part2-profiling/chapter6',
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
        path: '/part2-profiling/chapter7/',
        source: 'docs/part2-profiling/chapter7/index.md',
        code: 'code/part2-profiling/chapter7',
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
        path: '/part3-hip-kernels/chapter7/',
        source: 'docs/part3-hip-kernels/chapter7/index.md',
        code: 'code/part3-hip-kernels/chapter7',
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
        path: '/part3-hip-kernels/chapter8/',
        source: 'docs/part3-hip-kernels/chapter8/index.md',
        code: 'code/part3-hip-kernels/chapter8',
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
        path: '/part3-hip-kernels/chapter9/',
        source: 'docs/part3-hip-kernels/chapter9/index.md',
        code: 'code/part3-hip-kernels/chapter9',
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
        path: '/part3-hip-kernels/chapter10/',
        source: 'docs/part3-hip-kernels/chapter10/index.md',
        code: 'code/part3-hip-kernels/chapter10',
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
        path: '/part3-hip-kernels/chapter11/',
        source: 'docs/part3-hip-kernels/chapter11/index.md',
        code: 'code/part3-hip-kernels/chapter11',
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
        path: '/part3-hip-kernels/chapter12/',
        source: 'docs/part3-hip-kernels/chapter12/index.md',
        code: 'code/part3-hip-kernels/chapter12',
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
        path: '/part4-triton/chapter13/',
        source: 'docs/part4-triton/chapter13/index.md',
        code: 'code/part4-triton/chapter13',
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
        path: '/part4-triton/chapter14/',
        source: 'docs/part4-triton/chapter14/index.md',
        code: 'code/part4-triton/chapter14',
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
        path: '/part4-triton/chapter15/',
        source: 'docs/part4-triton/chapter15/index.md',
        code: 'code/part4-triton/chapter15',
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
        path: '/part4-triton/chapter16/',
        source: 'docs/part4-triton/chapter16/index.md',
        code: 'code/part4-triton/chapter16',
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
        path: '/part4-triton/chapter17/',
        source: 'docs/part4-triton/chapter17/index.md',
        code: 'code/part4-triton/chapter17',
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
    navText: '推理优化',
    title: '推理优化与模型部署',
    readmeTitle: '第 5 篇：推理优化与模型部署',
    chapters: [
      {
        path: '/part5-inference/chapter16/',
        source: 'docs/part5-inference/chapter16/index.md',
        code: 'code/part5-inference/chapter16',
        title: '推理优化全景',
        summary: '延迟、吞吐、精度、batch、并发、端到端 pipeline',
        status: '🚧',
        lead: '本章从端到端服务视角理解推理优化，不只盯着单个 kernel。读完后，你应该能区分模型计算、预处理、后处理、数据传输和服务调度各自的开销。',
        sections: [
          ['训练和推理的差异', '说明推理为什么更关注延迟、吞吐、稳定性和资源利用率。'],
          ['Latency 与 Throughput', '定义端到端延迟、单请求延迟和系统吞吐。'],
          ['Batch、并发与吞吐', '理解 batch size 和并发请求对性能的影响。'],
          ['FP32 / FP16 / BF16 / INT8', '介绍精度选择和量化对推理性能的影响边界。'],
          ['模型加载、预处理、后处理与服务开销', '把非模型计算也纳入性能分析范围。'],
          ['端到端推理 pipeline', '画出从输入到输出的完整路径。']
        ]
      },
      {
        path: '/part5-inference/chapter17/',
        source: 'docs/part5-inference/chapter17/index.md',
        code: 'code/part5-inference/chapter17',
        title: 'ONNX Runtime / MIGraphX 工具实战',
        summary: 'ONNX 导出、ROCm 推理、MIGraphX 运行、工具层性能对比',
        status: '🚧',
        lead: '本章只站在工具使用者角度，带你把模型导出、加载、运行并做性能对比。图优化为什么有效会留到编译器篇讲，这里重点是怎么正确使用和记录结果。',
        sections: [
          ['ONNX 模型导出', '从 PyTorch 模型导出 ONNX，并检查输入输出签名。'],
          ['ONNX Runtime on ROCm', '在 ROCm 环境下运行 ONNX Runtime，并记录基线性能。'],
          ['MIGraphX 基础运行', '使用 MIGraphX 加载和执行模型，观察工具链差异。'],
          ['开启工具层优化选项', '只介绍如何使用优化选项，不在本章展开优化原理。'],
          ['模型推理性能对比', '用相同输入和指标比较 PyTorch、ONNX Runtime、MIGraphX。'],
          ['结果记录与风险说明', '标注硬件、版本、模型和已验证边界。']
        ]
      },
      {
        path: '/part5-inference/chapter18/',
        source: 'docs/part5-inference/chapter18/index.md',
        code: 'code/part5-inference/chapter18',
        title: 'Triton Inference Server on AMD',
        summary: 'Model Repository、Backend、HTTP / gRPC、动态 batching、端到端测试',
        status: '🚧',
        lead: '本章把单机推理程序放进服务化框架中，观察服务层如何影响端到端性能。读完后，你应该能搭建最小 Model Repository，并用统一请求方式测试服务性能。',
        sections: [
          ['Triton Server 是什么', '理解推理服务和本地脚本的差异。'],
          ['Model Repository 结构', '组织模型文件、配置和版本目录。'],
          ['Python Backend / ONNX Backend / 自定义 Backend', '比较不同 backend 的定位和适用场景。'],
          ['gRPC / HTTP 调用', '用客户端发起请求并记录响应延迟。'],
          ['动态 batching', '观察动态 batch 对吞吐和延迟的影响。'],
          ['端到端性能测试', '把模型计算和服务开销放在同一份报告里。']
        ]
      },
      {
        path: '/part5-inference/chapter19/',
        source: 'docs/part5-inference/chapter19/index.md',
        code: 'code/part5-inference/chapter19',
        title: 'YOLO 推理优化案例',
        summary: '图像预处理、NMS、batch 推理、pipeline profiling、性能报告',
        status: '🚧',
        lead: '本章选择 YOLO 作为端到端视觉推理案例，练习从模型外部开销到 GPU kernel 的完整分析。读完后，你应该能知道一个推理 pipeline 慢，不一定是模型本身慢。',
        sections: [
          ['YOLO 模型部署', '准备模型、输入图片和最小推理脚本。'],
          ['图像预处理优化', '分析 resize、normalize、layout transform 等 CPU/GPU 开销。'],
          ['后处理 NMS 优化', '观察 NMS 是否成为端到端瓶颈。'],
          ['Batch 推理', '测试 batch size 对吞吐和延迟的影响。'],
          ['Pipeline profiling', '把预处理、模型、后处理和服务开销拆开记录。'],
          ['性能报告', '输出一份包含瓶颈判断和下一步计划的案例报告。']
        ]
      },
      {
        path: '/part5-inference/chapter20/',
        source: 'docs/part5-inference/chapter20/index.md',
        code: 'code/part5-inference/chapter20',
        title: 'LLM 推理性能分析入门',
        summary: 'Prefill、Decode、TTFT、TPOT、KV Cache、显存观测、batch / 并发',
        status: '🚧',
        lead: '本章降低对完整 vLLM 优化的承诺，先建立 LLM 推理性能分析的基本框架。读完后，你应该能观察 TTFT、TPOT、KV Cache 和显存占用，并理解后续是否引入 vLLM 需要先实测确认。',
        sections: [
          ['LLM 推理流程', '拆解 tokenizer、prefill、decode、采样和输出后处理。'],
          ['Prefill 与 Decode', '理解两个阶段的计算形态和性能指标差异。'],
          ['TTFT、TPOT 与吞吐', '定义 LLM 推理中更常用的延迟和吞吐指标。'],
          ['KV Cache 与显存占用', '观察上下文长度、batch 和 cache 对显存的影响。'],
          ['小模型推理 benchmark', '选择能在当前硬件上稳定运行的模型做可复现实验。'],
          ['Batch 与并发', '分析批处理和并发请求对延迟、吞吐和显存的影响。'],
          ['vLLM on AMD 的验证边界', '把 vLLM 作为待实测或可选路线，不在未验证前承诺优化结果。']
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
        path: '/part6-compiler/chapter21/',
        source: 'docs/part6-compiler/chapter21/index.md',
        code: 'code/part6-compiler/chapter21',
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
        path: '/part6-compiler/chapter22/',
        source: 'docs/part6-compiler/chapter22/index.md',
        code: 'code/part6-compiler/chapter22',
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
        path: '/part6-compiler/chapter23/',
        source: 'docs/part6-compiler/chapter23/index.md',
        code: 'code/part6-compiler/chapter23',
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
        path: '/part6-compiler/chapter24/',
        source: 'docs/part6-compiler/chapter24/index.md',
        code: 'code/part6-compiler/chapter24',
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
  },
  {
    prefix: '/part7-agent/',
    navText: 'Agent',
    title: 'AutoInfra Agent 自动优化系统',
    readmeTitle: '第 7 篇：AutoInfra Agent 自动优化系统',
    chapters: [
      {
        path: '/part7-agent/chapter25/',
        source: 'docs/part7-agent/chapter25/index.md',
        code: 'code/part7-agent/chapter25',
        title: '为什么 AI Infra 需要 Agent',
        summary: '人类优化流程、可自动化环节、LLM 角色、AutoInfra 总架构',
        status: '🚧',
        lead: '本章解释为什么 Agent 适合作为全书终点：它不是替代 profiling，而是把已经学过的 benchmark、profiling、优化假设和报告流程自动串起来。',
        sections: [
          ['人类优化专家的工作流', '复盘一个工程师从发现慢到验证优化的完整流程。'],
          ['哪些步骤可以自动化', '区分命令执行、日志解析、报告生成和代码修改等自动化边界。'],
          ['LLM 在优化流程里的角色', '说明 LLM 适合做解释、规划和代码候选，但不能替代实测。'],
          ['Agent 不能替代 profiling，但可以放大 profiling 的价值', '强调实验真实性和硬件反馈仍然是判断依据。'],
          ['AutoInfra Agent 总架构', '预览后续章节将逐步搭建的系统闭环。']
        ]
      },
      {
        path: '/part7-agent/chapter26/',
        source: 'docs/part7-agent/chapter26/index.md',
        code: 'code/part7-agent/chapter26',
        title: '实验资产与数据结构',
        summary: '硬件画像、实验配置、benchmark 结果、profiling 输出、报告数据模型',
        status: '🚧',
        lead: '本章先定义 Agent 系统要读写哪些数据，而不是急着写多个 Agent 类。读完后，你应该能理解硬件信息、实验配置、日志和报告之间如何关联。',
        sections: [
          ['硬件画像', '记录 GPU 型号、ROCm 版本、显存和关键能力。'],
          ['实验配置', '描述输入规模、命令、参数、版本和运行时间。'],
          ['Benchmark 结果', '规范 latency、throughput、memory 等指标的保存方式。'],
          ['Profiling 输出', '保存 timeline、kernel 表和硬件计数器摘要。'],
          ['报告数据模型', '把实验资产组织成后续 Agent 可以消费的结构。']
        ]
      },
      {
        path: '/part7-agent/chapter27/',
        source: 'docs/part7-agent/chapter27/index.md',
        code: 'code/part7-agent/chapter27',
        title: '自动 Benchmark 与 Profiling 数据管线',
        summary: '自动运行 benchmark、调用 profiling、保存结果、对比版本',
        status: '🚧',
        lead: '本章搭建系统闭环的第一段：让程序能自动运行实验并收集性能证据。读完后，你应该能把手工 benchmark 和 profiling 变成可重复调用的数据管线。',
        sections: [
          ['自动运行 benchmark', '把固定命令封装成可重复执行的任务。'],
          ['自动记录 latency / throughput / memory', '统一指标格式和日志路径。'],
          ['自动调用 profiling 工具', '根据配置调用 rocprof、PyTorch Profiler 或 Omniperf。'],
          ['自动保存实验结果', '把原始输出和摘要结果一起落盘。'],
          ['自动对比不同版本', '比较 baseline 与候选版本的关键指标。']
        ]
      },
      {
        path: '/part7-agent/chapter28/',
        source: 'docs/part7-agent/chapter28/index.md',
        code: 'code/part7-agent/chapter28',
        title: '瓶颈判断与优化计划',
        summary: 'Memory-bound / Compute-bound 判断、优化候选、优先级排序、实验计划',
        status: '🚧',
        lead: '本章让 Agent 从“收集数据”走向“提出下一步”。读完后，你应该能把 profiling 摘要转成结构化瓶颈判断和可执行实验计划。',
        sections: [
          ['判断 Memory-bound / Compute-bound', '把指标和方法论映射到瓶颈类型。'],
          ['生成优化候选', '根据瓶颈类型生成可能的改动方向。'],
          ['给优化策略排序', '根据预期收益、实现成本和风险排列优先级。'],
          ['生成实验计划', '把优化候选变成可运行命令、输入和验收指标。'],
          ['记录不确定性', '明确哪些判断只是假设，必须通过实验确认。']
        ]
      },
      {
        path: '/part7-agent/chapter29/',
        source: 'docs/part7-agent/chapter29/index.md',
        code: 'code/part7-agent/chapter29',
        title: '代码修改、运行与回滚',
        summary: '生成候选代码、自动插入 benchmark、检查编译错误、失败回滚',
        status: '🚧',
        lead: '本章进入最敏感的一步：让 Agent 修改代码。读完后，你应该理解为什么自动改代码必须和编译、测试、benchmark、回滚绑定在一起。',
        sections: [
          ['自动修改 Triton 参数', '从低风险的参数搜索开始练习代码改动。'],
          ['自动生成 HIP kernel 候选', '生成候选实现，但必须经过编译和运行验证。'],
          ['自动插入 benchmark 代码', '保证候选版本能用统一脚本测量。'],
          ['自动检查编译错误', '把失败信息转成下一步修复或放弃的依据。'],
          ['自动回滚失败版本', '避免失败候选污染后续实验。']
        ]
      },
      {
        path: '/part7-agent/chapter30/',
        source: 'docs/part7-agent/chapter30/index.md',
        code: 'code/part7-agent/chapter30',
        title: '报告生成与实验追踪',
        summary: 'before / after 表格、失败尝试、证据链、下一步优化方向',
        status: '🚧',
        lead: '本章把自动优化的结果变成可读、可审查、可复现的报告。读完后，你应该能让 Agent 不只给结论，还给出支撑结论的命令、日志和失败尝试。',
        sections: [
          ['自动生成优化报告', '把实验配置、硬件上下文和结果汇总成 Markdown。'],
          ['自动生成 before / after 表格', '清楚展示候选版本与 baseline 的差异。'],
          ['自动记录失败尝试', '保留失败路径，避免重复试错。'],
          ['自动总结下一步优化方向', '根据当前证据提出后续实验，而不是给绝对结论。'],
          ['报告的可信度边界', '说明哪些结论来自实测，哪些只是待验证假设。']
        ]
      },
      {
        path: '/part7-agent/chapter31/',
        source: 'docs/part7-agent/chapter31/index.md',
        code: 'code/part7-agent/chapter31',
        title: '最小 AutoInfra Agent 系统',
        summary: '把硬件采集、benchmark、profiling、计划、执行和报告串成闭环',
        status: '🚧',
        lead: '本章把前面分散的模块串成一个能跑的最小系统。读完后，你应该能从一个输入任务开始，让系统完成一次小规模自动性能分析和报告生成。',
        sections: [
          ['系统输入输出设计', '定义用户任务、代码路径、实验配置和报告输出。'],
          ['工作流编排', '把硬件采集、benchmark、profiling、计划、执行和报告串起来。'],
          ['状态与中间产物', '管理日志、候选版本、实验结果和失败记录。'],
          ['最小端到端运行', '选择一个低风险算子跑通完整闭环。'],
          ['系统限制', '明确当前系统不能自动承诺性能提升，必须依赖实测。']
        ]
      },
      {
        path: '/part7-agent/chapter32/',
        source: 'docs/part7-agent/chapter32/index.md',
        code: 'code/part7-agent/chapter32',
        title: '毕业项目：AutoInfra Agent 完整案例',
        summary: '完整系统、Triton Softmax、YOLO pipeline、AMD GPU 性能诊断报告',
        status: '🚧',
        lead: '本章是全书毕业项目，把前面所有能力组合成 AutoInfra Agent 的完整案例。读完后，你应该能理解一个自动优化系统如何从问题、实验、代码、数据和报告五个方面闭环。',
        sections: [
          ['项目目标', '定义毕业项目要解决的问题和验收标准。'],
          ['系统架构', '展示各模块如何连接成完整系统。'],
          ['输入输出设计', '明确用户输入、实验产物、代码候选和最终报告。'],
          ['完整运行流程', '串联一次端到端自动优化尝试。'],
          ['案例一：自动调优 Triton Softmax', '从较可控的 Triton 参数搜索开始。'],
          ['案例二：自动优化 YOLO 推理 pipeline', '把 Agent 应用到端到端推理链路。'],
          ['案例三：自动生成 AMD GPU 性能诊断报告', '生成包含硬件、profiling 和建议的诊断报告。']
        ]
      }
    ]
  }
]

export function numberedChapters() {
  let number = 0
  return parts.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part, number: number++ })))
}

export const chapters = numberedChapters()
export const chapterCount = chapters.length
export const bodyPartCount = parts.length - 1

export const navItems = parts.map((part) => ({
  text: part.navText,
  link: part.chapters[0].path,
}))

export const sidebar = Object.fromEntries(
  parts.map((part) => [
    part.prefix,
    [
      {
        text: part.title,
        items: part.chapters.map((chapter) => {
          const numbered = chapters.find((item) => item.path === chapter.path)
          return {
            text: `第 ${numbered.number} 章 ${chapter.title}`,
            link: chapter.path,
          }
        }),
      },
    ],
  ]),
)
