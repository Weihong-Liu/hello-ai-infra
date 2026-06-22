export const repoBaseUrl = 'https://github.com/datawhalechina/hello-gpu/blob/main'

export const parts = [
  {
    prefix: '/part0-intro/',
    navText: '入门',
    title: '入门与硬件速通',
    readmeTitle: '第 0 篇：入门与硬件速通',
    chapters: [
      {
        title: '写给读者的话',
        summary: '教程定位、为什么选 9070XT、和市面教程差异、学习路线',
        status: '🚧',
        lead: '本章是整本书的入口，先说明 hello-gpu 为什么存在、适合谁读，以及你最终会做出什么。读完后，你应该能判断自己是否适合继续学下去，并理解后续章节为什么围绕「算子从慢到快 + Agent 自动化」这条主线展开。',
        sections: [
          ['为什么写这本入门书', '从大模型推理和算子优化的真实痛点出发，说明只会调框架还不够，理解 GPU 底层对性能工作意味着什么。'],
          ['这本书适合谁', '说明读者需要的 Python、Linux、深度学习基础，以及什么样的读者收益最大。'],
          ['为什么选 AMD Radeon RX 9070 XT', '解释选消费卡 9070XT（RDNA4）作为主线的原因：易获得、成本低、Linux + ROCm 可跑；和数据中心卡的定位差异。'],
          ['这本书和市面教程有什么不同', '强调三条主线：Profiling-Driven（先会看数据再改代码）、刷题导向（每个算子都是从 naive 到优化的完整案例）、Agent-Driven（把优化流程交给 Agent 自动化）。'],
          ['你将完成的事', '预告全书闭环：学会几个经典算子的优化思路，最后亲手搭一个能自动做算子/模型优化的 Agent。'],
          ['学习路线图', '展示从环境 → profiling → 算子 → 刷题 → Agent 的学习路径，标出每篇大约需要的时间。'],
          ['和其他 datawhale 教程的边界', '说明本书聚焦单卡算子/Agent 优化；多卡通信、服务化、动态 batching 指向 hello-mlsys / hello-ai-infra。']
        ]
      },
      {
        title: '环境准备',
        summary: '9070XT + Linux + ROCm 6.4 验证、Windows 劝退、uv 环境、最小 smoke test',
        status: '🚧',
        lead: '本章不深入讲 ROCm 软件栈原理，只用最短路径帮你确认实验环境能不能继续往后跑。读完后，你应该能通过 uv sync 复现本篇环境，确认 ROCm 能看到 GPU，并跑通最小 HIP 程序。',
        sections: [
          ['本教程的实验基线', '明确所有实验默认在 Radeon RX 9070 XT + ROCm 6.4.x（Linux）上验证，其他设备只参考方法。'],
          ['为什么不用 Windows', '说明 9070XT 无官方 Windows ROCm 支持，劝退 Windows 路线，建议用 Linux 主机或 WSL2（但优先原生 Linux）。'],
          ['同步本篇 uv 环境', '进入 code/part0-intro 后运行 uv sync，并用 activate-rocm.sh 激活 ROCm wheel 环境。'],
          ['验证 GPU 可见性', '用 rocminfo 和 rocm-smi 检查 GPU、驱动、显存和运行状态。'],
          ['验证 PyTorch ROCm', '运行最小 PyTorch ROCm smoke test，确认框架能看到 GPU。'],
          ['验证最小 HIP 程序', '直接用 hipcc 编译并运行最小 vector add 程序。'],
          ['环境不通时先收集什么', '列出报错、版本、命令输出、硬件信息和日志，避免盲目排错。']
        ]
      },
      {
        title: 'GPU 体系结构速通',
        summary: 'CU/Wavefront/LDS/寄存器/显存层次，RDNA4 视角，不讲 MFMA/CDNA/HBM',
        status: '🚧',
        lead: '本章建立后续优化会反复用到的 GPU 硬件最小模型，但只讲 9070XT（RDNA4）用得到的部分。读完后，你应该能用自己的话讲清楚一个 kernel 从 launch 到执行经过哪些硬件单元，并知道 LDS、寄存器、wavefront 为什么是优化的核心资源。',
        sections: [
          ['GPU 为什么能并行', '用一张图说明 SIMT 执行模型，理解为什么 GPU 适合大规模数据并行。'],
          ['Compute Unit（CU）的内部结构', '逐层拆解 CU：SIMD 单元 / VALU / SALU / 标量与向量寄存器堆。'],
          ['Wavefront 与 SIMT 执行', '理解 32 或 64 线程一组的执行方式，以及分支收敛的代价。'],
          ['VGPR、SGPR 与 LDS 资源', '解释片上寄存器与共享内存为什么是 kernel 优化的核心资源。'],
          ['显存层次：寄存器 → LDS → L1/L2 → GDDR6', '说明 9070XT 的 16GB GDDR6（非 HBM）和各级缓存的容量、带宽、延迟分层。'],
          ['WMMA：RDNA4 的矩阵加速单元', '介绍 RDNA3+ 引入的 WMMA（不是 CDNA 的 MFMA），以及它在 GEMM/Attention 算子里的角色。'],
          ['Roofline 的硬件来源', '把硬件参数（峰值算力、带宽）翻译成 Roofline 上的两条线，建立「理论上限」直觉。']
        ]
      },
      {
        title: '第一个程序 + Roofline 心智模型',
        summary: 'vector add 跑通、建立性能上限直觉、benchmark 习惯',
        status: '🚧',
        lead: '本章在已经验证环境可用、且对硬件有基本心智模型的基础上，带你跑通第一个真正的 GPU 程序——vector add。重点不是算子本身，而是借此建立后续所有实验都会复用的计时习惯和 Roofline 心智模型。本章的 vector add 会贯穿整个 Part 1 profiling 篇。',
        sections: [
          ['从已经验证的环境开始', '复用第 1 章的环境验证结果，直接进入本章代码目录。'],
          ['跑通 vector add', '编写、编译并运行一个最小 HIP kernel，确认结果正确。'],
          ['建立 baseline benchmark', '用固定输入、热身、重复运行和 GPU event 建立可复查的计时 baseline。'],
          ['Roofline 心智模型', '用 vector add 的实测带宽和理论上限对比，建立「这个算子离极限有多远」的直觉。'],
          ['留下实验底稿', '说明源码、命令输出、benchmark 配置和 EXPERIMENT.md 应该如何对应。']
        ]
      }
    ]
  },
  {
    prefix: '/part1-profiling/',
    navText: 'Profiling',
    title: 'Profiling 实战',
    readmeTitle: '第 1 篇：Profiling 实战',
    chapters: [
      {
        title: 'benchmark 与可信计时',
        summary: '热身、重复、GPU event、避免测量陷阱',
        status: '🚧',
        lead: '本章继续用 vector add 做主线，但重点从「跑通」转向「量准」。读完后，你应该能解释为什么不能只跑一次就下结论，以及热身、重复、同步、日志记录这些「无聊」的细节如何决定数字是否可信。',
        sections: [
          ['为什么不能凭感觉优化', '用常见误区说明没有可信数据的优化为什么容易走偏。'],
          ['热身与缓存效应', '理解第一次运行为什么总是慢，以及为什么要丢掉前几次计时。'],
          ['重复运行与统计', '说明重复次数、取中位数还是最小值、如何报告方差。'],
          ['GPU event 计时', '用 HIP event 而不是 CPU wall clock 测量 kernel 真实耗时。'],
          ['避免测量陷阱', '识别数据拷贝遗漏、测量范围错误、缓存偶然命中导致的伪提升。'],
          ['可信 benchmark 的检查清单', '形成一份每次实验都该过一遍的检查清单。']
        ]
      },
      {
        title: 'rocprof + Omniperf 定位瓶颈',
        summary: 'kernel 耗时、访存/占用率计数器、strided 非合并访存反例',
        status: '🚧',
        lead: '本章用一个固定案例（vector add 的合并版 vs strided 非合并版）做主线，让每个 profiling 工具服务同一个问题：它到底慢在哪里。读完后，你应该能完成一次从 benchmark 到优化假设的最小闭环。',
        sections: [
          ['两个版本的 vector add', '准备合并访存版和 strided 非合并版，作为对照案例。'],
          ['运行 baseline benchmark', '用统一脚本记录两个版本的延迟、吞吐和硬件上下文。'],
          ['用 rocprof 看 kernel 时间', '采集 kernel 级耗时，找出主要开销来自哪里。'],
          ['用 Omniperf 看硬件计数器', '观察带宽利用率、访存合并、L2 命中率等底层信号。'],
          ['Occupancy 与 Wavefront 行为', '理解占用率、寄存器压力对性能的影响。'],
          ['从 profiling 结果反推优化方向', '把观测结果转成下一步实验假设，而不是直接拍脑袋改代码。']
        ]
      },
      {
        title: 'Roofline 曲线详解 + 性能报告',
        summary: '把算子点画到 Roofline、解释差距、报告模板',
        status: '🚧',
        lead: '本章把前两章的 profiling 过程整理成一份可复查的报告，并用 Roofline 曲线把算子性能可视化。读完后，你应该能把命令、日志、关键数字、瓶颈判断和下一步计划写成别人能复现的 Markdown。',
        sections: [
          ['Roofline 曲线怎么读', '复习 Ch3 的心智模型，这次用真实 profiling 数据画点。'],
          ['把 vector add 画到 Roofline 上', '用实测带宽和算力定位算子在曲线上的位置，解释它离极限有多远。'],
          ['解释差距来自哪里', '结合 Omniperf 计数器，说明为什么没达到理论上限。'],
          ['性能报告模板', '形成一份包含硬件上下文、命令、结果、瓶颈判断和风险说明的报告。'],
          ['本章小结：profiling 闭环', '回顾 Part 1 建立的「测量 → 定位 → 假设」闭环，预告 Part 2 算子篇会反复用到。']
        ]
      }
    ]
  },
  {
    prefix: '/part2-kernels/',
    navText: '算子优化',
    title: '算子优化 + 刷题',
    readmeTitle: '第 2 篇：算子优化 + 刷题',
    chapters: [
      {
        title: 'Reduction：从 naive 到 wavefront',
        summary: '跨线程归约、LDS 协作、多阶段、HIP/Triton 对比',
        status: '🚧',
        lead: '本章进入第一个真正体现 GPU 层级协作的算子：Reduction。读完后，你应该能理解为什么跨线程汇总需要 LDS、同步和多阶段设计，并能用 HIP 和 Triton 各写一版做对比。',
        sections: [
          ['Reduction 为什么重要', '说明求和、归约、归一化、注意力中为什么经常出现 reduction。'],
          ['Naive Reduction', '从简单但低效的实现开始，用 profiling 观察瓶颈。'],
          ['Shared Memory / LDS 优化', '使用 LDS 减少全局内存访问并组织 block 内归约。'],
          ['Wavefront 级优化', '理解 wavefront 内协作和分支收敛对 reduction 的影响。'],
          ['多阶段 Reduction', '把大规模输入拆成 block 内归约和跨 block 合并。'],
          ['Triton 版本对比', '用 Triton 写一版 reduction，从代码复杂度和性能两方面对比 HIP。'],
          ['性能对比与思考题', '用 benchmark 和 profiling 对比不同版本，分析输入长度和 block size 的影响。']
        ]
      },
      {
        title: 'Softmax：数值稳定 + 融合',
        summary: '减最大值、block 并行、减少写回、HIP/Triton 对比',
        status: '🚧',
        lead: '本章用 Softmax 把 reduction、数值稳定性和访存优化串起来。读完后，你应该能写出一个结果正确、能被 benchmark 和 profiling 验证的教学版 Softmax。',
        sections: [
          ['Softmax 在 Transformer 中的位置', '说明为什么 Softmax 是理解注意力性能的重要入口。'],
          ['Naive Softmax', '从直接实现开始，观察重复访存和数值问题。'],
          ['数值稳定性', '使用减去最大值的形式避免指数溢出。'],
          ['访存优化与融合', '减少多次读取和写回，理解中间结果如何组织。'],
          ['Block 级并行', '用 block 内协作处理一行或一段数据。'],
          ['Triton 版本对比', '用 Triton 写一版 softmax，对比 HIP 实现的复杂度和瓶颈。'],
          ['与 PyTorch 结果对齐', '确认数值误差、输入范围和边界条件。']
        ]
      },
      {
        title: 'GEMM：tiling + LDS',
        summary: '分块复用、寄存器 blocking、不追 rocBLAS、HIP/Triton 对比',
        status: '🚧',
        lead: '本章用教学版 GEMM 理解矩阵乘为什么是 AI 计算的核心。目标不是追平 rocBLAS，而是通过 tiling、LDS 和寄存器复用看懂高性能 GEMM 的基本方向。注意 9070XT 是 RDNA4，靠 WMMA 而非 MFMA。',
        sections: [
          ['GEMM 为什么是核心算子', '说明矩阵乘在神经网络和注意力计算中的地位。'],
          ['Naive Matmul', '写出最直接的一线程计算一个输出元素的实现。'],
          ['Tiling', '把矩阵拆块，理解数据复用的第一步。'],
          ['LDS 缓存', '用 LDS 缓存 tile，减少全局内存重复读取。'],
          ['Register Blocking', '观察每个线程计算多个输出时的寄存器复用。'],
          ['Triton 版本对比', '用 Triton 写一版 GEMM，对比 tiling 表达的简洁性。'],
          ['与 rocBLAS 对比（只看差距方向）', '只观察差距和方向，不承诺达到库级性能。']
        ]
      },
      {
        title: 'Flash Attention 思路',
        summary: '分块 + 在线 softmax、不物化中间矩阵（算子篇压轴）',
        status: '🚧',
        lead: '本章是算子篇的压轴，把前面 Reduction、Softmax、GEMM 的思路综合起来，理解 FlashAttention 为什么能大幅减少显存访问。重点是「思路」而非追平官方实现。',
        sections: [
          ['Attention 计算流程', '拆解 QK^T、Softmax、PV 三个阶段，理解为什么朴素实现慢。'],
          ['朴素 Attention 的显存瓶颈', '用 profiling 观察中间矩阵 S=QK^T、P=softmax(S) 物化带来的显存压力。'],
          ['分块计算（Tiling）', '把 Q、K、V 分块，理解为什么分块能减少全局访存。'],
          ['在线 Softmax（Online Softmax）', '理解为什么可以一边算一边归一化，不需要先算完整行。'],
          ['不物化中间矩阵', '把分块 + 在线 softmax 结合，避免写出 S、P 两个大矩阵。'],
          ['本章小结：算子篇的方法论回顾', '回顾 Reduction/Softmax/GEMM/Attention 共同的优化主线：减少访存、提升数据复用。']
        ]
      },
      {
        title: '怎么刷 LeetGPU',
        summary: '平台题型/评分、本地评测器、调试策略、性能闭环（硬件无关方法论）',
        status: '🚧',
        lead: '本章把前面四个算子积累的经验系统化成「刷题方法论」。重要前提：LeetGPU 目前仅支持 CUDA/Triton/PyTorch，不在 AMD 上提交；但刷题的方法论是硬件无关的——怎么读题、怎么搭本地评测、怎么用 profiling 驱动迭代，这些在哪个平台都通用。',
        sections: [
          ['GPU 算子题库长什么样', '介绍 LeetGPU / Tensara / GPU MODE 等平台的题目结构和评分机制（读题层，不依赖提交）。'],
          ['题型套路分类', '把题目分成 elementwise / reduction / GEMM-like / 融合型，对应本书 Ch4 / Ch7 / Ch9 / Ch10。'],
          ['本地评测器怎么搭', '复用 gpu-queue 思路，搭一个喂输入、跑 kernel、计时、对答案的本地评测器（跑在 9070XT 上）。'],
          ['刷题策略：正确性 → 带宽 → 计算强度', '说明每一步该用什么 profiling 工具验证，避免一上来就盲目优化。'],
          ['调试常见问题', '列出边界条件、数值误差、bank 冲突、occupancy 不足等常见坑。'],
          ['关于 AMD 平台的现状', '诚实说明 LeetGPU 当前是 CUDA-only；等 AMD 等价平台出现，本章方法论迁移成本很低。']
        ]
      }
    ]
  },
  {
    prefix: '/part3-agent/',
    navText: 'Agent',
    title: 'Agent（算子层）',
    readmeTitle: '第 3 篇：Agent（算子层）',
    chapters: [
      {
        title: 'Agent 入门',
        summary: '参考 hello-agents、LLM Agent 基本范式、工具调用',
        status: '🚧',
        lead: '本章是 Agent 篇的入口，参考 hello-agents 的概念铺垫节奏，讲清楚 LLM Agent 的基本范式。但本书的 Agent 场景是「算子/模型优化」，不是通用智能体——这是和 hello-agents 的关键区别。',
        sections: [
          ['什么是 LLM Agent', '用最简模型理解 Agent = LLM + 工具 + 循环，参考 hello-agents 第 1 章。'],
          ['为什么 Agent 适合算子优化', '说明算子优化天然适合 Agent：有明确目标（性能）、有可调用工具（编译/跑分/profiling）、有可验证反馈（benchmark 数字）。'],
          ['ReAct 范式简介', '理解 Reason-Act-Observe 循环，这是后续算子优化 Agent 的基本骨架。'],
          ['工具调用（Tool Use）', '理解 Agent 如何通过结构化接口调用外部工具。'],
          ['本书 Agent 的边界', '明确本书 Agent 聚焦算子/模型优化，不做通用代码生成或对话助手。'],
          ['和 hello-agents 的关系', '说明本书 Agent 篇假设你已了解 Agent 基本概念；零基础建议先读 hello-agents。']
        ]
      },
      {
        title: '工具封装',
        summary: 'benchmark/profiling/编译包成 Agent 可调用工具',
        status: '🚧',
        lead: '本章把 Part 1 学过的 benchmark、rocprof、Omniperf 以及编译流程，封装成 Agent 能调用的标准化工具。这是让 Agent「能动手」的前提——没有工具的 Agent 只会空谈。',
        sections: [
          ['为什么要封装工具', '说明 Agent 不能直接操作 shell，需要结构化、可解析的工具接口。'],
          ['封装 benchmark 工具', '把 Part 1 的计时脚本包成输入 kernel → 输出延迟/带宽的标准化工具。'],
          ['封装 profiling 工具', '把 rocprof/Omniperf 包成输入 kernel → 输出瓶颈信号的标准化工具。'],
          ['封装编译工具', '把 hipcc/triton 编译流程包成输入代码 → 输出编译成功/失败的标准化工具。'],
          ['工具的输入输出 schema', '用 JSON schema 定义每个工具的接口，让 Agent 能正确调用。'],
          ['错误处理与重试', '说明工具失败时如何把错误信息回传给 Agent 触发反思。']
        ]
      },
      {
        title: '算子优化 Agent 设计',
        summary: '读题→生成 kernel→跑分→反思迭代',
        status: '🚧',
        lead: '本章把前面封装的工具组装成一个完整的算子优化 Agent。读完后，你应该能理解 Agent 如何从一道算子题目出发，自动生成 kernel、跑 benchmark、根据结果反思改写。',
        sections: [
          ['Agent 的整体架构', '画出 读题 → 生成 → 编译 → 跑分 → 反思 的循环架构图。'],
          ['读题与问题理解', '让 Agent 解析题目规格（输入形状、数据类型、期望性能）。'],
          ['生成初始 kernel', '让 Agent 根据题目生成第一版 naive kernel 作为 baseline。'],
          ['跑分与性能反馈', '调用 benchmark 工具拿到延迟/带宽，转成 Agent 能理解的反馈。'],
          ['反思与改写', '让 Agent 根据 profiling 信号（访存瓶颈？计算瓶颈？）决定下一步优化方向。'],
          ['迭代终止条件', '说明什么时候停（达到目标性能、迭代轮次上限、连续无提升）。']
        ]
      },
      {
        title: '多轮优化实战',
        summary: 'Agent 把 naive kernel 优化 3-5x、失败回退、对比报告',
        status: '🚧',
        lead: '本章是 Agent 算子层篇的高潮：让 Agent 对一个 naive kernel 跑完整的多轮优化，观察它如何从慢版本一步步优化到 3-5 倍。重点是看 Agent 的行为轨迹，以及失败时如何回退。',
        sections: [
          ['选一个教学算子', '挑一个优化空间大的算子（如 reduction 或 matmul naive 版）作为 Agent 的优化对象。'],
          ['记录每轮优化的轨迹', '把 Agent 每轮的生成代码、benchmark 结果、反思内容完整落盘。'],
          ['观察性能提升曲线', '画出 Agent 多轮优化的性能变化，理解哪几轮提升最大、为什么。'],
          ['失败回退机制', '当某轮优化反而变慢或编译失败时，Agent 如何识别并回退到上一版。'],
          ['和人工优化的对比', '把 Agent 优化结果和人工优化的版本对比，讨论 Agent 的优势与局限。'],
          ['生成对比报告', '输出一份包含轨迹、性能曲线、关键决策的优化报告。']
        ]
      }
    ]
  },
  {
    prefix: '/part4-models-agent/',
    navText: '模型 + Agent',
    title: '真实模型 + Agent',
    readmeTitle: '第 4 篇：真实模型 + Agent',
    chapters: [
      {
        title: 'YOLO 部署 + Agent 自动优化',
        summary: 'ONNX/MIGraphX 部署、Agent profiling 找瓶颈、改配置/算子、对比',
        status: '🚧',
        lead: '本章把 Agent 的优化对象从单个教学算子升级为真实模型——YOLO。先快速部署 YOLO，再让 Agent 对它跑 profiling、识别瓶颈、给配置建议（batch/精度/算子融合）、改配置跑对比。注意：Agent 能改配置和换算子，但不能自动写新算子集成进 ONNX——这个边界在章首点明。',
        sections: [
          ['Agent 能力边界：算子层 vs 模型层', '明确本章 Agent 在模型层能做什么（profiling、改配置、换算子、出报告）、不能做什么（自动写新算子集成进 ONNX/MIGraphX）。'],
          ['YOLO 模型部署', '准备 YOLOv8 模型，导出 ONNX，用 MIGraphX 或 ONNX Runtime-ROCm 跑通推理。'],
          ['建立推理 baseline', '用第 4 章的 benchmark 习惯记录端到端延迟、吞吐和硬件上下文。'],
          ['Agent 跑推理 profiling', '让 Agent 自动 profiling，识别瓶颈在预处理、NMS、还是模型 kernel。'],
          ['Agent 给配置优化建议', '让 Agent 根据 profiling 信号给出 batch size、精度、算子融合等配置建议。'],
          ['改配置跑对比', '让 Agent 自动改配置、重跑 benchmark、对比优化前后性能。'],
          ['输出优化报告', '形成一份包含瓶颈判断、优化尝试、对比数据的 YOLO 优化报告。']
        ]
      },
      {
        title: '小模型 LLM 解码 + Agent 自动优化',
        summary: 'Qwen 0.5B/1.8B 量化、decode 算子视角、Agent 优化 KV cache/精度',
        status: '🚧',
        lead: '本章把 Agent 应用到 LLM 解码场景。受 9070XT 16GB 显存限制，只能量化小模型（Qwen 0.5B/1.8B）。重点是 decode 阶段的算子视角：Agent 如何优化 KV cache 访问、精度选择等。注意：PagedAttention、多卡 TP 等留给 hello-mlsys。',
        sections: [
          ['显存约束下的模型选型', '说明 9070XT 16GB 显存为什么选 Qwen 0.5B/1.8B 量化，跑不了 7B+。'],
          ['LLM 推理流程拆解', '拆解 prefill 和 decode 两个阶段，理解 decode 为什么是算子密集型。'],
          ['建立 decode baseline', '测量 TTFT、TPOT、KV cache 显存占用作为 baseline。'],
          ['decode 的算子视角', '把 decode 拆成 attention（KV cache 读取）+ matmul（投影）两个核心算子。'],
          ['Agent 优化 KV cache 访问', '让 Agent 分析 KV cache 的访存模式，给出精度/布局优化建议。'],
          ['Agent 优化精度选择', '让 Agent 对比 fp16/int8/int4 在 decode 性能和显存上的权衡。'],
          ['单卡边界与下一步', '明确 PagedAttention、多卡 TP、并发调度等留给 hello-mlsys。']
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

// 附录：独立成篇，不进 numberedChapters 的连续正文编号。
// 附录编号用 appendix-A / appendix-B 这样的形式，避免和正文章号冲突。
export const appendices = [
  {
    title: '附录 A · 环境安装细节与常见坑',
    summary: '本篇环境文件是怎么来的、为什么 AMD wheel 源要 explicit、rocm-sdk init 的坑',
    lead: '主线内容只要求你会跑 uv sync 和几个验证命令。但很多读者还会想知道——这套环境文件到底是怎么来的？本附录从这个问题出发，按步骤拆开本篇环境的生成过程，顺便把几个反复出现的坑提前指出来。',
    slug: 'appendix-a',
    dir: 'appendix-a-env-install',
    path: '/appendix/appendix-a-env-install/',
    source: 'docs/appendix/appendix-a-env-install/index.md',
  },
  {
    title: '附录 B · 换一张卡：从 gfx120X-all 迁移到 gfx1151',
    summary: 'AMD wheel 源按架构分开打包，换卡时需要改哪些地方、为什么这么改',
    lead: '本教程的实验基线是 gfx120X-all（RX 9070 XT / gfx1201）+ ROCm 7.13.0。但如果你手上的是其它架构（比如 RDNA 3.5 的 gfx1151 / AI MAX 395），照着本教程的 pyproject.toml 抄下来，uv sync 很可能直接报错。本附录只回答一个问题：换一张卡，环境文件到底要动哪几行？',
    slug: 'appendix-b',
    dir: 'appendix-b-switch-gpu',
    path: '/appendix/appendix-b-switch-gpu/',
    source: 'docs/appendix/appendix-b-switch-gpu/index.md',
  },
]

export const chapters = numberedChapters()
export const chapterCount = chapters.length
export const appendixCount = appendices.length
export const bodyPartCount = parts.length - 1

export const navItems = [
  { text: '首页', link: '/' },
  { text: '全书目录', link: '/part0-intro/chapter0/' },
  { text: '实验环境', link: '/part0-intro/chapter1/' },
  { text: 'GitHub', link: 'https://github.com/datawhalechina/hello-gpu' },
]

export const sidebar = [
  ...parts.map((part) => ({
    text: part.readmeTitle,
    collapsed: false,
    items: chapters
      .filter((chapter) => chapter.part.prefix === part.prefix)
      .map((chapter) => ({
        text: `第 ${chapter.number} 章 ${chapter.title}`,
        link: chapter.path,
      })),
  })),
  ...(appendixCount > 0
    ? [
        {
          text: '附录',
          collapsed: false,
          items: appendices.map((a) => ({
            text: a.title,
            link: a.path,
          })),
        },
      ]
    : []),
]
