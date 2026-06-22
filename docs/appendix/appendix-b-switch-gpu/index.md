---
title: "附录 B · 换一张卡：从 gfx120X-all 迁移到 gfx1151"
description: "Hello GPU 附录 · AMD wheel 源按架构分开打包，换卡时需要改哪些地方、为什么这么改"
---

# 附录 B · 换一张卡：从 gfx120X-all 迁移到 gfx1151

## 本附录导读

> 本教程的实验基线是 **gfx120X-all（RX 9070 XT / gfx1201）+ ROCm 7.13.0**。但如果你手上的是 RDNA 3.5 的 gfx1151（比如 AI MAX 395），照着本教程的 `pyproject.toml` 抄下来，`uv sync` 很可能直接报错。这一节就是给这种情况准备的——它只回答一个问题：**换一张卡，环境文件到底要动哪几行？**

好消息是，这件事比看起来简单得多。AMD 的 wheel 是按 GPU 架构单独编译、放在不同目录里的——换卡时，只要把 wheel 源和 libraries 包名这两个地方改对，**版本号一行都不用动**。

| 要改的两处 | gfx120X-all（本教程基线） | gfx1151（你的卡） |
| ---- | ---- | ---- |
| wheel 源路径 | `gfx120X-all/` | `gfx1151/` |
| libraries 包名 | `rocm-sdk-libraries-gfx120x-all` | `rocm-sdk-libraries-gfx1151` |
| 版本组合 | ROCm 7.13.0 / torch 2.11.0 / triton 3.6.0 | 同左，不改 |

下面三步，就是怎么把这张表填对、用上。

## 为什么换卡要改这么多：别被"万能包"误导

第一次换卡的人，十有八九会犯同一个错——以为换个架构改个版本号就行了，结果被 `uv sync` 甩一脸 `No solution found`。问题不在版本号，而在于 AMD 的 ROCm wheel 根本不是一份"万能包"。

它是**为每种 GPU 架构单独编译一套**的。原因是底层指令集不一样：gfx1201 是 RDNA 4，gfx1151 是 RDNA 3.5，两者的 machine code、寄存器布局、Tensor 单元都不一样。AMD 把这些不同架构的 wheel 放在不同的目录里：

```text
https://repo.amd.com/rocm/whl/
├── gfx120X-all/          ← 本教程基线（RX 9070 XT / gfx1200 / gfx1201 通用合并包）
└── gfx1151/              ← AI MAX 395 等使用
```

但有一个好消息：**ROCm 7.13.0 这套版本组合，gfx1151 源上同样有**——`rocm`、`rocm-sdk-core`、`rocm-sdk-devel`、`torch==2.11.0+rocm7.13.0`、`triton==3.6.0+rocm7.13.0` 一应俱全。所以你不用纠结要不要降版本，只需要改两件事：换源、换 libraries 包名。

## 第一步：查表，确认你的架构参数

动手之前，先回答自己两个问题。这两个答案决定了后面所有改动。

打开 [AMD GPU 架构对照表](https://datawhalechina.github.io/hello-rocm/zh/00-environment/rocm-gpu-architecture-table)，找到你的显卡型号，对照下表：

| 你要确认的事 | gfx120X-all 上的答案 | gfx1151 上的答案 | 为什么重要 |
| ---- | ---- | ---- | ---- |
| **gfx 编号**（LLVM target） | gfx1201 | gfx1151 | 决定 `rocminfo` 输出该看到什么，编译器 target 该写什么 |
| **wheel 源路径** | `gfx120X-all/` | `gfx1151/` | 决定 `[[tool.uv.index]].url` 写什么 |
| **libraries 包名后缀** | `gfx120x-all` | `gfx1151` | 决定 `dependencies` 和 `[tool.uv.sources]` 里的包名 |

> ⚠️ **最容易栽跟头的一个细节**：gfx 编号里那个字母 X 的大小写。
>
> - **gfx1201**（小写）是 GPU 型号编号，出现在 `rocminfo`、编译器 target 里。
> - **gfx120X-all**（大写 X）是 AMD 的 wheel 目录命名习惯，X 代表"这一系列所有型号通用的合并包"。它出现在 wheel 源 URL 和 `rocm-sdk-libraries` 的包名后缀里。
>
> 这两套命名只在 gfx1151 这种"一个型号独占一个源"的情况下才长得一样。一旦你的卡属于一个系列（gfx1200/1201 共用 gfx120X-all），它们就分开了——抄包名时**必须和 wheel 源里实际的目录名、文件名一字不差**。

## 第二步：确认源上有这一套版本

虽然 ROCm 7.13.0 在 gfx1151 源上可用，但动手前还是值得亲自打开源看一眼——AMD 的不同源，版本支持进度偶尔会有差异，凭记忆抄版本号容易翻车。

打开浏览器，访问你这张卡对应的源：

```text
https://repo.amd.com/rocm/whl/gfx1151/
```

逐个点开下面这些目录，确认 7.13.0 那一套都在：

```text
rocm/
rocm-sdk-core/
rocm-sdk-devel/
rocm-sdk-libraries-gfx1151/   ← 注意目录名后缀，它就是包名后缀
torch/
triton/
```

下面是本教程写作时（2026-06），**gfx1151 源上 Python 3.12（cp312）** 实际存在的版本组合，供你对照：

| 组件 | gfx1151 源上可用版本 |
| ---- | ---- |
| rocm / rocm-sdk-core / rocm-sdk-devel | 7.9.0 / 7.10.0 / 7.11.0 / 7.12.0 / **7.13.0** |
| rocm-sdk-libraries-gfx1151 | 7.9.0 / 7.10.0 / 7.11.0 / 7.12.0 / **7.13.0** |
| torch | 2.7.1 / 2.8.0 / 2.9.1 / 2.10.0 / **2.11.0** |
| triton | 3.3.1 / 3.4.0 / 3.5.1 / **3.6.0** |

版本号要成套：选了 ROCm 7.13.0，torch 就得是 `torch==2.11.0+rocm7.13.0`，triton 就得是 `triton==3.6.0+rocm7.13.0`。后缀里的 `+rocmX.Y.Z` 不是装饰——它和 ROCm 主版本是一一绑定的，错了 uv 就会告诉你"找不到这个版本"。

确认 7.13.0 那套齐全之后，你就可以直接照搬教程基线的版本号，只改源和包名。

## 第三步：逐行改 pyproject.toml

前面两步准备好之后，改动本身非常少。下面这份是 gfx1151 的完整 `pyproject.toml`，我从教程基线 gfx120X-all 的配置出发，把改动的地方用 `← 改X` 标了出来——注意，没有任何一行在改版本号。

```toml
[project]
name = "part0-preface"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.4.4",
    "rocm==7.13.0",
    "rocm-sdk-core==7.13.0",
    "rocm-sdk-devel==7.13.0",
    "rocm-sdk-libraries-gfx1151==7.13.0",              # ← 改②：gfx120x-all → gfx1151
    "torch==2.11.0+rocm7.13.0",
    "triton==3.6.0+rocm7.13.0",
]

[[tool.uv.index]]
name = "rocm-amd"
url = "https://repo.amd.com/rocm/whl/gfx1151/"        # ← 改①：gfx120X-all/ → gfx1151/
explicit = true

[[tool.uv.index]]
name = "pypi-mirror"
url = "https://mirrors.bfsu.edu.cn/pypi/web/simple"
default = true

[tool.uv.sources]
rocm = { index = "rocm-amd" }
rocm-sdk-core = { index = "rocm-amd" }
rocm-sdk-devel = { index = "rocm-amd" }
rocm-sdk-libraries-gfx1151 = { index = "rocm-amd" }   # ← 改②：和 dependencies 里包名一字不差
torch = { index = "rocm-amd" }
triton = { index = "rocm-amd" }
```

只有两处要改：

### 改①：换 wheel 源 URL（gfx120X-all/ → gfx1151/）

`[[tool.uv.index]].url` 是下载地址。这条最直接——改错了会直接 404，uv 报错信息也最明确，反而是最不容易踩坑的一处。

### 改②：换 libraries 包名（gfx120x-all → gfx1151）

这是**最容易抄错的一行**。`rocm-sdk-libraries-` 后面那段后缀，必须和你目标源里的目录名完全一致：

| 目标源 | 目录名 | 包名后缀 |
| ---- | ---- | ---- |
| gfx120X-all/ | `rocm-sdk-libraries-gfx120x-all/` | `gfx120x-all` |
| gfx1151/ | `rocm-sdk-libraries-gfx1151/` | `gfx1151` |

而且 `dependencies` 和 `[tool.uv.sources]` 两处的包名必须一字不差。很多人只改了 `dependencies` 忘了改 sources，uv 就会去默认源（bfsu 镜像）找这个包，然后告诉你"找不到"——不是包不存在，是它去了错的地方。

至于版本号为什么一行都不用改：因为 AMD 发布 ROCm 7.13.0 时，gfx1151 和 gfx120X-all 两个源是同步出包的，同一份 7.13.0 源码只是分别针对两种架构各编译了一套 wheel。你把教程基线里那段版本约束整段复制过来就行。

## 改完之后：执行安装

改完 `pyproject.toml`，先把旧的 `uv.lock` 删掉再重新同步——不然 uv 可能还在用旧锁文件里记录的源和版本：

```bash
rm uv.lock
uv sync
```

<details>
<summary>输出：gfx1151 环境 uv sync</summary>

```text
Using CPython 3.12.12
Creating virtual environment at: .venv
Resolved 17 packages in 0.61ms
Installed 16 packages in 123ms
 + filelock==3.29.4
 + fsspec==2026.6.0
 + jinja2==3.1.6
 + markupsafe==3.0.3
 + mpmath==1.3.0
 + networkx==3.6.1
 + numpy==2.5.0
 + rocm==7.13.0
 + rocm-sdk-core==7.13.0
 + rocm-sdk-devel==7.13.0
 + rocm-sdk-libraries-gfx1151==7.13.0
 + setuptools==82.0.0
 + sympy==1.14.0
 + torch==2.11.0+rocm7.13.0
 + triton==3.6.0+rocm7.13.0
 + typing-extensions==4.15.0
```

</details>

看到 `Resolved N packages` 且退出码为 0，迁移的第一关就过了。剩下的激活和验证步骤，完全照 [第 1 章 环境准备](../../part0-intro/chapter1/index.md) 走就行——换卡不换流程：

```bash
source ./activate-rocm.sh
rocminfo | grep gfx1151      # 这次应该看到 gfx1151，而不是 gfx1201
```

## 报错对照表：把模糊的"不行"翻译成具体原因

换卡时如果 `uv sync` 没一次过，大概率是下面这几种情况之一。每种都列出了根因和对症办法，对照着查就行。

| 报错信息 | 根本原因 | 解决办法 |
| ---- | ---- | ---- |
| `No solution found ... rocm-sdk-core==X.Y.Z` | 用了 `rocm[devel]==X.Y.Z` 这种 extras 写法，uv 把 `rocm-sdk-core` 当成传递依赖，跑去默认源（bfsu）找 | 改用拆分式写法（本文做法），把 `rocm-sdk-core` 等组件显式列为直接依赖 |
| `rocm-sdk-libraries-xxx` 找不到 | `dependencies` 和 `[tool.uv.sources]` 两处包名不一致（漏后缀、大小写错） | 两处严格一致，以源里目录名为准 |
| torch / triton `No solution found` | 版本号或 `+rocmX.Y.Z` 后缀没和 ROCm 主版本对齐 | torch 的 `+rocm7.13.0` 必须对应 ROCm 7.13.0；triton 同理 |
| 下载 404 | wheel 源 URL 写错，或这个版本在该源上不存在 | URL 用目标架构目录；回到第二步确认版本存在 |
| `was found on mirrors.bfsu.edu.cn ... but not at requested version` | bfsu 上有同名占位包，uv 默认的 `first-index` 策略锁在 bfsu 不走 AMD 源 | 确保 AMD 源 `explicit = true`，且包名在 `[tool.uv.sources]` 里被点名映射到 `rocm-amd` |

第一条值得单独说一句。用 `rocm[devel]==X.Y.Z` 这种 extras 写法看起来更简洁，但 uv 会把 `rocm-sdk-core` 当成 `rocm` 的传递依赖，解析时不走你指定的 AMD 源，而是去默认源（bfsu）找——那里根本没有这个版本，于是报错。这就是为什么本教程所有章节的 `pyproject.toml` 都采用"拆分式"写法：把 `rocm-sdk-core`、`rocm-sdk-devel`、`rocm-sdk-libraries-*` 各自单独列为直接依赖，而不是图省事合并成 `rocm[devel,libraries]`。这不是啰嗦，是专门为了绕开这个坑。迁移到任何架构，都请保留这种写法。

## 小结

换卡这件事，说到底就是：把 wheel 源和 libraries 包名这两个绑定在架构上的东西一起改掉，版本组合跨架构通用、不用动。

```text
目标架构  ──→  wheel 源 URL        （改①）
         ──→  libraries 包名后缀   （改②）
         ──×→  版本组合            （不用改，7.13.0 一套通用）
```

只改源不改包名，或者反过来，uv 报出来的错往往指向"找不到版本"这种表层现象——你盯着版本号调半天，其实根子在源或包名上。记住版本不动、源和包名一起改，这事儿就从玄学变成了机械操作。

如果你是反过来——手上是 gfx120X-all、想迁到 gfx1151 之外的架构——流程完全对称：源换成目标架构目录、包名换成对应的目录名后缀，版本照旧 `7.13.0` 一套不动。

## 延伸阅读

- [AMD GPU 架构对照表](https://datawhalechina.github.io/hello-rocm/zh/00-environment/rocm-gpu-architecture-table)
- [AMD ROCm wheel 源（gfx1151）](https://repo.amd.com/rocm/whl/gfx1151/)
- [uv 索引配置文档](https://docs.astral.sh/uv/concepts/projects/config/#index-configuration)
- 本教程 [第 1 章 环境准备](../../part0-intro/chapter1/index.md)
