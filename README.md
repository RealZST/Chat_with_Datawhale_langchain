# 个人知识库助手项目（复现）

## 环境配置

环境：Triton

创建及激活 Conda 环境
```shell
conda create -n llm-universe python==3.9.0
conda activate llm-universe
```

对requirements.txt中的packages版本修改如下：
```
langchain
langsmith==0.1.0
langchain-community==0.0.31
packaging
```

安装依赖项
```shell
pip install -r requirements.txt
```

若报错：
```
RuntimeError: Unsupported compiler -- at least C++11 support is needed!
```
说明没有安装 C++ 编译器（`g++`）或版本太低，可以运行下面指令验证：
```bash
g++ --version
```

对于受限的 HPC 环境（不允许安装 `g++`），可以切换到具有 `g++` 的环境：
```bash
module load gcc
```

此时再安装依赖项不会再报错。