# Reproduction for Personal Knowledge Base Assistant

## **Environment Setup**

**1️⃣ Create and activate the Conda environment**

```bash
conda create -n llm-universe python==3.9.0
conda activate llm-universe
```


**2️⃣ Modify package versions in `requirements.txt`**

Update the following package versions in `requirements.txt` ([reference](https://zhuanlan.zhihu.com/p/694891334)):
```
langchain
langsmith==0.1.0
langchain-community==0.0.31
packaging
tiktoken
```


**3️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

If you encounter the following error:
```
RuntimeError: Unsupported compiler -- at least C++11 support is needed!
```
It indicates that the C++ compiler (`g++`) is either missing or outdated. You can verify this by running:
```bash
g++ --version
```

For restricted HPC environments (like Triton) where `g++` installation is not allowed, you can switch to an environment that includes `g++` by using:
```bash
module load gcc
```

Then reinstall dependencies.


---

## **Getting Started**

**1️⃣ Activate the environment and navigate to the `serve` directory**

```bash
conda activate llm-universe
cd Chat_with_Datawhale_langchain/serve
```

**2️⃣ Add your API Key to `.env`**

Make sure to add your API Key inside the `.env` file before running.


**3️⃣ Run the Gradio frontend**

```bash
python run_gradio.py
```
