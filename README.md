# Time Series Classification

## 环境安装

创建并激活环境

```bash
conda create -n ts_robustness python=3.10 -y
conda activate ts_robustness
```

安装核心编译器依赖（可选但推荐）

```bash
conda install -c conda-forge numpy pandas scikit-learn -y
```

通过 pip 安装 aeon 和其他工具

```bash
pip install -r requirements.txt
```