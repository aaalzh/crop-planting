# 种植推荐

**一个面向农作物种植推荐、价格/产量/成本预测和可视化展示的本地项目，包含前端页面、FastAPI 后端、训练流水线和示例数据。**

## 目录

- `前端/`：静态页面、样式、脚本和图片资源
- `后端/`：FastAPI 服务、推荐逻辑、认证与可视化接口
- `训练流水线/`：训练入口、特征工程、评估和模型融合
- `环境推荐/`：环境推荐相关数据与模型
- `价格数据/`、`产量数据/`、`成本数据/`、`数据/`：项目数据与映射文件
- `示例输入/`：最小可测输入与样例数据
- `可视化图片/`：项目文档和可视化图片资源

## 环境

- Windows 10/11 + PowerShell
- Python 3.10 或 3.11
- 浏览器：Chrome / Edge
- 前端为静态资源，通常不需要 `npm install`

安装 Python 依赖：

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn joblib pyyaml fastapi uvicorn pydantic
```

如需运行价格数据爬虫，再额外安装：

```powershell
pip install playwright
python -m playwright install chromium
```

## 启动

启动本地后端服务：

```powershell
python 后端/入口/本地服务.py --host 127.0.0.1 --port 8000
```

启动训练流程：

```powershell
python -m 训练流水线.运行训练 --config 训练流水线/配置/高精度.yaml
```

## GitHub 仓库说明

- `输出/` 为训练产物、发布包、诊断结果和模型文件，体积很大，已从版本控制中排除
- `数据/系统/` 为本地运行态用户数据，已从版本控制中排除
- 如需启用 AI 助手 / 决策支持链路，请配置 `DEEPSEEK_API_KEY`

如果你需要复现实验或重新生成发布产物，请在本地运行训练或发布流程，而不是将生成结果直接提交到 Git 仓库。
