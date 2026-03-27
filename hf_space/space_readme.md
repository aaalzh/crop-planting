---
title: Crop Planting
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
fullWidth: true
header: default
startup_duration_timeout: 1h
short_description: 农作物种植推荐、预测与可视化服务
datasets:
  - __ARTIFACT_REPO_ID__
---

# Crop Planting Space

这是一个基于 FastAPI + 静态前端的农作物种植推荐服务。

## 部署结构

- 当前 Space 只保存应用代码
- 训练/发布产物保存在独立的 Hugging Face 数据仓库：`__ARTIFACT_REPO_ID__`
- 容器启动时会把产物同步到本地运行目录，再按 `champion` 发布版本加载

## 运行说明

- Space 启动入口：`hf_space/start_space.py`
- Docker 运行端口：`7860`
- 运行时会自动生成独立配置文件，不会改写仓库内的原始 `后端/配置.yaml`

## 需要的 Space 变量 / Secrets

- Variable: `HF_ARTIFACTS_REPO_ID=__ARTIFACT_REPO_ID__`
- Variable: `HF_ARTIFACTS_REPO_TYPE=dataset`
- Variable: `HF_ARTIFACTS_REVISION=main`
- Variable: `CROP_ACTIVE_RELEASE_POLICY=champion`
- Secret: `HF_TOKEN`

如果需要启用 LLM 决策支持链路，请额外配置：

- Secret: `DEEPSEEK_API_KEY`
