# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


def 定位项目根目录() -> Path:
    脚本目录 = Path(__file__).resolve().parent
    候选目录 = [脚本目录, 脚本目录.parent, Path.cwd(), Path.cwd().parent]
    for 路径 in 候选目录:
        if (路径 / "输出").exists() and (路径 / "数据").exists():
            return 路径
    return 脚本目录.parent


根目录 = 定位项目根目录()
输出目录 = 根目录 / "可视化图片" / "项目图片"

环境指标文件 = 根目录 / "输出" / "环境回测.json"
价格指标文件 = 根目录 / "输出" / "价格回测.csv"
成本指标文件 = 根目录 / "输出" / "成本回测.csv"
产量指标文件 = 根目录 / "输出" / "产量回测.json"
概率历史文件 = 根目录 / "数据" / "历史" / "概率历史.csv"
环境训练数据 = 根目录 / "环境推荐" / "数据" / "作物推荐数据.csv"
产量历史文件 = 根目录 / "产量数据" / "原始" / "产量历史数据.csv"


作物中文映射: Dict[str, str] = {
    "apple": "苹果",
    "banana": "香蕉",
    "blackgram": "黑豆",
    "chickpea": "鹰嘴豆",
    "cocount": "椰子",
    "coconut": "椰子",
    "coffee": "咖啡",
    "cotton": "棉花",
    "grapes": "葡萄",
    "jute": "黄麻",
    "kidneybeans": "芸豆",
    "lentil": "扁豆",
    "maize": "玉米",
    "mango": "芒果",
    "mothbeans": "木豆",
    "mungbean": "绿豆",
    "muskmelon": "香瓜",
    "orange": "橙子",
    "papaya": "木瓜",
    "pigeonpeas": "豌豆",
    "pomegranate": "石榴",
    "rice": "水稻",
    "water melon": "西瓜",
    "watermelon": "西瓜",
    "Cotton": "棉花",
    "Gram": "鹰嘴豆",
    "Jute": "黄麻",
    "Lentil": "扁豆",
    "Maize": "玉米",
    "Masur": "红扁豆",
    "Paddy": "水稻",
}


def 设置中文绘图() -> None:
    sns.set_theme(style="whitegrid")

    可选中文字体 = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
    ]
    已安装 = {f.name for f in font_manager.fontManager.ttflist}
    已命中 = [f for f in 可选中文字体 if f in 已安装]

    if 已命中:
        plt.rcParams["font.sans-serif"] = 已命中 + ["Arial Unicode MS", "sans-serif"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "sans-serif"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def 转中文作物名(name: str) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    return 作物中文映射.get(text, text)


def 保存图片(文件名: str) -> Path:
    输出路径 = 输出目录 / 文件名
    plt.tight_layout()
    plt.savefig(输出路径, dpi=220, bbox_inches="tight")
    plt.close()
    return 输出路径


def 绘制环境模型指标() -> Path:
    数据 = json.loads(环境指标文件.read_text(encoding="utf-8"))
    指标源 = 数据.get("holdout_metrics") or 数据.get("metrics") or {}
    指标 = {
        "准确率": float(指标源.get("accuracy", 0.0)),
        "平衡准确率": float(指标源.get("balanced_accuracy", 0.0)),
        "宏平均F1": float(指标源.get("macro_f1", 0.0)),
        "Top3准确率": float(指标源.get("top3_accuracy", 0.0)),
    }

    plt.figure(figsize=(9, 6))
    颜色 = sns.color_palette("Blues", len(指标))
    柱 = plt.bar(list(指标.keys()), list(指标.values()), color=颜色)
    plt.ylim(max(0.0, min(指标.values()) - 0.05), 1.005)
    plt.title("环境推荐模型分类指标", fontsize=15)
    plt.ylabel("指标值")
    for 元素, 值 in zip(柱, 指标.values()):
        plt.text(元素.get_x() + 元素.get_width() / 2, 值 + 0.001, f"{值:.4f}", ha="center", va="bottom", fontsize=10)
    return 保存图片("环境推荐模型_分类指标柱状图.png")


def 绘制价格模型指标() -> List[Path]:
    df = pd.read_csv(价格指标文件)
    df["作物"] = df["crop"].apply(转中文作物名)
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x="mae", y="作物", hue="作物", palette="YlOrBr", dodge=False, legend=False)
    plt.title("价格模型 MAE 对比（越低越好）", fontsize=15)
    plt.xlabel("MAE")
    plt.ylabel("作物")
    均值 = df["mae"].mean()
    plt.axvline(均值, color="red", linestyle="--", linewidth=1.2, label=f"平均值: {均值:.2f}")
    plt.legend()
    图1 = 保存图片("价格模型_MAE对比图.png")

    df["mape_pct"] = df["mape"] * 100.0
    df2 = df.sort_values("mape_pct", ascending=True).reset_index(drop=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df2, x="mape_pct", y="作物", hue="作物", palette="Greens", dodge=False, legend=False)
    plt.title("价格模型 MAPE 对比（百分比，越低越好）", fontsize=15)
    plt.xlabel("MAPE (%)")
    plt.ylabel("作物")
    均值2 = df2["mape_pct"].mean()
    plt.axvline(均值2, color="red", linestyle="--", linewidth=1.2, label=f"平均值: {均值2:.2f}%")
    plt.legend()
    图2 = 保存图片("价格模型_MAPE对比图.png")
    return [图1, 图2]


def 绘制成本模型指标() -> Path:
    df = pd.read_csv(成本指标文件)
    df["作物"] = df["crop"].apply(转中文作物名)
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="mae", y="作物", hue="作物", palette="rocket", dodge=False, legend=False)
    plt.title("成本模型 MAE 对比（越低越好）", fontsize=15)
    plt.xlabel("MAE")
    plt.ylabel("作物")
    for i, 值 in enumerate(df["mae"].tolist()):
        plt.text(值, i, f" {值:.1f}", va="center", fontsize=9)
    return 保存图片("成本模型_MAE对比图.png")


def 绘制产量模型指标() -> List[Path]:
    数据 = json.loads(产量指标文件.read_text(encoding="utf-8"))
    指标 = 数据["metrics"]

    指标名 = ["MAE", "RMSE", "MAPE(%)"]
    指标值 = [指标["mae"], 指标["rmse"], 指标["mape"] * 100.0]
    plt.figure(figsize=(8, 6))
    颜色 = sns.color_palette("mako", 3)
    柱 = plt.bar(指标名, 指标值, color=颜色)
    plt.title("产量模型核心指标", fontsize=15)
    plt.ylabel("数值")
    for 元素, 值 in zip(柱, 指标值):
        plt.text(元素.get_x() + 元素.get_width() / 2, 值, f"{值:.2f}", ha="center", va="bottom")
    图1 = 保存图片("产量模型_核心指标图.png")

    成员 = [m["name"].upper() for m in 指标.get("ensemble_members", [])]
    权重 = 指标.get("ensemble_weights", [])
    if 成员 and 权重:
        plt.figure(figsize=(7, 5))
        plt.pie(权重, labels=成员, autopct="%1.1f%%", startangle=120, colors=sns.color_palette("Set2", len(成员)))
        plt.title("产量模型集成权重占比", fontsize=14)
        图2 = 保存图片("产量模型_集成权重图.png")
        return [图1, 图2]
    return [图1]


def 绘制环境数据分析() -> List[Path]:
    df = pd.read_csv(环境训练数据)
    特征列 = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    相关矩阵 = df[特征列].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(相关矩阵, annot=True, fmt=".2f", cmap="YlGnBu", square=True, cbar_kws={"shrink": 0.85})
    plt.title("环境变量相关性热力图", fontsize=15)
    图1 = 保存图片("环境变量_相关性热力图.png")

    计数 = df["label"].value_counts().head(12).reset_index()
    计数.columns = ["作物", "样本数"]
    计数["作物"] = 计数["作物"].apply(转中文作物名)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=计数, x="样本数", y="作物", hue="作物", palette="Spectral", dodge=False, legend=False)
    plt.title("环境训练集样本量 Top12", fontsize=15)
    plt.xlabel("样本数")
    plt.ylabel("作物")
    图2 = 保存图片("环境训练数据_样本量分布图.png")
    return [图1, 图2]


def 绘制时序与分布分析() -> List[Path]:
    df = pd.read_csv(概率历史文件)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["作物"] = df["crop_name"].apply(转中文作物名)

    最近截止 = df["date"].max()
    最近开始 = 最近截止 - pd.Timedelta(days=365)
    df近一年 = df[df["date"] >= 最近开始].copy()
    top作物 = df近一年["作物"].value_counts().head(6).index.tolist()
    df近一年 = df近一年[df近一年["作物"].isin(top作物)].copy()

    plt.figure(figsize=(12, 6))
    for 作物 in top作物:
        子集 = df近一年[df近一年["作物"] == 作物].sort_values("date")
        plt.plot(子集["date"], 子集["price_real"], linewidth=1.4, label=作物)
    plt.title("近一年主要作物真实价格走势", fontsize=15)
    plt.xlabel("日期")
    plt.ylabel("价格（Rs/quintal）")
    plt.legend(ncol=3, fontsize=9)
    图1 = 保存图片("数据分析_近一年价格走势图.png")

    df["真实利润"] = df["price_real"] * df["yield_real"] - df["cost_real"]
    top作物2 = df["作物"].value_counts().head(6).index.tolist()
    df利润 = df[df["作物"].isin(top作物2)].copy()

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df利润, x="真实利润", y="作物", hue="作物", palette="coolwarm", showfliers=False, legend=False)
    plt.title("主要作物真实利润分布（历史样本）", fontsize=15)
    plt.xlabel("真实利润（Rs/hectare）")
    plt.ylabel("作物")
    图2 = 保存图片("数据分析_真实利润分布图.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(df["risk_score"], bins=30, kde=True, color="#0077b6")
    plt.title("风险分数分布", fontsize=15)
    plt.xlabel("风险分数")
    plt.ylabel("样本数")
    图3 = 保存图片("数据分析_风险分数分布图.png")
    return [图1, 图2, 图3]


def 绘制模型效果散点(df: pd.DataFrame, x列: str, y列: str, 标题: str, x标签: str, y标签: str, 文件名: str) -> Path:
    数据 = df[[x列, y列]].dropna().copy()
    if len(数据) > 6000:
        数据 = 数据.sample(n=6000, random_state=42)

    x = 数据[x列].to_numpy()
    y = 数据[y列].to_numpy()
    下界 = min(np.min(x), np.min(y))
    上界 = max(np.max(x), np.max(y))
    mae = float(np.mean(np.abs(x - y)))
    corr = float(np.corrcoef(x, y)[0, 1]) if len(数据) > 1 else np.nan

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=12, alpha=0.45, color="#2a9d8f", edgecolor="none")
    plt.plot([下界, 上界], [下界, 上界], color="red", linestyle="--", linewidth=1.3, label="理想线 y=x")
    plt.title(标题, fontsize=14)
    plt.xlabel(x标签)
    plt.ylabel(y标签)
    plt.legend(loc="upper left")
    plt.text(0.03, 0.95, f"MAE: {mae:.2f}\n相关系数: {corr:.4f}", transform=plt.gca().transAxes, va="top")
    return 保存图片(文件名)


def 绘制模型效果() -> List[Path]:
    df = pd.read_csv(概率历史文件)
    df["profit_real"] = df["price_real"] * df["yield_real"] - df["cost_real"]

    图1 = 绘制模型效果散点(
        df=df,
        x列="price_real",
        y列="price_pred",
        标题="模型效果：价格预测与真实值对比",
        x标签="真实价格",
        y标签="预测价格",
        文件名="模型效果_价格预测对比图.png",
    )
    图2 = 绘制模型效果散点(
        df=df,
        x列="yield_real",
        y列="yield_pred",
        标题="模型效果：产量预测与真实值对比",
        x标签="真实产量",
        y标签="预测产量",
        文件名="模型效果_产量预测对比图.png",
    )
    图3 = 绘制模型效果散点(
        df=df,
        x列="profit_real",
        y列="profit_pred",
        标题="模型效果：利润预测与真实值对比",
        x标签="真实利润",
        y标签="预测利润",
        文件名="模型效果_利润预测对比图.png",
    )
    return [图1, 图2, 图3]


def 绘制产量历史分析() -> Path | None:
    if not 产量历史文件.exists():
        return None

    df = pd.read_csv(产量历史文件)
    df["作物"] = df["crop_name"].apply(转中文作物名)
    top作物 = df["作物"].value_counts().head(6).index.tolist()
    df = df[df["作物"].isin(top作物)].copy()

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=df.sort_values(["作物", "year"]),
        x="year",
        y="yield_quintal_per_hectare",
        hue="作物",
        marker="o",
    )
    plt.title("产量历史趋势（Top6 作物）", fontsize=15)
    plt.xlabel("年份")
    plt.ylabel("单位面积产量（quintal/hectare）")
    plt.legend(title="作物", ncol=3, fontsize=9)
    return 保存图片("数据分析_产量历史趋势图.png")


def main() -> None:
    设置中文绘图()
    输出目录.mkdir(parents=True, exist_ok=True)

    生成文件: List[Path] = []
    生成文件.append(绘制环境模型指标())
    生成文件.extend(绘制价格模型指标())
    生成文件.append(绘制成本模型指标())
    生成文件.extend(绘制产量模型指标())
    生成文件.extend(绘制环境数据分析())
    生成文件.extend(绘制时序与分布分析())
    生成文件.extend(绘制模型效果())
    产量图 = 绘制产量历史分析()
    if 产量图 is not None:
        生成文件.append(产量图)

    print(f"图片已生成：{len(生成文件)} 张")
    for 路径 in 生成文件:
        print(路径.relative_to(根目录).as_posix())


if __name__ == "__main__":
    main()
