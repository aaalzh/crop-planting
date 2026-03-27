import { cropName, fmt, fmtPct, initProtectedPage, notify, refreshInteractions } from "./推荐页公共.js";
import { fetchCropVisuals, fetchDefaultEnv, fetchRecommend } from "./推荐接口.js";
import { chartAnimationBase, clamp, hasEcharts, safeNumber, seriesMotion, toNumber } from "./推荐图表工具.js";
import { buildScoreCompareOption, buildScoreDecompOption } from "./推荐图表配置.js";
import {
  buildYearSeries,
  CONFIDENCE_MAP,
  emptyVisualPayload,
  evaluateInputHealth,
  fitLinearPoint,
  INPUT_KEYS,
  marginToRadarPercent,
  normalizeRows,
  probabilityToRadarPercent,
  profitStrengthPercent,
  riskToSafetyPercent,
  SCORE_VIEW_DESCRIPTIONS,
  scoreToLevel,
  scoreToRadarPercent,
  sampleStabilityPercent,
  sortRows,
  toDatePairs,
  topCompareRows,
  VISUAL_CHART_IDS,
  yieldStrengthPercent,
  rowMarginPct
} from "./推荐领域工具.js";
import {
  AXIS_LABELS,
  MODEL_TERM_LABELS,
  UNIT_SYSTEM_NOTE,
  formatMetricValue,
  metricExplain,
  metricLabelWithUnit,
  metricSourceNote
} from "./界面指标.js";

if (!window.Vue || typeof window.Vue.createApp !== "function") {
  throw new Error("Vue runtime missing. Please load /assets/前端/资源/依赖/视图框架生产版.js first.");
}

const { createApp } = window.Vue;

createApp({
  data() {
    return {
      loading: false,
      statusText: "",
      statusError: false,
      env: {
        N: null,
        P: null,
        K: null,
        temperature: null,
        humidity: null,
        ph: null,
        rainfall: null
      },
      rows: [],
      sort: {
        key: "recommend_strength",
        direction: "desc"
      },
      result: null,
      selectedCrop: "",
      scoreView: "summary",
      visualView: "price",
      loadingVisual: false,
      visualError: false,
      visualErrorText: "",
      visualStatusText: "请先生成推荐结果。",
      visuals: emptyVisualPayload(),
      charts: {
        priceDaily: null,
        priceForecastDetail: null,
        priceWindowSeasonal: null,
        yield: null,
        cost: null,
        profile: null,
        decisionRisk: null,
        decisionConfidence: null,
        scoreDecomp: null,
        scoreCompare: null
      },
      profileBoard: {
        overall: 0,
        grade: "-",
        gradeTone: "mid",
        sampleText: "-",
        strengthText: "-",
        cautionText: "-"
      },
      priceDailyLegendNames: [],
      priceDailyLegendSelected: {},
      priceWindowLegendNames: [],
      priceWindowLegendSelected: {},
      resizeHandler: null,
      visualRequestSeq: 0
    };
  },
  computed: {
    health() {
      return evaluateInputHealth(this.env);
    },
    sortedRows() {
      return sortRows(this.rows, this.sort);
    },
    scoreBars() {
      if (!this.rows.length) return [];
      const ordered = [...this.rows];
      const maxScore = Math.max(...ordered.map((row) => toNumber(row?.recommend_strength)), 1);
      return ordered.map((row) => ({
        ...row,
        width: clamp((toNumber(row?.recommend_strength) / maxScore) * 100, 8, 100)
      }));
    },
    topkItems() {
      const fromFinalTopk =
        Array.isArray(this.result?.final_topk) && this.result.final_topk.length
          ? this.result.final_topk.map((item) => [item.crop, item.env_prob])
          : [];
      const topk = fromFinalTopk.length ? fromFinalTopk : Array.isArray(this.result?.env?.topk) ? this.result.env.topk : [];
      return topk.slice(0, 8).map(([crop, prob], index) => ({
        rank: index + 1,
        crop,
        prob
      }));
    },
    scoreViewDescription() {
      return SCORE_VIEW_DESCRIPTIONS[this.scoreView] || SCORE_VIEW_DESCRIPTIONS.summary;
    },
    unitSystemNote() {
      return UNIT_SYSTEM_NOTE;
    },
    bestCrop() {
      return cropName(this.result?.env?.best_label);
    },
    bestRow() {
      const bestLabel = this.result?.env?.best_label;
      return this.rows.find((row) => row?.crop === bestLabel) || this.rows[0] || null;
    },
    bestProb() {
      return this.result?.env?.best_prob ?? null;
    },
    bestRisk() {
      return this.bestRow?.risk ?? null;
    },
    bestStrength() {
      return this.bestRow?.recommend_strength ?? null;
    },
    bestStrengthLevel() {
      return this.bestRow?.recommend_strength_level || "-";
    },
    runtimeMs() {
      return this.result?.runtime?.elapsed_ms ?? null;
    },
    cacheHit() {
      return Boolean(this.result?.cache?.hit);
    },
    confidenceText() {
      return CONFIDENCE_MAP[this.result?.env_confidence_norm] || this.result?.env_confidence_norm || "-";
    },
    selectedRow() {
      return this.rows.find((row) => row?.crop === this.selectedCrop) || null;
    },
    selectedProfile() {
      return this.selectedRow && typeof this.selectedRow.decision_profile === "object" ? this.selectedRow.decision_profile : {};
    },
    selectedReasons() {
      return Array.isArray(this.selectedProfile.reasons) ? this.selectedProfile.reasons : [];
    },
    selectedNextSteps() {
      return Array.isArray(this.selectedProfile.next_steps) ? this.selectedProfile.next_steps : [];
    },
    selectedRiskFocus() {
      return Array.isArray(this.selectedProfile.risk_focus) ? this.selectedProfile.risk_focus : [];
    },
    selectedComparisonMarks() {
      return Array.isArray(this.selectedProfile.comparison_marks) ? this.selectedProfile.comparison_marks : [];
    },
    decisionInsightCards() {
      if (!this.selectedRow) return [];
      const row = this.selectedRow;
      const profile = this.selectedProfile;
      const margin = safeNumber(row?.margin_pct ?? this.rowMarginPct(row));
      return [
        {
          label: "推荐强度",
          value: this.formatMetric("recommend_strength", row?.recommend_strength),
          hint: row?.recommend_strength_level ? `当前属于${row.recommend_strength_level}推荐。` : "用于面向用户展示的综合强度。"
        },
        {
          label: "环境匹配",
          value: this.formatMetric("env_prob", row?.env_prob),
          hint: profile.fit_summary || this.metricHelp("env_prob")
        },
        {
          label: "胜出概率",
          value: this.formatMetric("prob_best", row?.prob_best),
          hint: `来源：${row?.prob_best_source || "综合校准"}`
        },
        {
          label: "预计利润率",
          value: margin === null ? "-" : this.formatMetric("margin_pct", margin),
          hint: profile?.profit_band?.summary ? `收益区间：${profile.profit_band.summary}` : this.metricHelp("margin_pct")
        },
        {
          label: "行情信号",
          value: profile?.market_signal?.direction || "-",
          hint: profile?.market_signal?.summary || "暂无明确的行情方向提示。"
        }
      ];
    },
    decisionSignalCards() {
      if (!this.selectedRow) return [];
      const row = this.selectedRow;
      const riskFocus = this.selectedRiskFocus.length ? this.selectedRiskFocus.join("、") : "暂无明显焦点";
      return [
        {
          label: "当前作物",
          value: cropName(row?.crop),
          hint: this.selectedProfile?.style || "综合候选"
        },
        {
          label: "风险焦点",
          value: riskFocus,
          hint: "优先关注当前最突出的两类风险来源。"
        },
        {
          label: "样本覆盖",
          value: this.sampleCoverageText(),
          hint: `过滤后/过滤前候选，校准来源：${row?.prob_best_source || "综合"}`
        },
        {
          label: "推荐强度",
          value: this.formatMetric("recommend_strength", row?.recommend_strength),
          hint: `内部排序分：${this.formatMetric("score", row?.score)}`
        }
      ];
    },
    visualPrediction() {
      return this.visuals?.prediction || {};
    },
    visualPriceStats() {
      return this.visuals?.price?.stats || {};
    },
    visualYieldStats() {
      return this.visuals?.yield?.stats || {};
    },
    visualCostStats() {
      return this.visuals?.cost?.stats || {};
    },
    visualViewTitle() {
      const titleMap = {
        price: "价格历史、均线与预测（₹/公担）",
        yield: "产量历史趋势与预测（吨/公顷）",
        cost: "成本历史趋势与预测（₹/公顷）",
        profile: "推荐画像雷达（绝对锚点评分）"
      };
      return titleMap[this.visualView] || titleMap.price;
    }
  },
  methods: {
    cropName,
    fmt,
    fmtPct,
    scoreToLevel,
    isPriceDailySeriesHidden(name) {
      if (!name) return false;
      return this.priceDailyLegendSelected[name] === false;
    },
    togglePriceDailySeries(name) {
      if (!name || !this.charts.priceDaily) return;
      this.charts.priceDaily.dispatchAction({
        type: "legendToggleSelect",
        name
      });
    },
    applyPriceDailyLegendSelection(selected, datasets) {
      const chart = this.charts.priceDaily;
      if (!chart) return;

      const nextSelected = selected && typeof selected === "object" ? { ...selected } : {};
      const historyData = Array.isArray(datasets?.history) ? datasets.history : [];
      const ma30Data = Array.isArray(datasets?.ma30) ? datasets.ma30 : [];
      const ma90Data = Array.isArray(datasets?.ma90) ? datasets.ma90 : [];
      const bandBaseData = Array.isArray(datasets?.bandBase) ? datasets.bandBase : [];
      const bandWidthData = Array.isArray(datasets?.bandWidth) ? datasets.bandWidth : [];
      const p10Data = Array.isArray(datasets?.p10) ? datasets.p10 : [];
      const p90Data = Array.isArray(datasets?.p90) ? datasets.p90 : [];
      const forecastData = Array.isArray(datasets?.forecast) ? datasets.forecast : [];
      const isVisible = (name) => nextSelected[name] !== false;

      chart.setOption(
        {
          legend: {
            selected: nextSelected
          },
          series: [
            {
              name: "历史价格（₹/公担）",
              data: isVisible("历史价格（₹/公担）") ? historyData : []
            },
            {
              name: "30日均线",
              data: isVisible("30日均线") ? ma30Data : []
            },
            {
              name: "90日均线",
              data: isVisible("90日均线") ? ma90Data : []
            },
            {
              name: "_预测区间基线",
              data: isVisible("预测区间（p10-p90）") ? bandBaseData : []
            },
            {
              name: "预测区间（p10-p90）",
              data: isVisible("预测区间（p10-p90）") ? bandWidthData : []
            },
            {
              name: "预测下沿（p10）",
              data: isVisible("预测区间（p10-p90）") ? p10Data : []
            },
            {
              name: "预测上沿（p90）",
              data: isVisible("预测区间（p10-p90）") ? p90Data : []
            },
            {
              name: "预测价格（₹/公担）",
              data: isVisible("预测价格（₹/公担）") ? forecastData : []
            }
          ]
        },
        false
      );
      this.priceDailyLegendSelected = nextSelected;
    },
    isPriceWindowSeriesHidden(name) {
      if (!name) return false;
      return this.priceWindowLegendSelected[name] === false;
    },
    togglePriceWindowSeries(name) {
      if (!name || !this.charts.priceWindowSeasonal) return;
      this.charts.priceWindowSeasonal.dispatchAction({
        type: "legendToggleSelect",
        name
      });
      const selected =
        this.charts.priceWindowSeasonal.getOption?.()?.legend?.[0]?.selected || this.priceWindowLegendSelected;
      this.priceWindowLegendSelected = { ...selected };
    },
    modelTerm(key) {
      return MODEL_TERM_LABELS[key] || key;
    },
    metricLabel(key) {
      return metricLabelWithUnit(key);
    },
    metricHelp(key) {
      return metricExplain(key);
    },
    metricSource(key) {
      return metricSourceNote(key);
    },
    formatMetric(key, value) {
      return formatMetricValue(key, value, { fmt, fmtPct });
    },
    activeScoreChartKey() {
      if (this.scoreView === "decomp") return "scoreDecomp";
      if (this.scoreView === "compare") return "scoreCompare";
      return null;
    },
    resizeChartOnFrame(chartKey, rafDepth = 2) {
      const chart = this.charts?.[chartKey];
      if (!chart || typeof chart.resize !== "function") return;
      const frames = Math.max(0, Number(rafDepth) || 0);
      const run = (left) => {
        if (left <= 0 || typeof window.requestAnimationFrame !== "function") {
          chart.resize();
          return;
        }
        window.requestAnimationFrame(() => run(left - 1));
      };
      run(frames);
    },
    setScoreView(viewKey) {
      this.scoreView = SCORE_VIEW_DESCRIPTIONS[viewKey] ? viewKey : "summary";
      this.$nextTick(() => {
        this.renderScoreCharts();
        const activeKey = this.activeScoreChartKey();
        if (activeKey) {
          this.resizeChartOnFrame(activeKey, 2);
        }
        this.refreshUi();
      });
    },
    async setVisualView(viewKey) {
      const allowed = new Set(["price", "yield", "cost", "profile"]);
      const nextView = allowed.has(viewKey) ? viewKey : "price";
      this.visualView = nextView;
      await this.$nextTick();
      this.renderVisualCharts();
      if (nextView === "price") {
        this.resizeChartOnFrame("priceDaily", 2);
        this.resizeChartOnFrame("priceForecastDetail", 2);
        this.resizeChartOnFrame("priceWindowSeasonal", 2);
      } else {
        this.resizeChartOnFrame(nextView, 2);
      }
      const activeScoreKey = this.activeScoreChartKey();
      if (activeScoreKey) {
        this.resizeChartOnFrame(activeScoreKey, 2);
      }
      await this.refreshUi();
    },
    async refreshUi() {
      await this.$nextTick();
      refreshInteractions(document);
    },
    setStatus(text, isError = false) {
      this.statusText = String(text || "");
      this.statusError = Boolean(isError);
      if (this.statusError && this.statusText) {
        notify(this.statusText, "error", 4200);
      }
    },
    setVisualStatus(text, isError = false) {
      this.visualStatusText = String(text || "");
      this.visualError = Boolean(isError);
      this.visualErrorText = isError ? this.visualStatusText : "";
    },
    assignEnv(env) {
      INPUT_KEYS.forEach((key) => {
        this.env[key] = Number(env?.[key]);
      });
    },
    buildEnvPayload() {
      return INPUT_KEYS.reduce((payload, key) => {
        payload[key] = Number(this.env[key]);
        return payload;
      }, {});
    },
    sortIndicator(key) {
      if (this.sort.key !== key) return "↕";
      return this.sort.direction === "asc" ? "▲" : "▼";
    },
    async setSort(key) {
      if (this.sort.key === key) {
        this.sort.direction = this.sort.direction === "asc" ? "desc" : "asc";
      } else {
        this.sort.key = key;
        this.sort.direction = key === "crop" ? "asc" : "desc";
      }
      await this.refreshUi();
    },
    async fillDefault(showSuccess = true) {
      try {
        const payload = await fetchDefaultEnv();
        this.assignEnv(payload || {});
        if (showSuccess) {
          this.setStatus("已填充默认环境参数。", false);
        }
      } catch (err) {
        this.setStatus(`读取默认参数失败：${err.message}`, true);
      } finally {
        await this.refreshUi();
      }
    },
    initVisualCharts() {
      if (!hasEcharts()) {
        this.setVisualStatus("缺少 ECharts，无法渲染可视化图表。", true);
        return false;
      }

      Object.entries(VISUAL_CHART_IDS).forEach(([key, id]) => {
        const node = document.getElementById(id);
        if (!node) return;
        const isVisible = node.offsetParent !== null || node.getClientRects().length > 0;
        if (!this.charts[key] && isVisible) {
          this.charts[key] = window.echarts.init(node);
        }
      });

      if (!this.resizeHandler) {
        this.resizeHandler = () => {
          Object.values(this.charts).forEach((chart) => chart?.resize());
        };
        window.addEventListener("resize", this.resizeHandler, { passive: true });
      }
      return true;
    },
    disposeVisualCharts() {
      Object.keys(this.charts).forEach((key) => {
        this.charts[key]?.dispose?.();
        this.charts[key] = null;
      });
      if (this.resizeHandler) {
        window.removeEventListener("resize", this.resizeHandler);
        this.resizeHandler = null;
      }
    },
    tooltipDataPair(item, dataIndex, categories) {
      const raw =
        item && typeof item === "object" && !Array.isArray(item) && Object.prototype.hasOwnProperty.call(item, "value")
          ? item.value
          : item;
      if (raw === null || raw === undefined) return null;
      if (Array.isArray(raw)) {
        if (raw.length >= 2) {
          const xVal = raw[0];
          const yVal = Number(raw[1]);
          if (xVal === null || xVal === undefined || !Number.isFinite(yVal)) return null;
          return [xVal, yVal];
        }
        if (raw.length === 1 && Array.isArray(categories) && categories[dataIndex] !== undefined) {
          const yVal = Number(raw[0]);
          if (!Number.isFinite(yVal)) return null;
          return [categories[dataIndex], yVal];
        }
        return null;
      }
      if (!Array.isArray(categories) || categories[dataIndex] === undefined) return null;
      const yVal = Number(raw);
      if (!Number.isFinite(yVal)) return null;
      return [categories[dataIndex], yVal];
    },
    pointToSegmentDistance(point, start, end) {
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) {
        return { distance: Math.hypot(point[0] - start[0], point[1] - start[1]), t: 0 };
      }
      const t =
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) /
        (dx * dx + dy * dy);
      const clamped = clamp(t, 0, 1);
      const px = start[0] + dx * clamped;
      const py = start[1] + dy * clamped;
      return { distance: Math.hypot(point[0] - px, point[1] - py), t: clamped };
    },
    extractEcData(target) {
      let current = target || null;
      while (current) {
        const key = Object.keys(current).find((name) => {
          const payload = current[name];
          return (
            name.startsWith("__ec_inner_") &&
            payload &&
            typeof payload === "object" &&
            (payload.seriesIndex !== undefined || payload.dataIndex !== undefined)
          );
        });
        if (key) {
          return {
            target: current,
            data: current[key]
          };
        }
        current = current.parent || null;
      }
      return null;
    },
    searchHoverTarget(chart, point, radius = 0) {
      const zr = chart?.getZr?.();
      const handler = zr?.handler;
      if (!handler || !Array.isArray(point)) return null;
      const rawRadius = Math.max(0, Math.floor(Number(radius) || 0));
      const offsets = [[0, 0]];
      if (rawRadius > 0) {
        const half = Math.max(1, Math.floor(rawRadius / 2));
        [
          [-half, 0],
          [half, 0],
          [0, -half],
          [0, half],
          [-rawRadius, 0],
          [rawRadius, 0],
          [0, -rawRadius],
          [0, rawRadius],
          [-half, -half],
          [half, -half],
          [-half, half],
          [half, half]
        ].forEach((item) => offsets.push(item));
      }

      let best = null;
      offsets.forEach(([dx, dy]) => {
        const hover = handler.findHover(point[0] + dx, point[1] + dy);
        const matched = this.extractEcData(hover?.target);
        if (!matched) return;
        const distance = Math.hypot(dx, dy);
        if (!best || distance < best.distance) {
          best = {
            distance,
            point: [point[0] + dx, point[1] + dy],
            target: matched.target,
            ecData: matched.data
          };
        }
      });
      return best;
    },
    ensureFloatingChartTooltip() {
      let node = document.getElementById("recommendFloatingChartTooltip");
      if (node) return node;
      node = document.createElement("div");
      node.id = "recommendFloatingChartTooltip";
      Object.assign(node.style, {
        position: "fixed",
        zIndex: "2200",
        pointerEvents: "none",
        display: "none",
        maxWidth: "280px",
        padding: "8px 10px",
        borderRadius: "10px",
        border: "1px solid rgba(197, 214, 205, 0.85)",
        background: "rgba(255, 255, 255, 0.98)",
        color: "#294236",
        fontSize: "12px",
        lineHeight: "1.45",
        boxShadow: "0 14px 32px rgba(26, 48, 37, 0.18)"
      });
      document.body.appendChild(node);
      return node;
    },
    hideFloatingChartTooltip() {
      const node = document.getElementById("recommendFloatingChartTooltip");
      if (!node) return;
      node.style.display = "none";
      node.innerHTML = "";
    },
    showFloatingChartTooltip(chart, point, html) {
      if (!html) {
        this.hideFloatingChartTooltip();
        return;
      }
      const node = this.ensureFloatingChartTooltip();
      const rect = chart.getDom().getBoundingClientRect();
      node.innerHTML = html;
      node.style.display = "block";
      node.style.left = `${Math.round(rect.left + point[0] + 14)}px`;
      node.style.top = `${Math.round(rect.top + point[1] + 14)}px`;
    },
    resolveChartPoint(chart, evt) {
      const rect = chart?.getDom?.().getBoundingClientRect?.();
      const raw = evt?.event || evt;
      const xCandidates = [evt?.offsetX, evt?.zrX, raw?.offsetX, raw?.clientX != null && rect ? raw.clientX - rect.left : null];
      const yCandidates = [evt?.offsetY, evt?.zrY, raw?.offsetY, raw?.clientY != null && rect ? raw.clientY - rect.top : null];
      const x = xCandidates.find((value) => Number.isFinite(Number(value)));
      const y = yCandidates.find((value) => Number.isFinite(Number(value)));
      if (!Number.isFinite(Number(x)) || !Number.isFinite(Number(y))) return null;
      return [Number(x), Number(y)];
    },
    bindSeriesItemTooltip(chartKey, formatter) {
      const chart = this.charts?.[chartKey];
      if (!chart || typeof chart.on !== "function") return;
      chart.__itemTooltipFormatter = formatter;
      if (chart.__itemTooltipBound) return;

      const hideTip = () => {
        this.hideFloatingChartTooltip();
      };
      const moveTip = (params) => {
        if (params?.componentType !== "series") {
          hideTip();
          return;
        }
        const point = this.resolveChartPoint(chart, params?.event);
        if (!point) {
          hideTip();
          return;
        }
        const html = typeof chart.__itemTooltipFormatter === "function" ? chart.__itemTooltipFormatter(params) : "";
        if (!html) {
          hideTip();
          return;
        }
        this.showFloatingChartTooltip(chart, point, html);
      };

      chart.on("mouseover", moveTip);
      chart.on("mousemove", moveTip);
      chart.on("mouseout", hideTip);
      chart.getZr?.().on?.("globalout", hideTip);
      chart.__itemTooltipBound = true;
    },
    bindPreciseBarTooltip(chartKey, formatter, tolerance = 8) {
      const chart = this.charts?.[chartKey];
      if (!chart || typeof chart.getDom !== "function") return;
      chart.__preciseBarTooltipFormatter = formatter;
      chart.__preciseBarTooltipTolerance = tolerance;
      if (chart.__preciseBarTooltipBound) return;

      const hideTip = () => {
        this.hideFloatingChartTooltip();
      };
      const moveTip = (params) => {
        if (params?.componentType && params.componentType !== "series") {
          hideTip();
          return;
        }
        const point = this.resolveChartPoint(chart, params?.event || params);
        if (!point) {
          hideTip();
          return;
        }
        const option = chart.getOption?.();
        if (!option) {
          hideTip();
          return;
        }
        const hit = this.searchHoverTarget(chart, point, chart.__preciseBarTooltipTolerance);
        const seriesIndex = hit?.ecData?.seriesIndex;
        const dataIndex = hit?.ecData?.dataIndex;
        if (!Number.isInteger(seriesIndex) || !Number.isInteger(dataIndex)) {
          hideTip();
          return;
        }
        const xAxis = Array.isArray(option.xAxis) ? option.xAxis[0] : option.xAxis;
        const yAxis = Array.isArray(option.yAxis) ? option.yAxis[0] : option.yAxis;
        const xCategories = Array.isArray(xAxis?.data) ? xAxis.data : null;
        const yCategories = Array.isArray(yAxis?.data) ? yAxis.data : null;
        const isVertical = xAxis?.type === "category" && yAxis?.type === "value";
        const isHorizontal = xAxis?.type === "value" && yAxis?.type === "category";
        const seriesList = Array.isArray(option.series) ? option.series : [];
        const series = seriesList[seriesIndex];
        if (series?.type !== "bar") {
          hideTip();
          return;
        }
        const rawItem = Array.isArray(series?.data) ? series.data[dataIndex] : null;
        const rawValue =
          rawItem && typeof rawItem === "object" && !Array.isArray(rawItem) && Object.prototype.hasOwnProperty.call(rawItem, "value")
            ? rawItem.value
            : rawItem;
        const numericValue = Array.isArray(rawValue)
          ? Number(rawValue[1] ?? rawValue[0])
          : Number(rawValue);
        if (!Number.isFinite(numericValue)) {
          hideTip();
          return;
        }
        const xLabel = isVertical
          ? String(xCategories?.[dataIndex] ?? "-")
          : isHorizontal
          ? String(yCategories?.[dataIndex] ?? "-")
          : "-";
        const html =
          typeof chart.__preciseBarTooltipFormatter === "function"
            ? chart.__preciseBarTooltipFormatter({
                seriesIndex,
                dataIndex,
                xLabel,
                yValue: numericValue,
                seriesName: String(series?.name || "")
              })
            : "";
        if (!html) {
          hideTip();
          return;
        }
        this.showFloatingChartTooltip(chart, hit.point, html);
      };

      chart.getZr?.().on?.("mousemove", moveTip);
      chart.getZr?.().on?.("globalout", hideTip);
      chart.getDom().addEventListener("mousemove", moveTip);
      chart.getDom().addEventListener("mouseleave", hideTip);
      chart.__preciseBarTooltipBound = true;
    },
    bindPreciseLineTooltip(chartKey, formatterOrConfig, tolerance = 10) {
      const chart = this.charts?.[chartKey];
      if (!chart || typeof chart.getZr !== "function") return;
      chart.__preciseTooltipTolerance = tolerance;
      chart.__preciseTooltipConfig =
        typeof formatterOrConfig === "function" ? { formatter: formatterOrConfig } : formatterOrConfig || {};
      if (chart.__preciseTooltipBound) return;

      const hideTip = () => {
        chart.__preciseTooltipSig = "";
        this.hideFloatingChartTooltip();
      };

      const moveTip = (params) => {
        if (params?.componentType !== "series") {
          hideTip();
          return;
        }
        const point = this.resolveChartPoint(chart, params?.event || params);
        if (!point) {
          hideTip();
          return;
        }
        const option = chart.getOption?.();
        if (!option) {
          hideTip();
          return;
        }
        const xAxis = Array.isArray(option.xAxis) ? option.xAxis[0] : option.xAxis;
        const categories = Array.isArray(xAxis?.data) ? xAxis.data : null;
        const seriesList = Array.isArray(option.series) ? option.series : [];
        const tooltipConfig = chart.__preciseTooltipConfig || {};
        const includeSeries =
          typeof tooltipConfig.includeSeries === "function"
            ? tooltipConfig.includeSeries
            : (series) => !String(series?.name || "").startsWith("_") && series?.tooltip?.show !== false;
        const selected = {};
        const legends = Array.isArray(option.legend) ? option.legend : option.legend ? [option.legend] : [];
        legends.forEach((legend) => Object.assign(selected, legend?.selected || {}));
        const directMatched = this.extractEcData(params?.event?.target || params?.event?.topTarget || null);
        const hit = directMatched
          ? {
              distance: 0,
              point,
              target: directMatched.target,
              ecData: directMatched.data
            }
          : this.searchHoverTarget(chart, point, chart.__preciseTooltipTolerance);
        const seriesIndex = Number.isInteger(params?.seriesIndex) ? params.seriesIndex : hit?.ecData?.seriesIndex;
        if (!Number.isInteger(seriesIndex)) {
          hideTip();
          return;
        }
        const series = seriesList[seriesIndex];
        if (series?.type !== "line" || !includeSeries(series, seriesIndex) || selected[series?.name] === false) {
          hideTip();
          return;
        }
        let dataIndex = Number.isInteger(params?.dataIndex)
          ? params.dataIndex
          : Number.isInteger(hit?.ecData?.dataIndex)
          ? hit.ecData.dataIndex
          : null;
        if (!Number.isInteger(dataIndex)) {
          const shapePoints = hit?.target?.shape?.points;
          const data = Array.isArray(series?.data) ? series.data : [];
          if (!shapePoints || typeof shapePoints.length !== "number" || !data.length) {
            hideTip();
            return;
          }
          let bestPoint = null;
          let prev = null;
          for (let idx = 0; idx + 1 < shapePoints.length; idx += 2) {
            const pointIndex = idx / 2;
            const px = Number(shapePoints[idx]);
            const py = Number(shapePoints[idx + 1]);
            if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
            const currentPoint = [px, py];
            const pointDistance = Math.hypot(point[0] - px, point[1] - py);
            if (!bestPoint || pointDistance < bestPoint.distance) {
              bestPoint = { distance: pointDistance, dataIndex: pointIndex };
            }
            if (prev) {
              const segment = this.pointToSegmentDistance(point, prev.point, currentPoint);
              if ((!bestPoint || segment.distance < bestPoint.distance) && segment.distance <= chart.__preciseTooltipTolerance) {
                bestPoint = {
                  distance: segment.distance,
                  dataIndex: segment.t <= 0.5 ? prev.dataIndex : pointIndex
                };
              }
            }
            prev = { point: currentPoint, dataIndex: pointIndex };
          }
          if (!bestPoint || bestPoint.distance > chart.__preciseTooltipTolerance) {
            hideTip();
            return;
          }
          dataIndex = Math.min(data.length - 1, Math.max(0, Math.round(bestPoint.dataIndex)));
        }
        const pair = this.tooltipDataPair(series?.data?.[dataIndex], dataIndex, categories);
        if (!series || !pair) {
          hideTip();
          return;
        }

        const html =
          typeof tooltipConfig.formatter === "function"
            ? tooltipConfig.formatter({
                chartKey,
                seriesName: String(series?.name || ""),
                xLabel: String(pair[0] ?? "-"),
                yValue: Number(pair[1]),
                dataIndex,
                option,
                series
              })
            : "";
        if (!html) {
          hideTip();
          return;
        }

        const sig = `${seriesIndex}:${dataIndex}`;
        chart.__preciseTooltipSig = sig;
        this.showFloatingChartTooltip(chart, point, html);
      };

      chart.on("mousemove", moveTip);
      chart.on("mouseout", hideTip);
      chart.getZr().on("globalout", hideTip);
      chart.getDom().addEventListener("mouseleave", hideTip);
      chart.__preciseTooltipBound = true;
    },
    toDatePairs,
    toDatePairsByKey(rows, key) {
      if (!Array.isArray(rows)) return [];
      return rows
        .map((item) => {
          const date = String(item?.date || "");
          const value = safeNumber(item?.[key]);
          if (!date || value === null) return null;
          return [date, value];
        })
        .filter(Boolean);
    },
    buildIntervalBand(rows) {
      const base = [];
      const width = [];
      if (!Array.isArray(rows)) return { base, width };
      rows.forEach((item) => {
        const date = String(item?.date || "");
        const low = safeNumber(item?.p10);
        const high = safeNumber(item?.p90);
        if (!date || low === null || high === null || high < low) return;
        base.push([date, low]);
        width.push([date, high - low]);
      });
      return { base, width };
    },
    buildYearSeries,
    fitLinearPoint,
    probabilityToRadarPercent,
    riskToSafetyPercent,
    marginToRadarPercent,
    scoreToRadarPercent,
    sampleStabilityPercent() {
      return sampleStabilityPercent(this.rows.length, this.result?.runtime?.filters?.before);
    },
    sampleCoverageText() {
      const before = Math.max(this.rows.length, toNumber(this.result?.runtime?.filters?.before));
      return `${this.rows.length}/${before}`;
    },
    calibrationStabilityPercent(probBestVal) {
      const support = this.sampleStabilityPercent();
      const probBest = safeNumber(probBestVal);
      if (probBest === null) return support;
      const probScore = this.probabilityToRadarPercent(probBest);
      // Blend calibrated probability with sample support to avoid over-penalizing sparse multi-class outputs.
      const blended = support * 0.82 + probScore * 0.18;
      const adaptiveFloor = clamp(12 + support * 0.4, 12, 55);
      return clamp(Math.max(adaptiveFloor, blended), 3, 97);
    },
    rowMarginPct(row) {
      return rowMarginPct(row);
    },
    profitStrengthPercent(profit, refAbs) {
      return profitStrengthPercent(profit, refAbs);
    },
    yieldStrengthPercent(yieldValue, refMax) {
      return yieldStrengthPercent(yieldValue, refMax);
    },
    riskBreakdownColor(score) {
      const value = toNumber(score);
      if (value >= 65) return "#b85c3b";
      if (value >= 40) return "#d39a2f";
      return "#2f7b54";
    },
    certaintyColor(score) {
      const value = toNumber(score);
      if (value >= 75) return "#2f7b54";
      if (value >= 55) return "#76a06d";
      if (value >= 40) return "#c59534";
      return "#b65b3b";
    },
    buildDecisionConfidenceItems(row) {
      const profile = this.selectedProfile;
      const margin = safeNumber(row?.margin_pct ?? this.rowMarginPct(row));
      const probBest = safeNumber(row?.prob_best);
      return [
        {
          name: "环境把握",
          value: safeNumber(row?.env_prob) === null ? 0 : this.probabilityToRadarPercent(row?.env_prob),
          raw: this.formatMetric("env_prob", row?.env_prob),
          hint: profile.fit_summary || this.metricHelp("env_prob")
        },
        {
          name: "胜出把握",
          value: this.calibrationStabilityPercent(probBest),
          raw:
            probBest === null
              ? `样本覆盖 ${this.sampleCoverageText()}`
              : `${this.formatMetric("prob_best", probBest)} · ${row?.prob_best_source || "综合校准"}`,
          hint: "把校准概率和样本覆盖一起看，避免只盯单点概率。"
        },
        {
          name: "风险安全度",
          value: this.riskToSafetyPercent(row?.risk),
          raw: this.formatMetric("risk", row?.risk),
          hint: this.metricHelp("risk")
        },
        {
          name: "利润缓冲",
          value: margin === null ? 0 : this.marginToRadarPercent(margin),
          raw: margin === null ? "-" : this.formatMetric("margin_pct", margin),
          hint: profile?.profit_band?.summary ? `收益区间：${profile.profit_band.summary}` : this.metricHelp("margin_pct")
        },
        {
          name: "样本稳定度",
          value: this.sampleStabilityPercent(),
          raw: `候选覆盖 ${this.sampleCoverageText()}`,
          hint: "保留候选越完整，说明当前输入越接近历史经验分布。"
        }
      ];
    },
    topCompareRows(limit = 6) {
      return topCompareRows(this.rows, limit);
    },
    buildScoreDecompItems(row, topRows) {
      const profitAbsMax = Math.max(...topRows.map((item) => Math.abs(toNumber(item?.profit))), 1);
      const yieldMax = Math.max(...topRows.map((item) => toNumber(item?.yield)), 1);
      const envProb = safeNumber(row?.env_prob);
      const probBest = safeNumber(row?.prob_best);
      const margin = this.rowMarginPct(row);
      const risk = safeNumber(row?.risk);
      const score = safeNumber(row?.score);
      const sampleSupportText = this.sampleCoverageText();
      const calibrationScore = this.calibrationStabilityPercent(probBest);

      return [
        {
          name: "环境适配",
          value: envProb === null ? 0 : this.probabilityToRadarPercent(envProb),
          raw: envProb === null ? "-" : this.formatMetric("env_prob", envProb),
          explain: this.metricHelp("env_prob")
        },
        {
          name: "校准稳定性",
          value: calibrationScore,
          raw:
            probBest === null
              ? `样本覆盖 ${sampleSupportText}`
              : `${this.formatMetric("prob_best", probBest)} · 样本覆盖 ${sampleSupportText}`,
          explain: `${this.metricHelp("prob_best")}（已融合样本覆盖稳定度）`
        },
        {
          name: "收益强度",
          value: this.profitStrengthPercent(row?.profit, profitAbsMax),
          raw: this.formatMetric("profit", row?.profit),
          explain: this.metricHelp("profit")
        },
        {
          name: "风险安全度",
          value: this.riskToSafetyPercent(risk),
          raw: this.formatMetric("risk", risk),
          explain: this.metricHelp("risk")
        },
        {
          name: "产量强度",
          value: this.yieldStrengthPercent(row?.yield, yieldMax),
          raw: this.formatMetric("yield", row?.yield),
          explain: this.metricHelp("yield")
        },
        {
          name: "综合评分强度",
          value: this.scoreToRadarPercent(score),
          raw: this.formatMetric("score", score),
          explain: "内部排序信号，用于候选排序，不直接代表面向用户的推荐强度。"
        },
        {
          name: "利润率健康度",
          value: margin === null ? 0 : this.marginToRadarPercent(margin),
          raw: margin === null ? "-" : this.formatMetric("margin_pct", margin),
          explain: this.metricHelp("margin_pct")
        }
      ];
    },
    renderDecisionSupportCharts() {
      if (!this.initVisualCharts()) return;

      const row = this.selectedRow;
      if (!row) {
        this.charts.decisionRisk?.clear?.();
        this.charts.decisionConfidence?.clear?.();
        return;
      }

      const profile = this.selectedProfile;
      const breakdown = Array.isArray(profile.risk_breakdown) ? profile.risk_breakdown : [];
      const riskChart = this.charts.decisionRisk;
      riskChart?.clear?.();
      if (riskChart && breakdown.length) {
        riskChart.setOption(
          {
            ...chartAnimationBase(0),
            grid: {
              left: 118,
              right: 44,
              top: 16,
              bottom: 8
            },
            xAxis: {
              type: "value",
              min: 0,
              max: 100,
              splitNumber: 4,
              axisLabel: {
                color: "#6b8275",
                formatter: "{value}"
              },
              splitLine: {
                lineStyle: {
                  color: "rgba(205, 220, 212, 0.75)"
                }
              }
            },
            yAxis: {
              type: "category",
              inverse: true,
              data: breakdown.map((item) => item.label),
              axisTick: { show: false },
              axisLine: { show: false },
              axisLabel: {
                color: "#355648",
                interval: 0,
                width: 92,
                overflow: "truncate"
              }
            },
            tooltip: { show: false },
            series: [
              {
                type: "bar",
                barWidth: 16,
                showBackground: true,
                backgroundStyle: {
                  color: "rgba(229, 237, 232, 0.9)",
                  borderRadius: 999
                },
                itemStyle: {
                  borderRadius: [0, 999, 999, 0]
                },
                label: {
                  show: true,
                  position: "right",
                  color: "#355648",
                  formatter: ({ value, data }) => `${fmt(value, 1)} · ${data.level || "-"}`
                },
                data: breakdown.map((item) => ({
                  value: toNumber(item?.score),
                  level: item?.level || "-",
                  detail: item?.detail || "",
                  itemStyle: {
                    color: this.riskBreakdownColor(item?.score)
                  }
                }))
              }
            ]
          },
          true
        );
        this.bindPreciseBarTooltip("decisionRisk", (params) => {
          const item = breakdown[params?.dataIndex ?? -1];
          if (!item) return "";
          return [
            `<strong>${item.label}</strong>`,
            `风险评分：${fmt(item.score, 1)} / 100`,
            `等级：${item.level || "-"}`,
            item.detail || "暂无补充说明"
          ].join("<br/>");
        });
        this.resizeChartOnFrame("decisionRisk", 2);
      }

      const confidenceItems = this.buildDecisionConfidenceItems(row);
      const confidenceChart = this.charts.decisionConfidence;
      confidenceChart?.clear?.();
      if (confidenceChart && confidenceItems.length) {
        confidenceChart.setOption(
          {
            ...chartAnimationBase(60),
            grid: {
              left: 106,
              right: 34,
              top: 16,
              bottom: 8
            },
            xAxis: {
              type: "value",
              min: 0,
              max: 100,
              splitNumber: 4,
              axisLabel: {
                color: "#6b8275",
                formatter: "{value}"
              },
              splitLine: {
                lineStyle: {
                  color: "rgba(205, 220, 212, 0.75)"
                }
              }
            },
            yAxis: {
              type: "category",
              inverse: true,
              data: confidenceItems.map((item) => item.name),
              axisTick: { show: false },
              axisLine: { show: false },
              axisLabel: {
                color: "#355648",
                interval: 0,
                width: 84,
                overflow: "truncate"
              }
            },
            tooltip: { show: false },
            series: [
              {
                type: "bar",
                barWidth: 16,
                showBackground: true,
                backgroundStyle: {
                  color: "rgba(229, 237, 232, 0.9)",
                  borderRadius: 999
                },
                itemStyle: {
                  borderRadius: [0, 999, 999, 0]
                },
                label: {
                  show: true,
                  position: "right",
                  color: "#355648",
                  formatter: ({ value }) => `${fmt(value, 1)} / 100`
                },
                data: confidenceItems.map((item) => ({
                  value: toNumber(item.value),
                  raw: item.raw,
                  hint: item.hint,
                  itemStyle: {
                    color: this.certaintyColor(item.value)
                  }
                }))
              }
            ]
          },
          true
        );
        this.bindPreciseBarTooltip("decisionConfidence", (params) => {
          const item = confidenceItems[params?.dataIndex ?? -1];
          if (!item) return "";
          return [
            `<strong>${item.name}</strong>`,
            `标准化得分：${fmt(item.value, 1)} / 100`,
            `原值：${item.raw}`,
            item.hint || "暂无补充说明"
          ].join("<br/>");
        });
        this.resizeChartOnFrame("decisionConfidence", 2);
      }
    },
    renderScoreCharts() {
      if (!this.initVisualCharts()) return;
      const activeKey = this.activeScoreChartKey();
      if (!activeKey) return;

      const topRows = this.topCompareRows(6);
      if (!topRows.length) {
        this.charts[activeKey]?.clear?.();
        return;
      }

      if (activeKey === "scoreDecomp") {
        const focusRow = this.selectedRow || topRows[0];
        const decompItems = this.buildScoreDecompItems(focusRow, topRows);
        const chart = this.charts.scoreDecomp;
        chart?.clear?.();
        chart?.setOption(buildScoreDecompOption(decompItems), true);
        this.bindPreciseBarTooltip("scoreDecomp", (params) => {
          const item = decompItems[params?.dataIndex ?? -1];
          if (!item) return "";
          return [`<strong>${item.name}</strong>`, `标准化得分：${fmt(item.value, 1)} / 100`, `原值：${item.raw}`, item.explain].join(
            "<br/>"
          );
        });
        this.resizeChartOnFrame("scoreDecomp", 2);
        return;
      }

      const profitAbsMax = Math.max(...topRows.map((item) => Math.abs(toNumber(item?.profit))), 1);
      const compareRows = topRows.map((row) => ({
        row,
        crop: cropName(row?.crop),
        scoreStrength: safeNumber(row?.recommend_strength) ?? this.scoreToRadarPercent(row?.score),
        profitStrength: this.profitStrengthPercent(row?.profit, profitAbsMax),
        riskSafety: this.riskToSafetyPercent(row?.risk)
      }));

      const chart = this.charts.scoreCompare;
      chart?.clear?.();
      chart?.setOption(buildScoreCompareOption(compareRows, (key, value) => this.formatMetric(key, value)), true);
      this.bindPreciseBarTooltip("scoreCompare", (params) => {
        const pick = compareRows[params?.dataIndex ?? -1];
        if (!pick) return "";
        return [
          `<strong>${pick.crop}</strong>`,
          `推荐强度：${fmt(pick.scoreStrength, 1)} / 100（内部排序分 ${this.formatMetric("score", pick.row?.score)}）`,
          `收益强度：${fmt(pick.profitStrength, 1)} / 100（原值 ${this.formatMetric("profit", pick.row?.profit)}）`,
          `风险安全度：${fmt(pick.riskSafety, 1)} / 100（原值 ${this.formatMetric("risk", pick.row?.risk)}）`
        ].join("<br/>");
      });
      this.resizeChartOnFrame("scoreCompare", 2);
    },
    renderVisualCharts() {
      if (!this.initVisualCharts()) return;

      const priceHistory = this.toDatePairs(this.visuals?.price?.history);
      const priceMa30 = this.toDatePairs(this.visuals?.price?.ma30);
      const priceMa90 = this.toDatePairs(this.visuals?.price?.ma90);
      const priceForecastRows = Array.isArray(this.visuals?.price?.forecast) ? this.visuals.price.forecast : [];
      const priceForecastRaw = this.toDatePairs(priceForecastRows);
      const priceForecastP10 = this.toDatePairsByKey(priceForecastRows, "p10");
      const priceForecastP90 = this.toDatePairsByKey(priceForecastRows, "p90");
      const priceForecastBand = this.buildIntervalBand(priceForecastRows);
      const priceForecast =
        priceForecastRaw.length === 1 && priceHistory.length
          ? (() => {
              const lastHistory = priceHistory[priceHistory.length - 1];
              const onlyForecast = priceForecastRaw[0];
              if (!lastHistory || !onlyForecast) return priceForecastRaw;
              if (String(lastHistory[0]) === String(onlyForecast[0])) return priceForecastRaw;
              return [lastHistory, onlyForecast];
            })()
          : priceForecastRaw;
      const priceRawHistory = this.toDatePairs(this.visuals?.price?.raw?.history);
      const priceRawMa30 = this.toDatePairs(this.visuals?.price?.raw?.ma30);
      const priceRawMa90 = this.toDatePairs(this.visuals?.price?.raw?.ma90);
      const priceRawForecastBase = this.toDatePairs(this.visuals?.price?.raw?.forecast);
      const priceRawForecast =
        priceRawForecastBase.length && priceRawHistory.length
          ? (() => {
              const lastHistory = priceRawHistory[priceRawHistory.length - 1];
              const firstForecast = priceRawForecastBase[0];
              if (!lastHistory || !firstForecast) return priceRawForecastBase;
              if (String(lastHistory[0]) === String(firstForecast[0])) return priceRawForecastBase;
              return [lastHistory, ...priceRawForecastBase];
            })()
          : priceRawForecastBase;
      const connectFromLastHistory = (historyLine, targetLine) => {
        if (!Array.isArray(historyLine) || !Array.isArray(targetLine) || !targetLine.some((x) => x !== null)) {
          return targetLine;
        }
        const firstTargetIdx = targetLine.findIndex((x) => x !== null);
        if (firstTargetIdx <= 0) return targetLine;

        let prevHistoryIdx = -1;
        for (let idx = firstTargetIdx - 1; idx >= 0; idx -= 1) {
          if (Number.isFinite(Number(historyLine[idx]))) {
            prevHistoryIdx = idx;
            break;
          }
        }
        if (prevHistoryIdx < 0) return targetLine;

        const connected = [...targetLine];
        if (connected[prevHistoryIdx] === null || connected[prevHistoryIdx] === undefined) {
          connected[prevHistoryIdx] = historyLine[prevHistoryIdx];
        }
        return connected;
      };

      this.charts.price?.setOption(
        {
          ...chartAnimationBase(0),
          tooltip: {
            trigger: "item",
            show: true,
            triggerOn: "mousemove|click",
            appendToBody: true,
            confine: false,
            formatter: (item) => {
              const row = Array.isArray(item) ? item[0] : item;
              const seriesName = String(row?.seriesName || "");
              if (!seriesName || seriesName.startsWith("_")) return "";
              const valueRaw = Array.isArray(row?.data) ? row.data?.[1] : Array.isArray(row?.value) ? row.value?.[1] : row?.value;
              const value = Number(valueRaw);
              if (!Number.isFinite(value)) return "";
              const dateLabel =
                row?.axisValueLabel ||
                (Array.isArray(row?.data) ? row.data?.[0] : Array.isArray(row?.value) ? row.value?.[0] : "-") ||
                "-";
              return [
                `日期：${dateLabel}`,
                `${row?.marker || ""}${seriesName}：${fmt(value, 2)} ₹/公担`,
                "数据来源：价格序列（₹/公担）。"
              ].join("<br/>");
            }
          },
          legend: {
            type: "scroll",
            top: 0,
            left: 0,
            right: 0,
            data: ["历史价格（₹/公担）", "30日均线", "90日均线", "预测区间（p10-p90）", "预测价格（₹/公担）"]
          },
          grid: { left: 58, right: 18, top: 56, bottom: 32, containLabel: true },
          xAxis: { type: "time", axisLabel: { hideOverlap: true } },
          yAxis: {
            type: "value",
            name: "价格（₹/公担）",
            axisLabel: { formatter: (value) => Number(value).toLocaleString("zh-CN"), hideOverlap: true }
          },
          series: [
            {
              ...seriesMotion(0, 10, 64),
              name: "历史价格（₹/公担）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#5470C6",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 2.1, color: "#5470C6" },
              itemStyle: { color: "#5470C6" },
              areaStyle: { opacity: 0.08, color: "#5470C6" },
              data: priceHistory
            },
            {
              ...seriesMotion(1, 9, 56),
              name: "30日均线",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#91CC75",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 1, color: "#91CC75" },
              data: priceMa30
            },
            {
              ...seriesMotion(2, 9, 56),
              name: "90日均线",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#FAC858",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 1, color: "#FAC858" },
              data: priceMa90
            },
            {
              ...seriesMotion(3, 8, 44),
              name: "_预测区间基线",
              type: "line",
              triggerLineEvent: true,
              symbol: "none",
              showSymbol: false,
              stack: "pred_ci",
              lineStyle: { opacity: 0, width: 0 },
              areaStyle: { opacity: 0 },
              tooltip: { show: false },
              emphasis: { disabled: true },
              data: priceForecastBand.base
            },
            {
              ...seriesMotion(4, 8, 44),
              name: "预测区间（p10-p90）",
              type: "line",
              triggerLineEvent: true,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              stack: "pred_ci",
              lineStyle: { opacity: 0, width: 0 },
              areaStyle: { opacity: 0.16, color: "#73C0DE" },
              tooltip: { show: false },
              emphasis: { disabled: true },
              data: priceForecastBand.width
            },
            {
              ...seriesMotion(5, 8, 48),
              name: "预测下沿（p10）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              lineStyle: { type: "solid", width: 1.0, opacity: 0.65, color: "#73C0DE" },
              data: priceForecastP10
            },
            {
              ...seriesMotion(6, 8, 48),
              name: "预测上沿（p90）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              lineStyle: { type: "solid", width: 1.0, opacity: 0.65, color: "#73C0DE" },
              data: priceForecastP90
            },
            {
              ...seriesMotion(7, 8, 48),
              name: "预测价格（₹/公担）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#9A60B4",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              universalTransition: true,
              lineStyle: { type: "solid", width: 1.3, opacity: 0.95, color: "#9A60B4" },
              data: priceForecast
            }
          ]
        },
        true
      );
      this.bindPreciseLineTooltip(
        "price",
        ({ xLabel, seriesName, yValue }) =>
          [`日期：${xLabel}`, `${seriesName}：${fmt(yValue, 2)} ₹/公担`, "数据来源：价格序列（₹/公担）。"].join("<br/>"),
        10
      );

      const dailyHistoryData = priceRawHistory.length ? priceRawHistory : priceHistory;
      const dailyMa30Data = priceRawMa30.length ? priceRawMa30 : priceMa30;
      const dailyMa90Data = priceRawMa90.length ? priceRawMa90 : priceMa90;
      const dailyForecastRows =
        Array.isArray(this.visuals?.price?.raw?.forecast) && this.visuals.price.raw.forecast.length
          ? this.visuals.price.raw.forecast
          : priceForecastRows;
      const dailyForecastData = priceRawForecast.length ? priceRawForecast : priceForecast;
      const dailyForecastP10 = this.toDatePairsByKey(dailyForecastRows, "p10");
      const dailyForecastP90 = this.toDatePairsByKey(dailyForecastRows, "p90");
      const dailyForecastBand = this.buildIntervalBand(dailyForecastRows);
      const priceDailyLegendNames = [
        "历史价格（₹/公担）",
        "30日均线",
        "90日均线",
        "预测区间（p10-p90）",
        "预测价格（₹/公担）"
      ];
      const priceDailyLegendSelected = priceDailyLegendNames.reduce((acc, name) => {
        acc[name] = this.priceDailyLegendSelected[name] !== false;
        return acc;
      }, {});
      const showPriceDailySeries = (name) => priceDailyLegendSelected[name] !== false;
      const priceDailyLegendDataSet = {
        history: dailyHistoryData,
        ma30: dailyMa30Data,
        ma90: dailyMa90Data,
        bandBase: dailyForecastBand.base,
        bandWidth: dailyForecastBand.width,
        p10: dailyForecastP10,
        p90: dailyForecastP90,
        forecast: dailyForecastData
      };
      this.priceDailyLegendNames = priceDailyLegendNames;
      this.charts.priceDaily?.setOption(
        {
          ...chartAnimationBase(20),
          tooltip: {
            trigger: "item",
            show: true,
            triggerOn: "mousemove|click",
            appendToBody: true,
            confine: false,
            formatter: (item) => {
              const row = Array.isArray(item) ? item[0] : item;
              const seriesName = String(row?.seriesName || "");
              if (!seriesName || seriesName.startsWith("_")) return "";
              const valueRaw = Array.isArray(row?.data) ? row.data?.[1] : Array.isArray(row?.value) ? row.value?.[1] : row?.value;
              const value = Number(valueRaw);
              if (!Number.isFinite(value)) return "";
              const dateLabel =
                row?.axisValueLabel ||
                (Array.isArray(row?.data) ? row.data?.[0] : Array.isArray(row?.value) ? row.value?.[0] : "-") ||
                "-";
              return [
                `日期：${dateLabel}`,
                `${row?.marker || ""}${seriesName}：${fmt(value, 2)} ₹/公担`,
                "数据来源：日度价格序列（₹/公担）。"
              ].join("<br/>");
            }
          },
          legend: {
            type: "scroll",
            top: 0,
            left: 0,
            right: 0,
            selectedMode: true,
            inactiveColor: "#9aa9a0",
            selected: priceDailyLegendSelected,
            data: priceDailyLegendNames
          },
          grid: { left: 58, right: 18, top: 56, bottom: 32, containLabel: true },
          xAxis: { type: "time", axisLabel: { hideOverlap: true } },
          yAxis: {
            type: "value",
            name: "价格（₹/公担）",
            axisLabel: { formatter: (value) => Number(value).toLocaleString("zh-CN"), hideOverlap: true }
          },
          series: [
            {
              ...seriesMotion(0, 10, 64),
              name: "历史价格（₹/公担）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#5470C6",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 2.1, color: "#5470C6" },
              itemStyle: { color: "#5470C6" },
              areaStyle: { opacity: 0.06, color: "#5470C6" },
              data: showPriceDailySeries("历史价格（₹/公担）") ? dailyHistoryData : []
            },
            {
              ...seriesMotion(1, 9, 56),
              name: "30日均线",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#91CC75",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 1.1, color: "#91CC75" },
              data: showPriceDailySeries("30日均线") ? dailyMa30Data : []
            },
            {
              ...seriesMotion(2, 9, 56),
              name: "90日均线",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#FAC858",
              symbol: "none",
              universalTransition: true,
              lineStyle: { width: 1.1, color: "#FAC858" },
              data: showPriceDailySeries("90日均线") ? dailyMa90Data : []
            },
            {
              ...seriesMotion(3, 8, 44),
              name: "_预测区间基线",
              type: "line",
              triggerLineEvent: true,
              symbol: "none",
              showSymbol: false,
              stack: "pred_ci_daily",
              lineStyle: { opacity: 0, width: 0 },
              areaStyle: { opacity: 0 },
              tooltip: { show: false },
              emphasis: { disabled: true },
              data: showPriceDailySeries("预测区间（p10-p90）") ? dailyForecastBand.base : []
            },
            {
              ...seriesMotion(4, 8, 44),
              name: "预测区间（p10-p90）",
              type: "line",
              triggerLineEvent: true,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              stack: "pred_ci_daily",
              lineStyle: { opacity: 0, width: 0 },
              areaStyle: { opacity: 0.16, color: "#73C0DE" },
              tooltip: { show: false },
              emphasis: { disabled: true },
              data: showPriceDailySeries("预测区间（p10-p90）") ? dailyForecastBand.width : []
            },
            {
              ...seriesMotion(5, 8, 48),
              name: "预测下沿（p10）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              lineStyle: { type: "solid", width: 1.0, opacity: 0.65, color: "#73C0DE" },
              data: showPriceDailySeries("预测区间（p10-p90）") ? dailyForecastP10 : []
            },
            {
              ...seriesMotion(6, 8, 48),
              name: "预测上沿（p90）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#73C0DE",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              lineStyle: { type: "solid", width: 1.0, opacity: 0.65, color: "#73C0DE" },
              data: showPriceDailySeries("预测区间（p10-p90）") ? dailyForecastP90 : []
            },
            {
              ...seriesMotion(7, 8, 48),
              name: "预测价格（₹/公担）",
              type: "line",
              triggerLineEvent: true,
              smooth: false,
              color: "#9A60B4",
              symbol: "none",
              showSymbol: false,
              connectNulls: true,
              universalTransition: true,
              lineStyle: { type: "solid", width: 1.2, opacity: 0.92, color: "#9A60B4" },
              data: showPriceDailySeries("预测价格（₹/公担）") ? dailyForecastData : []
            }
          ]
        },
        true
      );
      this.bindPreciseLineTooltip(
        "priceDaily",
        ({ xLabel, seriesName, yValue }) =>
          [`日期：${xLabel}`, `${seriesName}：${fmt(yValue, 2)} ₹/公担`, "数据来源：日度价格序列（₹/公担）。"].join("<br/>"),
        10
      );
      if (this.charts.priceDaily && typeof this.charts.priceDaily.off === "function") {
        this.charts.priceDaily.off("legendselectchanged");
        this.charts.priceDaily.on("legendselectchanged", (params) => {
          const selected = params?.selected && typeof params.selected === "object" ? params.selected : {};
          this.applyPriceDailyLegendSelection(selected, priceDailyLegendDataSet);
        });
      }
      this.applyPriceDailyLegendSelection(
        this.charts.priceDaily?.getOption?.()?.legend?.[0]?.selected || priceDailyLegendSelected,
        priceDailyLegendDataSet
      );

      const toTs = (dateVal) => {
        const ts = new Date(String(dateVal || "")).getTime();
        return Number.isFinite(ts) ? ts : null;
      };
      const dayMs = 24 * 60 * 60 * 1000;
      const detailContextDays = 50;
      const detailForecastPairs = this.toDatePairs(dailyForecastRows);
      const detailForecastStartDate = detailForecastPairs.length ? detailForecastPairs[0][0] : null;
      const detailForecastEndDate = detailForecastPairs.length ? detailForecastPairs[detailForecastPairs.length - 1][0] : null;
      const detailStartTs = detailForecastStartDate ? toTs(detailForecastStartDate) : null;
      const detailEndTs = detailForecastEndDate ? toTs(detailForecastEndDate) : null;
      const detailHistoryData = dailyHistoryData.filter((item) => {
        if (!Array.isArray(item) || !item.length) return false;
        const ts = toTs(item[0]);
        if (ts === null) return false;
        if (detailStartTs === null) return true;
        return ts >= detailStartTs - detailContextDays * dayMs;
      });
      const detailForecastRows = Array.isArray(dailyForecastRows)
        ? dailyForecastRows.filter((item) => {
            const ts = toTs(item?.date);
            if (ts === null) return false;
            if (detailStartTs === null || detailEndTs === null) return true;
            return ts >= detailStartTs && ts <= detailEndTs + dayMs;
          })
        : [];
      const detailActualData = this.toDatePairs(this.visuals?.price?.raw?.actual);
      const detailForecastP50 = this.toDatePairs(detailForecastRows);
      const detailForecastP10 = this.toDatePairsByKey(detailForecastRows, "p10");
      const detailForecastP90 = this.toDatePairsByKey(detailForecastRows, "p90");
      const detailForecastBand = this.buildIntervalBand(detailForecastRows);
      const detailValues = [...detailHistoryData, ...detailActualData, ...detailForecastP10, ...detailForecastP90, ...detailForecastP50]
        .map((item) => Number(item?.[1]))
        .filter(Number.isFinite);
      const detailMin = detailValues.length ? Math.min(...detailValues) : null;
      const detailMax = detailValues.length ? Math.max(...detailValues) : null;
      const detailPadding =
        detailMin === null || detailMax === null ? null : Math.max((detailMax - detailMin) * 0.08, 80);
      const detailWindowStart = detailStartTs === null ? null : detailStartTs - detailContextDays * dayMs;
      const detailWindowEnd = detailEndTs === null ? null : detailEndTs + 2 * dayMs;

      if (detailForecastP50.length) {
        const detailLegendNames = detailActualData.length
          ? ["近段历史", "真实价格", "预测区间（p10-p90）", "预测价格"]
          : ["近段历史", "预测区间（p10-p90）", "预测价格"];
        const detailSeries = [
          {
            ...seriesMotion(0, 8, 36),
            name: "近段历史",
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            color: "#5470C6",
            symbol: "none",
            lineStyle: { width: 1.8, color: "#5470C6" },
            itemStyle: { color: "#5470C6" },
            areaStyle: { opacity: 0.07, color: "#5470C6" },
            markArea:
              detailForecastStartDate && detailForecastEndDate
                ? {
                    silent: true,
                    itemStyle: { color: "rgba(47,123,84,0.06)" },
                    data: [[{ xAxis: detailForecastStartDate }, { xAxis: detailForecastEndDate }]]
                  }
                : undefined,
            data: detailHistoryData
          }
        ];
        if (detailActualData.length) {
          detailSeries.push({
            ...seriesMotion(1, 8, 36),
            name: "真实价格",
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            color: "#EE6666",
            symbol: "none",
            showSymbol: false,
            lineStyle: { width: 2.0, type: "solid", color: "#EE6666" },
            itemStyle: { color: "#EE6666" },
            data: detailActualData
          });
        }
        detailSeries.push(
          {
            ...seriesMotion(detailSeries.length, 8, 36),
            name: "_细化区间基线",
            type: "line",
            triggerLineEvent: true,
            symbol: "none",
            showSymbol: false,
            stack: "pred_ci_detail",
            lineStyle: { opacity: 0, width: 0 },
            areaStyle: { opacity: 0 },
            tooltip: { show: false },
            emphasis: { disabled: true },
            data: detailForecastBand.base
          },
          {
            ...seriesMotion(detailSeries.length + 1, 8, 36),
            name: "预测区间（p10-p90）",
            type: "line",
            triggerLineEvent: true,
            color: "#73C0DE",
            symbol: "none",
            showSymbol: false,
            stack: "pred_ci_detail",
            lineStyle: { opacity: 0, width: 0 },
            areaStyle: { opacity: 0.2, color: "#73C0DE" },
            tooltip: { show: false },
            emphasis: { disabled: true },
            data: detailForecastBand.width
          },
          {
            ...seriesMotion(detailSeries.length + 2, 8, 36),
            name: "_预测下沿",
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            symbol: "none",
            showSymbol: false,
            lineStyle: { width: 0.9, type: "solid", opacity: 0.62, color: "#73C0DE" },
            data: detailForecastP10
          },
          {
            ...seriesMotion(detailSeries.length + 3, 8, 36),
            name: "_预测上沿",
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            symbol: "none",
            showSymbol: false,
            lineStyle: { width: 0.9, type: "solid", opacity: 0.62, color: "#73C0DE" },
            data: detailForecastP90
          },
          {
            ...seriesMotion(detailSeries.length + 4, 8, 36),
            name: "预测价格",
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            color: "#9A60B4",
            symbol: "none",
            showSymbol: false,
            lineStyle: { width: 2.1, type: "solid", color: "#9A60B4" },
            areaStyle: { opacity: 0.06, color: "#9A60B4" },
            data: detailForecastP50
          }
        );
        this.charts.priceForecastDetail?.setOption(
          {
            ...chartAnimationBase(40),
            tooltip: {
              trigger: "item",
              show: true,
              triggerOn: "mousemove|click",
              appendToBody: true,
              confine: false,
              formatter: (item) => {
                const row = Array.isArray(item) ? item[0] : item;
                const seriesName = String(row?.seriesName || "");
                if (!seriesName || seriesName.startsWith("_")) return "";
                const valueRaw = Array.isArray(row?.data) ? row.data?.[1] : Array.isArray(row?.value) ? row.value?.[1] : row?.value;
                const value = Number(valueRaw);
                if (!Number.isFinite(value)) return "";
                const dateLabel =
                  row?.axisValueLabel ||
                  (Array.isArray(row?.data) ? row.data?.[0] : Array.isArray(row?.value) ? row.value?.[0] : "-") ||
                  "-";
                return [`细节日期：${dateLabel}`, `${row?.marker || ""}${seriesName}：${fmt(value, 2)} ₹/公担`].join("<br/>");
              }
            },
            legend: {
              type: "scroll",
              top: 0,
              left: 0,
              right: 0,
              data: detailLegendNames
            },
            grid: { left: 58, right: 18, top: 48, bottom: 30, containLabel: true },
            xAxis: {
              type: "time",
              min: detailWindowStart ?? undefined,
              max: detailWindowEnd ?? undefined,
              axisLabel: { hideOverlap: true }
            },
            yAxis: {
              type: "value",
              name: "价格（₹/公担）",
              min: detailPadding === null ? undefined : Math.max(0, detailMin - detailPadding),
              max: detailPadding === null ? undefined : detailMax + detailPadding,
              axisLabel: { formatter: (value) => Number(value).toLocaleString("zh-CN"), hideOverlap: true }
            },
            series: detailSeries
          },
          true
        );
        this.bindPreciseLineTooltip(
          "priceForecastDetail",
          {
            includeSeries: (series) =>
              ["近段历史", "真实价格", "预测价格", "_预测下沿", "_预测上沿"].includes(String(series?.name || "")),
            areaBands: [{ baseSeriesName: "_细化区间基线", bandSeriesName: "预测区间（p10-p90）", label: "预测区间（p10-p90）" }],
            formatter: ({ xLabel, seriesName, yValue }) => {
              const labelMap = {
                "近段历史": "近段历史",
                "真实价格": "真实价格",
                "预测价格": "预测价格",
                "_预测下沿": "预测下沿（p10）",
                "_预测上沿": "预测上沿（p90）"
              };
              return [`细节日期：${xLabel}`, `${labelMap[seriesName] || seriesName}：${fmt(yValue, 2)} ₹/公担`].join("<br/>");
            },
            bandFormatter: ({ xLabel, seriesName, lowValue, highValue }) => {
              return [
                `细节日期：${xLabel}`,
                `${seriesName}：${fmt(lowValue, 2)} - ${fmt(highValue, 2)} ₹/公担`
              ].join("<br/>");
            }
          },
          12
        );
        if (this.visualView === "price") {
          this.resizeChartOnFrame("priceForecastDetail", 2);
        }
      } else {
        this.charts.priceForecastDetail?.clear?.();
      }

      const toMonthDayKey = (dateVal) => {
        const dt = new Date(String(dateVal || ""));
        if (!Number.isFinite(dt.getTime())) return "";
        const mm = String(dt.getMonth() + 1).padStart(2, "0");
        const dd = String(dt.getDate()).padStart(2, "0");
        return `${mm}-${dd}`;
      };
      const windowMonthDays = [];
      const windowMonthDaySet = new Set();
      (dailyForecastRows || []).forEach((item) => {
        const key = toMonthDayKey(item?.date);
        if (!key || windowMonthDaySet.has(key)) return;
        windowMonthDaySet.add(key);
        windowMonthDays.push(key);
      });
      const windowLabels = windowMonthDays.map((key) => {
        const [mm, dd] = String(key).split("-");
        return `${Number(mm)}/${Number(dd)}`;
      });
      const forecastWindowYear = (() => {
        const firstDate = dailyForecastRows?.[0]?.date;
        const dt = new Date(String(firstDate || ""));
        return Number.isFinite(dt.getTime()) ? String(dt.getFullYear()) : "";
      })();
      const historyByYear = new Map();
      (dailyHistoryData || []).forEach((item) => {
        if (!Array.isArray(item) || item.length < 2) return;
        const dateVal = item[0];
        const value = safeNumber(item[1]);
        if (value === null) return;
        const dt = new Date(String(dateVal || ""));
        if (!Number.isFinite(dt.getTime())) return;
        const key = toMonthDayKey(dateVal);
        if (!windowMonthDaySet.has(key)) return;
        const year = String(dt.getFullYear());
        if (!historyByYear.has(year)) {
          historyByYear.set(year, new Map());
        }
        historyByYear.get(year).set(key, value);
      });
      const minCoverage = 3;
      const maxSeasonalYears = 4;
      const forecastYearNum = Number(forecastWindowYear);
      const seasonalYears = Array.from(historyByYear.entries())
        .map(([year, points]) => ({ year, points, count: points.size }))
        .filter((item) => item.year !== forecastWindowYear && item.count >= minCoverage)
        .sort((a, b) => Number(a.year) - Number(b.year))
        .filter((item) => !Number.isFinite(forecastYearNum) || Number(item.year) >= forecastYearNum - maxSeasonalYears)
        .slice(-maxSeasonalYears)
        .map((item) => item.year);
      const forecastWindowMap = new Map();
      (dailyForecastRows || []).forEach((item) => {
        const key = toMonthDayKey(item?.date);
        const value = safeNumber(item?.value);
        if (!key || value === null || forecastWindowMap.has(key)) return;
        forecastWindowMap.set(key, value);
      });
      const forecastWindowLine = windowMonthDays.map((key) => (forecastWindowMap.has(key) ? forecastWindowMap.get(key) : null));
      const seasonalCanRender =
        windowMonthDays.length >= 7 && seasonalYears.length > 0 && forecastWindowLine.some((x) => Number.isFinite(Number(x)));
      if (seasonalCanRender) {
        const seasonalPalette = [
          "#5470C6",
          "#91CC75",
          "#FAC858",
          "#EE6666",
          "#73C0DE",
          "#3BA272",
          "#FC8452",
          "#9A60B4",
          "#EA7CCC",
          "#6E7074"
        ];
        const seasonalSeries = seasonalYears.map((year, idx) => {
          const map = historyByYear.get(year) || new Map();
          return {
            ...seriesMotion(idx, 10, 42),
            name: `${year}年同期`,
            type: "line",
            triggerLineEvent: true,
            smooth: false,
            symbol: "none",
            showSymbol: false,
            connectNulls: false,
            color: seasonalPalette[idx % seasonalPalette.length],
            lineStyle: {
              width: 1.5,
              opacity: 0.9,
              color: seasonalPalette[idx % seasonalPalette.length]
            },
            data: windowMonthDays.map((key) => (map.has(key) ? map.get(key) : null))
          };
        });
        seasonalSeries.push({
          ...seriesMotion(seasonalSeries.length, 12, 44),
          name: `${forecastWindowYear || "当前"}年预测`,
          type: "line",
          triggerLineEvent: true,
          smooth: false,
          symbol: "none",
          showSymbol: false,
          connectNulls: false,
          color: "#9A60B4",
          lineStyle: { width: 2.1, type: "solid", color: "#9A60B4" },
          areaStyle: { opacity: 0.05, color: "#9A60B4" },
          data: forecastWindowLine
        });
        const seasonalLegendNames = seasonalSeries.map((item) => item.name);
        const seasonalLegendSelected = seasonalLegendNames.reduce((acc, name) => {
          acc[name] = this.priceWindowLegendSelected[name] !== false;
          return acc;
        }, {});
        if (!Object.values(seasonalLegendSelected).some(Boolean) && seasonalLegendNames.length) {
          seasonalLegendSelected[seasonalLegendNames[seasonalLegendNames.length - 1]] = true;
        }
        this.priceWindowLegendNames = seasonalLegendNames;
        this.charts.priceWindowSeasonal?.setOption(
          {
            ...chartAnimationBase(60),
            tooltip: {
              trigger: "item",
              show: true,
              triggerOn: "mousemove|click",
              appendToBody: true,
              confine: false,
              formatter: (item) => {
                const row = Array.isArray(item) ? item[0] : item;
                const value = Number(row?.data);
                if (!Number.isFinite(value)) return "";
                const dateLabel = row?.axisValueLabel || "-";
                return [`窗口日期：${dateLabel}`, `${row?.marker || ""}${row?.seriesName || "-"}：${fmt(value, 2)} ₹/公担`].join(
                  "<br/>"
                );
              }
            },
            legend: {
              type: "plain",
              top: 0,
              left: 0,
              right: 0,
              selectedMode: true,
              inactiveColor: "#9aa9a0",
              selected: seasonalLegendSelected,
              data: seasonalSeries.map((item) => item.name)
            },
            grid: { left: 58, right: 18, top: 50, bottom: 30, containLabel: true },
            xAxis: {
              type: "category",
              data: windowLabels,
              boundaryGap: false,
              axisLabel: { hideOverlap: true }
            },
            yAxis: {
              type: "value",
              name: "价格（₹/公担）",
              axisLabel: { formatter: (value) => Number(value).toLocaleString("zh-CN"), hideOverlap: true }
            },
            series: seasonalSeries
          },
          false
        );
        this.bindPreciseLineTooltip(
          "priceWindowSeasonal",
          ({ xLabel, seriesName, yValue }) => [`窗口日期：${xLabel}`, `${seriesName}：${fmt(yValue, 2)} ₹/公担`].join("<br/>"),
          10
        );
        if (this.charts.priceWindowSeasonal && typeof this.charts.priceWindowSeasonal.off === "function") {
          this.charts.priceWindowSeasonal.off("legendselectchanged");
          this.charts.priceWindowSeasonal.on("legendselectchanged", (params) => {
            const selected = params?.selected && typeof params.selected === "object" ? params.selected : {};
            this.priceWindowLegendSelected = { ...selected };
          });
        }
        this.priceWindowLegendSelected = {
          ...(this.charts.priceWindowSeasonal?.getOption?.()?.legend?.[0]?.selected || seasonalLegendSelected)
        };
        if (this.visualView === "price") {
          this.resizeChartOnFrame("priceWindowSeasonal", 2);
        }
      } else {
        this.priceWindowLegendNames = [];
        this.priceWindowLegendSelected = {};
        this.charts.priceWindowSeasonal?.clear?.();
      }

      const yieldHistory = Array.isArray(this.visuals?.yield?.history) ? this.visuals.yield.history : [];
      const yieldTrend = Array.isArray(this.visuals?.yield?.trend) ? this.visuals.yield.trend : [];
      const yieldActual = Array.isArray(this.visuals?.yield?.actual) ? this.visuals.yield.actual : [];
      const yieldForecast = this.visuals?.yield?.forecast || {};
      const yieldYearSet = new Set([
        ...yieldHistory.map((x) => String(x?.year ?? "")),
        ...yieldTrend.map((x) => String(x?.year ?? "")),
        ...yieldActual.map((x) => String(x?.year ?? "")),
        String(yieldForecast?.year ?? "")
      ]);
      const yieldYears = Array.from(yieldYearSet)
        .filter((x) => x)
        .sort((a, b) => Number(a) - Number(b));
      const yieldLabelRotate = yieldYears.length > 8 ? 30 : 0;
      const toYearLine = (rows) => {
        const map = new Map(
          (rows || [])
            .map((x) => [String(x?.year ?? ""), safeNumber(x?.value)])
            .filter((x) => x[0] && x[1] !== null)
        );
        return yieldYears.map((year) => (map.has(year) ? map.get(year) : null));
      };
      const yieldHistoryLine = toYearLine(yieldHistory);
      const yieldTrendLine = toYearLine(yieldTrend);
      const yieldActualLineRaw = toYearLine(yieldActual);
      const yieldForecastLineRaw = (() => {
        const year = String(yieldForecast?.year ?? "");
        const value = safeNumber(yieldForecast?.value);
        return yieldYears.map((x) => (x === year ? value : null));
      })();
      const yieldActualLine = connectFromLastHistory(yieldHistoryLine, yieldActualLineRaw);
      const yieldForecastLine = connectFromLastHistory(yieldHistoryLine, yieldForecastLineRaw);
      const hasYieldActual = yieldActualLine.some((x) => x !== null);
      const yieldLegendData = hasYieldActual
        ? ["历史产量（吨/公顷）", "趋势线", "真实产量（吨/公顷）", "预测产量（吨/公顷）"]
        : ["历史产量（吨/公顷）", "趋势线", "预测产量（吨/公顷）"];

      this.charts.yield?.setOption(
        {
          ...chartAnimationBase(80),
          tooltip: {
            trigger: "item",
            show: true,
            triggerOn: "mousemove|click",
            appendToBody: true,
            confine: false,
            formatter: (item) => {
              const row = Array.isArray(item) ? item[0] : item;
              const value = Number(row?.data);
              if (!Number.isFinite(value)) return "";
              const year = row?.axisValueLabel || "-";
              return [
                `年份：${year}`,
                `${row?.marker || ""}${row?.seriesName || "-"}：${fmt(value, 2)} 吨/公顷`,
                "数据来源：产量序列（吨/公顷）。"
              ].join("<br/>");
            }
          },
          legend: {
            type: "scroll",
            top: 0,
            left: 0,
            right: 0,
            data: yieldLegendData
          },
          grid: { left: 56, right: 18, top: 56, bottom: 32, containLabel: true },
          xAxis: {
            type: "category",
            name: AXIS_LABELS.year,
            data: yieldYears,
            axisLabel: { hideOverlap: true, rotate: yieldLabelRotate }
          },
          yAxis: {
            type: "value",
            name: "产量（吨/公顷）",
            axisLabel: {
              formatter: (value) => Number(value).toLocaleString("zh-CN"),
              hideOverlap: true
            }
          },
          series: [
            {
              ...seriesMotion(0, 18, 36),
              name: "历史产量（吨/公顷）",
              type: "line",
              triggerLineEvent: true,
              smooth: 0.18,
              color: "#5470C6",
              itemStyle: { color: "#5470C6" },
              lineStyle: { color: "#5470C6" },
              symbolSize: 6,
              universalTransition: true,
              data: yieldHistoryLine
            },
            {
              ...seriesMotion(1, 16, 32),
              name: "趋势线",
              type: "line",
              triggerLineEvent: true,
              smooth: true,
              color: "#91CC75",
              symbol: "none",
              universalTransition: true,
              lineStyle: { type: "dashed", color: "#91CC75" },
              data: yieldTrendLine
            },
            {
              ...seriesMotion(2, 20, 28),
              name: "预测产量（吨/公顷）",
              type: "line",
              triggerLineEvent: true,
              color: "#FAC858",
              itemStyle: { color: "#FAC858" },
              lineStyle: { color: "#FAC858", width: 2 },
              symbolSize: 10,
              universalTransition: true,
              data: yieldForecastLine
            },
            ...(hasYieldActual
              ? [
                  {
                    ...seriesMotion(3, 18, 28),
                    name: "真实产量（吨/公顷）",
                    type: "line",
                    triggerLineEvent: true,
                    smooth: false,
                    color: "#EE6666",
                    symbol: "diamond",
                    symbolSize: 11,
                    z: 6,
                    itemStyle: { color: "#EE6666", borderColor: "#FFFFFF", borderWidth: 1.5 },
                    lineStyle: { color: "#EE6666", width: 2.2 },
                    universalTransition: true,
                    data: yieldActualLine
                  }
                ]
              : [])
          ]
        },
        true
      );
      this.bindPreciseLineTooltip(
        "yield",
        ({ xLabel, seriesName, yValue }) =>
          [`年份：${xLabel}`, `${seriesName}：${fmt(yValue, 2)} 吨/公顷`, "数据来源：产量序列（吨/公顷）。"].join("<br/>"),
        12
      );

      const costHistory = Array.isArray(this.visuals?.cost?.history) ? this.visuals.cost.history : [];
      const costTrend = Array.isArray(this.visuals?.cost?.trend) ? this.visuals.cost.trend : [];
      const costActual = Array.isArray(this.visuals?.cost?.actual) ? this.visuals.cost.actual : [];
      const costForecast = this.visuals?.cost?.forecast || {};
      const costYearSet = new Set([
        ...costHistory.map((x) => String(x?.year ?? "")),
        ...costTrend.map((x) => String(x?.year ?? "")),
        ...costActual.map((x) => String(x?.year ?? "")),
        String(costForecast?.year ?? "")
      ]);
      const costYears = Array.from(costYearSet)
        .filter((x) => x)
        .sort((a, b) => Number(a) - Number(b));
      const costLabelRotate = costYears.length > 8 ? 30 : 0;
      const toCostLine = (rows) => {
        const map = new Map(
          (rows || [])
            .map((x) => [String(x?.year ?? ""), safeNumber(x?.value)])
            .filter((x) => x[0] && x[1] !== null)
        );
        return costYears.map((year) => (map.has(year) ? map.get(year) : null));
      };
      const costHistoryLine = toCostLine(costHistory);
      const costTrendLine = toCostLine(costTrend);
      const costActualLineRaw = toCostLine(costActual);
      const costForecastLineRaw = (() => {
        const year = String(costForecast?.year ?? "");
        const value = safeNumber(costForecast?.value);
        return costYears.map((x) => (x === year ? value : null));
      })();
      const costActualLine = connectFromLastHistory(costHistoryLine, costActualLineRaw);
      const costForecastLine = connectFromLastHistory(costHistoryLine, costForecastLineRaw);
      const costForecastYear = String(costForecast?.year ?? "");
      const costPredRaw = safeNumber(this.visualPrediction?.cost_pred_raw);
      const costRawForecastLine = costYears.map((year) => (year === costForecastYear ? costPredRaw : null));
      const hasCostActual = costActualLine.some((x) => x !== null);
      const hasRawCostPoint = (() => {
        const cur = safeNumber(costForecast?.value);
        if (costPredRaw === null || cur === null) return false;
        return Math.abs(costPredRaw - cur) > 1e-9;
      })();
      const costLegendData = [
        "历史成本（₹/公顷）",
        "趋势线",
        ...(hasCostActual ? ["真实成本（₹/公顷）"] : []),
        "预测成本（当前推荐）",
        ...(hasRawCostPoint ? ["预测成本（原值）"] : [])
      ];

      this.charts.cost?.setOption(
        {
          ...chartAnimationBase(140),
          tooltip: {
            trigger: "item",
            show: true,
            triggerOn: "mousemove|click",
            appendToBody: true,
            confine: false,
            formatter: (item) => {
              const row = Array.isArray(item) ? item[0] : item;
              const value = Number(row?.data);
              if (!Number.isFinite(value)) return "";
              const year = row?.axisValueLabel || "-";
              return [
                `年份：${year}`,
                `${row?.marker || ""}${row?.seriesName || "-"}：${fmt(value, 2)} ₹/公顷`,
                "数据来源：成本序列（₹/公顷）。"
              ].join("<br/>");
            }
          },
          legend: {
            type: "scroll",
            top: 0,
            left: 0,
            right: 0,
            data: costLegendData
          },
          grid: { left: 86, right: 18, top: 62, bottom: 32, containLabel: true },
          xAxis: {
            type: "category",
            name: AXIS_LABELS.year,
            data: costYears,
            axisLabel: { hideOverlap: true, rotate: costLabelRotate }
          },
          yAxis: {
            type: "value",
            name: "成本（₹/公顷）",
            axisLabel: {
              formatter: (value) => Number(value).toLocaleString("zh-CN"),
              hideOverlap: true
            }
          },
          series: [
            {
              ...seriesMotion(0, 18, 36),
              name: "历史成本（₹/公顷）",
              type: "line",
              triggerLineEvent: true,
              smooth: 0.15,
              color: "#5470C6",
              itemStyle: { color: "#5470C6" },
              lineStyle: { color: "#5470C6" },
              symbolSize: 6,
              universalTransition: true,
              data: costHistoryLine
            },
            {
              ...seriesMotion(1, 16, 32),
              name: "趋势线",
              type: "line",
              triggerLineEvent: true,
              smooth: true,
              color: "#91CC75",
              symbol: "none",
              universalTransition: true,
              lineStyle: { type: "dashed", color: "#91CC75" },
              data: costTrendLine
            },
            {
              ...seriesMotion(2, 20, 28),
              name: "预测成本（当前推荐）",
              type: "line",
              triggerLineEvent: true,
              color: "#FAC858",
              itemStyle: { color: "#FAC858" },
              lineStyle: { color: "#FAC858", width: 2 },
              symbolSize: 10,
              universalTransition: true,
              data: costForecastLine
            },
            ...(hasRawCostPoint
              ? [
                  {
                    ...seriesMotion(3, 20, 28),
                    name: "预测成本（原值）",
                    type: "line",
                    triggerLineEvent: true,
                    color: "#73C0DE",
                    itemStyle: { color: "#73C0DE" },
                    symbolSize: 8,
                    symbol: "rect",
                    universalTransition: true,
                    lineStyle: { type: "dotted", width: 1.6, opacity: 0.82, color: "#73C0DE" },
                    data: costRawForecastLine
                  }
                ]
              : []),
            ...(hasCostActual
              ? [
                  {
                    ...seriesMotion(4, 18, 28),
                    name: "真实成本（₹/公顷）",
                    type: "line",
                    triggerLineEvent: true,
                    smooth: false,
                    color: "#EE6666",
                    symbol: "diamond",
                    symbolSize: 11,
                    z: 6,
                    itemStyle: { color: "#EE6666", borderColor: "#FFFFFF", borderWidth: 1.5 },
                    lineStyle: { color: "#EE6666", width: 2.2 },
                    universalTransition: true,
                    data: costActualLine
                  }
                ]
              : []),
          ]
        },
        true
      );
      this.bindPreciseLineTooltip(
        "cost",
        ({ xLabel, seriesName, yValue }) =>
          [`年份：${xLabel}`, `${seriesName}：${fmt(yValue, 2)} ₹/公顷`, "数据来源：成本序列（₹/公顷）。"].join("<br/>"),
        12
      );

      const selected = this.selectedRow || {};
      const envProbRaw = safeNumber(this.visuals?.profile?.env_prob ?? selected?.env_prob);
      const probBestRaw = safeNumber(this.visuals?.profile?.prob_best ?? selected?.prob_best);
      const riskRaw = safeNumber(this.visuals?.profile?.risk ?? selected?.risk);
      const scoreRaw = safeNumber(selected?.recommend_strength);
      const marginRaw = safeNumber(this.visualPrediction?.margin_pct ?? this.rowMarginPct(selected));
      const stabilityRaw = this.sampleStabilityPercent();
      const sampleSupportText = this.sampleCoverageText();
      const calibrationScore = this.calibrationStabilityPercent(probBestRaw);

      const radarMeta = [
        {
          name: "环境适配概率",
          score: envProbRaw === null ? 0 : this.probabilityToRadarPercent(envProbRaw),
          rawText: envProbRaw === null ? "-" : `${fmt(envProbRaw * 100, 2)}%`
        },
        {
          name: "校准稳定性",
          score: calibrationScore,
          rawText: probBestRaw === null ? `样本覆盖 ${sampleSupportText}` : `${fmt(probBestRaw * 100, 2)}% · 样本覆盖 ${sampleSupportText}`
        },
        {
          name: "风险安全度",
          score: riskRaw === null ? 0 : this.riskToSafetyPercent(riskRaw),
          rawText: riskRaw === null ? "-" : fmt(riskRaw, 3)
        },
        {
          name: "利润率健康度",
          score: marginRaw === null ? 0 : this.marginToRadarPercent(marginRaw),
          rawText: marginRaw === null ? "-" : `${fmt(marginRaw, 2)}%`
        },
        {
          name: "推荐强度",
          score: scoreRaw === null ? 0 : scoreRaw,
          rawText: scoreRaw === null ? "-" : `${fmt(scoreRaw, 1)} / 100`
        },
        {
          name: "样本覆盖度",
          score: stabilityRaw,
          rawText: sampleSupportText
        }
      ];
      const radarValues = radarMeta.map((x) => x.score);
      const radarAreaColor =
        window.echarts?.graphic?.RadialGradient &&
        new window.echarts.graphic.RadialGradient(0.5, 0.4, 0.82, [
          { offset: 0, color: "rgba(46,123,84,0.56)" },
          { offset: 0.75, color: "rgba(46,123,84,0.30)" },
          { offset: 1, color: "rgba(46,123,84,0.08)" }
        ]);

      this.charts.profile?.setOption(
        {
          ...chartAnimationBase(220),
          tooltip: {
            trigger: "item",
            show: true,
            triggerOn: "mousemove|click",
            appendToBody: true,
            formatter: () => {
              const lines = radarMeta.map((item) => {
                const explainKey =
                  item.name === "环境适配概率"
                    ? "env_prob"
                    : item.name === "校准稳定性"
                    ? "prob_best"
                    : item.name === "风险安全度"
                    ? "risk"
                    : item.name === "综合收益强度"
                    ? "score"
                    : item.name === "利润率健康度"
                    ? "margin_pct"
                    : "yield";
                return `- ${item.name}：${fmt(item.score, 1)} / 100（原值 ${item.rawText}）<br/>${this.metricHelp(explainKey)}`;
              });
              const sparseNote =
                this.rows.length < 3
                  ? "<br/><span style=\"color:#7d8b82;\">当前候选样本偏少，优先看“原值”而不是形状对比。</span>"
                  : "";
              return `${cropName(this.selectedCrop)}<br/>${lines.join("<br/>")}${sparseNote}`;
            },
            confine: false
          },
          radar: {
            center: ["50%", "54%"],
            radius: "70%",
            splitNumber: 6,
            axisNameGap: 14,
            axisName: {
              color: "#2f5644",
              fontSize: 13,
              fontWeight: 600
            },
            axisLine: {
              lineStyle: {
                color: "rgba(75,110,92,0.25)"
              }
            },
            splitLine: {
              lineStyle: {
                color: "rgba(122,153,136,0.28)"
              }
            },
            splitArea: {
              areaStyle: {
                color: ["rgba(245,251,247,0.88)", "rgba(232,244,237,0.65)"]
              }
            },
            indicator: radarMeta.map((item) => ({ name: item.name, max: 100 }))
          },
          series: [
            {
              ...seriesMotion(0, 0, 0),
              type: "radar",
              data: [
                {
                  value: radarValues,
                  name: cropName(this.selectedCrop)
                }
              ],
              symbol: "circle",
              symbolSize: 7,
              lineStyle: {
                width: 3,
                color: "#2f7b54",
                shadowBlur: 6,
                shadowColor: "rgba(47,123,84,0.22)"
              },
              itemStyle: {
                color: "#2f7b54",
                borderColor: "#ffffff",
                borderWidth: 1.2
              },
              areaStyle: {
                opacity: 0.44,
                color: radarAreaColor || "rgba(46,123,84,0.34)"
              },
              emphasis: {
                lineStyle: { width: 3.4 },
                areaStyle: { opacity: 0.5 }
              }
            }
          ]
        },
        true
      );
      this.bindSeriesItemTooltip("profile", () => {
        const lines = radarMeta.map((item) => `- ${item.name}：${fmt(item.score, 1)} / 100（原值 ${item.rawText}）`);
        return `${cropName(this.selectedCrop)}<br/>${lines.join("<br/>")}`;
      });
      this.resizeChartOnFrame("profile", 2);
      this.renderScoreCharts();
    },
    async selectCrop(row) {
      const crop = String(row?.crop || "");
      if (!crop) return;
      this.selectedCrop = crop;
      await this.$nextTick();
      this.renderDecisionSupportCharts();
      await this.loadCropVisuals();
    },
    async loadCropVisuals() {
      const row = this.selectedRow;
      if (!row) {
        this.visuals = emptyVisualPayload();
        this.setVisualStatus("请先生成推荐结果。", false);
        return;
      }

      if (!hasEcharts()) {
        this.setVisualStatus("缺少 ECharts，无法渲染可视化图表。", true);
        return;
      }

      const seq = ++this.visualRequestSeq;
      this.loadingVisual = true;
      this.setVisualStatus(`正在加载 ${cropName(row.crop)} 的可视化数据...`, false);

      try {
        const payload = await fetchCropVisuals({
          crop: row.crop,
          price_pred: row.price_pred,
          price_forecast: Array.isArray(row.price_forecast) ? row.price_forecast : null,
          yield_pred: row.yield,
          cost_pred: row.cost_pred,
          cost_pred_raw: row.cost_pred_raw ?? null,
          profit_pred: row.profit,
          env_prob: row.env_prob,
          prob_best: row.prob_best,
          risk: row.risk,
          score: row.score,
          horizon_days: this.result?.runtime?.prediction_window?.price_horizon_days ?? null,
          target_year: row.target_year ?? null,
          history_years: 12
        });

        if (seq !== this.visualRequestSeq) return;
        this.visuals = payload || emptyVisualPayload();

        const warnings = Array.isArray(this.visuals?.warnings) ? this.visuals.warnings : [];
        const warningHint = warnings.length ? `（警告：${warnings.join("、")}）` : "";
        const targetYear = this.visuals?.time_meta?.target_year;
        const timeHint = Number.isFinite(Number(targetYear)) ? `（目标年份：${targetYear}）` : "";
        const alignmentNotes = Array.isArray(this.visuals?.alignment_notes) ? this.visuals.alignment_notes : [];
        const alignmentHint = alignmentNotes.length ? `（时间对齐：${alignmentNotes.join("；")}）` : "";
        const adjustmentTags = Array.isArray(row?.cost_adjustment?.applied) ? row.cost_adjustment.applied : [];
        const adjustHint = adjustmentTags.length ? `（成本调整：${adjustmentTags.join("+")}）` : "";
        this.setVisualStatus(`已加载 ${cropName(row.crop)} 的可视化数据。${timeHint}${warningHint}${alignmentHint}${adjustHint}`, false);

        await this.$nextTick();
        this.renderVisualCharts();
        await this.refreshUi();
      } catch (err) {
        if (seq !== this.visualRequestSeq) return;
        this.visuals = emptyVisualPayload();
        this.setVisualStatus(`加载作物可视化失败：${err.message}`, true);
        this.renderVisualCharts();
      } finally {
        if (seq === this.visualRequestSeq) {
          this.loadingVisual = false;
        }
      }
    },
    async runRecommend() {
      if (this.loading) return;
      this.loading = true;
      this.setStatus("正在调用模型服务，请稍候...", false);

      try {
        const payload = await fetchRecommend(this.buildEnvPayload());

        this.result = payload || null;
        this.rows = normalizeRows(payload?.results);

        const filterInfo = payload?.runtime?.filters;
        const filterHint =
          filterInfo && Number.isFinite(filterInfo.before) && Number.isFinite(filterInfo.after)
            ? ` 候选过滤：${filterInfo.before} -> ${filterInfo.after}`
            : "";
        this.setStatus(`推荐完成，共 ${this.rows.length} 个候选作物。${filterHint}`, false);

        if (this.rows.length) {
          this.selectedCrop = payload?.env?.best_label || this.rows[0].crop;
          await this.$nextTick();
          this.renderDecisionSupportCharts();
          await this.loadCropVisuals();
        } else {
          this.selectedCrop = "";
          this.visuals = emptyVisualPayload();
          this.setVisualStatus("当前推荐结果为空，暂无可视化数据。", false);
          this.renderDecisionSupportCharts();
          this.renderVisualCharts();
        }
      } catch (err) {
        this.setStatus(`推荐失败：${err.message}`, true);
      } finally {
        this.loading = false;
        await this.refreshUi();
      }
    }
  },
  async mounted() {
    try {
      await initProtectedPage();
      await this.fillDefault(false);
      this.initVisualCharts();
      this.renderDecisionSupportCharts();
      this.renderVisualCharts();
      if (!this.statusError) {
        this.setStatus("页面已就绪，可直接生成推荐。", false);
      }
    } catch (err) {
      this.setStatus(`初始化失败：${err.message}`, true);
    } finally {
      await this.refreshUi();
    }
  },
  beforeUnmount() {
    this.disposeVisualCharts();
  }
}).mount("#recommendApp");





