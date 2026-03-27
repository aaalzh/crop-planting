import time
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

URL = "https://agmarknet.ceda.ashoka.edu.in/"

NAV_TIMEOUT = 60_000
ACTION_TIMEOUT = 30_000

SCROLL_WAIT_BASE = 80
SCROLL_WAIT_SLOW = 160
IDLE_STOP = 8
MAX_ROUNDS = 1500

HEARTBEAT_EVERY = 1.0  # 每隔多少秒打印一次“还活着”


def _parse_date_any(s: str):
    if s is None:
        return None
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    return None


def wait_table_ready(page):
    # 不要只等 tr，有时表格先出来但 tr 还在加载
    page.wait_for_selector("table", timeout=ACTION_TIMEOUT)
    page.wait_for_selector("table tbody", timeout=ACTION_TIMEOUT)


def get_rows_fast(page):
    """
    在浏览器端一次性把 tbody 里的行读出来（避免逐格 inner_text 超慢卡死）
    """
    return page.evaluate("""
    () => {
      const table = document.querySelector("table");
      if (!table) return [];
      const rows = Array.from(table.querySelectorAll("tbody tr"));
      return rows.map(tr => Array.from(tr.querySelectorAll("td")).map(td => (td.innerText || "").trim()));
    }
    """)


def find_scroll_container(page):
    # 找到表格外围滚动容器（如果没有就滚 body）
    return page.evaluate_handle("""
    () => {
      const table = document.querySelector("table");
      if (!table) return document.body;
      let el = table;
      while (el && el !== document.body) {
        const style = window.getComputedStyle(el);
        const oy = style.overflowY;
        if (oy === 'auto' || oy === 'scroll') return el;
        el = el.parentElement;
      }
      return document.body;
    }
    """)


def scroll_down(page, scroll_handle):
    page.evaluate("(el) => { el.scrollTop = el.scrollHeight; }", scroll_handle)


def scrape_table_scroll(page, out_csv="prices_manual.csv"):
    wait_table_ready(page)

    scroll_handle = find_scroll_container(page)

    seen = set()
    idle = 0
    t0 = time.time()
    last_hb = t0

    print("\n🚀 开始爬取（实时进度）：")
    print("round=轮次 total=累计行数 +new=本轮新增 idle=连续无新增 elapsed=用时(s)\n")

    for r in range(1, MAX_ROUNDS + 1):
        # 心跳：防止你以为卡死
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            elapsed = int(now - t0)
            print(f"…仍在运行中（round={r} elapsed={elapsed}s total={len(seen)}）")
            last_hb = now

        rows = get_rows_fast(page)   # ✅ 快速抓取
        before = len(seen)

        for row in rows:
            if row:
                seen.add(tuple(row))

        after = len(seen)
        new_cnt = after - before

        if new_cnt == 0:
            idle += 1
        else:
            idle = 0

        elapsed = int(time.time() - t0)
        print(f"round={r:4d} | total={after:6d} | +new={new_cnt:4d} | idle={idle:2d} | elapsed={elapsed:4d}s")

        if idle >= IDLE_STOP:
            print(f"\n🛑 连续 {IDLE_STOP} 轮无新增，停止。")
            break

        # 滚动
        try:
            scroll_down(page, scroll_handle)
        except:
            page.mouse.wheel(0, 1600)

        # 自适应等待
        page.wait_for_timeout(SCROLL_WAIT_SLOW if new_cnt <= 1 else SCROLL_WAIT_BASE)

    print("\n✅ 抓取结束，开始整理数据并保存...")

    df = pd.DataFrame([list(x) for x in seen])

    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["date", "modal_price", "min_price", "max_price", "change"]

    # 清洗
    for c in ["modal_price", "min_price", "max_price", "change"]:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 排序
    if "date" in df.columns:
        df["date_dt"] = df["date"].apply(_parse_date_any)
        if df["date_dt"].notna().any():
            df = df.dropna(subset=["date_dt"]).sort_values("date_dt").drop(columns=["date_dt"])

    df = df.drop_duplicates()
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"💾 已保存：{out_csv} 行数={len(df)}")
    if len(df) > 0 and "date" in df.columns:
        print(f"📅 日期范围：{df['date'].min()} -> {df['date'].max()}")

    return df


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1500, "height": 900})
        page = context.new_page()
        page.set_default_timeout(ACTION_TIMEOUT)
        page.set_default_navigation_timeout(NAV_TIMEOUT)

        page.goto(URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)

        # 等筛选区出现
        page.wait_for_selector("text=Category", timeout=ACTION_TIMEOUT)

        print("\n✅ 网页已打开。你手动选好筛选条件后回到这里：")
        input("👉 按回车开始抓取...")

        out_csv = input("输出文件名（回车默认 prices_manual.csv）: ").strip()
        if not out_csv:
            out_csv = "prices_manual.csv"
        if not out_csv.lower().endswith(".csv"):
            out_csv += ".csv"

        print("⏳ 开始抓取...")
        scrape_table_scroll(page, out_csv=out_csv)

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
