import { initProtectedPage, notify, refreshInteractions } from "./公共.js";
import { fetchStoreSummary } from "./决策接口.js";
import { cardEmpty, escapeHtml } from "./决策界面.js";
import { cartCount, hydrateCart, loadCart, saveCart } from "./商城购物车.js";

const state = {
  products: [],
  keyword: "",
  category: "全部",
  cart: loadCart()
};

function categories() {
  return ["全部", ...new Set(state.products.map((item) => item.crop_label).filter(Boolean))];
}

function filteredProducts() {
  const keyword = state.keyword.trim().toLowerCase();
  return state.products.filter((item) => {
    const categoryOk = state.category === "全部" || item.crop_label === state.category;
    if (!categoryOk) return false;
    if (!keyword) return true;
    const haystack = [item.title, item.crop_label, item.shop_name, ...(item.tags || [])]
      .join(" ")
      .toLowerCase();
    return haystack.includes(keyword);
  });
}

function goDetail(id) {
  window.location.assign(`/store/product/${encodeURIComponent(id)}`);
}

function renderCategories() {
  const node = document.getElementById("storeCategoryChips");
  if (!node) return;
  node.innerHTML = categories()
    .map(
      (label) => `
        <button class="quick-chip ${label === state.category ? "is-active" : ""}" type="button" data-category="${escapeHtml(label)}">
          ${escapeHtml(label)}
        </button>
      `
    )
    .join("");
  node.querySelectorAll("[data-category]").forEach((button) => {
    button.addEventListener("click", () => {
      state.category = button.dataset.category || "全部";
      renderList();
      renderCategories();
    });
  });
}

function renderList() {
  const container = document.getElementById("storeBundleList");
  const metaNode = document.getElementById("storeProductMeta");
  const cartNode = document.getElementById("storeCartBadge");
  if (!container || !metaNode || !cartNode) return;

  const rows = filteredProducts();
  metaNode.textContent = `共 ${rows.length} 件商品`;
  cartNode.textContent = `购物车 ${cartCount(state.cart)} 件`;

  container.innerHTML = rows.length
    ? rows
        .map(
          (item) => `
            <article class="marketplace-card interactive-card" data-product-id="${escapeHtml(item.id)}" tabindex="0" role="link" aria-label="查看 ${escapeHtml(item.title || "商品")}">
              <div class="marketplace-card-media">
                <img src="${escapeHtml(item.image || "")}" alt="${escapeHtml(item.image_alt || item.title || "商品图片")}" loading="lazy">
              </div>
              <div class="marketplace-card-body">
                <h3>${escapeHtml(item.title || "")}</h3>
                <p class="marketplace-card-copy">${escapeHtml(item.reason || "")}</p>
                <div class="marketplace-card-tags">
                  ${(item.tags || []).slice(0, 3).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}
                </div>
                <div class="marketplace-price-row">
                  <strong>${escapeHtml(item.price || "-")}</strong>
                  <span>${escapeHtml(item.market_price || "")}</span>
                </div>
                <div class="marketplace-subline">${escapeHtml(item.coupon_text || "")}</div>
                <div class="marketplace-meta">
                  <span>${escapeHtml(item.crop_label || "商品")}</span>
                  <span>${escapeHtml(item.sold || "现货")}</span>
                  <span>${escapeHtml(item.origin || "")}</span>
                </div>
                <div class="marketplace-shop-row">
                  <div>
                    <strong>${escapeHtml(item.shop_name || item.seller || "店铺")}</strong>
                    <p>${escapeHtml(item.shop_badge || "")} ${item.rating ? `· 评分 ${escapeHtml(item.rating)}` : ""}</p>
                  </div>
                </div>
              </div>
            </article>
          `
        )
        .join("")
    : cardEmpty("没有匹配到商品。");

  container.querySelectorAll("[data-product-id]").forEach((card) => {
    const openDetail = () => goDetail(card.dataset.productId || "");
    card.addEventListener("click", openDetail);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openDetail();
      }
    });
  });
}

async function main() {
  await initProtectedPage();
  try {
    const data = await fetchStoreSummary();
    state.products = data.bundles || [];
    state.cart = saveCart(hydrateCart(loadCart(), state.products));

    document.getElementById("storeSummaryTitle").textContent = data.headline || "农资小商城";
    document.getElementById("storeSummaryNote").textContent = data.subheadline || "";

    document.getElementById("storeSearchInput")?.addEventListener("input", (event) => {
      state.keyword = event.target.value || "";
      renderList();
    });

    window.addEventListener("storage", () => {
      state.cart = loadCart();
      renderList();
    });

    renderCategories();
    renderList();
    refreshInteractions(document);
  } catch (error) {
    notify(error.message || "商城页面加载失败", "error");
  }
}

main();
