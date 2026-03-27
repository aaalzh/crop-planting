import { initProtectedPage, notify, refreshInteractions } from "./公共.js";
import { fetchStoreSummary } from "./决策接口.js";
import { cardEmpty, escapeHtml } from "./决策界面.js";
import {
  addCartItem,
  cartCount,
  cartLineShipping,
  cartLineSubtotal,
  cartSummary,
  clearCart,
  createCartItem,
  hydrateCart,
  itemKey,
  loadCart,
  removeCartItem,
  saveCart,
  updateCartItemQuantity
} from "./商城购物车.js";

const LAST_ORDER_KEY = "store_last_order_v1";

const state = {
  products: [],
  product: null,
  activeImage: 0,
  activeVariant: "",
  quantity: 1,
  activeTab: "detail",
  cart: loadCart(),
  cartOpen: false,
  lastOrder: loadLastOrder()
};

function parseProductId() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return decodeURIComponent(parts[parts.length - 1] || "");
}

function loadLastOrder() {
  try {
    const raw = window.localStorage.getItem(LAST_ORDER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveLastOrder(order) {
  state.lastOrder = order;
  window.localStorage.setItem(LAST_ORDER_KEY, JSON.stringify(order));
}

function syncCart() {
  state.cart = saveCart(hydrateCart(state.cart, state.products));
}

function formatMoney(value) {
  return `¥${Number(value || 0).toLocaleString("zh-CN")}`;
}

function selectedVariant() {
  return (state.product?.variants || []).find((item) => item.id === state.activeVariant) || state.product?.variants?.[0] || null;
}

function currentImages() {
  const gallery = state.product?.gallery || [];
  return gallery.length
    ? gallery
    : [{ image: state.product?.image || "", alt: state.product?.image_alt || state.product?.title || "商品图片" }];
}

function totals() {
  const variant = selectedVariant();
  const subtotal = Number(variant?.price || state.product?.unit_price || 0) * state.quantity;
  const shipping = subtotal >= 199 ? 0 : Number(state.product?.shipping_fee || 0);
  return {
    subtotal,
    shipping,
    total: subtotal + shipping
  };
}

function currentSelectionKey() {
  return itemKey(state.product?.id, selectedVariant()?.id || "");
}

function currentSelectionCartQty() {
  const current = state.cart.items.find((item) => item.key === currentSelectionKey());
  return Number(current?.quantity || 0);
}

function createSelectionItem() {
  return createCartItem(state.product, selectedVariant(), state.quantity);
}

function renderFloatingCart(summary = cartSummary(state.cart)) {
  const button = document.getElementById("storeFloatingCartBtn");
  if (!button) return;
  const badge = button.querySelector(".store-cart-toggle-badge");
  if (badge) {
    badge.textContent = summary.quantity > 99 ? "99+" : String(summary.quantity);
    badge.hidden = summary.quantity <= 0;
  }
  button.setAttribute("aria-expanded", String(state.cartOpen));
  button.setAttribute("aria-label", `购物车，${summary.quantity}件，点击${state.cartOpen ? "收起" : "展开"}`);
}

function setCartOpen(next) {
  state.cartOpen = Boolean(next);
  renderFloatingCart();
  renderCartPopover();
}

function goBackToStore() {
  try {
    const referrer = document.referrer ? new URL(document.referrer) : null;
    if (referrer && referrer.origin === window.location.origin && referrer.pathname.startsWith("/store")) {
      window.history.back();
      return;
    }
  } catch {
    // ignore invalid referrer
  }
  window.location.assign("/store");
}

function submitOrder(items, mode) {
  const summary = cartSummary({ items });
  if (!summary.quantity) {
    notify("当前没有可下单的商品。", "info");
    return;
  }
  saveLastOrder({
    id: `ORD-${Date.now()}`,
    mode,
    createdAt: new Date().toLocaleString("zh-CN", { hour12: false }),
    items: items.map((item) => ({
      title: item.title,
      variantLabel: item.variantLabel,
      quantity: item.quantity,
      unitLabel: item.unitLabel,
      amount: cartLineSubtotal(item)
    })),
    summary
  });
  if (mode === "cart") {
    state.cart = clearCart();
    syncCart();
  }
  setCartOpen(true);
  renderDetail();
  renderRelated();
  notify(`已生成本地订单，共 ${summary.quantity} 件，应付 ${formatMoney(summary.total)}`, "success");
}

function addToCart() {
  if (!state.product) return;
  state.cart = addCartItem(state.cart, createSelectionItem());
  syncCart();
  renderDetail();
  renderRelated();
  notify("已加入购物车", "success");
}

function buyNow() {
  if (!state.product) return;
  submitOrder([createSelectionItem()], "direct");
}

function checkoutCart() {
  if (!state.cart.items.length) {
    notify("购物车还是空的。", "info");
    return;
  }
  submitOrder(state.cart.items, "cart");
}

function renderRelated() {
  const node = document.getElementById("storeRelatedList");
  if (!node || !state.product) return;
  const rows = state.products.filter((item) => item.id !== state.product.id).slice(0, 3);
  node.innerHTML = rows.length
    ? rows
        .map(
          (item) => `
            <article class="marketplace-card interactive-card" data-related-id="${escapeHtml(item.id)}" tabindex="0" role="link" aria-label="查看 ${escapeHtml(item.title || "商品")}">
              <div class="marketplace-card-media">
                <img src="${escapeHtml(item.image || "")}" alt="${escapeHtml(item.image_alt || item.title || "商品图片")}" loading="lazy">
              </div>
              <div class="marketplace-card-body">
                <h3>${escapeHtml(item.title || "")}</h3>
                <p class="marketplace-card-copy">${escapeHtml(item.reason || "")}</p>
                <div class="marketplace-price-row">
                  <strong>${escapeHtml(item.price || "-")}</strong>
                  <span>${escapeHtml(item.market_price || "")}</span>
                </div>
              </div>
            </article>
          `
        )
        .join("")
    : cardEmpty("暂时没有更多推荐商品。");

  node.querySelectorAll("[data-related-id]").forEach((card) => {
    const openDetail = () => {
      window.location.assign(`/store/product/${encodeURIComponent(card.dataset.relatedId || "")}`);
    };
    card.addEventListener("click", openDetail);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openDetail();
      }
    });
  });
  refreshInteractions(node);
}

function renderTabPanel() {
  const variant = selectedVariant();
  if (!state.product) return "";
  if (state.activeTab === "params") {
    return `
      <div class="store-tab-panel">
        <div class="store-param-grid">
          ${(state.product.specs || [])
            .map(
              (row) => `
                <div class="store-param-card">
                  <span>${escapeHtml(row.label || "")}</span>
                  <strong>${escapeHtml(row.value || "")}</strong>
                </div>
              `
            )
            .join("")}
        </div>
      </div>
    `;
  }
  if (state.activeTab === "gallery") {
    return `
      <div class="store-tab-panel store-gallery-panel">
        ${currentImages()
          .map(
            (item) => `
              <figure class="store-gallery-card">
                <img src="${escapeHtml(item.image || "")}" alt="${escapeHtml(item.alt || "商品图片")}" loading="lazy">
                <figcaption>${escapeHtml(item.alt || "")}</figcaption>
              </figure>
            `
          )
          .join("")}
      </div>
    `;
  }
  if (state.activeTab === "service") {
    return `
      <div class="store-tab-panel">
        <ul class="text-list">
          ${(state.product.delivery_lines || []).map((row) => `<li>${escapeHtml(row)}</li>`).join("")}
          ${(state.product.service || []).map((row) => `<li>${escapeHtml(row)}</li>`).join("")}
          ${variant?.note ? `<li>${escapeHtml(variant.note)}</li>` : ""}
        </ul>
        <div class="store-detail-license">
          <span>图片署名：${escapeHtml(state.product.credit || "可再用素材")}</span>
          <div class="store-source-links">
            <a class="text-link" href="${escapeHtml(state.product.source_url || "#")}" target="_blank" rel="noreferrer">图片来源</a>
            <a class="text-link" href="${escapeHtml(state.product.license_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(state.product.license_label || "许可信息")}</a>
          </div>
        </div>
      </div>
    `;
  }
  return `
    <div class="store-tab-panel">
      ${(state.product.description || []).map((row) => `<p>${escapeHtml(row)}</p>`).join("")}
      <div class="store-rich-highlight">
        <h3>购买建议</h3>
        <ul class="text-list">${(state.product.overview || []).map((row) => `<li>${escapeHtml(row)}</li>`).join("")}</ul>
      </div>
    </div>
  `;
}

function renderCartPopover() {
  const root = document.getElementById("storeCartPopoverRoot");
  if (!root) return;

  const summary = cartSummary(state.cart);
  renderFloatingCart(summary);

  if (!state.cartOpen) {
    root.innerHTML = "";
    return;
  }

  const cartItems = state.cart.items;
  const lastOrderItems = (state.lastOrder?.items || [])
    .map(
      (item) => `
        <li>${escapeHtml(item.title || "")} ${escapeHtml(item.variantLabel || "")} × ${escapeHtml(String(item.quantity || 0))}${escapeHtml(item.unitLabel || "件")}</li>
      `
    )
    .join("");

  root.innerHTML = `
    <div class="store-cart-popover-shell">
      <aside class="card store-cart-popover" role="dialog" aria-modal="false" aria-label="购物车明细">
        <div class="store-cart-popover-head">
          <div>
            <p class="eyebrow">购物车</p>
            <h3>已选清单</h3>
          </div>
          <button class="store-cart-close" type="button" id="storeCartCloseBtn">关闭</button>
        </div>

        <div class="store-cart-popover-body">
          ${
            cartItems.length
              ? `
                <div class="store-cart-list">
                  ${cartItems
                    .map(
                      (item) => `
                        <article class="store-cart-item">
                          <div class="store-cart-line">
                            <div>
                              <strong>${escapeHtml(item.title || "商品")}</strong>
                              <p>${escapeHtml(item.variantLabel || "默认规格")} · ${escapeHtml(item.shopName || "店铺")}</p>
                            </div>
                            <button class="text-link store-inline-action" type="button" data-cart-remove="${escapeHtml(item.key)}">移除</button>
                          </div>
                          <div class="store-cart-line">
                            <strong>${formatMoney(item.unitPrice || 0)}</strong>
                            <span class="muted">小计 ${formatMoney(cartLineSubtotal(item))} · ${cartLineShipping(item) ? `运费 ${formatMoney(cartLineShipping(item))}` : "免运费"}</span>
                          </div>
                          <div class="store-cart-line">
                            <div class="qty-stepper qty-stepper-sm">
                              <button class="qty-btn" type="button" data-cart-step="${escapeHtml(item.key)}:-1">-</button>
                              <span>${escapeHtml(String(item.quantity || 0))}</span>
                              <button class="qty-btn" type="button" data-cart-step="${escapeHtml(item.key)}:1">+</button>
                            </div>
                            <span class="muted">库存 ${escapeHtml(String(item.stock || 0))} ${escapeHtml(item.unitLabel || "件")}</span>
                          </div>
                        </article>
                      `
                    )
                    .join("")}
                </div>

                <div class="store-cart-summary">
                  <div class="store-summary-row"><span>商品件数</span><strong>${summary.quantity} 件</strong></div>
                  <div class="store-summary-row"><span>商品小计</span><strong>${formatMoney(summary.subtotal)}</strong></div>
                  <div class="store-summary-row"><span>预计运费</span><strong>${formatMoney(summary.shipping)}</strong></div>
                  <div class="store-summary-row is-total"><span>购物车总价</span><strong>${formatMoney(summary.total)}</strong></div>
                  <p class="store-checkout-note">本页为前端本地下单演示，结算后会清空本地购物车。</p>
                </div>

                <div class="store-detail-actions">
                  <button class="btn btn-secondary" type="button" id="storeClearCartBtn">清空购物车</button>
                  <button class="btn btn-primary" type="button" id="storeCheckoutCartBtn">结算购物车</button>
                </div>
              `
              : `<div class="store-cart-empty">还没有加入商品。先选规格和数量，再放进购物车。</div>`
          }

          ${
            state.lastOrder
              ? `
                <div id="storeLastOrderCard" class="store-rich-highlight store-cart-last-order">
                  <h3>最近一次下单</h3>
                  <p>订单号 ${escapeHtml(state.lastOrder.id || "-")} · ${escapeHtml(state.lastOrder.createdAt || "")}</p>
                  <ul class="text-list">${lastOrderItems}</ul>
                  <div class="store-summary-row"><span>订单总价</span><strong>${formatMoney(state.lastOrder.summary?.total || 0)}</strong></div>
                </div>
              `
              : ""
          }
        </div>
      </aside>
    </div>
  `;

  root.querySelector("#storeCartCloseBtn")?.addEventListener("click", () => setCartOpen(false));
  root.querySelectorAll("[data-cart-step]").forEach((button) => {
    button.addEventListener("click", () => {
      const [key, step] = String(button.dataset.cartStep || "").split(":");
      const current = state.cart.items.find((item) => item.key === key);
      state.cart = updateCartItemQuantity(state.cart, key, Number(current?.quantity || 0) + Number(step || 0));
      syncCart();
      renderDetail();
      renderRelated();
    });
  });
  root.querySelectorAll("[data-cart-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      state.cart = removeCartItem(state.cart, button.dataset.cartRemove || "");
      syncCart();
      renderDetail();
      renderRelated();
    });
  });
  root.querySelector("#storeClearCartBtn")?.addEventListener("click", () => {
    state.cart = clearCart();
    syncCart();
    renderDetail();
    renderRelated();
  });
  root.querySelector("#storeCheckoutCartBtn")?.addEventListener("click", checkoutCart);
  refreshInteractions(root);
}

function renderDetail() {
  const node = document.getElementById("storeDetailRoot");
  if (!node || !state.product) return;
  const variant = selectedVariant();
  const total = totals();
  const images = currentImages();
  const activeImage = images[state.activeImage] || images[0];
  const breadcrumb = document.getElementById("storeDetailBreadcrumb");
  const alreadyInCart = currentSelectionCartQty();

  if (breadcrumb) breadcrumb.textContent = state.product.title || "商品详情";

  node.innerHTML = `
    <section class="store-product-page-shell">
      <div class="store-visual-column">
        <div class="store-thumb-strip">
          ${images
            .map(
              (item, index) => `
                <button class="store-thumb ${index === state.activeImage ? "is-active" : ""}" type="button" data-thumb-index="${index}">
                  <img src="${escapeHtml(item.image || "")}" alt="${escapeHtml(item.alt || "商品缩略图")}" loading="lazy">
                </button>
              `
            )
            .join("")}
        </div>
        <div class="store-preview-frame">
          <img src="${escapeHtml(activeImage.image || "")}" alt="${escapeHtml(activeImage.alt || state.product.title || "商品大图")}" loading="lazy">
        </div>
      </div>

      <div class="store-detail-column">
        <div class="store-detail-topline">
          <button class="text-link store-back-link" type="button" id="storeDetailBackBtn">返回上一页</button>
        </div>

        <div class="store-shop-head">
          <div>
            <p class="eyebrow">${escapeHtml(state.product.shop_badge || "品牌店铺")}</p>
            <h2>${escapeHtml(state.product.shop_name || state.product.seller || "店铺")}</h2>
          </div>
          <span class="tone-pill tone-safe">评分 ${escapeHtml(state.product.rating || "4.8")}</span>
        </div>

        <div class="store-product-title">
          <h1>${escapeHtml(state.product.title || "")}</h1>
          <p>${escapeHtml(state.product.reason || "")}</p>
        </div>

        <div class="store-price-board">
          <div class="store-price-main">
            <strong>${formatMoney(variant?.price || state.product.unit_price || 0)}</strong>
            <span>${escapeHtml(state.product.market_price || "")}</span>
          </div>
          <div class="store-price-note">${escapeHtml(state.product.coupon_text || "支持优惠组合")}</div>
          <div class="store-service-row">
            ${(state.product.delivery_lines || []).map((row) => `<span>${escapeHtml(row)}</span>`).join("")}
          </div>
        </div>

        <div class="store-option-block">
          <p class="store-option-title">规格分类</p>
          <div class="store-variant-list">
            ${(state.product.variants || [])
              .map(
                (item) => `
                  <button class="store-variant-btn ${item.id === state.activeVariant ? "is-active" : ""}" type="button" data-variant-id="${escapeHtml(item.id)}">
                    <strong>${escapeHtml(item.label || "")}</strong>
                    <span>${formatMoney(item.price || 0)}</span>
                    <small>${escapeHtml(item.note || "")}</small>
                  </button>
                `
              )
              .join("")}
          </div>
        </div>

        <div class="store-option-block">
          <p class="store-option-title">数量</p>
          <div class="store-qty-row">
            <div class="qty-stepper">
              <button class="qty-btn" type="button" data-qty-step="-1">-</button>
              <span>${state.quantity}</span>
              <button class="qty-btn" type="button" data-qty-step="1">+</button>
            </div>
            <span class="muted">库存 ${escapeHtml(String(state.product.stock || 0))} ${escapeHtml(state.product.unit_label || "件")}</span>
          </div>
        </div>

        <div class="store-calc-card">
          <div class="store-summary-row"><span>当前规格</span><strong>${escapeHtml(variant?.label || state.product.spec || "")}</strong></div>
          <div class="store-summary-row"><span>已在购物车</span><strong>${alreadyInCart} ${escapeHtml(state.product.unit_label || "件")}</strong></div>
          <div class="store-summary-row"><span>商品小计</span><strong>${formatMoney(total.subtotal)}</strong></div>
          <div class="store-summary-row"><span>预计运费</span><strong>${formatMoney(total.shipping)}</strong></div>
          <div class="store-summary-row is-total"><span>应付总价</span><strong>${formatMoney(total.total)}</strong></div>
        </div>

        <div class="store-detail-actions">
          <button class="btn btn-secondary" type="button" id="storeDetailAddCartBtn">加入购物车</button>
          <button class="btn btn-primary" type="button" id="storeDetailBuyBtn">立即购买</button>
        </div>
      </div>
    </section>

    <section class="store-detail-tabs">
      <div class="store-tab-buttons">
        <button class="preview-card-btn ${state.activeTab === "detail" ? "is-active" : ""}" type="button" data-tab-id="detail">图文详情</button>
        <button class="preview-card-btn ${state.activeTab === "params" ? "is-active" : ""}" type="button" data-tab-id="params">参数信息</button>
        <button class="preview-card-btn ${state.activeTab === "gallery" ? "is-active" : ""}" type="button" data-tab-id="gallery">图集</button>
        <button class="preview-card-btn ${state.activeTab === "service" ? "is-active" : ""}" type="button" data-tab-id="service">发货服务</button>
      </div>
      ${renderTabPanel()}
    </section>
  `;

  node.querySelector("#storeDetailBackBtn")?.addEventListener("click", goBackToStore);
  node.querySelectorAll("[data-thumb-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeImage = Number(button.dataset.thumbIndex || 0);
      renderDetail();
    });
  });
  node.querySelectorAll("[data-variant-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeVariant = button.dataset.variantId || state.activeVariant;
      renderDetail();
    });
  });
  node.querySelectorAll("[data-qty-step]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = state.quantity + Number(button.dataset.qtyStep || 0);
      state.quantity = Math.max(1, Math.min(Number(state.product.stock || 99), next));
      renderDetail();
    });
  });
  node.querySelectorAll("[data-tab-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = button.dataset.tabId || "detail";
      renderDetail();
    });
  });
  node.querySelector("#storeDetailAddCartBtn")?.addEventListener("click", addToCart);
  node.querySelector("#storeDetailBuyBtn")?.addEventListener("click", buyNow);
  refreshInteractions(node);
  renderCartPopover();
}

async function main() {
  await initProtectedPage();
  try {
    const data = await fetchStoreSummary();
    state.products = data.bundles || [];
    state.product = state.products.find((item) => item.id === parseProductId()) || null;
    if (!state.product) {
      notify("商品不存在，已返回商城列表。", "info");
      window.location.assign("/store");
      return;
    }

    state.activeVariant = state.product.variants?.[0]?.id || "";
    syncCart();

    document.getElementById("storeBackBtn")?.addEventListener("click", goBackToStore);
    document.getElementById("storeFloatingCartBtn")?.addEventListener("click", (event) => {
      event.stopPropagation();
      setCartOpen(!state.cartOpen);
    });

    document.addEventListener("click", (event) => {
      if (!state.cartOpen) return;
      const target = event.target;
      const button = document.getElementById("storeFloatingCartBtn");
      const popover = document.getElementById("storeCartPopoverRoot");
      if (button?.contains(target) || popover?.contains(target)) return;
      setCartOpen(false);
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.cartOpen) {
        setCartOpen(false);
      }
    });

    renderFloatingCart();
    renderDetail();
    renderRelated();
    refreshInteractions(document);

    window.addEventListener("storage", () => {
      state.cart = loadCart();
      state.lastOrder = loadLastOrder();
      syncCart();
      renderDetail();
      renderRelated();
    });
  } catch (error) {
    notify(error.message || "商品详情加载失败", "error");
  }
}

main();
