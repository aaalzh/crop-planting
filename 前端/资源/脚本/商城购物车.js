const CART_KEY = "store_cart_v2";
const CART_VERSION = 3;

function buildCart(items = []) {
  return {
    version: CART_VERSION,
    items
  };
}

function clampQuantity(value) {
  return Math.max(0, Number.parseInt(String(value), 10) || 0);
}

export function itemKey(productId, variantId = "") {
  return `${String(productId || "").trim()}::${String(variantId || "").trim()}`;
}

function normalizeItem(item = {}) {
  const productId = String(item.productId || item.product_id || "").trim();
  const variantId = String(item.variantId || item.variant_id || "").trim();
  const quantity = clampQuantity(item.quantity);
  return {
    key: item.key || itemKey(productId, variantId),
    productId,
    variantId,
    title: String(item.title || "").trim(),
    variantLabel: String(item.variantLabel || "").trim(),
    quantity,
    unitPrice: Number(item.unitPrice || 0) || 0,
    marketPrice: String(item.marketPrice || "").trim(),
    image: String(item.image || "").trim(),
    imageAlt: String(item.imageAlt || "").trim(),
    shopName: String(item.shopName || "").trim(),
    unitLabel: String(item.unitLabel || "件").trim(),
    shippingFee: Number(item.shippingFee || 0) || 0,
    stock: Number(item.stock || 0) || 0
  };
}

function normalizeCart(raw) {
  if (raw && Array.isArray(raw.items)) {
    return buildCart(
      raw.items
        .map((item) => normalizeItem(item))
        .filter((item) => item.productId && item.quantity > 0)
    );
  }
  if (raw && typeof raw === "object") {
    return buildCart(
      Object.entries(raw)
        .map(([productId, quantity]) => normalizeItem({ productId, quantity }))
        .filter((item) => item.productId && item.quantity > 0)
    );
  }
  return buildCart();
}

export function loadCart() {
  try {
    const raw = window.localStorage.getItem(CART_KEY);
    return normalizeCart(raw ? JSON.parse(raw) : null);
  } catch {
    return buildCart();
  }
}

export function saveCart(cart) {
  const next = normalizeCart(cart);
  window.localStorage.setItem(CART_KEY, JSON.stringify(next));
  return next;
}

function stockLimit(item) {
  const stock = Number(item.stock);
  return Number.isFinite(stock) && stock > 0 ? stock : 9999;
}

export function cartCount(cart) {
  return normalizeCart(cart).items.reduce((sum, item) => sum + item.quantity, 0);
}

export function cartDistinctCount(cart) {
  return normalizeCart(cart).items.length;
}

export function cartLineSubtotal(item) {
  return (Number(item.unitPrice || 0) || 0) * clampQuantity(item.quantity);
}

export function cartLineShipping(item) {
  const subtotal = cartLineSubtotal(item);
  const shippingFee = Number(item.shippingFee || 0) || 0;
  return subtotal >= 199 ? 0 : shippingFee;
}

export function cartSummary(cart) {
  const normalized = normalizeCart(cart);
  const subtotal = normalized.items.reduce((sum, item) => sum + cartLineSubtotal(item), 0);
  const shipping = normalized.items.reduce((sum, item) => sum + cartLineShipping(item), 0);
  return {
    itemCount: cartDistinctCount(normalized),
    quantity: cartCount(normalized),
    subtotal,
    shipping,
    total: subtotal + shipping
  };
}

export function createCartItem(product, variant, quantity) {
  const resolvedProduct = product || {};
  const resolvedVariant = variant || {};
  return normalizeItem({
    productId: resolvedProduct.id,
    variantId: resolvedVariant.id || "",
    title: resolvedProduct.title || "商品",
    variantLabel: resolvedVariant.label || resolvedProduct.spec || "默认规格",
    quantity,
    unitPrice: resolvedVariant.price || resolvedProduct.unit_price || 0,
    marketPrice: resolvedProduct.market_price || "",
    image: resolvedProduct.image || "",
    imageAlt: resolvedProduct.image_alt || resolvedProduct.title || "商品图片",
    shopName: resolvedProduct.shop_name || resolvedProduct.seller || "店铺",
    unitLabel: resolvedProduct.unit_label || "件",
    shippingFee: resolvedProduct.shipping_fee || 0,
    stock: resolvedProduct.stock || 0
  });
}

export function addCartItem(cart, item) {
  const normalized = normalizeCart(cart);
  const nextItem = normalizeItem(item);
  if (!nextItem.productId || nextItem.quantity <= 0) {
    return normalized;
  }
  const index = normalized.items.findIndex((row) => row.key === nextItem.key);
  if (index < 0) {
    normalized.items.push({
      ...nextItem,
      quantity: Math.min(stockLimit(nextItem), nextItem.quantity)
    });
    return normalized;
  }
  const current = normalized.items[index];
  normalized.items[index] = {
    ...current,
    ...nextItem,
    quantity: Math.min(stockLimit({ ...current, ...nextItem }), current.quantity + nextItem.quantity)
  };
  return normalized;
}

export function updateCartItemQuantity(cart, key, quantity) {
  const normalized = normalizeCart(cart);
  normalized.items = normalized.items
    .map((item) => {
      if (item.key !== key) return item;
      return {
        ...item,
        quantity: Math.min(stockLimit(item), clampQuantity(quantity))
      };
    })
    .filter((item) => item.quantity > 0);
  return normalized;
}

export function removeCartItem(cart, key) {
  const normalized = normalizeCart(cart);
  normalized.items = normalized.items.filter((item) => item.key !== key);
  return normalized;
}

export function clearCart() {
  return buildCart();
}

export function hydrateCart(cart, products = []) {
  const normalized = normalizeCart(cart);
  const productMap = new Map((products || []).filter(Boolean).map((product) => [String(product.id || ""), product]));
  normalized.items = normalized.items.map((item) => {
    const product = productMap.get(item.productId);
    if (!product) return item;
    const variant = (product.variants || []).find((row) => row.id === item.variantId) || product.variants?.[0] || null;
    const hydrated = createCartItem(product, variant, item.quantity);
    return {
      ...item,
      ...hydrated,
      key: itemKey(item.productId, hydrated.variantId),
      quantity: Math.min(stockLimit({ ...item, stock: product.stock || item.stock }), item.quantity)
    };
  });
  return normalized;
}
