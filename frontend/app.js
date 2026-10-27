/* MELB/VALUATION — frontend controller.
   Loads model metadata, keeps the form pre-filled with realistic suburb
   defaults, and sends the property to POST /api/predict. */

const TYPE_LABELS = { h: "House", t: "Townhouse", u: "Unit / Apartment" };
const METHOD_LABELS = {
  S: "Sold",
  SP: "Sold prior to auction",
  PI: "Passed in",
  VB: "Vendor bid",
  SA: "Sold after auction",
};

const $ = (id) => document.getElementById(id);
const money = (n) =>
  new Intl.NumberFormat("en-AU", {
    style: "currency",
    currency: "AUD",
    maximumFractionDigits: 0,
  }).format(n);

let METADATA = null;

/* ── boot ─────────────────────────────────────────────── */
async function boot() {
  await Promise.all([checkHealth(), loadMetadata()]);
  $("predict-form").addEventListener("submit", onSubmit);
}

async function checkHealth() {
  const pill = $("status-pill");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    pill.classList.add("ok");
    pill.innerHTML = `<span class="dot"></span> api online`;
    $("model-tag").textContent = `model · ${data.model} · ${data.features} features`;
  } catch (err) {
    pill.classList.add("bad");
    pill.innerHTML = `<span class="dot"></span> api offline`;
  }
}

async function loadMetadata() {
  const res = await fetch("/api/metadata");
  METADATA = await res.json();
  const { options, defaults, metrics } = METADATA;

  // Hero statistics.
  $("stat-r2").textContent = (metrics.r2 * 100).toFixed(1) + "%";
  $("stat-mae").textContent = money(metrics.mae);
  $("stat-mape").textContent = metrics.mape.toFixed(1) + "%";
  $("stat-rows").textContent = Number(metrics.records || 13580).toLocaleString();

  // Suburb autocomplete.
  const list = $("suburb-list");
  list.innerHTML = options.Suburb
    .map((s) => `<option value="${s}"></option>`)
    .join("");

  fillSelect("f-Regionname", options.Regionname, (v) => v);
  fillSelect("f-CouncilArea", options.CouncilArea, (v) => v);
  fillSelect("f-SellerG", options.SellerG, (v) => v);
  fillSelect("f-Type", options.Type, (v) => TYPE_LABELS[v] || v);
  fillSelect("f-Method", options.Method, (v) => METHOD_LABELS[v] || v);

  // Global numeric defaults.
  for (const [key, value] of Object.entries(defaults)) {
    const el = $("f-" + key);
    if (el && el.tagName === "INPUT" && el.type !== "date" && !el.value) {
      el.value = Math.round(value * 100) / 100;
    }
  }
  // Categorical defaults (selects default to their first option otherwise).
  $("f-Type").value = defaults.Type;
  $("f-Method").value = defaults.Method;
  $("f-SellerG").value = options.SellerG.includes("Biggin") ? "Biggin" : options.SellerG[0];

  // Pre-fill with the first suburb so the form is immediately usable.
  const first = options.Suburb[0];
  $("f-Suburb").value = first;
  applySuburbDefaults(first);

  $("f-Suburb").addEventListener("change", (e) =>
    applySuburbDefaults(e.target.value.trim())
  );
}

function fillSelect(id, values, labeler) {
  const el = $(id);
  el.innerHTML = values
    .map((v) => `<option value="${v}">${labeler(v)}</option>`)
    .join("");
}

function applySuburbDefaults(suburb) {
  const stats = METADATA.suburb_stats[suburb];
  if (!stats) return;
  const set = (id, value) => {
    const el = $(id);
    if (el && value != null) el.value = Math.round(value * 100) / 100;
  };
  set("f-Distance", stats.median_distance);
  set("f-Postcode", stats.median_postcode);
  set("f-Lattitude", stats.median_lat);
  set("f-Longtitude", stats.median_lon);
  set("f-Propertycount", stats.median_propertycount);
  set("f-Rooms", stats.median_rooms);
  set("f-Bedroom2", stats.median_bedroom2);
  set("f-Bathroom", stats.median_bathroom);
  set("f-Car", stats.median_car);
  set("f-Landsize", stats.median_land);
  set("f-BuildingArea", stats.median_building);
  if (stats.council) $("f-CouncilArea").value = stats.council;
  if (stats.region) $("f-Regionname").value = stats.region;
}

/* ── submit ───────────────────────────────────────────── */
function collect() {
  const form = $("predict-form");
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    if (value !== "") payload[key] = value;
  }
  return payload;
}

async function onSubmit(event) {
  event.preventDefault();
  const suburb = $("f-Suburb").value.trim();
  if (!suburb) {
    showError("Please choose a suburb from the list.");
    return;
  }

  setState("loading");
  $("submit-btn").disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong while scoring the property.");
  } finally {
    $("submit-btn").disabled = false;
  }
}

/* ── render ───────────────────────────────────────────── */
function setState(state) {
  $("result-empty").hidden = state !== "empty";
  $("result-loading").hidden = state !== "loading";
  $("result-error").hidden = state !== "error";
  $("result-body").hidden = state !== "ready";
}

function showError(message) {
  setState("error");
  $("result-error").textContent = message;
}

function renderResult(data) {
  setState("ready");
  const body = $("result-body");
  body.classList.remove("reveal");
  void body.offsetWidth; // restart reveal animation
  body.classList.add("reveal");

  $("result-price").textContent = money(data.predicted_price);
  $("result-range").textContent =
    `${money(data.range.low)}  —  ${money(data.range.high)}`;

  const { low, high } = data.range;
  const span = Math.max(high - low, 1);
  const pct = Math.min(100, Math.max(0, ((data.predicted_price - low) / span) * 100));
  $("band-fill").style.width = "100%";
  $("band-marker").style.left = pct + "%";
  $("band-low").textContent = money(low);
  $("band-high").textContent = money(high);

  $("fact-per-room").textContent = money(data.per_room);
  $("fact-comp").textContent = data.comparable_median
    ? money(data.comparable_median)
    : "—";
  $("fact-conf").textContent = (data.confidence * 100).toFixed(1) + "%";

  const tbody = document.querySelector("#comp-table tbody");
  tbody.innerHTML = (data.comparables || [])
    .map(
      (c) => `<tr>
        <td>${c.address}</td>
        <td>${TYPE_LABELS[c.type] || c.type}</td>
        <td>${c.rooms}</td>
        <td class="ta-r">${money(c.price)}</td>
      </tr>`
    )
    .join("");
}

document.addEventListener("DOMContentLoaded", boot);
