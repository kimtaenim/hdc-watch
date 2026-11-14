// 거래량 이상 그래프: 60일 주가 선(이상·주의 날 표시)과, 고른 날의 30분 주가 선 + 거래량 막대.
// 주가와 거래량은 단위가 달라 한 그래프에 두 축으로 겹치지 않고, 같은 가로축으로 위아래에 따로 그린다.

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};
const won = (v) => (v == null ? "-" : `${Math.round(v).toLocaleString("ko-KR")}원`);
const num = (v) => (v == null ? "-" : Math.round(v).toLocaleString("ko-KR"));
const pct = (a, b) => (a && b ? `${b >= a ? "+" : ""}${(((b - a) / a) * 100).toFixed(1)}%` : "-");
const VERDICT_CLASS = { "이상": "v-alert", "주의": "v-watch", "평소": "v-normal" };

// 눈금: 보기 좋은 간격 3-5개
function ticks(lo, hi, n = 4) {
  const span = hi - lo || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function frame(w, h, pad) {
  return { w, h, pad, iw: w - pad.l - pad.r, ih: h - pad.t - pad.b };
}

function tooltipHost() {
  const wrap = el("div", { class: "chart" });
  const tip = el("div", { class: "tip", hidden: "" });
  wrap.append(tip);
  return { wrap, tip, show(x, y, html) {
    tip.replaceChildren(...html); tip.hidden = false;
    const W = wrap.clientWidth, tw = tip.offsetWidth;
    tip.style.left = `${Math.min(Math.max(0, x - tw / 2), W - tw)}px`; tip.style.top = `${Math.max(0, y - tip.offsetHeight - 10)}px`;
  }, hide() { tip.hidden = true; } };
}

// 60일 주가 선. days: [{date, close, alert, watch}], selected: 날짜
export function overviewChart(days, selected, onPick) {
  const F = frame(340, 150, { l: 44, r: 8, t: 10, b: 22 });
  const ys = days.map((d) => d.close).filter((v) => v != null);
  const lo = Math.min(...ys), hi = Math.max(...ys), padY = (hi - lo) * 0.08 || 10;
  const y0 = lo - padY, y1 = hi + padY;
  const X = (i) => F.pad.l + (days.length === 1 ? F.iw / 2 : (i / (days.length - 1)) * F.iw);
  const Y = (v) => F.pad.t + (1 - (v - y0) / (y1 - y0)) * F.ih;
  const s = svg("svg", { viewBox: `0 0 ${F.w} ${F.h}`, class: "plot", role: "img", "aria-label": "60일 주가와 이상·주의가 있던 날" });
  for (const t of ticks(y0, y1, 3)) {
    s.append(svg("line", { x1: F.pad.l, x2: F.w - F.pad.r, y1: Y(t), y2: Y(t), class: "grid" }),
      svg("text", { x: F.pad.l - 6, y: Y(t) + 3, class: "axis", "text-anchor": "end" }, document.createTextNode(num(t))));
  }
  for (const i of [0, Math.floor((days.length - 1) / 2), days.length - 1]) {
    s.append(svg("text", { x: X(i), y: F.h - 6, class: "axis", "text-anchor": i === 0 ? "start" : i === days.length - 1 ? "end" : "middle" },
      document.createTextNode(days[i].date.slice(5))));
  }
  const si = days.findIndex((d) => d.date === selected);
  if (si >= 0) s.append(svg("line", { x1: X(si), x2: X(si), y1: F.pad.t, y2: F.pad.t + F.ih, class: "sel-rule" }));
  const pts = days.map((d, i) => (d.close == null ? null : `${X(i)},${Y(d.close)}`)).filter(Boolean).join(" ");
  s.append(svg("polyline", { points: pts, class: "price-line" }));
  days.forEach((d, i) => {
    if (d.close == null || !(d.alert || d.watch)) return;
    s.append(svg("circle", { cx: X(i), cy: Y(d.close), r: 4.5, class: `dot ${d.alert ? "v-alert" : "v-watch"}` }));
  });
  const host = tooltipHost();
  const hit = svg("rect", { x: F.pad.l, y: 0, width: F.iw, height: F.h, class: "hit" });
  const near = (ev) => {
    const r = s.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * F.w;
    return Math.max(0, Math.min(days.length - 1, Math.round(((x - F.pad.l) / F.iw) * (days.length - 1))));
  };
  hit.addEventListener("pointermove", (ev) => {
    const i = near(ev), d = days[i], r = s.getBoundingClientRect();
    host.show((X(i) / F.w) * r.width, (Y(d.close ?? y1) / F.h) * r.height, [
      el("b", {}, d.date), el("div", {}, `종가 ${won(d.close)}`),
      el("div", {}, d.alert ? `이상 ${d.alert}개` : d.watch ? `주의 ${d.watch}개` : "모두 평소"), el("div", { class: "tip-hint" }, "눌러서 그날 보기")]);
  });
  hit.addEventListener("pointerleave", () => host.hide());
  hit.addEventListener("click", (ev) => onPick(days[near(ev)].date));
  s.append(hit);
  host.wrap.prepend(s);
  return host.wrap;
}

// 고른 날: 위에 30분 주가 선, 아래에 거래량 막대(판정 색) + 평소 거래량 눈금
export function dayCharts(bars, medians) {
  const n = bars.length;
  const P = frame(340, 120, { l: 44, r: 8, t: 10, b: 6 });
  const V = frame(340, 150, { l: 44, r: 8, t: 8, b: 22 });
  const band = P.iw / n;
  const X = (i) => P.pad.l + band * (i + 0.5);

  // 주가
  const closes = bars.flatMap((b) => [b.open, b.close]).filter((v) => v != null);
  const host = tooltipHost();
  const tip = (i) => {
    const b = bars[i];
    return [el("b", {}, b.slot), el("div", {}, `가격 ${won(b.open)} → ${won(b.close)} (${pct(b.open, b.close)})`),
      el("div", {}, `거래량 ${num(b.volume)}주, 평소(${num(medians[b.slot])}주)의 ${b.ratio == null ? "-" : b.ratio.toFixed(1)}배`),
      el("div", {}, `판정 ${b.verdict ?? "-"}${b.status && b.status !== "확정" ? ` (${b.status})` : ""}`)];
  };
  const ps = svg("svg", { viewBox: `0 0 ${P.w} ${P.h}`, class: "plot", role: "img", "aria-label": "30분마다 주가" });
  if (closes.length) {
    const lo = Math.min(...closes), hi = Math.max(...closes), pad = (hi - lo) * 0.15 || 5;
    const y0 = lo - pad, y1 = hi + pad, Y = (v) => P.pad.t + (1 - (v - y0) / (y1 - y0)) * P.ih;
    for (const t of ticks(y0, y1, 3)) ps.append(svg("line", { x1: P.pad.l, x2: P.w - P.pad.r, y1: Y(t), y2: Y(t), class: "grid" }),
      svg("text", { x: P.pad.l - 6, y: Y(t) + 3, class: "axis", "text-anchor": "end" }, document.createTextNode(num(t))));
    ps.append(svg("polyline", { points: bars.map((b, i) => (b.close == null ? null : `${X(i)},${Y(b.close)}`)).filter(Boolean).join(" "), class: "price-line" }));
    bars.forEach((b, i) => { if (b.close != null) ps.append(svg("circle", { cx: X(i), cy: Y(b.close), r: 3.5, class: "price-dot" })); });
    ps.append(svg("text", { x: P.pad.l, y: P.pad.t + 2, class: "axis" }, document.createTextNode("주가(30분 종가)")));
  } else {
    ps.append(svg("text", { x: P.w / 2, y: P.h / 2, class: "axis", "text-anchor": "middle" }, document.createTextNode("이날 가격 자료 없음")));
  }

  // 거래량
  const vmax = Math.max(1, ...bars.map((b) => b.volume || 0), ...bars.map((b) => medians[b.slot] || 0));
  const vy1 = vmax * 1.08, VY = (v) => V.pad.t + (1 - v / vy1) * V.ih;
  const vs = svg("svg", { viewBox: `0 0 ${V.w} ${V.h}`, class: "plot", role: "img", "aria-label": "30분마다 거래량과 판정" });
  for (const t of ticks(0, vy1, 3)) vs.append(svg("line", { x1: V.pad.l, x2: V.w - V.pad.r, y1: VY(t), y2: VY(t), class: "grid" }),
    svg("text", { x: V.pad.l - 6, y: VY(t) + 3, class: "axis", "text-anchor": "end" }, document.createTextNode(t >= 10000 ? `${t / 10000}만` : t >= 1000 ? `${t / 1000}천` : num(t))));
  const bw = Math.min(16, band * 0.62);
  bars.forEach((b, i) => {
    const x = X(i) - bw / 2, top = VY(b.volume || 0), base = VY(0), h = Math.max(0, base - top), rr = Math.min(4, h, bw / 2);
    if (h > 0) vs.append(svg("path", { class: `vbar ${VERDICT_CLASS[b.verdict] || "v-normal"}${b.status === "잠정" ? " provisional" : ""}`,
      d: `M${x},${base} V${top + rr} Q${x},${top} ${x + rr},${top} H${x + bw - rr} Q${x + bw},${top} ${x + bw},${top + rr} V${base} Z` }));
    const m = medians[b.slot];
    if (m) vs.append(svg("line", { x1: X(i) - band * 0.45, x2: X(i) + band * 0.45, y1: VY(m), y2: VY(m), class: "median-tick" }));
    if (b.verdict === "이상" || b.verdict === "주의") vs.append(svg("text", { x: X(i), y: top - 4, class: "mark-label", "text-anchor": "middle" },
      document.createTextNode(b.verdict)));
    if (i % 2 === 0) vs.append(svg("text", { x: X(i), y: V.h - 6, class: "axis", "text-anchor": "middle" }, document.createTextNode(b.slot)));
  });
  vs.append(svg("line", { x1: V.pad.l, x2: V.w - V.pad.r, y1: VY(0), y2: VY(0), class: "baseline" }));

  // 두 그래프에 같은 가로 위치로 툴팁
  for (const [s, F] of [[ps, P], [vs, V]]) {
    const hit = svg("rect", { x: F.pad.l, y: 0, width: F.iw, height: F.h, class: "hit" });
    hit.addEventListener("pointermove", (ev) => {
      const r = s.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * F.w;
      const i = Math.max(0, Math.min(n - 1, Math.floor((x - F.pad.l) / band)));
      const hr = host.wrap.getBoundingClientRect();
      host.show(((X(i) / F.w) * r.width) + (r.left - hr.left), r.top - hr.top + 20, tip(i));
    });
    hit.addEventListener("pointerleave", () => host.hide());
    s.append(hit);
  }
  const legend = el("div", { class: "legend" },
    el("span", {}, el("i", { class: "sw v-normal" }), "평소"), el("span", {}, el("i", { class: "sw v-watch" }), "주의"),
    el("span", {}, el("i", { class: "sw v-alert" }), "이상"), el("span", {}, el("i", { class: "sw median" }), "평소 거래량(중앙값)"));
  host.wrap.prepend(ps, vs, legend);
  return host.wrap;
}

// 그날의 표: 시간, 거래량, 평소의 몇 배, 가격, 등락, 판정
export function dayTable(bars, medians, onlyFlagged) {
  const rows = onlyFlagged ? bars.filter((b) => b.verdict === "이상" || b.verdict === "주의") : bars;
  const noData = bars.filter((b) => b.status === "자료 없음" || b.volume == null).map((b) => b.slot);
  if (!rows.length) return el("p", { class: "hint" }, "이날은 모든 30분이 평소였습니다.", noData.length ? ` (자료 없는 칸: ${noData.join(", ")})` : "");
  return el("div", { class: "table-wrap flat" }, el("table", { class: "coltable" },
    el("thead", {}, el("tr", {}, ["시간", "거래량", "평소의", "가격", "등락", "판정"].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.map((b) => el("tr", {},
      el("th", {}, b.slot), el("td", {}, num(b.volume)), el("td", {}, b.ratio == null ? "-" : `${b.ratio.toFixed(1)}배`),
      el("td", {}, won(b.close)), el("td", {}, pct(b.open, b.close)),
      el("td", {}, el("span", { class: `badge ${VERDICT_CLASS[b.verdict] || ""}` }, b.verdict ? (b.verdict === "이상" ? "▲ 이상" : b.verdict === "주의" ? "● 주의" : "평소") : "-",
        b.status && b.status !== "확정" ? ` (${b.status})` : "")))))));
}
