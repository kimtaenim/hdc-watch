// 화면: 추적 기록(trace)만 읽어 그린다. 설명 문구는 모두 추적 기록에서 온다.
// 한 건 따라가기는 같은 HDC 부품(hdc.js)으로 브라우저에서 다시 계산하고, 파이썬이 낸 점수와 맞춰 본다.
import * as H from "./hdc.js";
import { renderLab, renderCheck } from "./lab.js";
import { overviewChart, dayCharts, dayTable } from "./volchart.js";

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};
const getJSON = async (p) => {
  const r = await fetch(p, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${p}: ${r.status}`);
  return r.json();
};
const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("ko-KR") : String(+v.toFixed(4))) : Array.isArray(v) ? v.join(" - ") : v ?? "");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const OP_COLOR = { "벡터 생성": "--op-gen", "뒤집기": "--op-flip", "묶기": "--op-bind", "더하기": "--op-bundle", "비교": "--op-cmp", "일반 계산": "--op-plain" };

let SITE = { repo: "" }, GLOSS = {}, RUNS = [];
const usedTerms = new Set();

// ---------- 용어 풀이: 처음 나올 때만 ----------
function glossText(text) {
  const frag = document.createDocumentFragment();
  let rest = String(text);
  while (rest) {
    let best = null;
    for (const t of Object.keys(GLOSS)) {
      if (t.startsWith("_") || usedTerms.has(t)) continue;
      const i = rest.indexOf(t);
      if (i >= 0 && (!best || i < best.i)) best = { t, i };
    }
    if (!best) { frag.append(rest); break; }
    frag.append(rest.slice(0, best.i));
    usedTerms.add(best.t);
    const note = el("span", { class: "term-note", hidden: "" }, `${best.t}: ${GLOSS[best.t]}`);
    frag.append(el("button", { class: "term", onclick: () => (note.hidden = !note.hidden) }, best.t), note);
    rest = rest.slice(best.i + best.t.length);
  }
  return frag;
}

// ---------- 비트 격자: 1만 비트를 100×100 ----------
function gridCanvas(bits, highlight = null) {
  const side = Math.ceil(Math.sqrt(bits.length));
  const c = el("canvas", { width: side, height: side });
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(side, side);
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const [c1, c0, ch] = [rgb(css("--bit1")), rgb(css("--bit0")), rgb(css("--hl"))];
  for (let i = 0; i < side * side; i++) {
    const col = i >= bits.length ? c0 : highlight && highlight[i] ? ch : bits[i] ? c1 : c0;
    img.data.set([...col, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function gridBox(name, bits, extra = "", highlight = null) {
  return el("div", { class: "grid-box" }, gridCanvas(bits, highlight),
    el("div", { class: "gname" }, name), el("div", {}, `1의 개수 ${H.ones(bits).toLocaleString("ko-KR")}`), extra ? el("div", {}, extra) : null);
}

// ---------- 숫자 줄: 앞 4칸 ··· 뒤 4칸을 위아래로 맞춰 ----------
function bitRows(rows, dim, n = 4, tail = 4) {
  const cols = [...Array(n).keys(), null, ...Array.from({ length: tail }, (_, i) => dim - tail + i)];
  const head = el("tr", {}, el("th", {}, "칸"), cols.map((c) => el("th", { class: "pos" }, c === null ? "" : c + 1)));
  const body = rows.map((r) => el("tr", { class: r.line ? "result" : "" }, el("th", { class: "rname" }, r.name),
    cols.map((c) => c === null ? el("td", { class: "dots" }, "···")
      : el("td", { class: r.mark && r.mark[c] ? "hl" : "" },
        r.vals ? (r.vals[c] > 0 ? `+${r.vals[c]}` : `${r.vals[c]}`) : r.bits[c]))));
  return el("div", { class: "bitrows" }, el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
}

// ---------- 표 ----------
function tableEl(cols, rows, limit = null) {
  const shown = limit ? rows.slice(0, limit) : rows;
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, c)))),
    el("tbody", {}, shown.map((r) => el("tr", {}, r.map((v) => el("td", {}, Array.isArray(v) ? v.join(", ") : fmt(v))))))));
}

// ---------- 소스 링크 ----------
function srcLink(source, code) {
  if (!source) return null;
  const ref = code?.commit || "HEAD";
  const url = `https://github.com/${SITE.repo}/blob/${ref}/${source.file}#L${source.line}`;
  return el("div", { class: "src" }, el("a", { href: url, target: "_blank", rel: "noopener" },
    `소스: ${source.file} ${source.line}번째 줄 (${source.function})`),
    code?.dirty ? el("span", { class: "hint" }, " 실행 당시 이 커밋에서 수정 중이던 코드라 줄이 조금 다를 수 있음") : null);
}

// ---------- 톱니바퀴 한 칸 ----------
function renderGear(step, code) {
  const li = el("li", { class: "gear", style: `--op-color: var(${OP_COLOR[step.op] || "--line"})` });
  li.append(el("div", { class: "gear-head" },
    el("span", { class: "gear-no" }, `${step.index}.`),
    el("span", { class: "gear-name" }, step.name),
    el("span", { class: "op" }, step.op)));

  const all = [...(step.inputs || []).map((x) => ({ ...asObj(x), _in: true })), ...(step.outputs || []).map(asObj)];
  const values = all.filter((o) => o.type === "value");
  const vectors = all.filter((o) => o.type === "vector");
  const tables = all.filter((o) => o.type === "table");
  if (values.length) li.append(el("div", { class: "values" }, values.map((v) =>
    el("span", { class: "chip" }, `${v.name} `, el("b", {}, fmt(v.value))))));

  const withBits = vectors.filter((v) => v.bits_b64);
  if (withBits.length) li.append(bitRows(withBits.slice(0, 4).map((v) => ({ name: v.name, bits: H.bitsFromB64(v.bits_b64, v.dim) })), withBits[0].dim));
  if (step.check) {
    const c = step.check;
    li.append(el("div", { class: "check" }, `확인: ${c.question}. 단순 카운트 정답 `, el("b", {}, fmt(c.count_answer)),
      `, HDC `, el("b", {}, fmt(c.hdc_answer)), " ", el("span", { class: c.agree ? "ok" : "bad" }, c.agree ? "일치" : "어긋남")));
  }
  if (tables.length) {
    const t = tables[0];
    li.append(el("div", { class: "tname" }, t.name), tableEl(t.columns || [], t.rows, 6));
  }

  // 펼치면 숫자 전체
  const more = el("details", { class: "more" }, el("summary", {}, "숫자 전체, 격자 그림, 소스 코드"));
  more.addEventListener("toggle", () => {
    if (!more.open || more.dataset.done) return;
    more.dataset.done = "1";
    if (withBits.length > 4) more.append(bitRows(withBits.slice(4).map((v) => ({ name: v.name, bits: H.bitsFromB64(v.bits_b64, v.dim) })), withBits[0].dim));
    if (withBits.length) more.append(el("div", { class: "grids" }, withBits.map((v) => gridBox(v.name, H.bitsFromB64(v.bits_b64, v.dim)))));
    for (const v of vectors) more.append(el("div", { class: "head-bits" }, `${v.name}: 1의 개수 ${v.ones}, 앞 64비트 ${v.head}`));
    for (const t of tables) more.append(el("div", { class: "tname" }, `${t.name} (${t.rows.length}행)`), tableEl(t.columns || [], t.rows));
    for (const o of all.filter((o) => o.type === "flips")) more.append(el("div", { class: "head-bits" }, `${o.name}: ${o.cells.length}칸 (${o.cells.slice(0, 40).join(", ")} ...)`));
    more.append(srcLink(step.source, code) || "");
  });
  li.append(more);
  const w = whyBox(step);
  if (w) li.append(w);
  return li;
}
const asObj = (x) => (typeof x === "object" && x ? x : { type: "value", name: "", value: x });

function renderGears(trace, path) {
  const ol = $("#gears");
  ol.replaceChildren();
  const sec = new Map((trace.sections || []).map((s) => [s.from_index, s]));
  for (const step of trace.steps) {
    const s = sec.get(step.index);
    if (s) ol.append(el("li", {}, el("div", { class: "section-title" }, s.title), s.text ? el("p", { class: "section-text" }, glossText(s.text)) : null));
    ol.append(renderGear(step, trace.code));
  }
  const fails = trace.log?.failures?.length || 0;
  $("#gears-meta").replaceChildren(`${trace.started_at} 실행, 단계 ${trace.steps.length}개, 기록된 실패 ${fails}건. `,
    el("a", { href: path, target: "_blank" }, "추적 기록 원문(JSON)"));
}

// ---------- 운행 기록 ----------
function renderRuns(runs) {
  const rows = [...runs].reverse().map((r) => [
    r.at.replace("T", " ").slice(0, 19), r.kind, r.subject,
    Object.entries(r.fetched || {}).map(([k, v]) => `${k} ${v}`).join(", "),
    r.failures?.length ? `${r.failures.length}건: ${r.failures.slice(0, 2).join(" / ")}${r.failures.length > 2 ? " ..." : ""}` : "없음",
    r.delay_min ?? "", r.trace, r.note || ""]);
  const t = $("#runs");
  t.replaceChildren(el("thead", {}, el("tr", {}, ["시각", "종류", "종목", "받은 건수", "실패", "예약보다 늦음(분)", "추적 기록", "메모"].map((c) => el("th", {}, c)))),
    el("tbody", {}, rows.map((r) => el("tr", {},
      r.map((v, i) => i === 6 ? el("td", {}, v ? el("a", { href: v, target: "_blank" }, "열기") : "사라짐")
        : el("td", { class: i === 4 && v !== "없음" ? "fail" : "" }, v))))));
}

// ---------- 톱니바퀴: 고른 날의 실제 자료가 단계마다 바뀌는 모습 ----------
const F = { cache: {}, trace: null, model: null, raw: null, bars: [], date: null, slot: null };

const r2 = (v, d) => (v == null ? "" : Number(v.toFixed(d)));
const stepOf = (t, test) => t.steps.find(test) || { why: [] };
function whyBox(step) {
  if (!step?.why?.length) return null;
  return el("details", { class: "why-box" }, el("summary", {}, "왜?"),
    el("ul", { class: "why" }, step.why.map((w) => el("li", {}, el("span", { class: "kind", "data-k": w.kind }, w.kind), glossText(w.text)))));
}
function card(no, title, op, why, ...body) {
  return el("li", { class: "gear", style: `--op-color: var(${OP_COLOR[op] || "--line"})` },
    el("div", { class: "gear-head" }, el("span", { class: "gear-no" }, no), el("span", { class: "gear-name" }, title), el("span", { class: "op" }, op)),
    ...body, whyBox(why));
}
// 세로 표: 줄마다 시간 하나. 줄을 누르면 그 시간 봉을 따라간다.
function colTable(head, rows, selIdx, onPick) {
  return el("div", { class: "table-wrap flat" }, el("table", { class: "coltable" },
    el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.map((r, i) => el("tr", { class: i === selIdx ? "sel" : "", onclick: () => onPick(i) },
      r.map((v, j) => (j === 0 ? el("th", {}, v) : el("td", {}, fmt(v)))))))));
}
async function cached(key, fn) {
  if (!(key in F.cache)) F.cache[key] = await fn();
  return F.cache[key];
}

async function setupFlow(trace, stockDir) {
  F.trace = trace;
  F.model = await getJSON(`${stockDir}/model.json`);
  F.raw = await getJSON(`${stockDir}/raw.json`);
  const tbl = trace.steps.flatMap((x) => x.outputs).find((o) => o.name === "봉별 점수");
  const c = Object.fromEntries(tbl.columns.map((k, i) => [k, i]));
  F.bars = tbl.rows.map((r) => ({ date: r[c["날짜"]], wd: r[c["요일"]], slot: r[c["시간대"]], vol: r[c["거래량"]], ratio: r[c["배율"]],
    level: r[c["단계"]], score: r[c["점수"]], scoreBin: r[c["이진 점수"]], rank: r[c["순위"]], verdict: r[c["판정"]] }));
  const dates = [...new Set(F.bars.map((b) => b.date))].sort().reverse();
  const rare = F.bars.find((b) => b.rank === 1);
  const sel = $("#pick-date");
  sel.replaceChildren(...dates.map((d) => {
    const day = F.bars.filter((b) => b.date === d);
    const worst = day.reduce((a, b) => (b.score < a.score ? b : a));
    return el("option", { value: d }, `${d}(${day[0].wd}) ${worst.verdict === "평소" ? "" : `${worst.verdict} 있음`}`);
  }));
  sel.onchange = () => { F.date = sel.value; F.slot = null; drawFlow(); };
  $("#pick-rare").onclick = () => { sel.value = rare.date; F.date = rare.date; F.slot = rare.slot; drawFlow(); };
  $("#flow-wrap").ontoggle = () => { if ($("#flow-wrap").open) drawFlow(); };
  F.date = dates[0];
  sel.value = F.date;
  if ($("#flow-wrap").open) await drawFlow();
}

async function drawFlow() {
  const m = F.model, t = F.trace, dim = m.dim;
  const day = F.bars.filter((b) => b.date === F.date).sort((a, b) => a.slot.localeCompare(b.slot));
  if (F.slot == null) F.slot = day.reduce((a, b) => (b.score < a.score ? b : a)).slot;
  const bar = day.find((b) => b.slot === F.slot) || day[0];
  const slots = day.map((b) => b.slot);
  const hi = slots.indexOf(bar.slot);
  const go = (i) => { F.slot = slots[i]; drawFlow(); };
  const raw = Object.fromEntries(F.raw.slots.filter((r) => r.date === F.date).map((r) => [r.slot, r]));
  const ol = $("#flow");
  ol.replaceChildren(el("li", { class: "hint" }, "벡터를 만드는 중..."));

  const ladder = await cached("ladder", () => H.levelVectors(m.ladder, m.levels, dim));
  const dayV = await cached(`d${bar.wd}`, () => H.makeVector(`요일:${bar.wd}`, dim));
  const hv = {};
  for (const h of slots) hv[h] = await cached(`h${h}`, () => H.makeVector(`시간대:${h}`, dim));
  const full = H.Memory.fromJSON(m.memory_train || m.memory);
  // 그날 봉마다 브라우저에서 다시 채점(자기 제외)
  const scoreOf = async (b) => {
    const v = H.bind(H.bind(dayV, hv[b.slot]), ladder.vectors[b.level - 1]);
    const mem = H.Memory.fromJSON(m.memory_train || m.memory); mem.add(v, -1);
    let dot = 0, sq = 0;
    for (let i = 0; i < dim; i++) { const c = mem.counts[i]; dot += (v[i] ? 1 : -1) * c; sq += c * c; }
    const proto = await mem.toVector();
    return { v, mem, proto, dot, sim: dot / (Math.sqrt(dim) * Math.sqrt(sq)), simBin: H.similarity(v, proto) };
  };
  const res = {};
  for (const b of day) res[b.slot] = await scoreOf(b);
  const r = res[bar.slot];
  const lv = ladder.vectors[bar.level - 1];
  const ds = H.bind(dayV, hv[bar.slot]);
  const diff = H.bind(r.v, r.proto);
  const same = (a, b) => Math.abs(a - b) < 1e-4;
  const allOk = day.every((b) => same(res[b.slot].sim, b.score) && same(res[b.slot].simBin, b.scoreBin));
  const th = m.thresholds;
  const S = (test) => stepOf(t, test);

  ol.replaceChildren(
    card("①", "받은 자료 (5분봉을 시간대별로 합침)", "일반 계산", S((s) => s.name.startsWith("원자료")),
      colTable(["시간", "거래량", "5분봉 수"], slots.map((h) => [h, raw[h]?.volume ?? 0, raw[h]?.n5 ?? 0]), hi, go)),
    card("②", "평소 거래량 (학습 기간 중앙값)", "일반 계산", S((s) => s.name.includes("중앙값")),
      colTable(["시간", "중앙값"], slots.map((h) => [h, Math.round(m.medians[h].median)]), hi, go)),
    card("③", "평소의 몇 배 → 12단계", "일반 계산", S((s) => s.name.includes("단계 나누기")),
      colTable(["시간", "배율", "단계"], day.map((b) => [b.slot, r2(b.ratio, 2), b.level]), hi, go),
      el("p", { class: "hint" }, "줄을 누르면 그 시간 봉을 아래 ④-⑦에서 따라갑니다.")),
    card("④", `${bar.slot} 봉의 재료 벡터 꺼내기`, "벡터 생성", S((s) => s.op === "벡터 생성"),
      bitRows([{ name: `요일:${bar.wd}`, bits: dayV }, { name: `시간대:${bar.slot}`, bits: hv[bar.slot] },
        { name: `${bar.level}단계`, bits: lv }], dim)),
    card("⑤", "묶기 (XOR)", "묶기", S((s) => s.op === "묶기"),
      bitRows([{ name: `요일:${bar.wd}`, bits: dayV }, { name: `시간대:${bar.slot}`, bits: hv[bar.slot] },
        { name: "요일 ⊕ 시간대", bits: ds, line: true }], dim),
      bitRows([{ name: "요일 ⊕ 시간대", bits: ds }, { name: `${bar.level}단계`, bits: lv },
        { name: "봉 벡터", bits: r.v, line: true }], dim)),
    card("⑥", `프로토타입 (${fmt(r.mem.n)}개 봉의 합, 이 봉 제외)`, "더하기", S((s) => s.op === "더하기"),
      bitRows([{ name: "정수 프로토타입", vals: r.mem.counts }, { name: "이진 프로토타입", bits: r.proto, line: true }], dim, 3, 3)),
    card("⑦", "비교 → 점수", "비교", S((s) => s.op === "비교"),
      bitRows([{ name: "봉 ±1", vals: Array.from(r.v, (x) => (x ? 1 : -1)) }, { name: "정수 프로토타입", vals: r.mem.counts },
        { name: "곱", vals: Array.from(r.v, (x, i) => (x ? 1 : -1) * r.mem.counts[i]), line: true }], dim, 3, 3),
      el("div", { class: "values" }, el("span", { class: "chip" }, "곱의 합 ", el("b", {}, fmt(r.dot))),
        el("span", { class: "chip" }, "정수 점수(코사인) ", el("b", {}, fmt(r.sim)))),
      bitRows([{ name: "봉 벡터", bits: r.v }, { name: "이진 프로토타입", bits: r.proto },
        { name: "다른 칸", bits: diff, mark: diff, line: true }], dim),
      el("div", { class: "values" }, el("span", { class: "chip" }, "다른 칸 ", el("b", {}, fmt(H.ones(diff)))),
        el("span", { class: "chip" }, "이진 점수 ", el("b", {}, fmt(r.simBin))))),
    card("⑧", "그날의 점수와 판정", "일반 계산", S((s) => s.name.startsWith("판정")),
      colTable(["시간", "정수 점수", "이진 점수", "판정", "순위"],
        day.map((b) => [b.slot, r2(res[b.slot].sim, 3), r2(res[b.slot].simBin, 3), b.verdict, b.rank]), hi, go),
      el("p", { class: "hint" }, `순위는 학습 기간 ${F.bars.length}개 봉 중 정수 점수가 낮은 순.`),
      el("div", { class: "values" }, el("span", { class: "chip" }, "이상 경계 ", el("b", {}, fmt(th.alert))),
        el("span", { class: "chip" }, "주의 경계 ", el("b", {}, fmt(th.watch)))),
      el("div", { class: "check" }, `파이썬 계산과 이 브라우저 계산(그날 ${day.length}개 봉, 두 점수) `,
        el("span", { class: allOk ? "ok" : "bad" }, allOk ? "모두 일치" : "어긋남 있음"))),
  );
}

// ---------- 거래량 이상: 그래프와 표 ----------
const V = { days: [], byDate: {}, medians: {}, selected: null };

async function setupVol(trace, stockDir, model, today) {
  const tbl = trace.steps.flatMap((x) => x.outputs).find((o) => o.name === "봉별 점수");
  const c = Object.fromEntries(tbl.columns.map((k, i) => [k, i]));
  const raw = await getJSON(`${stockDir}/raw.json`);
  const price = Object.fromEntries((raw.slots || []).map((r) => [`${r.date} ${r.slot}`, r]));
  const by = {};
  for (const r of tbl.rows) {
    const d = r[c["날짜"]], s = r[c["시간대"]], p = price[`${d} ${s}`];
    (by[d] ||= []).push({ date: d, slot: s, volume: r[c["거래량"]], ratio: r[c["배율"]], verdict: r[c["판정"]], status: "확정",
      open: p?.open ?? null, close: p?.close ?? null });
  }
  const liveDates = (model.history || []).filter((h) => h.kind === "장 마감 갱신").map((h) => h.date);
  if (today) liveDates.push(today.date);
  for (const d of [...new Set(liveDates)]) {
    if (by[d]) continue;
    try {
      const day = d === today?.date ? today : await getJSON(`${stockDir}/live/${d}.json`);
      by[d] = day.bars.filter((b) => b.status !== "아직").map((b) => ({ ...b }));
    } catch {}
  }
  for (const k of Object.keys(by)) by[k].sort((a, b) => a.slot.localeCompare(b.slot));
  V.byDate = by;
  V.medians = Object.fromEntries(Object.entries(model.medians).map(([k, v]) => [k, v.median]));
  V.days = Object.keys(by).sort().map((d) => {
    const bs = by[d], last = [...bs].reverse().find((b) => b.close != null);
    return { date: d, close: last?.close ?? null, alert: bs.filter((b) => b.verdict === "이상").length, watch: bs.filter((b) => b.verdict === "주의").length };
  });
  $("#vol-title").textContent = `거래량 이상: ${trace.subject.name}`;
  pickVolDay(V.days.at(-1).date);
}

function pickVolDay(date) {
  V.selected = date;
  $("#vol-over").replaceChildren(overviewChart(V.days, date, pickVolDay),
    el("p", { class: "hint" }, "빨간 점: 이상이 있던 날, 주황 점: 주의가 있던 날. 그래프를 누르면 그날을 아래에 보입니다."));
  const bs = V.byDate[date] || [];
  const first = bs.find((b) => b.open != null), last = [...bs].reverse().find((b) => b.close != null);
  const wd = "일월화수목금토"[new Date(date + "T00:00:00Z").getUTCDay()];  // 보는 사람의 시간대와 무관하게
  const nA = bs.filter((b) => b.verdict === "이상").length, nW = bs.filter((b) => b.verdict === "주의").length;
  $("#vol-day-title").replaceChildren(el("b", {}, `${date} (${wd})`),
    ` 시가 ${first ? Math.round(first.open).toLocaleString("ko-KR") : "-"}원 → 종가 ${last ? Math.round(last.close).toLocaleString("ko-KR") : "-"}원`,
    first && last ? ` (${last.close >= first.open ? "+" : ""}${(((last.close - first.open) / first.open) * 100).toFixed(1)}%)` : "",
    `, 이상 ${nA}개, 주의 ${nW}개`);
  $("#vol-day").replaceChildren(dayCharts(bs, V.medians), el("div", { class: "tname" }, "주의·이상인 30분"), dayTable(bs, V.medians, true),
    el("details", { class: "more" }, el("summary", {}, "이날 30분 모두 보기"), dayTable(bs, V.medians, false)));
  if (V.byDate[date] && $("#pick-date") && [...$("#pick-date").options].some((o) => o.value === date)) {
    $("#pick-date").value = date; F.date = date; F.slot = null;
    if ($("#flow-wrap").open) drawFlow();
  }
}

// ---------- 트렌드: 키워드 탭, 낱말별 펼치기, HDC 계산 과정 ----------
const titleLi = (date, source, title, link) => el("li", {}, el("span", { class: "hint" }, `${date} ${source} `),
  el("a", { href: link, target: "_blank", rel: "noopener" }, title));

async function setupTrend(trace) {
  const ids = trace.subject.id;
  const days = {};
  for (const kid of ids) { try { days[kid] = await getJSON(`data/trend/${kid}/day.json`); } catch {} }
  const have = ids.filter((k) => days[k]);
  const d0 = days[have[0]];
  if (d0) $("#trend-when").textContent = `최근 ${d0.recent_days}일(${d0.recent_from} - ${d0.date})을 그 전 4주(${d0.past_from} - ${d0.past_to})와 비교한다.`;
  const first = [...have].sort((a, b) => (days[b].titles_recent * 1000 + days[b].titles_past) - (days[a].titles_recent * 1000 + days[a].titles_past))[0];
  const pick = async (kid) => {
    $("#kw-tabs").querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x.dataset.kid === kid)));
    const day = days[kid];
    const fresh = new Set(day.new.map((x) => x.word));
    // 낱말별 펼치기: 누르면 그 낱말이 키워드와 함께 나온 제목(최근, 과거)과 HDC 점수
    const words = day.rows.map((r) => {
      const wt = day.word_titles?.[r[0]] || { recent: [], past: [] };
      return el("details", { class: "word" },
        el("summary", {}, el("b", {}, r[0]), ` 최근 ${r[1]}번 · 과거 ${r[3]}번`, fresh.has(r[0]) ? el("span", { class: "chip new" }, "새로 붙음") : null),
        el("div", { class: "values" }, el("span", { class: "chip" }, "과거 점수 ", el("b", {}, fmt(r[4]))),
          el("span", { class: "chip" }, "잡음 수준 ", el("b", {}, fmt(day.threshold)))),
        el("div", { class: "tname" }, "최근 제목"),
        el("ul", { class: "titles" }, wt.recent.map((i) => { const x = day.titles[i]; return titleLi(x[0], x[1], x[2], x[3]); })),
        wt.past.length ? [el("div", { class: "tname" }, "과거 제목(최근 것부터)"),
          el("ul", { class: "titles" }, wt.past.map((x) => titleLi(x[0].slice(5), x[1], x[2], x[3])))] : el("p", { class: "hint" }, "과거 4주에는 이 키워드와 함께 나온 적이 없다."));
    });
    $("#kw-summary").replaceChildren(
      el("div", { class: "kw-head" }, el("b", {}, day.name), ` 최근 ${day.recent_days}일 제목 ${day.titles_recent}개 · 그 전 4주 ${day.titles_past}개`,
        day.warmup ? el("span", { class: "hint" }, ` (예열 중: 과거 자료가 있는 날 ${day.past_days_with}일)`) : null),
      el("div", { class: "kw-line" }, el("span", { class: "kw-label" }, "새로 붙은 낱말"),
        fresh.size ? [...fresh].map((w) => el("span", { class: "chip new" }, w)) : el("span", { class: "hint" }, "없음")),
      el("div", { class: "tname" }, `같이 나온 낱말 ${day.rows.length}개 (누르면 제목이 펼쳐짐)`),
      words.length ? el("div", { class: "words" }, words) : el("p", { class: "hint" }, "없음"));
    await drawTrend(kid, trace, $("#trend-flow"));
  };
  $("#kw-tabs").replaceChildren(...have.map((kid) => el("button", { role: "tab", "data-kid": kid, onclick: () => pick(kid) },
    `${days[kid].name} ${days[kid].titles_recent}`)));
  if (first) await pick(first);
}

async function drawTrend(kid, trace, ol) {
  ol.replaceChildren(el("li", { class: "hint" }, "불러오는 중..."));
  const day = await getJSON(`data/trend/${kid}/day.json`);
  const proto = await getJSON(`data/trend/${kid}/proto.json`);
  const dim = proto.recent.dim;
  const kv = await H.makeVector(`키워드:${kid}`, dim);
  const sec = (trace.sections || []).find((x) => x.title.endsWith(day.name));
  const steps = sec ? trace.steps.filter((s) => s.index >= sec.from_index).slice(0, 4) : [];
  const S = (op) => steps.find((s) => s.op === op);
  const RL = `최근 ${day.recent_days}일`;
  const top = day.rows[0]?.[0];
  const wv = top ? await H.makeVector(`단어:${top}`, dim) : null;
  const pair = wv ? H.bind(kv, wv) : null;
  // 브라우저에서 과거 점수를 다시 계산(정수 프로토타입을 키워드로 풀고 코사인)
  const past = proto.past.counts, recent = proto.recent.counts;
  const norm = Math.sqrt(past.reduce((a, c) => a + c * c, 0));
  const askPast = async (w) => {
    if (!norm) return 0;
    const v = await H.makeVector(`단어:${w}`, dim);
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += -past[i] * (kv[i] ? 1 : -1) * (v[i] ? 1 : -1);
    return dot / (Math.sqrt(dim) * norm);
  };
  const rows = day.rows.slice(0, 15);
  const mine = [];
  for (const r of rows) mine.push(await askPast(r[0]));
  const ok = rows.every((r, i) => Math.abs(r[4] - mine[i]) < 1e-4);
  ol.replaceChildren(...[
    el("li", { class: "section-title" }, "HDC 계산 과정"),
    card("①", `${RL} '${day.name}'이 든 제목 ${day.titles_recent}개 → 명사`, "일반 계산", S("일반 계산"),
      day.titles.length ? el("ul", { class: "titles" }, day.titles.map((x) => el("li", {},
        el("span", { class: "hint" }, `${x[0]} ${x[1]} `), el("a", { href: x[3], target: "_blank", rel: "noopener" }, x[2]),
        el("div", { class: "nouns" }, "→ 명사: ", x[4].join(", ") || "(없음)"))))
        : el("p", { class: "hint" }, `${RL} 동안 이 키워드가 든 제목이 없다.`)),
    pair ? card("②", `묶기: 키워드 ⊕ '${top}'`, "묶기", S("묶기"),
      bitRows([{ name: `키워드:${day.name}`, bits: kv }, { name: `단어:${top}`, bits: wv }, { name: "묶은 결과", bits: pair, line: true }], dim)) : null,
    card("③", `정수 프로토타입: ${RL} ${proto.recent.n}개, 과거 ${proto.past.n}개 조합`, "더하기", S("묶기"),
      bitRows([{ name: "최근", vals: recent }, { name: "과거", vals: past }], dim, 3, 3)),
    card("④", "과거에 물어보기", "비교", S("비교"),
      colTable(["단어", "최근", "과거 횟수", "과거 점수", "판정"],
        rows.map((r, i) => [r[0], r[1], r[3], Number(mine[i].toFixed(3)), r[5] || ""]), -1, () => {}),
      el("div", { class: "values" }, el("span", { class: "chip" }, "잡음 수준(판정선) ", el("b", {}, fmt(day.threshold))),
        day.one_past ? el("span", { class: "chip" }, "과거에 한 번 나온 단어의 점수(약) ", el("b", {}, fmt(Number(day.one_past.toFixed(4))))) : null),
      el("div", { class: "check" }, "파이썬 계산과 이 브라우저 계산(과거 점수) ", el("span", { class: ok ? "ok" : "bad" }, ok ? "모두 일치" : "어긋남 있음"))),
    card("⑤", `${RL} 문맥과 과거 문맥 전체 비교`, "비교", steps.filter((s) => s.op === "비교")[1],
      el("div", { class: "values" }, el("span", { class: "chip" }, "HDC 코사인 ", el("b", {}, day.cos_hdc == null ? "없음" : fmt(Number(day.cos_hdc.toFixed(3))))),
        el("span", { class: "chip" }, "단어 횟수(BoW) 코사인 ", el("b", {}, day.cos_bow == null ? "없음" : fmt(Number(day.cos_bow.toFixed(3))))))),
  ].filter(Boolean));
}

// ---------- 오늘(장중 채점) ----------
async function renderToday(stockDir) {
  const wrap = $("#today-wrap");
  let day;
  try { day = await getJSON(`${stockDir}/live.json`); } catch { wrap.hidden = true; return null; }
  wrap.hidden = true;  // 오늘 자료는 거래량 그래프에서 보인다
  $("#today-title").textContent = `오늘 ${day.date} (${day.as_of.slice(11, 16)} 기준${day.closing ? ", 마감" : ""}${day.simulated ? ", 모의" : ""})`;
  const shown = day.bars.filter((b) => b.status !== "아직");
  $("#today").replaceChildren(
    colTable(["시간", "거래량", "단계", "정수 점수", "판정", "상태"],
      shown.map((b) => [b.slot, b.volume, b.level, b.score, b.verdict, b.status]), -1, () => {}),
    el("p", { class: "hint" }, "잠정: 아직 끝나지 않은 30분 칸이라 거래량이 덜 찼다. ",
      el("a", { href: day.trace, target: "_blank" }, "이 실행의 추적 기록")));
  return day;
}

// ---------- 문자판 ----------
async function renderDial(stockId, trainTrace, today = null) {
  const s = trainTrace?.subject || { name: stockId, code: stockId };
  $("#dial-stock").textContent = `${s.name} (${s.code})`;
  const lastToday = today?.bars.filter((b) => b.score != null).at(-1);
  if (lastToday) {
    $("#dial-score").replaceChildren(`거래량 점수 `, el("b", {}, fmt(lastToday.score)),
      ` ${lastToday.verdict} (${today.date} ${lastToday.slot} ${lastToday.status})`);
  } else if (trainTrace) {
    const tbl = trainTrace.steps.flatMap((x) => x.outputs).find((o) => o.name === "봉별 점수");
    const c = Object.fromEntries(tbl.columns.map((k, i) => [k, i]));
    const last = [...tbl.rows].sort((a, b) => (a[c["날짜"]] + a[c["시간대"]]).localeCompare(b[c["날짜"]] + b[c["시간대"]])).at(-1);
    $("#dial-score").replaceChildren(`거래량 점수 `, el("b", {}, fmt(last[c["점수"]])),
      ` ${last[c["판정"]]} (${last[c["날짜"]]} ${last[c["시간대"]]})`);
  }
  try {
    const sm = await getJSON("data/trend/summary.json");
    $("#dial-trend").replaceChildren("최근 7일 새 낱말 ", el("b", {}, `${sm.new_total}개`), sm.warmup ? " (예열 중)" : "");
  } catch { $("#dial-trend").replaceChildren("최근 7일 새 낱말 ", el("b", {}, "자료 없음")); }
}

// ---------- 시작 ----------
async function showLab(kind, stock, trainRun) {
  for (const id of ["#flow-wrap", "#trend-wrap", "#gears-wrap", "#vol-wrap"]) $(id).hidden = true;
  $("#lab-wrap").hidden = false;
  $("#lab-title").textContent = kind === "lab" ? "실험대: 같은 부품을 직접 만져 보기" : "확인: HDC 답과 단순히 센 답";
  $("#lab").replaceChildren(el("li", { class: "hint" }, "계산 중..."));
  const trace = await getJSON(trainRun.trace);
  const tbl = trace.steps.flatMap((x) => x.outputs).find((o) => o.name === "봉별 점수");
  const c = Object.fromEntries(tbl.columns.map((k, i) => [k, i]));
  const bars = tbl.rows.map((r) => ({ date: r[c["날짜"]], wd: r[c["요일"]], slot: r[c["시간대"]], level: r[c["단계"]],
    score: r[c["점수"]], countSum: r[c["카운트 가까움 합"]], verdict: r[c["판정"]] }));
  const model = await getJSON(`data/volume/${stock}/model.json`);
  let keywords = [];
  try { keywords = await getJSON("data/trend/keywords.json"); } catch {}
  await (kind === "lab" ? renderLab : renderCheck)($("#lab"), { model, bars, keywords });
}

async function show() {
  const stock = $("#pick-stock").value, kind = $("#pick-view").value;
  if (kind === "lab" || kind === "check") {
    const trainRun = [...RUNS].reverse().find((r) => r.subject === stock && r.kind === "volume.train" && r.trace);
    await renderDial(stock, trainRun ? await getJSON(trainRun.trace) : null, await renderToday(`data/volume/${stock}`));
    return showLab(kind, stock, trainRun);
  }
  $("#lab-wrap").hidden = true;
  $("#gears-wrap").hidden = false;
  usedTerms.clear();
  const pick = (k) => [...RUNS].reverse().find((r) => (k === "trend" || r.subject === stock) && r.kind === k && r.trace && !(r.failures || []).some((f) => f.startsWith("수집 실패")));
  const trainRun = pick("volume.train"), run = pick(kind);
  const trainTrace = trainRun ? await getJSON(trainRun.trace) : null;
  const today = await renderToday(`data/volume/${stock}`);
  await renderDial(stock, trainTrace, today);
  if (!run) { $("#gears").replaceChildren(el("li", {}, "아직 실행 기록이 없습니다.")); return; }
  const trace = run === trainRun ? trainTrace : await getJSON(run.trace);
  const isTrain = kind === "volume.train";
  $("#flow-wrap").hidden = !isTrain;
  $("#vol-wrap").hidden = !isTrain;
  $("#trend-wrap").hidden = kind !== "trend";
  if (kind === "trend") await setupTrend(trace);
  $("#gears-title").textContent = isTrain ? "전체 계산 기록 (가장 최근 실행, 숫자 전체와 소스)" : "단계별 실제 자료 (가장 최근 실행)";
  $("#gears-wrap").open = !isTrain;
  if (isTrain) {
    await setupFlow(trace, `data/volume/${stock}`);
    await setupVol(trace, `data/volume/${stock}`, F.model, today);
  }
  renderGears(trace, run.trace);
}

async function main() {
  [SITE, GLOSS, RUNS] = await Promise.all([getJSON("data/site.json"), getJSON("data/glossary.json"), getJSON("data/runs.json")]);
  renderRuns(RUNS);
  const stocks = [...new Set(RUNS.map((r) => r.subject))];
  $("#pick-stock").replaceChildren(...stocks.map((s) => el("option", { value: s }, s)));
  $("#pick-stock").onchange = show;
  $("#pick-view").onchange = show;
  document.querySelectorAll(".tabs button").forEach((t) => (t.onclick = () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === t)));
    $("#pick-view").value = t.dataset.view;
    show();
  }));
  await show();
}

main().catch((e) => {
  document.querySelector("main").prepend(el("div", { class: "panel" }, `불러오기 실패: ${e.message}`));
  console.error(e);
});
