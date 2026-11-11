// 화면: 추적 기록(trace)만 읽어 그린다. 설명 문구는 모두 추적 기록에서 온다.
// 한 건 따라가기는 같은 HDC 부품(hdc.js)으로 브라우저에서 다시 계산하고, 파이썬이 낸 점수와 맞춰 본다.
import * as H from "./hdc.js";

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
  li.append(el("ul", { class: "why" }, step.why.map((w) =>
    el("li", {}, el("span", { class: "kind", "data-k": w.kind }, w.kind), glossText(w.text)))));

  const all = [...(step.inputs || []).map((x) => ({ ...asObj(x), _in: true })), ...(step.outputs || []).map(asObj)];
  const values = all.filter((o) => o.type === "value");
  const vectors = all.filter((o) => o.type === "vector");
  const tables = all.filter((o) => o.type === "table");
  if (values.length) li.append(el("div", { class: "values" }, values.map((v) =>
    el("span", { class: "chip" }, `${v.name} `, el("b", {}, fmt(v.value))))));

  const withBits = vectors.filter((v) => v.bits_b64);
  if (withBits.length) li.append(el("div", { class: "grids" }, withBits.slice(0, 4).map((v) =>
    gridBox(v.name, H.bitsFromB64(v.bits_b64, v.dim)))));
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
  const more = el("details", { class: "more" }, el("summary", {}, "숫자 전체와 소스 코드"));
  more.addEventListener("toggle", () => {
    if (!more.open || more.dataset.done) return;
    more.dataset.done = "1";
    if (withBits.length > 4) more.append(el("div", { class: "grids" }, withBits.slice(4).map((v) => gridBox(v.name, H.bitsFromB64(v.bits_b64, v.dim)))));
    for (const v of vectors) more.append(el("div", { class: "head-bits" }, `${v.name}: 1의 개수 ${v.ones}, 앞 64비트 ${v.head}`));
    for (const t of tables) more.append(el("div", { class: "tname" }, `${t.name} (${t.rows.length}행)`), tableEl(t.columns || [], t.rows));
    for (const o of all.filter((o) => o.type === "flips")) more.append(el("div", { class: "head-bits" }, `${o.name}: ${o.cells.length}칸 (${o.cells.slice(0, 40).join(", ")} ...)`));
    more.append(srcLink(step.source, code) || "");
  });
  li.append(more);
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
    r.trace, r.note || ""]);
  const t = $("#runs");
  t.replaceChildren(el("thead", {}, el("tr", {}, ["시각", "종류", "종목", "받은 건수", "실패", "추적 기록", "메모"].map((c) => el("th", {}, c)))),
    el("tbody", {}, rows.map((r) => el("tr", {},
      r.map((v, i) => i === 5 ? el("td", {}, v ? el("a", { href: v, target: "_blank" }, "열기") : "사라짐")
        : el("td", { class: i === 4 && v !== "없음" ? "fail" : "" }, v))))));
}

// ---------- 한 건 따라가기 (시간대 모델) ----------
const F = { frames: [], i: 0, bars: [], cache: {} };

function stepOf(trace, op, nameHas = "") {
  return trace.steps.find((s) => s.op === op && s.name.includes(nameHas)) || { why: [] };
}
function whyList(step) {
  return el("ul", { class: "why" }, (step.why || []).map((w) => el("li", {}, el("span", { class: "kind", "data-k": w.kind }, w.kind), w.text)));
}

async function setupFollow(trace, stockDir) {
  const wrap = $("#follow-wrap");
  if (trace.kind !== "volume.train") { wrap.hidden = true; return; }
  wrap.hidden = false;
  const scoreStep = trace.steps.find((s) => s.outputs.some((o) => o.name === "봉별 점수"));
  const tbl = scoreStep.outputs.find((o) => o.name === "봉별 점수");
  const col = Object.fromEntries(tbl.columns.map((c, i) => [c, i]));
  F.bars = tbl.rows.map((r) => ({ date: r[col["날짜"]], wd: r[col["요일"]], hour: r[col["시간대"]], vol: r[col["거래량"]],
    ratio: r[col["배율"]], level: r[col["단계"]], score: r[col["점수"]], rank: r[col["순위"]], verdict: r[col["판정"]] }));
  F.model = await getJSON(`${stockDir}/model.json`);
  F.trace = trace;
  $("#follow-hint").replaceChildren(`봉 하나를 골라 HDC 연산을 한 칸씩 넘깁니다. 계산은 이 브라우저에서 같은 부품(hdc.js)으로 다시 하고, 마지막에 파이썬이 낸 점수와 맞춰 봅니다.`);
  const sel = $("#follow-select");
  const sorted = [...F.bars].sort((a, b) => (b.date + b.hour).localeCompare(a.date + a.hour));
  sel.replaceChildren(...sorted.map((b) => el("option", { value: F.bars.indexOf(b) },
    `${b.date}(${b.wd}) ${b.hour}시대 ${fmt(b.vol)}주, ${b.level}단계, ${b.rank}위 ${b.verdict}`)));
  sel.onchange = () => buildFrames(F.bars[+sel.value]);
  document.querySelectorAll("[data-quick]").forEach((btn) => (btn.onclick = () => {
    const q = btn.dataset.quick;
    const b = q === "worst" ? F.bars.find((x) => x.rank === 1) : q === "best" ? F.bars.reduce((a, x) => (x.rank > a.rank ? x : a))
      : sorted[0];
    sel.value = F.bars.indexOf(b);
    buildFrames(b);
  }));
  $("#follow-prev").onclick = () => showFrame(F.i - 1);
  $("#follow-next").onclick = () => showFrame(F.i + 1);
  sel.value = F.bars.indexOf(F.bars.find((x) => x.rank === 1));
  await buildFrames(F.bars[+sel.value]);
}

async function cached(key, fn) {
  if (!(key in F.cache)) F.cache[key] = await fn();
  return F.cache[key];
}

async function buildFrames(bar) {
  $("#follow-stage").replaceChildren(el("p", { class: "hint" }, "브라우저에서 벡터를 만드는 중..."));
  const m = F.model, dim = m.dim, t = F.trace;
  const dayV = await cached(`d${bar.wd}`, () => H.makeVector(`요일:${bar.wd}`, dim));
  const hourV = await cached(`h${bar.hour}`, () => H.makeVector(`시간대:${bar.hour}`, dim));
  const ladder = await cached("ladder", () => H.levelVectors(m.ladder, m.levels, dim));
  const lv1 = ladder.vectors[0], lvk = ladder.vectors[bar.level - 1];
  const flipped = new Uint8Array(dim);
  for (let k = 0; k < bar.level - 1; k++) for (const p of ladder.flips[k]) flipped[p] = 1;
  const ds = H.bind(dayV, hourV);
  const barV = H.bind(ds, lvk);
  // 기억에서 이 봉 자신을 뺀다(파이썬과 같은 조건)
  const mem = H.Memory.fromJSON(m.memory);
  mem.add(barV, -1);
  const memV = await mem.toVector();
  const diff = H.bind(barV, memV);
  const d = H.ones(diff);
  const sim = H.similarity(barV, memV);
  const th = m.thresholds;
  const verdict = sim <= th.alert ? "이상" : sim <= th.watch ? "주의" : "평소";
  const genStep = stepOf(t, "벡터 생성"), flipStep = stepOf(t, "뒤집기"), bindStep = stepOf(t, "묶기"),
    addStep = stepOf(t, "더하기"), cmpStep = stepOf(t, "비교"), judge = stepOf(t, "일반 계산", "판정");
  const label = `${bar.date}(${bar.wd}) ${bar.hour}시대, ${fmt(bar.vol)}주, 그 시간대 중앙값의 ${fmt(bar.ratio)}배 → ${bar.level}단계`;

  F.frames = [
    { title: "① 요일 벡터 꺼내기", op: "벡터 생성", why: genStep,
      body: () => [el("div", { class: "grids" }, gridBox(`요일:${bar.wd}`, dayV))] },
    { title: "② 시간대 벡터 꺼내기", op: "벡터 생성", why: genStep,
      body: () => [el("div", { class: "grids" }, gridBox(`시간대:${bar.hour}`, hourV),
        el("div", { class: "chip" }, `요일 벡터와의 유사도 `, el("b", {}, fmt(H.similarity(dayV, hourV)))))] },
    { title: `③ ${bar.level}단계 벡터 꺼내기`, op: "뒤집기", why: flipStep,
      body: () => [el("div", { class: "grids" }, gridBox("1단계", lv1),
        gridBox(`${bar.level}단계`, lvk, `1단계에서 뒤집힌 칸 ${H.ones(flipped).toLocaleString("ko-KR")}개(주황)`, flipped)),
        el("div", { class: "chip" }, `1단계와의 유사도 `, el("b", {}, fmt(H.similarity(lv1, lvk))))] },
    { title: "④ 묶기: 요일 ⊕ 시간대", op: "묶기", why: bindStep,
      body: () => [el("div", { class: "grids" }, gridBox("A 요일", dayV), el("span", { class: "grid-op" }, "⊕"),
        gridBox("B 시간대", hourV), el("span", { class: "grid-op" }, "="), gridBox("A ⊕ B", ds, "두 칸이 다르면 1")),
        el("div", { class: "chip" }, `결과와 A의 유사도 `, el("b", {}, fmt(H.similarity(ds, dayV))))] },
    { title: "⑤ 묶기: (요일 ⊕ 시간대) ⊕ 단계 = 봉 벡터", op: "묶기", why: bindStep,
      body: () => [el("div", { class: "grids" }, gridBox("요일 ⊕ 시간대", ds), el("span", { class: "grid-op" }, "⊕"),
        gridBox(`${bar.level}단계`, lvk), el("span", { class: "grid-op" }, "="), gridBox("봉 벡터", barV)),
        el("div", { class: "chip" }, "되돌려 보기: 봉 벡터 ⊕ (요일 ⊕ 시간대) = ", el("b", {},
          H.hamming(H.bind(barV, ds), lvk) === 0 ? `${bar.level}단계 벡터와 모든 칸이 같음` : "다름"))] },
    { title: "⑥ 기억 꺼내기(이 봉 자신은 뺀다)", op: "더하기", why: addStep,
      body: () => [el("div", { class: "grids" }, gridBox("기억(다수결)", memV)),
        el("div", { class: "values" }, el("span", { class: "chip" }, "더해 둔 봉 ", el("b", {}, fmt(m.memory.n))),
          el("span", { class: "chip" }, "이 봉을 뺀 뒤 ", el("b", {}, fmt(mem.n))))] },
    { title: "⑦ 비교: 봉 벡터와 기억", op: "비교", why: cmpStep,
      body: () => [el("div", { class: "grids" }, gridBox("봉 벡터", barV), el("span", { class: "grid-op" }, "vs"),
        gridBox("기억", memV), el("span", { class: "grid-op" }, "→"), gridBox("다른 칸(주황)", barV, `${d.toLocaleString("ko-KR")}칸이 다름`, diff)),
        el("div", { class: "values" },
          el("span", { class: "chip" }, "해밍 거리 ", el("b", {}, fmt(d))),
          el("span", { class: "chip" }, `유사도 = 1 - 2 × ${d} / ${dim} = `, el("b", {}, fmt(sim))),
          el("span", { class: "chip" }, "잡음 수준 ", el("b", {}, fmt(H.noiseLevel(dim))))),
        el("div", { class: "check" }, `확인: 파이썬이 낸 점수 ${fmt(bar.score)}, 이 브라우저 ${fmt(Math.round(sim * 1e4) / 1e4)} `,
          el("span", { class: Math.abs(sim - bar.score) < 1e-4 ? "ok" : "bad" }, Math.abs(sim - bar.score) < 1e-4 ? "일치" : "어긋남"))] },
    { title: "⑧ 판정", op: "일반 계산", why: judge,
      body: () => [el("div", { class: "values" },
        el("span", { class: "chip" }, "점수 ", el("b", {}, fmt(sim))),
        el("span", { class: "chip" }, "이상 경계 ", el("b", {}, fmt(th.alert))),
        el("span", { class: "chip" }, "주의 경계 ", el("b", {}, fmt(th.watch))),
        el("span", { class: "chip" }, "판정 ", el("b", {}, verdict)),
        el("span", { class: "chip" }, `반년 ${F.bars.length}개 중 순위 `, el("b", {}, `${bar.rank}위`)))] },
  ];
  F.label = label;
  showFrame(0);
}

function showFrame(i) {
  F.i = Math.max(0, Math.min(F.frames.length - 1, i));
  const f = F.frames[F.i];
  $("#follow-stage").replaceChildren(
    el("div", { class: "frame-step" }, F.label),
    el("div", { class: "gear-head", style: `--op-color: var(${OP_COLOR[f.op]})` },
      el("span", { class: "frame-title" }, f.title), el("span", { class: "op" }, f.op)),
    whyList(f.why), ...f.body());
  $("#follow-pos").textContent = `${F.i + 1} / ${F.frames.length}`;
  $("#follow-prev").disabled = F.i === 0;
  $("#follow-next").disabled = F.i === F.frames.length - 1;
}

// ---------- 문자판 ----------
async function renderDial(stockId, trainTrace) {
  const s = trainTrace?.subject || { name: stockId, code: stockId };
  $("#dial-stock").textContent = `${s.name} (${s.code})`;
  if (trainTrace) {
    const tbl = trainTrace.steps.flatMap((x) => x.outputs).find((o) => o.name === "봉별 점수");
    const c = Object.fromEntries(tbl.columns.map((k, i) => [k, i]));
    const last = [...tbl.rows].sort((a, b) => (a[c["날짜"]] + a[c["시간대"]]).localeCompare(b[c["날짜"]] + b[c["시간대"]])).at(-1);
    $("#dial-score").replaceChildren(`거래량 점수 `, el("b", {}, fmt(last[c["점수"]])),
      ` ${last[c["판정"]]} (${last[c["날짜"]]} ${last[c["시간대"]]}시대)`);
  }
  $("#dial-trend").replaceChildren("오늘의 새 조합 ", el("b", {}, "준비 중(⑤ 단계)"));
}

// ---------- 시작 ----------
async function show() {
  const stock = $("#pick-stock").value, kind = $("#pick-view").value;
  usedTerms.clear();
  const pick = (k) => [...RUNS].reverse().find((r) => r.subject === stock && r.kind === k && r.trace && !(r.failures || []).some((f) => f.startsWith("수집 실패")));
  const trainRun = pick("volume.train"), run = pick(kind);
  const trainTrace = trainRun ? await getJSON(trainRun.trace) : null;
  await renderDial(stock, trainTrace);
  if (!run) { $("#gears").replaceChildren(el("li", {}, "아직 실행 기록이 없습니다.")); return; }
  const trace = run === trainRun ? trainTrace : await getJSON(run.trace);
  renderGears(trace, run.trace);
  await setupFollow(trace, `data/volume/${stock}`);
}

async function main() {
  [SITE, GLOSS, RUNS] = await Promise.all([getJSON("data/site.json"), getJSON("data/glossary.json"), getJSON("data/runs.json")]);
  renderRuns(RUNS);
  const stocks = [...new Set(RUNS.map((r) => r.subject))];
  $("#pick-stock").replaceChildren(...stocks.map((s) => el("option", { value: s }, s)));
  $("#pick-stock").onchange = show;
  $("#pick-view").onchange = show;
  await show();
}

main().catch((e) => {
  document.querySelector("main").prepend(el("div", { class: "panel" }, `불러오기 실패: ${e.message}`));
  console.error(e);
});
