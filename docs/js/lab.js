// 실험대와 확인 화면. 같은 HDC 부품(hdc.js)으로 브라우저에서 바로 계산한다.
import * as H from "./hdc.js";

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
const r = (v, d = 3) => (v == null ? "" : Number(v.toFixed(d)));
const chip = (label, v) => el("span", { class: "chip" }, label, " ", el("b", {}, v));
const table = (head, rows, mark = () => false) => el("div", { class: "table-wrap flat" }, el("table", { class: "coltable" },
  el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
  el("tbody", {}, rows.map((row, i) => el("tr", { class: mark(row, i) ? "sel" : "" }, row.map((v, j) => (j === 0 ? el("th", {}, v) : el("td", {}, v))))))));
const box = (no, title, ...body) => el("li", { class: "gear" }, el("div", { class: "gear-head" },
  el("span", { class: "gear-no" }, no), el("span", { class: "gear-name" }, title)), ...body);
const bars = (vals, max) => el("div", { class: "hbars" }, vals.map(([label, v, note]) => el("div", { class: "hbar" },
  el("span", { class: "hl" }, label), el("span", { class: "track" }, el("span", { class: "hb", style: `width:${Math.max(0, Math.min(1, v / max) * 100)}%` })),
  el("span", { class: "hv" }, note ?? r(v)))));
// 앞 4칸 ··· 뒤 4칸
function bitRows(rows, dim) {
  const cols = [0, 1, 2, 3, null, dim - 4, dim - 3, dim - 2, dim - 1];
  return el("div", { class: "bitrows" }, el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "칸"), cols.map((c) => el("th", { class: "pos" }, c === null ? "" : c + 1)))),
    el("tbody", {}, rows.map((x) => el("tr", { class: x.line ? "result" : "" }, el("th", { class: "rname" }, x.name),
      cols.map((c) => (c === null ? el("td", { class: "dots" }, "···") : el("td", { class: x.mark && x.mark[c] ? "hl" : "" }, x.bits[c]))))))));
}
// 순위(같은 값은 평균 순위)와 순위 상관
function ranks(xs) {
  const idx = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(xs.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) out[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return out;
}
function spearman(a, b) {
  const ra = ranks(a), rb = ranks(b), n = a.length, ma = (n + 1) / 2;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - ma); da += (ra[i] - ma) ** 2; db += (rb[i] - ma) ** 2; }
  return num / Math.sqrt(da * db);
}
// 정수 프로토타입을 요일·시간대로 풀어 단계마다 코사인. 봉 = 요일 ⊕ 시간대 ⊕ 단계(세 벡터 XOR = ±1 곱)
function askContext(counts, dv, sv, levels) {
  const dim = dv.length, q = new Float64Array(dim);
  for (let i = 0; i < dim; i++) q[i] = counts[i] * (dv[i] ? 1 : -1) * (sv[i] ? 1 : -1);
  const nq = Math.sqrt(q.reduce((a, x) => a + x * x, 0));
  return levels.map((L) => { let d = 0; for (let i = 0; i < dim; i++) d += q[i] * (L[i] ? 1 : -1); return d / (Math.sqrt(dim) * nq); });
}

export async function renderLab(root, ctx) {
  const { model } = ctx;
  const dim = model.dim;
  const ladder = await H.levelVectors(model.ladder, model.levels, dim);
  root.replaceChildren();

  // ① 두 이름
  const a = el("input", { value: "요일:수", "aria-label": "이름 A" }), b = el("input", { value: "시간대:14:00", "aria-label": "이름 B" });
  const out1 = el("div");
  const run1 = async () => {
    const va = await H.makeVector(a.value, dim), vb = await H.makeVector(b.value, dim), d = H.bind(va, vb);
    out1.replaceChildren(bitRows([{ name: "A", bits: va }, { name: "B", bits: vb }, { name: "다른 칸", bits: d, mark: d, line: true }], dim),
      el("div", { class: "values" }, chip("다른 칸", H.ones(d).toLocaleString("ko-KR")), chip("유사도", r(H.similarity(va, vb), 4)),
        chip("무관하면", "0 근처, 흔들림 약 0.01")));
  };
  root.append(box("①", "이름 두 개: 얼마나 닮았나", el("div", { class: "lab-in" }, a, b, el("button", { onclick: run1 }, "계산")), out1));
  await run1();

  // ② 단계 옮기기
  const base = el("select", {}, ladder.vectors.map((_, i) => el("option", { value: i }, `${i + 1}단계`)));
  base.value = "5";
  const out2 = el("div");
  const run2 = () => {
    const j = +base.value;
    out2.replaceChildren(bars(ladder.vectors.map((v, k) => [`${k + 1}단계`, H.similarity(ladder.vectors[j], v)]), 1));
  };
  base.onchange = run2;
  root.append(box("②", "거래량 단계를 옮기면", el("div", { class: "lab-in" }, "기준 ", base), out2));
  run2();

  // ③ 칸 수와 우연한 닮음
  const dsel = el("select", {}, [100, 1000, 10000].map((n) => el("option", { value: n }, `${n.toLocaleString("ko-KR")}칸`)));
  dsel.value = "1000";
  const out3 = el("div");
  const run3 = async () => {
    const n = +dsel.value, sims = [];
    out3.replaceChildren("계산 중...");
    for (let i = 0; i < 200; i++) sims.push(H.similarity(await H.makeVector(`실험a${i}`, n), await H.makeVector(`실험b${i}`, n)));
    const mean = sims.reduce((x, y) => x + y, 0) / sims.length;
    const sd = Math.sqrt(sims.reduce((x, y) => x + (y - mean) ** 2, 0) / sims.length);
    const edges = [-0.3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3];
    const hist = edges.slice(0, -1).map((e, i) => [`${e} - ${edges[i + 1]}`, sims.filter((s) => s >= e && s < edges[i + 1]).length]);
    const outside = sims.filter((s) => s < -0.3 || s >= 0.3).length;
    out3.replaceChildren(el("div", { class: "values" }, chip("무관한 쌍", 200), chip("흔들림(표준편차)", r(sd, 4)),
      chip("이론 1/√칸 수", r(1 / Math.sqrt(n), 4)), chip("가장 큰 |유사도|", r(Math.max(...sims.map(Math.abs)), 3))),
      bars([...hist.map(([l, c]) => [l, c, `${c}쌍`]), ["그 밖", outside, `${outside}쌍`]], 200));
  };
  dsel.onchange = run3;
  root.append(box("③", "칸 수를 바꾸면 우연한 닮음이", el("div", { class: "lab-in" }, dsel), out3));
  await run3();

  // ④ 프로토타입에 요일·시간대를 물으면
  const wsel = el("select", {}, [..."월화수목금"].map((w) => el("option", { value: w }, `${w}요일`)));
  const ssel = el("select", {}, model.sessions.map((s) => el("option", { value: s }, s)));
  wsel.value = "수"; ssel.value = model.sessions.includes("14:00") ? "14:00" : model.sessions[0];
  const out4 = el("div");
  const run4 = async () => {
    const dv = await H.makeVector(`요일:${wsel.value}`, dim), sv = await H.makeVector(`시간대:${ssel.value}`, dim);
    const sims = askContext(model.memory.counts, dv, sv, ladder.vectors);
    const best = sims.indexOf(Math.max(...sims)) + 1;
    const lv = ctx.bars.filter((x) => x.wd === wsel.value && x.slot === ssel.value).map((x) => x.level).sort((p, q) => p - q);
    const cnt = Array.from({ length: model.levels }, (_, k) => lv.filter((x) => x === k + 1).length);
    const med = lv.length ? lv[Math.floor((lv.length - 1) / 2)] : null;
    out4.replaceChildren(
      el("div", { class: "tname" }, "프로토타입에 물은 답(단계마다 닮음)"), bars(sims.map((s, k) => [`${k + 1}단계`, Math.max(0, s), r(s)]), Math.max(...sims)),
      el("div", { class: "tname" }, `실제로 센 답(이 칸의 봉 ${lv.length}개)`), bars(cnt.map((c, k) => [`${k + 1}단계`, c, `${c}개`]), Math.max(1, ...cnt)),
      el("div", { class: "values" }, chip("HDC 답", `${best}단계`), chip("센 답(중앙값)", med ? `${med}단계` : "없음")));
  };
  wsel.onchange = run4; ssel.onchange = run4;
  root.append(box("④", "프로토타입에 '요일 ⊕ 시간'을 물으면", el("div", { class: "lab-in" }, wsel, ssel), out4));
  await run4();
}

export async function renderCheck(root, ctx) {
  const { model, bars: B } = ctx;
  const dim = model.dim;
  root.replaceChildren();

  // ① 거래량 점수와 단순 셈
  const rho = spearman(B.map((x) => x.score), B.map((x) => x.countSum));
  const rh = ranks(B.map((x) => x.score)), rc = ranks(B.map((x) => x.countSum));
  const diff = B.map((x, i) => ({ ...x, rh: rh[i], rc: rc[i], gap: Math.abs(rh[i] - rc[i]) })).sort((p, q) => q.gap - p.gap).slice(0, 10);
  root.append(box("①", "거래량: HDC 점수 순서와 단순 셈 순서",
    el("div", { class: "values" }, chip("순위 상관", r(rho, 3)), chip("봉", B.length)),
    el("div", { class: "tname" }, "가장 많이 어긋난 10개"),
    table(["봉", "단계", "HDC 순위", "셈 순위"], diff.map((x) => [`${x.date.slice(5)} ${x.wd} ${x.slot}`, x.level, Math.round(x.rh), Math.round(x.rc)]))));

  // ② 요일·시간 칸마다 풀어 묻기
  const ladder = await H.levelVectors(model.ladder, model.levels, dim);
  const rows = [];
  let ok = 0, n = 0;
  for (const w of "월화수목금") {
    const dv = await H.makeVector(`요일:${w}`, dim);
    for (const s of model.sessions) {
      const sv = await H.makeVector(`시간대:${s}`, dim);
      const sims = askContext(model.memory.counts, dv, sv, ladder.vectors);
      const best = sims.indexOf(Math.max(...sims)) + 1;
      const lv = B.filter((x) => x.wd === w && x.slot === s).map((x) => x.level).sort((p, q) => p - q);
      if (!lv.length) continue;
      const med = lv[Math.floor((lv.length - 1) / 2)];
      n++; if (Math.abs(best - med) <= 1) ok++;
      rows.push([`${w} ${s}`, `${best}단계`, `${med}단계`, Math.abs(best - med) <= 1 ? "" : "어긋남"]);
    }
  }
  root.append(box("②", "요일·시간 칸마다 '평소 단계': 프로토타입에 물은 답과 센 답(중앙값)",
    el("div", { class: "values" }, chip("1단계 이내로 맞은 칸", `${ok} / ${n}`)),
    table(["칸", "HDC 답", "센 답", ""], rows.filter((x) => x[3]).concat(rows.filter((x) => !x[3]).slice(0, 5)), (x) => !!x[3])));

  // ③ 트렌드 새 낱말
  const tr = [];
  for (const k of ctx.keywords) {
    try {
      const d = await (await fetch(`data/trend/${k.id}/day.json`, { cache: "no-cache" })).json();
      const both = d.rows.filter((x) => x[5] && x[6]).length, hOnly = d.rows.filter((x) => x[5] && !x[6]).map((x) => x[0]),
        cOnly = d.rows.filter((x) => !x[5] && x[6]).map((x) => x[0]);
      tr.push([k.name, both, hOnly.join(", ") || "-", cOnly.join(", ") || "-"]);
    } catch { tr.push([k.name, "-", "자료 없음", ""]); }
  }
  root.append(box("③", "트렌드: HDC가 고른 새 낱말과 센 답(과거 0번)이 고른 새 낱말",
    table(["키워드", "둘 다", "HDC만", "센 답만"], tr, (x) => x[2] !== "-" || x[3] !== "-")));

  // ④ 날짜별 나란히
  const days = [...new Set(B.map((x) => x.date))].sort().reverse().slice(0, 20);
  let daily = {};
  try { daily = await (await fetch("data/trend/daily.json", { cache: "no-cache" })).json(); } catch {}
  root.append(box("④", "날짜별로 나란히: 거래량에서 가장 드문 봉과 그날 새 낱말 수",
    el("p", { class: "hint" }, "둘은 나란히 놓았을 뿐이고, 서로 관계가 있다고 말하지 않는다."),
    table(["날짜", "가장 드문 봉", "점수", "판정", "새 낱말"], days.map((d) => {
      const w = B.filter((x) => x.date === d).reduce((p, q) => (q.score < p.score ? q : p));
      const t = daily[d];
      return [d.slice(5), w.slot, w.score, w.verdict, t ? `${t.new_total}${t.warmup ? " (예열)" : ""}` : "-"];
    }), (x) => x[3] !== "평소")));
}
