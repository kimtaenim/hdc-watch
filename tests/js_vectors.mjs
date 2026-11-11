// 파이썬 테스트가 부르는 보조 스크립트. 요청 JSON을 받아 JS판 hdc.js로 계산한 결과를 16진수로 낸다.
import { readFileSync } from "node:fs";
import { makeVector, levelVectors, bind, Memory, toHex } from "../docs/js/hdc.js";

const req = JSON.parse(readFileSync(process.argv[2], "utf-8"));
const out = { vectors: [], levels: [], bundles: [] };

for (const { name, dim } of req.vectors) out.vectors.push(toHex(await makeVector(name, dim)));

for (const { name, levels, dim } of req.levels) {
  const { vectors } = await levelVectors(name, levels, dim);
  out.levels.push(vectors.map(toHex));
}

for (const { pairs, dim } of req.bundles) {
  const m = new Memory(dim);
  for (const [a, b] of pairs) m.add(bind(await makeVector(a, dim), await makeVector(b, dim)));
  out.bundles.push({ counts: Array.from(m.counts), vector: toHex(await m.toVector()) });
}

process.stdout.write(JSON.stringify(out));
