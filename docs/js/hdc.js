// hdc-watch 공용 HDC 부품, 브라우저판.
// 파이썬 hdc/vectors.py, hdc/ops.py와 같은 이름에서 완전히 같은 벡터를 만든다(tests/test_hdc.py가 확인).
// 해시는 브라우저 내장 Web Crypto의 SHA-256을 쓴다. 난수 생성기는 쓰지 않는다.

export const D = 10000;
export const TIE_NAME = "__동점깨기__";

const encoder = new TextEncoder();

async function sha256(text) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(text)));
}

// 이름 + "#" + 일련번호를 해시해 큰 자리 비트부터 이어 붙인다.
export async function makeVector(name, dim = D) {
  const bits = new Uint8Array(dim);
  let filled = 0;
  for (let i = 0; filled < dim; i++) {
    const digest = await sha256(`${name}#${i}`);
    for (const byte of digest) {
      for (let k = 7; k >= 0 && filled < dim; k--) bits[filled++] = (byte >> k) & 1;
    }
  }
  return bits;
}

// 칸마다 "뒤집기|이름|칸번호" 해시 앞 8바이트를 제비 번호로 주고 작은 순으로 줄 세운다.
export async function flipOrder(name, dim = D) {
  const digests = await Promise.all(
    Array.from({ length: dim }, (_, p) => sha256(`뒤집기|${name}|${p}`)));
  const keys = digests.map((h) => {
    let x = 0n;
    for (let j = 0; j < 8; j++) x = (x << 8n) | BigInt(h[j]);
    return x;
  });
  const order = Array.from({ length: dim }, (_, p) => p);
  order.sort((a, b) => (keys[a] < keys[b] ? -1 : keys[a] > keys[b] ? 1 : a - b));
  return order;
}

export function levelBoundaries(levels, dim = D) {
  const half = Math.floor(dim / 2);
  return Array.from({ length: levels }, (_, k) => Math.floor((k * half) / (levels - 1)));
}

// 1단계 = makeVector(name). 단계가 오를 때마다 겹치지 않는 칸 묶음을 뒤집는다.
export async function levelVectors(name, levels = 12, dim = D) {
  const order = await flipOrder(name, dim);
  const bounds = levelBoundaries(levels, dim);
  const vectors = [await makeVector(name, dim)];
  const flips = [];
  for (let k = 1; k < levels; k++) {
    const cells = order.slice(bounds[k - 1], bounds[k]);
    const v = vectors[vectors.length - 1].slice();
    for (const p of cells) v[p] ^= 1;
    vectors.push(v);
    flips.push(cells);
  }
  return { vectors, flips };
}

// 묶기 = XOR. 풀기도 같은 XOR.
export function bind(a, b) {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] ^ b[i];
  return out;
}
export const unbind = bind;

export function hamming(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i++) n += a[i] ^ b[i];
  return n;
}

export function similarity(a, b) {
  return 1 - (2 * hamming(a, b)) / a.length;
}

export function noiseLevel(dim) {
  return 1 / Math.sqrt(dim);
}

export function ones(v) {
  let n = 0;
  for (const b of v) n += b;
  return n;
}

// 더하기 = 칸마다 정수 카운터. 1이면 +1, 0이면 -1.
export class Memory {
  constructor(dim = D, counts = null, n = 0) {
    this.dim = dim;
    this.counts = counts ? Int32Array.from(counts) : new Int32Array(dim);
    this.n = n;
  }
  add(v, weight = 1) {
    for (let i = 0; i < this.dim; i++) this.counts[i] += weight * (v[i] ? 1 : -1);
    this.n += weight;
  }
  // 다수결. 0:0으로 비긴 칸은 동점깨기 벡터의 칸을 따른다.
  async toVector() {
    const tie = await makeVector(TIE_NAME, this.dim);
    const out = new Uint8Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const c = this.counts[i];
      out[i] = c > 0 ? 1 : c < 0 ? 0 : tie[i];
    }
    return out;
  }
  static fromJSON(data) {
    return new Memory(data.dim, data.counts, data.n);
  }
}

export function nearest(query, codebook) {
  return Object.entries(codebook)
    .map(([name, v]) => [name, similarity(query, v)])
    .sort((a, b) => b[1] - a[1]);
}

// 추적 기록의 bits_b64 → 0/1 배열
export function bitsFromB64(text, dim) {
  const raw = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  const bits = new Uint8Array(dim);
  for (let i = 0; i < dim; i++) bits[i] = (raw[i >> 3] >> (7 - (i & 7))) & 1;
  return bits;
}

export function toHex(bits) {
  let hex = "";
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0;
    for (let k = 0; k < 8; k++) byte = (byte << 1) | (bits[i + k] ?? 0);
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}
