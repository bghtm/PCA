/**
 * 校验 03_iris_pca_可视化.html：
 *   1) 页面里所有 <script> 块语法可通过编译；
 *   2) 从页面抽取 PCA 内核 + 内嵌数据，独立运行，和 numpy 的期望值对拍。
 * 用法: node build/verify_html.js
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const HTML = path.join(ROOT, "03_iris_pca_可视化.html");
const html = fs.readFileSync(HTML, "utf8");

let fail = 0;
const ok = (cond, msg, extra) => {
  if (cond) {
    console.log("  PASS  " + msg + (extra ? "  " + extra : ""));
  } else {
    fail++;
    console.log("  FAIL  " + msg + (extra ? "  " + extra : ""));
  }
};

/* ---------- 1. 语法检查 ---------- */
console.log("\n[1] <script> 语法检查");
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
console.log("  找到 " + scripts.length + " 个 script 块");
scripts.forEach((code, i) => {
  try {
    // 只编译不执行
    new Function(code);
    console.log("  PASS  script[" + i + "] 语法 OK（" + code.length + " 字符）");
  } catch (e) {
    fail++;
    console.log("  FAIL  script[" + i + "] 语法错误: " + e.message);
  }
});

/* ---------- 2. 抽取数学内核 + 数据 ---------- */
console.log("\n[2] 抽取 PCA 内核与数据");
function slice(src, a, b, name) {
  const i = src.indexOf(a);
  const j = src.indexOf(b);
  if (i < 0 || j < 0 || j <= i) throw new Error("找不到标记: " + name);
  return src.slice(i + a.length, j);
}
const dataBlock = slice(html, "/* IRIS-DATA-BEGIN */", "/* IRIS-DATA-END */", "IRIS-DATA");
const mathBlock = slice(html, "/* ==PCA-MATH-BEGIN== */", "/* ==PCA-MATH-END== */", "PCA-MATH");
console.log("  数据块 " + dataBlock.length + " 字符，数学块 " + mathBlock.length + " 字符");

const api = new Function(
  dataBlock + "\n" +
  mathBlock + "\n" +
  "return { IRIS, pcaFit, reconstruct, rmseOf, jacobiEigen, covariance, matMean, matStd, preprocess };"
)();

const IRIS = api.IRIS;
console.log("  样本数 = " + IRIS.x.length + "，特征数 = " + IRIS.x[0].length +
            "，类别分布 = " + JSON.stringify([0, 1, 2].map(c => IRIS.y.filter(v => v === c).length)));

/* ---------- 3. 与 numpy 对拍 ---------- */
console.log("\n[3] 与 numpy 期望值对拍");
const expected = JSON.parse(fs.readFileSync(path.join(ROOT, "build", "expected.json"), "utf8"));

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  }
  return m;
}

for (const mode of ["raw", "std"]) {
  const standardize = mode === "std";
  const fit = api.pcaFit(IRIS.x, { standardize });
  const exp = expected[mode];
  const tag = standardize ? "标准化" : "仅中心化";

  const dVal = Math.max(...fit.eigvals.map((v, i) => Math.abs(v - exp.eigvals[i])));
  ok(dVal < 1e-9, tag + " · 特征值一致", "最大误差 " + dVal.toExponential(2));

  const dRatio = Math.max(...fit.ratio.map((v, i) => Math.abs(v - exp.ratio[i])));
  ok(dRatio < 1e-9, tag + " · 方差贡献率一致", "最大误差 " + dRatio.toExponential(2));

  const dComp = Math.max(...fit.comps.map((row, c) => Math.max(...row.map((v, j) => Math.abs(v - exp.comps[c][j])))));
  ok(dComp < 1e-8, tag + " · 主成分方向一致（符号已规范）", "最大误差 " + dComp.toExponential(2));

  const dScore = Math.max(...fit.scores.map((row, i) => Math.max(...row.map((v, c) => Math.abs(v - exp.scores[i][c])))));
  ok(dScore < 1e-7, tag + " · 主成分得分一致", "最大误差 " + dScore.toExponential(2));

  /* 正交性：P Pᵀ = I */
  const p = fit.comps.length;
  let orthoErr = 0;
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) {
      let s = 0;
      for (let j = 0; j < 4; j++) s += fit.comps[a][j] * fit.comps[b][j];
      orthoErr = Math.max(orthoErr, Math.abs(s - (a === b ? 1 : 0)));
    }
  }
  ok(orthoErr < 1e-10, tag + " · 主成分两两正交（PPᵀ=I）", "最大误差 " + orthoErr.toExponential(2));

  /* 重构：k=1..4 的 RMSE 单调下降，k=4 归零 */
  const rmses = [1, 2, 3, 4].map(k => api.rmseOf(IRIS.x, api.reconstruct(fit, k)));
  const mono = rmses.every((v, i) => i === 0 || v < rmses[i - 1]);
  ok(mono, tag + " · RMSE 随 k 单调下降", rmses.map(v => v.toFixed(4)).join(" > "));
  ok(rmses[3] < 1e-9, tag + " · k=4 重构无损", "RMSE=" + rmses[3].toExponential(2));
  const expK2 = exp.ratio[0] + exp.ratio[1];
  ok(Math.abs(fit.ratio.slice(0, 2).reduce((a, b) => a + b, 0) - expK2) < 1e-6,
     tag + " · k=2 累计贡献率与 numpy 一致",
     (fit.ratio[0] + fit.ratio[1]).toFixed(6));
}

/* ---------- 4. 结构自检 ---------- */
console.log("\n[4] 页面结构自检");
["cRaw", "cPca", "cScree", "c3d", "k-slider", "std-toggle", "class-toggle", "loadings",
 "recon-table", "summary", "m-keep", "m-rmse", "m-comp", "m-k", "play-btn", "reset-btn"]
  .forEach(id => ok(html.includes('id="' + id + '"'), 'id="' + id + '" 存在'));
ok(!/https?:\/\//.test(html.replace(/<!--[\s\S]*?-->/g, "")), "无外部 http(s) 资源引用（离线可用）");
ok((html.match(/<canvas/g) || []).length === 4, "4 个 canvas",
   "实际 " + (html.match(/<canvas/g) || []).length);

console.log("\n==== 结果: " + (fail === 0 ? "全部通过 ✅" : fail + " 项失败 ❌") + " ====\n");
process.exit(fail === 0 ? 0 : 1);
