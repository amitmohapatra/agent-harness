// Every ```mermaid block in the given Markdown files parses (make docs-mermaid).
//
//   MERMAID_MODULES=<a node_modules with mermaid and jsdom> node scripts/check_mermaid.mjs *.md
//
// (npm install --prefix <dir> mermaid jsdom; MERMAID_MODULES=<dir>/node_modules). mermaid parses
// in a browser; jsdom stands in for one. Exit 1 when any diagram fails.
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

const modules = process.env.MERMAID_MODULES || path.resolve("node_modules");
const require = createRequire(path.join(modules, "noop.js"));
const { JSDOM } = require("jsdom");

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.DOMParser = dom.window.DOMParser;
const { default: mermaid } = await import(
  pathToFileURL(path.join(modules, "mermaid", "dist", "mermaid.core.mjs")).href
);
mermaid.initialize({ startOnLoad: false });

let failed = 0;
let total = 0;
for (const file of process.argv.slice(2)) {
  const text = fs.readFileSync(file, "utf8");
  const fence = /^```mermaid\n([\s\S]*?)^```/gm;
  let match;
  while ((match = fence.exec(text))) {
    total++;
    const line = text.slice(0, match.index).split("\n").length;
    try {
      await mermaid.parse(match[1]);
    } catch (error) {
      failed++;
      const message = String(error.message || error).split("\n").slice(0, 3).join(" | ");
      console.log(`${file}:${line}: ${message}`);
    }
  }
}
console.log(`${total} Mermaid diagrams, ${failed} failed`);
process.exit(failed ? 1 : 0);
