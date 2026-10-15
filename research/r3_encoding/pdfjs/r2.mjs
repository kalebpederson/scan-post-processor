// usage: node render.mjs in.pdf out.png dpi   (pdf.js legacy build + @napi-rs/canvas)
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { createCanvas } from "@napi-rs/canvas";
import fs from "fs";
const [,, inp, out, dpi] = process.argv;
const data = new Uint8Array(fs.readFileSync(inp));
const doc = await getDocument({ data, verbosity: 5 }).promise;
const page = await doc.getPage(1);
const vp = page.getViewport({ scale: Number(dpi) / 72 });
const canvas = createCanvas(Math.round(vp.width), Math.round(vp.height));
const ctx = canvas.getContext("2d");
ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
await page.render({ canvasContext: ctx, viewport: vp, canvas }).promise;
fs.writeFileSync(out, canvas.toBuffer("image/png"));
