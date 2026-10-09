import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const pkg = resolve(import.meta.dirname, "..");
const manifest = JSON.parse(readFileSync(resolve(pkg, "package.json"), "utf8")) as {
  types: string;
  exports: Record<string, Record<string, Record<string, string>> | string>;
};

describe("build output (npm run build)", () => {
  it("emits every file package.json points at, including .d.ts", () => {
    const targets = new Set<string>([manifest.types]);
    for (const entry of Object.values(manifest.exports)) {
      if (typeof entry === "string") continue;
      for (const condition of Object.values(entry)) for (const file of Object.values(condition)) targets.add(file);
    }
    const missing = [...targets].filter((file) => !existsSync(resolve(pkg, file)));
    expect(missing, "run `npm run build` first").toEqual([]);
    const dts = readFileSync(resolve(pkg, "dist/index.d.ts"), "utf8");
    expect(dts).toContain("declare class HypergraphClient");
    expect(readFileSync(resolve(pkg, "dist/ai-sdk.d.ts"), "utf8")).toContain("hypergraphMemoryMiddleware");
  });
});
