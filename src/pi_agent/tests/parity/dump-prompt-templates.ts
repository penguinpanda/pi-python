/**
 * Dump prompt-template functions (parseCommandArgs, substituteArgs,
 * expandPromptTemplate) by running the REAL TypeScript implementations from
 * the pi mono-repo (packages/coding-agent/src/core/prompt-templates.ts).
 *
 * Reads fixtures/prompt_templates.json (fixed inputs) and writes golden:
 *   - prompttemplates_substituteArgs_<i>.txt
 *   - prompttemplates_parseCommandArgs_<i>.txt (JSON array, compact separators)
 *   - prompttemplates_expandPromptTemplate_<i>.txt
 *
 * Usage (pi-python repo root; needs the pi checkout, see dump-system-prompt.ts):
 *
 *   node --experimental-strip-types \
 *     src/pi_agent/tests/parity/dump-prompt-templates.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "golden");

const TS_SRC =
	process.env.PI_TS_PROMPT_TEMPLATES_SRC ??
	"C:/coding/AI/Agent/pi/packages/coding-agent/src/core/prompt-templates.ts";

const { parseCommandArgs, substituteArgs, expandPromptTemplate } = await import(
	pathToFileURL(TS_SRC).href
);

mkdirSync(GOLDEN_DIR, { recursive: true });

const fixture = JSON.parse(readFileSync(join(HERE, "fixtures", "prompt_templates.json"), "utf-8"));

const writeCase = (prefix: string, index: number, output: string): void => {
	const file = join(GOLDEN_DIR, `${prefix}_${index}.txt`);
	writeFileSync(file, output, "utf-8");
	console.log(`wrote ${prefix}_${index}.txt (${output.length} chars)`);
};

for (const [index, c] of (fixture.substituteArgs as { content: string; args: string[] }[]).entries()) {
	writeCase("prompttemplates_substituteArgs", index, substituteArgs(c.content, c.args));
}

for (const [index, s] of (fixture.parseCommandArgs as string[]).entries()) {
	// JSON 数组（紧凑分隔符），Python 侧用相同序列化方式比较。
	writeCase("prompttemplates_parseCommandArgs", index, JSON.stringify(parseCommandArgs(s)));
}

for (const [index, c] of (
	fixture.expandPromptTemplate as {
		text: string;
		templates: { name: string; content: string }[];
	}[]
).entries()) {
	writeCase(
		"prompttemplates_expandPromptTemplate",
		index,
		expandPromptTemplate(c.text, c.templates as never),
	);
}
