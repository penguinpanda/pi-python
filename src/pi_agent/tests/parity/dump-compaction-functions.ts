/**
 * Dump compaction function outputs (formatFileOperations, serializeConversation)
 * by running the REAL TypeScript implementations from the pi mono-repo.
 *
 * Reads fixtures/compaction_functions.json (fixed inputs) and writes
 * golden/compaction_formatFileOperations_<i>.txt and
 * golden/compaction_serializeConversation_<i>.txt.
 *
 * Needs the pi mono-repo npm deps installed (see dump-system-prompt.ts) AND a
 * minimal shim for @earendil-works/pi-ai (which has no built dist/):
 *
 *   mkdir -p <pi>/node_modules/@earendil-works/pi-ai
 *   # package.json: {"name":"@earendil-works/pi-ai","type":"module",
 *   #                "main":"./index.ts","exports":{".":{"import":"./index.ts"}}}
 *   # index.ts: export { contentText } from "../../../packages/ai/src/utils/text.ts";
 *
 * The shim lives in gitignored node_modules/ and re-exports the real source.
 *
 * Usage (pi-python repo root):
 *
 *   node --experimental-strip-types \
 *     src/pi_agent/tests/parity/dump-compaction-functions.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "golden");

const TS_UTILS =
	process.env.PI_TS_COMPACTION_UTILS ??
	"C:/coding/AI/Agent/pi/packages/coding-agent/src/core/compaction/utils.ts";

const { formatFileOperations, serializeConversation } = await import(pathToFileURL(TS_UTILS).href);

mkdirSync(GOLDEN_DIR, { recursive: true });

const fixture = JSON.parse(readFileSync(join(HERE, "fixtures", "compaction_functions.json"), "utf-8"));

const writeCase = (prefix: string, index: number, output: string): void => {
	const file = join(GOLDEN_DIR, `${prefix}_${index}.txt`);
	writeFileSync(file, output, "utf-8");
	console.log(`wrote ${prefix}_${index}.txt (${output.length} chars)`);
};

for (const [index, c] of (fixture.formatFileOperations as { readFiles: string[]; modifiedFiles: string[] }[]).entries()) {
	writeCase("compaction_formatFileOperations", index, formatFileOperations(c.readFiles, c.modifiedFiles));
}

for (const [index, messages] of (fixture.serializeConversation as unknown[][]).entries()) {
	writeCase("compaction_serializeConversation", index, serializeConversation(messages as never));
}
