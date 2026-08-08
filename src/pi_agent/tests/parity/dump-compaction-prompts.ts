/**
 * Dump compaction/branch-summarization prompt constants to golden files.
 *
 * These template constants are NOT exported by the TS modules (the functions
 * that consume them trigger real LLM calls), so they are extracted verbatim
 * from the source with a regex — no normalization, no strip. The extracted
 * literal IS the runtime value (plain template literals without interpolation).
 *
 * Writes golden/compaction_<NAME>.txt per constant, consumed by
 * test_compaction_prompts_parity.py which compares against the Python runtime
 * constants.
 *
 * Usage (pi-python repo root; needs the pi checkout, see dump-system-prompt.ts):
 *
 *   node --experimental-strip-types \
 *     src/pi_agent/tests/parity/dump-compaction-prompts.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, "golden");

const TS_COMPACTION_DIR =
	process.env.PI_TS_COMPACTION_DIR ??
	"C:/coding/AI/Agent/pi/packages/coding-agent/src/core/compaction";

const CONSTANTS: Record<string, string> = {
	SUMMARIZATION_SYSTEM_PROMPT: "utils.ts",
	SUMMARIZATION_PROMPT: "compaction.ts",
	UPDATE_SUMMARIZATION_PROMPT: "compaction.ts",
	TURN_PREFIX_SUMMARIZATION_PROMPT: "compaction.ts",
	BRANCH_SUMMARY_PROMPT: "branch-summarization.ts",
	BRANCH_SUMMARY_PREAMBLE: "branch-summarization.ts",
};

mkdirSync(GOLDEN_DIR, { recursive: true });

for (const [name, file] of Object.entries(CONSTANTS)) {
	const source = readFileSync(join(TS_COMPACTION_DIR, file), "utf-8");
	const match = source.match(new RegExp(`(?:export )?const ${name}\\s*=\\s*\`([\\s\\S]*?)\`;`));
	if (!match) {
		throw new Error(`Cannot find const ${name} in ${file}`);
	}
	const literal = match[1];
	writeFileSync(join(GOLDEN_DIR, `compaction_${name}.txt`), literal, "utf-8");
	console.log(`wrote golden/compaction_${name}.txt (${literal.length} chars)`);
}
