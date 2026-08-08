/**
 * Dump buildSystemPrompt output for each parity fixture.
 *
 * Runs the REAL TypeScript buildSystemPrompt from the pi mono-repo and writes
 * golden/<name>.txt per fixture, so the Python port can be compared
 * character-for-character at runtime (not source-template level).
 *
 * Usage (from the pi-python repo root, with the pi checkout at the default
 * path, or override PI_TS_SYSTEM_PROMPT_SRC):
 *
 *   PI_PACKAGE_DIR=C:/pi-pkg \
 *     node --experimental-strip-types src/pi_agent/tests/parity/dump-system-prompt.ts
 *
 * PI_PACKAGE_DIR must match the value used by the Python side
 * (test_system_prompt_parity.py sets it to the same fixed path), otherwise the
 * "Pi documentation" paths in the prompt will differ between the two sides.
 * Requires Node >= 22.6 and that the pi mono-repo's npm dependencies are
 * installed (npm install in the pi repo root; only `ignore`, `yaml` and
 * `cross-spawn` are needed at runtime).
 */
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(HERE, "fixtures");
const GOLDEN_DIR = join(HERE, "golden");

// golden/ 可能不存在（例如刚 clone 或已清理），与 Python 侧
// dump_system_prompt.py 的 OUT_DIR.mkdir(parents=True, exist_ok=True) 对称。
mkdirSync(GOLDEN_DIR, { recursive: true });

const TS_SRC =
	process.env.PI_TS_SYSTEM_PROMPT_SRC ??
	"C:/coding/AI/Agent/pi/packages/coding-agent/src/core/system-prompt.ts";

const { buildSystemPrompt } = await import(pathToFileURL(TS_SRC).href);

interface SkillFixture {
	name: string;
	description: string;
	filePath: string;
	baseDir: string;
	source?: string;
	disableModelInvocation?: boolean;
}

function toSkill(s: SkillFixture): {
	name: string;
	description: string;
	filePath: string;
	baseDir: string;
	sourceInfo: Record<string, unknown>;
	disableModelInvocation: boolean;
} {
	return {
		name: s.name,
		description: s.description,
		filePath: s.filePath,
		baseDir: s.baseDir,
		sourceInfo: {
			path: s.filePath,
			source: s.source ?? "local",
			scope: "temporary",
			origin: "top-level",
			baseDir: s.baseDir,
		},
		disableModelInvocation: s.disableModelInvocation ?? false,
	};
}

function buildOptions(fixture: Record<string, unknown>): Record<string, unknown> {
	const skills = (fixture.skills ?? []) as SkillFixture[];
	return {
		customPrompt: fixture.customPrompt,
		selectedTools: fixture.selectedTools,
		toolSnippets: fixture.toolSnippets,
		promptGuidelines: fixture.promptGuidelines,
		appendSystemPrompt: fixture.appendSystemPrompt,
		cwd: fixture.cwd,
		contextFiles: fixture.contextFiles,
		skills: skills.map(toSkill),
	};
}

const names = readdirSync(FIXTURES_DIR)
	.filter((file) => file.endsWith(".json"))
	.map((file) => file.slice(0, -".json".length))
	.sort();

for (const name of names) {
	const fixture = JSON.parse(readFileSync(join(FIXTURES_DIR, `${name}.json`), "utf-8"));
	const prompt = buildSystemPrompt(buildOptions(fixture));
	writeFileSync(join(GOLDEN_DIR, `${name}.txt`), prompt, "utf-8");
	console.log(`wrote golden/${name}.txt (${prompt.length} chars)`);
}
