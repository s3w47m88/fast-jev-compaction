import type {
  CommandSpec,
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import type { CompactProgress } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  onProgress?: (progress: CompactProgress) => void,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(
    messages,
    jevAsker(fetchFn, config.apiKey, config.model),
    config,
    onProgress,
  );
  return { result, messages: toSessionMessages(messages, result.messages) };
}

/** The ordered stages the status line lights up as compaction runs. */
const STATUS_STAGES: readonly { key: CompactProgress['stage']; label: string }[] = [
  { key: 'scan', label: 'scan' },
  { key: 'score', label: 'score' },
  { key: 'prune', label: 'prune' },
];

/**
 * Renders one staged status line for the prompt toolbar, e.g.
 * `Jev ✓scan ◍score 2/4 ·prune`. A finished (`done`) or truncated (`charsAfter`)
 * result collapses to a one-line summary instead.
 */
export function stageStatus(progress: CompactProgress): string {
  if (progress.stage === 'done') return '';
  const activeIndex = STATUS_STAGES.findIndex((s) => s.key === progress.stage);
  const pips = STATUS_STAGES.map((s, index) => {
    if (index < activeIndex) return `✓${s.label}`;
    if (index > activeIndex) return `·${s.label}`;
    if (progress.stage === 'score') return `◍${s.label} ${progress.done}/${progress.total}`;
    return `◍${s.label}`;
  });
  return `Jev ${pips.join('  ')}`;
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

/**
 * A single durable line recording the stages that ran, e.g.
 * `Jev ✓scan ✓score 1 batch ✓prune · kept 5/6 (22%)`. Rendered through
 * `$.ui.log` so it persists in scrollback — the under-prompt status pips flash
 * by in a sub-second on a small (single-batch) session, so this is the signal
 * the user can actually read after the fact.
 */
export function stagedSummary(result: CompactResult, kept: number, total: number): string {
  const batches = result.stats.requests;
  return `Jev ✓scan ✓score ${batches} batch${batches === 1 ? '' : 'es'} ✓prune · kept ${kept}/${total} (${percent(reductionRatio(result))})`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const JEV_COMPACT_COMMAND = 'jevcompact';
export const JEV_COMPACT_DESCRIPTION =
  'Runs a Jev compaction of this session now, without waiting for the context threshold.';

/** The `$` surface `registerJevCompactCommand` needs, so it can be tested without an engine. */
export type CommandRegisterHook = {
  command: {
    register: (command: CommandSpec) => Promise<unknown>;
  };
};

/** Declares this plugin's `/jevcompact` slash command for the session. */
export async function registerJevCompactCommand($: CommandRegisterHook): Promise<void> {
  await $.command.register({ name: JEV_COMPACT_COMMAND, description: JEV_COMPACT_DESCRIPTION });
}

/**
 * Serves the `/jevcompact` command. The host forbids `session.compact()` from a
 * `command.run` hook (it would compact under the turn the hook is holding), so
 * the command only flags the request; the `turn.complete` hook runs it once this
 * turn ends.
 */
export function runJevCompactCommand(queue: () => void): string {
  queue();
  return 'Jev compaction queued; it runs when this turn completes.';
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  let manualRequested = false;
  // Scored in turn.complete (status visible) and reused by the session.compact
  // hook, which runs behind the host's compaction overlay where progress cannot show.
  let precomputed:
    | { sourceCount: number; result: CompactResult; messages: SessionMessage[]; summary: string }
    | null = null;
  // The line the session.compact hook wants pinned once the overlay clears.
  let pendingStatus: string | undefined;

  on('session.start', async ($, event, next) => {
    try {
      await registerJevCompactCommand($);
    } catch (error) {
      $.ui.log(
        `/${JEV_COMPACT_COMMAND} not registered (${error instanceof Error ? error.message : String(error)})`,
      );
    }
    return next(event);
  });

  on('command.run', { command: JEV_COMPACT_COMMAND }, async () => {
    return { text: runJevCompactCommand(() => { manualRequested = true; }) };
  });

  on('session.compact', { trigger: 'plugin' }, async ($, event, next) => {
    const stash = precomputed;
    precomputed = null;
    pendingStatus = undefined;
    try {
      let result: CompactResult;
      let messages: SessionMessage[];
      // Reuse the turn.complete scoring only when it scored the same transcript
      // the engine now hands us; a differing count means it was truncated, so
      // rescore here rather than drop older messages.
      if (stash && stash.sourceCount === event.messages.length) {
        ({ result, messages } = stash);
      } else {
        const config = { ...configured, apiKey: await getApiKey($, configured) };
        ({ result, messages } = await compactSession(
          event.messages,
          config,
          async (url, init) => {
            const response = await $.http.fetch(url, init);
            return { status: response.status, ok: response.ok, text: response.text };
          },
          (progress) => {
            const line = stageStatus(progress);
            $.ui.status(line.length > 0 ? line : undefined);
          },
        ));
      }
      $.ui.log(stagedSummary(result, messages.length, event.messages.length));
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < configured.minReductionRatio) {
        $.ui.status(undefined);
        notify(
          $,
          `fallback to built-in summary (below ${percent(configured.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      pendingStatus = `Jev ✓ kept ${messages.length}/${event.messages.length} · ${percent(reductionRatio(result))} smaller`;
      $.ui.status(pendingStatus);
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      $.ui.status(undefined);
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const manual = manualRequested;
      manualRequested = false;
      if (!manual) {
        const { context } = await $.session.usage();
        if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      }
      compacting = true;
      // Score with visible staged progress now, before session.compact draws the
      // host's compaction overlay over the status line.
      try {
        const config = { ...configured, apiKey: await getApiKey($, configured) };
        const source = await $.session.messages();
        const { result, messages } = await compactSession(
          source,
          config,
          async (url, init) => {
            const response = await $.http.fetch(url, init);
            return { status: response.status, ok: response.ok, text: response.text };
          },
          (progress) => {
            const line = stageStatus(progress);
            $.ui.status(line.length > 0 ? line : undefined);
          },
        );
        precomputed = {
          sourceCount: source.length,
          result,
          messages,
          summary: `Jev ✓ kept ${messages.length}/${source.length} · ${percent(reductionRatio(result))} smaller`,
        };
      } catch (error) {
        precomputed = null;
        $.ui.status(undefined);
        $.ui.log(
          `Jev pre-scan skipped (${error instanceof Error ? error.message : String(error)})`,
        );
      }
      await $.session.compact();
      // Re-pin the outcome: the post-compaction context reset drops the status set inside the hook.
      $.ui.status(pendingStatus);
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
      precomputed = null;
    }
    return next(event);
  });
};

export { resolveOptions };
