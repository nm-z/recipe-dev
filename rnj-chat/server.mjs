import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { StringDecoder } from 'node:string_decoder';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const state = '/home/nate/codex/rnj-chat';
await mkdir(state, { recursive: true });
let busy = false;
const server = http.createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(await readFile(new URL('./index.html', import.meta.url)));
  }
  if (req.method !== 'POST' || req.url !== '/chat') { res.writeHead(404); return res.end(); }
  if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) { res.writeHead(403); return res.end(); }
  if (busy) { res.writeHead(409); return res.end('Another reply is running.'); }
  busy = true;
  try {
    let body = '';
    for await (const chunk of req) {
      body += chunk;
      if (body.length > 2_000_000) throw Error('Conversation is too large.');
    }
    const { messages, tokens = 512 } = JSON.parse(body);
    if (!Array.isArray(messages) || !messages.length || !Number.isInteger(tokens) || tokens < 1 || tokens > 4096 ||
        messages.some(m => !['user', 'assistant'].includes(m.role) || typeof m.content !== 'string')) throw Error('Invalid conversation.');
    const turn = (role, text) => `<|start_header_id|>${role}<|end_header_id|>\n${text}<|eot_id|>`;
    const prompt = turn('system', 'You are rnj-1, a foundation model trained by Essential AI.') +
      messages.map(m => turn(m.role, m.content)).join('') + '<|start_header_id|>assistant<|end_header_id|>\n';
    await writeFile(`${state}/prompt.txt`, prompt);
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' });
    const send = value => { if (!res.destroyed) res.write(JSON.stringify(value) + '\n'); };
    send({ status: 'Loading model and preparing conversation…' });
    const child = spawn('flock', ['-x', '/home/nate/codex/rnj-estimate/gpu.lock', `${state}/rnj-chat`], {
      cwd: root, detached: true,
      env: { ...process.env, RECIPE_DEVICE: 'amd0', RECIPE_PACKED_DOT: '1', RECIPE_HOST_SPILL: '1', RECIPE_TIMINGS: '1', RNJ_CONTEXT: '32768',
        RNJ_TOKENS: String(tokens), RNJ_PROMPT_FILE: `${state}/prompt.txt`, RNJ_RAW_PROMPT: '1' },
      stdio: ['ignore', 'pipe', 'pipe']
    });
    const decoder = new StringDecoder('utf8');
    let log = '';
    child.stdout.on('data', chunk => send({ text: decoder.write(chunk) }));
    child.stderr.on('data', chunk => {
      log += chunk.toString();
      const match = log.match(/RNJ-1: (\d+) prompt tokens/);
      if (match) send({ promptTokens: Number(match[1]) });
    });
    res.on('close', () => { if (child.exitCode === null) { try { process.kill(-child.pid, 'SIGTERM'); } catch {} } });
    child.on('error', error => { busy = false; send({ error: error.message }); res.end(); });
    child.on('close', async code => {
      busy = false;
      send({ text: decoder.end() });
      await writeFile(`${state}/last-run.log`, log);
      send(code === 0 ? { done: true, stats: log.trim() } : { error: log.trim() || `Generation exited with code ${code}.` });
      res.end();
    });
  } catch (error) { busy = false; res.writeHead(400); res.end(error.message); }
});
server.listen(8766, '127.0.0.1', () => console.log('RNJ-1 chat: http://127.0.0.1:8766'));
