import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const state = '/home/nate/codex/rnj-chat';
await mkdir(state, { recursive: true });
let busy = false;
let pending;
let startupLog = '';
const routes = new Set();
const residentEnv = {
  ...process.env,
  RECIPE_DEVICE: 'amd0',
  RECIPE_TIMINGS: '1',
  RECIPE_CONTEXT: '1024',
};
const child = spawn('flock', ['-x', '/home/nate/codex/rnj-estimate/gpu.lock', `${state}/rnj-chat`], {
  cwd: root,
  env: residentEnv,
  stdio: ['pipe', 'pipe', 'pipe'],
});
let readyResolve;
let readyReject;
const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
createInterface({ input: child.stdout }).on('line', line => {
  if (line.startsWith('READY\t')) return readyResolve(Number(line.slice(6)));
  if (!line.startsWith('RESULT\t') || !pending) return;
  const [, promptTokens, prefill, decode, rate, encoded] = line.split('\t');
  const current = pending;
  pending = undefined;
  current.resolve({
    text: Buffer.from(encoded, 'hex').toString('utf8'),
    promptTokens: Number(promptTokens),
    prefill: Number(prefill),
    decode: Number(decode),
    rate: Number(rate),
  });
});
child.stderr.on('data', chunk => {
  startupLog = (startupLog + chunk.toString()).slice(-200_000);
  for (const line of startupLog.split('\n')) {
    if (line.includes(' -> ')) routes.add(line.trim());
  }
});
const residentFailed = error => {
  readyReject(error);
  if (pending) {
    pending.reject(error);
    pending = undefined;
  }
};
child.on('error', residentFailed);
child.on('close', code => residentFailed(Error(startupLog.trim() || `Resident model exited with code ${code}.`)));
const decode = (path, tokens) => new Promise((resolve, reject) => {
  pending = { resolve, reject };
  child.stdin.write(`${path}\t${tokens}\n`, error => {
    if (error && pending) {
      pending = undefined;
      reject(error);
    }
  });
});
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
    send({ status: 'Preparing the resident model and conversation…' });
    const context = await ready;
    send({ context });
    const result = await decode(`${state}/prompt.txt`, tokens);
    send({ promptTokens: result.promptTokens });
    send({ text: result.text });
    const instructions = [...routes].join('; ');
    const stats = `${residentEnv.RECIPE_DEVICE}; context ${context}; ${instructions}; prefill ${result.prefill.toFixed(3)} s; decode ${result.decode.toFixed(3)} s; ${result.rate.toFixed(2)} tok/s`;
    await writeFile(`${state}/last-run.log`, `${startupLog}\n${stats}\n`);
    send({ done: true, stats });
    res.end();
  } catch (error) {
    if (!res.headersSent) res.writeHead(400);
    else if (!res.destroyed) res.write(JSON.stringify({ error: error.message }) + '\n');
    res.end(error.message);
  } finally { busy = false; }
});
server.listen(8766, '127.0.0.1', () => console.log('RNJ-1 chat: http://127.0.0.1:8766'));
