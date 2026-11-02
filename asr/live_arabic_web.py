#!/usr/bin/env python3
"""Live Arabic speech-to-text with the browser as microphone and display (Whisper large-v3 + Egyptian adapter).

  python3 asr/live_arabic_web.py              # then open http://127.0.0.1:8765 in Edge or Chrome and press Start
  python3 asr/live_arabic_web.py --no-adapter --language auto --log out.txt --port 8765

The web page captures the microphone with the browser's own audio stack (device picker, optional noise
suppression and gain control), streams 16 kHz PCM to this script, and shows the live transcript with proper
right-to-left Arabic rendering. Nothing leaves the laptop: the page and the model both run here.
Phrase detection is Silero VAD (see live_arabic.py, which this script shares its parts with).
"""
import argparse
import collections
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_arabic import FIRM_CUT_S, FRAME, FRAME_MS, MAX_PHRASE_S, MIN_VOICED_MS, RATE, SOFT_CUT_S, SpeechDetector, Transcriber, is_hallucination  # noqa: E402
from denoise import Highpass, SpectralGate  # noqa: E402

HTML = r"""<!doctype html>
<html lang="ar"><head><meta charset="utf-8"><title>Live Arabic transcription</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 24px; max-width: 1100px; margin-inline: auto; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  .bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
  button { font-size: 16px; padding: 8px 18px; border-radius: 8px; border: 1px solid #8884; cursor: pointer; }
  #start { background: #1a7f37; color: white; } #stop { background: #b42318; color: white; } button:disabled { opacity: .5; cursor: default; }
  select, label { font-size: 15px; }
  #meter { height: 14px; width: 260px; background: #8883; border-radius: 7px; overflow: hidden; }
  #level { height: 100%; width: 0; background: #1a7f37; transition: width .1s; }
  #status { font-size: 14px; opacity: .8; min-height: 1.2em; }
  .ar { direction: rtl; text-align: right; font-family: "Noto Naskh Arabic", "Noto Sans Arabic", "Amiri", "Scheherazade New", "Segoe UI", sans-serif; }
  #partial { font-size: 30px; line-height: 1.7; min-height: 2.2em; opacity: .55; padding: 10px 14px; border: 1px dashed #8886; border-radius: 10px; margin: 14px 0; }
  .final { font-size: 30px; line-height: 1.7; padding: 8px 14px; border-bottom: 1px solid #8883; display: flex; gap: 16px; align-items: baseline; }
  .final .meta { direction: ltr; font-size: 12px; opacity: .55; font-family: monospace; white-space: nowrap; margin-left: auto; }
  .final.unclear { opacity: .5; }
</style></head>
<body>
<h1>Live Arabic transcription · Whisper large-v3 <span id="model"></span></h1>
<div class="bar">
  <button id="start">Start</button><button id="stop" disabled>Stop</button>
  <select id="mic"><option value="">default microphone</option></select>
  <label><input type="checkbox" id="proc" checked> browser noise suppression + auto gain</label>
  <div id="meter"><div id="level"></div></div>
  <button id="clear" style="margin-left:auto">Clear</button>
</div>
<div id="status">Press Start, then allow the microphone if asked. Best results with a headset or a microphone close to your mouth; grey lines with ؟ are low-confidence guesses.</div>
<div id="partial" class="ar"></div>
<div id="finals"></div>
<script>
const $ = id => document.getElementById(id);
let ctx, stream, node, running = false, es;
function setStatus(t) { $('status').textContent = t; }
async function listMics() {
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const sel = $('mic'); const cur = sel.value; sel.innerHTML = '<option value="">default microphone</option>';
    devs.filter(d => d.kind === 'audioinput' && d.deviceId !== 'default').forEach(d => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || ('microphone ' + d.deviceId.slice(0, 6)); sel.appendChild(o); });
    sel.value = cur;
  } catch (e) {}
}
function connectEvents() {
  es = new EventSource('/events');
  es.onmessage = m => {
    const ev = JSON.parse(m.data);
    if (ev.type === 'level') { $('level').style.width = Math.min(100, ev.level / Math.max(ev.thr, 1) * 35) + '%'; $('level').style.background = ev.level > ev.thr ? '#1a7f37' : '#888'; }
    else if (ev.type === 'partial') { $('partial').textContent = ev.text; }
    else if (ev.type === 'final') {
      $('partial').textContent = '';
      const d = document.createElement('div'); d.className = 'final ar' + (ev.unclear ? ' unclear' : '');
      d.innerHTML = '<span class="text"></span><span class="meta"></span>';
      d.querySelector('.text').textContent = (ev.unclear ? '؟ ' : '') + ev.text;
      d.querySelector('.meta').textContent = ev.time + ' · ' + ev.audio_s.toFixed(1) + 's audio · ' + ev.proc_s.toFixed(2) + 's · conf ' + (ev.conf === undefined ? '-' : ev.conf.toFixed(2)) + (ev.unclear ? ' (unclear)' : '');
      $('finals').prepend(d);
    }
    else if (ev.type === 'status') { setStatus(ev.text); if (ev.model) $('model').textContent = ev.model; }
  };
  es.onerror = () => setStatus('lost connection to the Python script (is it still running?)');
}
async function start() {
  const proc = $('proc').checked, dev = $('mic').value;
  const audio = { echoCancellation: proc, noiseSuppression: proc, autoGainControl: proc, voiceIsolation: proc, channelCount: 1 };
  if (dev) audio.deviceId = { exact: dev };
  try { stream = await navigator.mediaDevices.getUserMedia({ audio }); }
  catch (e) { setStatus('microphone error: ' + e.name + ' ' + e.message); return; }
  await listMics();
  ctx = new AudioContext({ sampleRate: 16000 });
  const src = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(4096, 1, 1);          // 256 ms blocks at 16 kHz
  const mute = ctx.createGain(); mute.gain.value = 0;
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  const label = stream.getAudioTracks()[0].label;
  await fetch('/control', { method: 'POST', body: JSON.stringify({ cmd: 'start', device: label, rate: ctx.sampleRate }) });
  node.onaudioprocess = e => {
    if (!running) return;
    const f = e.inputBuffer.getChannelData(0), pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) { const v = Math.max(-1, Math.min(1, f[i])); pcm[i] = v < 0 ? v * 32768 : v * 32767; }
    fetch('/audio', { method: 'POST', body: pcm.buffer, headers: { 'Content-Type': 'application/octet-stream' } }).catch(() => {});
  };
  running = true; $('start').disabled = true; $('stop').disabled = false;
  setStatus('listening on: ' + label + (proc ? '  (browser processing on)' : '  (raw)') + '  · context rate ' + ctx.sampleRate + ' Hz');
}
async function stop() {
  running = false; $('start').disabled = false; $('stop').disabled = true;
  try { node && node.disconnect(); stream && stream.getTracks().forEach(t => t.stop()); ctx && await ctx.close(); } catch (e) {}
  await fetch('/control', { method: 'POST', body: JSON.stringify({ cmd: 'stop' }) });
  setStatus('stopped');
}
$('start').onclick = start; $('stop').onclick = stop; $('clear').onclick = () => { $('finals').innerHTML = ''; $('partial').textContent = ''; };
connectEvents(); listMics();
</script></body></html>
"""


class Hub:
    """Audio frames in (from the browser), events out (to every open page)."""
    def __init__(self):
        self.audio = queue.Queue()
        self.clients, self.history, self.lock = [], [], threading.Lock()
        self.model_label = ""

    def publish(self, ev):
        if ev["type"] == "final":
            self.history = (self.history + [ev])[-200:]
        line = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        with self.lock:
            for q in list(self.clients):
                q.put(line)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = hub.subscribe()
                try:
                    self.wfile.write(("data: " + json.dumps({"type": "status", "text": "connected", "model": hub.model_label}, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    for ev in hub.history:
                        self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        try:
                            line = q.get(timeout=15)
                        except queue.Empty:
                            line = ": keepalive\n\n"
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unsubscribe(q)
                return
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(n)
            if self.path == "/audio":
                if n >= 2:
                    hub.audio.put(np.frombuffer(data[: n - (n % 2)], dtype=np.int16))
            elif self.path == "/control":
                try:
                    msg = json.loads(data.decode("utf-8"))
                except ValueError:
                    msg = {}
                hub.audio.put(msg)
            self.send_response(204)
            self.end_headers()
    return Handler


def frames(hub):
    """Re-chunk whatever the browser sends into fixed frames; pass control messages through."""
    rest = np.zeros(0, dtype=np.int16)
    while True:
        item = hub.audio.get()
        if isinstance(item, dict):
            yield item
            continue
        rest = np.concatenate([rest, item]) if len(rest) else item
        while len(rest) >= FRAME:
            yield rest[:FRAME]
            rest = rest[FRAME:]


def save_wav(path, audio_int16):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.ascontiguousarray(audio_int16).tobytes())


def worker(hub, asr, det, a, log):
    hp = Highpass(a.highpass) if a.highpass > 0 else None
    clean = (lambda x: x) if a.no_denoise else SpectralGate()
    pre = collections.deque(maxlen=300 // FRAME_MS)
    talking, buf, above, below, unsure, last_partial, voiced, status_t = False, [], 0, 0, 0, 0.0, 0, 0.0
    ambient, ambient_left = [], 0            # --dump: the first 10 s after Start, VAD or not
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)

    def finish(buf, below):
        keep = len(buf) - max(0, below - 200 // FRAME_MS)
        audio = np.concatenate(buf[:keep]) if keep > 0 else np.zeros(0, dtype=np.int16)
        if len(audio) / RATE < 0.4:
            return
        t = time.time()
        enhanced = clean(audio)
        text = asr(enhanced, a.language, beams=a.beams_final)
        if is_hallucination(text):
            print("(dropped, noise: %s)" % text, flush=True)
            hub.publish({"type": "partial", "text": ""})
            return
        ev = {"type": "final", "text": text, "time": time.strftime("%H:%M:%S"), "audio_s": len(audio) / RATE, "proc_s": time.time() - t,
              "conf": round(asr.last_conf, 3), "unclear": asr.last_conf < a.min_conf}
        if a.dump:
            stamp = time.strftime("%H%M%S")
            save_wav(os.path.join(a.dump, "phrase_%s_raw.wav" % stamp), audio)
            save_wav(os.path.join(a.dump, "phrase_%s_clean.wav" % stamp), enhanced)
        hub.publish(ev)
        print("[%s] %s   (%.1f s audio, %.2f s, conf %.2f%s)" % (ev["time"], text, ev["audio_s"], ev["proc_s"], ev["conf"], ", unclear" if ev["unclear"] else ""), flush=True)
        if log:
            log.write("[%s] %s\n" % (ev["time"], text))
            log.flush()

    for item in frames(hub):
        if isinstance(item, dict):
            if item.get("cmd") == "start":
                talking, buf, above, below, voiced = False, [], 0, 0, 0
                det.reset()
                if a.dump:
                    ambient, ambient_left = [], 10 * RATE
                hub.publish({"type": "status", "text": "listening on: %s  ·  %s" % (item.get("device", "?"), det.name)})
                print("browser started: %s @ %s Hz" % (item.get("device"), item.get("rate")), flush=True)
            elif item.get("cmd") == "stop":
                if talking and buf and voiced * FRAME_MS >= MIN_VOICED_MS:
                    finish(buf, below)
                talking, buf = False, []
                det.reset()
                hub.publish({"type": "partial", "text": ""})
                hub.publish({"type": "level", "level": 0, "thr": 50})
            continue
        frame = item
        if hp is not None:
            frame = hp(frame)
        if ambient_left > 0:
            ambient.append(frame)
            ambient_left -= FRAME
            if ambient_left <= 0:
                p = os.path.join(a.dump, "ambient_%s.wav" % time.strftime("%H%M%S"))
                save_wav(p, np.concatenate(ambient))
                print("saved 10 s of ambient audio to", p, flush=True)
        speech, quiet, meter = det.update(frame, talking)
        if not det.ready:
            continue
        if time.time() - status_t > 0.15:
            status_t = time.time()
            hub.publish({"type": "level", "level": meter, "thr": 50})
        if not talking:
            pre.append(frame)
            above = above + 1 if speech else 0
            if above >= 3:
                talking, buf, below, unsure, last_partial, voiced = True, list(pre), 0, 0, 0.0, above
            continue
        buf.append(frame)
        voiced += 1 if speech else 0
        below = below + 1 if quiet else 0              # frames clearly without voice
        unsure = unsure + 1 if not speech else 0        # frames without clear voice (noise can sit here)
        dur = len(buf) * FRAME_MS / 1000.0
        ended = (below * FRAME_MS >= a.silence or unsure * FRAME_MS >= 1.5 * a.silence or dur >= MAX_PHRASE_S
                 or (dur >= SOFT_CUT_S and below * FRAME_MS >= 250) or (dur >= FIRM_CUT_S and unsure * FRAME_MS >= 100))
        real = voiced * FRAME_MS >= MIN_VOICED_MS
        if real and not ended and dur >= 1.0 and dur - last_partial >= a.partial_every:
            last_partial = dur
            hub.publish({"type": "partial", "text": asr(clean(np.concatenate(buf)), a.language)})
        if ended:
            talking, above = False, 0
            det.reset()
            if real:
                finish(buf, below)
            else:
                hub.publish({"type": "partial", "text": ""})
            buf = []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-adapter", action="store_true", help="plain whisper-large-v3 without the Egyptian adapter")
    ap.add_argument("--language", default="ar", help="ar (default), en, ... or auto")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--partial-every", type=float, default=1.5, help="seconds between live updates while you talk")
    ap.add_argument("--silence", type=int, default=800, help="ms of quiet that ends a phrase")
    ap.add_argument("--threshold", type=float, default=0, help="energy threshold for the fallback detector (0 = automatic)")
    ap.add_argument("--beams", type=int, default=1, help="beam size for the live partial updates (1 = fastest)")
    ap.add_argument("--beams-final", type=int, default=1, help="beam size for the final line (1 = greedy; beam search tends to loop on long noisy phrases)")
    ap.add_argument("--vad", type=float, default=0.6, help="voice probability needed to start a phrase; raise towards 0.8 in a noisy room")
    ap.add_argument("--min-conf", type=float, default=-0.8, help="finals whose mean token log-prob is below this are marked (unclear)")
    ap.add_argument("--highpass", type=float, default=100, help="high-pass cutoff in Hz against hum and rumble (0 = off)")
    ap.add_argument("--no-denoise", action="store_true", help="skip the spectral noise gate before Whisper")
    ap.add_argument("--log", help="append final lines to this file")
    ap.add_argument("--dump", help="debug: save 10 s of ambient audio after Start and every phrase as WAV files in this folder")
    a = ap.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and a.partial_every < 4:
        a.partial_every = 4
    hub = Hub()
    asr = Transcriber(device, not a.no_adapter, a.beams)
    det = SpeechDetector(a.threshold, a.vad)
    print("phrase detection:", det.name)
    print("noise handling: %s%s" % ("high-pass %.0f Hz" % a.highpass if a.highpass > 0 else "no high-pass", ", spectral gate" if not a.no_denoise else ", no gate"))
    hub.model_label = "+ Egyptian adapter" if not a.no_adapter else "(no adapter)"
    log = io.open(a.log, "a", encoding="utf-8") if a.log else None
    threading.Thread(target=worker, args=(hub, asr, det, a, log), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub))
    srv.daemon_threads = True
    print("ready: open http://127.0.0.1:%d in Edge or Chrome and press Start  (Ctrl-C here to quit)" % a.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
