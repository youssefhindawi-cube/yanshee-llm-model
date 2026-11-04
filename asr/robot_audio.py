#!/usr/bin/env python3
"""The Yanshee's microphone and speakers over the network, for the live pipeline (the models stay on the PC).

  python3 asr/robot_audio.py 192.168.89.99 --record 5 out.wav          # 5 s from the robot's microphone
  python3 asr/robot_audio.py 192.168.89.99 --play asr/tts_samples/voicetut_Hassan.wav
  python3 asr/robot_audio.py 192.168.89.99 --latency                   # a beep through the speakers, timed on its own microphone
  python3 asr/robot_audio.py 192.168.89.99 --volume 70
  python3 asr/live_arabic_web.py --model cohere --brain Q_Brain_Brief_Yanshee_ar.md --speak voicetut:Hassan --robot 192.168.89.99

How it works. One ssh connection to the robot's Raspberry Pi (login pi/raspberry, the factory default that yanshee_sim's
tools also use; change it with --robot-user/--robot-password or YANSHEE_PASSWORD), multiplexed so the microphone stream
and the player are two sessions on it.
- Microphone: `arecord` on the robot streams 16 kHz mono PCM to us; it goes into the same queue the browser page feeds.
- Speakers: a small pacer script (PLAYER below) is copied to the robot and keeps `aplay` fed with silence; whatever PCM we
  send is played the moment it arrives, with no device open/close per sentence (one ssh session per sentence cost
  0.6-1.5 s). It plays 24 kHz mono (VoiceTut's and Munsit's rate); other rates are resampled here.
- The robot hears its own speakers at about -40 dBFS (measured 2026-11-04), so the microphone is gated from the moment a
  reply is sent until it has sounded, plus a short tail; the gate uses the measured send-to-microphone lag (0.85 s, --latency).
The sound card is a Microsemi codec (card 0) with dsnoop/dmix as ALSA's default, so the robot's own voice service can
still use it next to us.
"""
import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

MIC_RATE = 16000
MIC_BLOCK = 2048                       # bytes per read = 64 ms at 16 kHz
MIC_CMD = "arecord -q -D default -f S16_LE -c 1 -r %d -t raw -" % MIC_RATE
CLEANUP_CMD = "pkill -xf '%s'; pkill -f '[q]_player.py'; true" % MIC_CMD   # ours only; the robot's own services do not use these
SUDO = "sudo -n "                      # the robot's own services use the sound card as root; ALSA's dmix/dsnoop shared memory is then
                                       # root-only, so our two audio processes run as root too (pi has passwordless sudo on the Yanshee)
PLAY_RATE = 24000
BLOCK_MS = 20
BLOCK_BYTES = PLAY_RATE * 2 * BLOCK_MS // 1000
REMOTE_PLAYER = "/home/pi/q_player.py"
PLAYER_CMD = "python3 -u %s %d %d" % (REMOTE_PLAYER, PLAY_RATE, BLOCK_BYTES)
DEFAULT_USER, DEFAULT_PASSWORD = "pi", os.environ.get("YANSHEE_PASSWORD", "raspberry")
REST_PORT = 9090
ECHO_LAG_S = 0.85                      # send -> sounds on the robot -> its microphone -> here (measured 2026-11-04 with --latency)

# Runs on the robot (Python 3.5): stdin = PCM from us, stdout = "ready"/"idle" lines. Keeps aplay fed with silence so the
# device never closes; a small pipe to aplay (F_SETPIPE_SZ = 1031) keeps our lead over the DAC short.
PLAYER = r'''
import collections, fcntl, subprocess, sys, threading, time
rate, block = int(sys.argv[1]), int(sys.argv[2])
p = subprocess.Popen(["aplay", "-q", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", str(rate), "-t", "raw", "-B", "150000", "-"],
                     stdin=subprocess.PIPE)
try:
    fcntl.fcntl(p.stdin.fileno(), 1031, 8192)
except OSError:
    pass
silence = bytes(block)
try:
    p.stdin.write(silence * 3)
    time.sleep(0.4)
except BrokenPipeError:
    pass
if p.poll() is not None:
    sys.stdout.write("aplay exited with %s before playing anything\n" % p.returncode); sys.stdout.flush()
    sys.exit(1)
q, lock, eof = collections.deque(), threading.Lock(), [False]
def reader():
    src, carry = sys.stdin.buffer, b""
    while True:
        data = src.read1(65536)
        if not data:
            break
        carry += data
        n = len(carry) - len(carry) % block
        with lock:
            for i in range(0, n, block):
                q.append(carry[i:i + block])
        carry = carry[n:]
    if carry:
        with lock:
            q.append(carry + bytes(block - len(carry)))
    eof[0] = True
threading.Thread(target=reader, daemon=True).start()
busy = False
sys.stdout.write("ready\n"); sys.stdout.flush()
while True:
    with lock:
        chunk = q.popleft() if q else None
    if chunk is None:
        if eof[0]:
            break
        if busy:
            busy = False
            sys.stdout.write("idle\n"); sys.stdout.flush()
        chunk = silence
    else:
        busy = True
    try:
        p.stdin.write(chunk)
    except BrokenPipeError:
        break
p.stdin.close()
p.wait()
'''


def resample(pcm, sr, to):
    pcm = np.asarray(pcm, dtype=np.int16)
    if sr == to or len(pcm) == 0:
        return pcm
    n = int(round(len(pcm) * to / sr))
    x = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm.astype(np.float32))
    return np.clip(x, -32768, 32767).astype(np.int16)


class RobotAudio:
    """connect() then: frames and {"cmd": "start"} messages go to sink(item); play(sr, pcm) sounds on the robot; muted says
    whether the microphone is being ignored because the robot is (about to be) talking."""

    def __init__(self, host, sink=None, user=DEFAULT_USER, password=DEFAULT_PASSWORD, tail=0.6, lag=ECHO_LAG_S, log=None):
        self.host, self.user, self.sink, self.tail, self.lag = host, user, sink or (lambda item: None), tail, lag
        self.log = log or (lambda s: print("    robot: " + s, flush=True))
        self.env = dict(os.environ, SSHPASS=password)
        self.ssh = ["sshpass", "-e", "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                    "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=3",
                    "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/qy-ssh-%C", "-o", "ControlPersist=120", "%s@%s" % (user, host)]
        self.busy_until = 0.0          # when the robot will have finished sounding what we sent
        self.quiet_until = 0.0         # the microphone is ignored until then
        self.mic_proc, self.play_proc, self.sudo, self._player_t = None, None, "", 0.0
        self.gate = True               # False = keep listening while the robot talks (measurements only)
        self.stop, self.lock, self.sent_s = threading.Event(), threading.Lock(), 0.0

    # --- link ---
    def _run(self, remote, timeout=25, input=None):
        return subprocess.run(self.ssh + [remote], env=self.env, timeout=timeout, input=input, capture_output=True)

    def connect(self, player=True, mic=True):
        try:
            r = self._run("true")
        except FileNotFoundError:
            raise RuntimeError("sshpass is not installed (sudo apt install sshpass)")
        if r.returncode:
            raise RuntimeError("cannot ssh to %s@%s: %s" % (self.user, self.host, r.stderr.decode(errors="replace").strip()[-200:] or "no answer"))
        self.sudo = SUDO if self._run(SUDO + "true").returncode == 0 else ""
        if not self.sudo:
            self.log("no passwordless sudo on the robot: audio runs as %s and may be refused while the robot's own services use the sound card" % self.user)
        self._run(self._cleanup_cmd())                      # leftovers of an earlier run that did not close
        if player:
            up = self._run("cat > %s" % REMOTE_PLAYER, input=PLAYER.encode())
            if up.returncode:
                raise RuntimeError("cannot copy the player to the robot: %s" % up.stderr.decode(errors="replace").strip()[-200:])
            self._start_player()
        if mic:
            threading.Thread(target=self._mic_loop, daemon=True).start()
        return self

    def _cleanup_cmd(self):
        return CLEANUP_CMD.replace("pkill", self.sudo + "pkill")

    def _start_player(self):
        self._player_t = time.time()
        self.play_proc = subprocess.Popen(self.ssh + [self.sudo + PLAYER_CMD], env=self.env,
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        said, deadline = [], time.time() + 20
        while time.time() < deadline:
            if not select.select([self.play_proc.stdout], [], [], max(0.0, deadline - time.time()))[0]:
                break
            line = self.play_proc.stdout.readline()
            if not line:
                break
            if line.strip() == b"ready":
                threading.Thread(target=self._player_log, daemon=True).start()
                return
            said.append(line.decode(errors="replace").rstrip())
        try:
            self.play_proc.kill()
        except OSError:
            pass
        raise RuntimeError("the robot's player did not start: %s" % ("; ".join(said)[-300:] or "no 'ready' within 20 s"))

    def close(self):
        self.stop.set()
        for p in (self.mic_proc, self.play_proc):
            if p is not None and p.poll() is None:
                try:
                    if p.stdin:
                        p.stdin.close()
                    p.terminate()
                    p.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    p.kill()
        try:                                                 # a remote command outlives its ssh session until it writes; end it
            self._run(self._cleanup_cmd(), timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            pass
        subprocess.run(self.ssh[2:-1] + ["-O", "exit", self.ssh[-1]], env=self.env, capture_output=True, timeout=10)

    # --- microphone ---
    @property
    def muted(self):
        return self.gate and time.time() < self.quiet_until

    def _mic_loop(self):
        first = True
        while not self.stop.is_set():
            p = subprocess.Popen(self.ssh + [self.sudo + MIC_CMD], env=self.env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.mic_proc = p
            self.sink({"cmd": "start", "device": "Yanshee microphone at %s%s" % (self.host, "" if first else " (reconnected)"), "rate": MIC_RATE})
            first = False
            fd, rest = p.stdout.fileno(), b""
            while not self.stop.is_set():
                data = os.read(fd, MIC_BLOCK)               # unbuffered: the gate applies to each block as it arrives
                if not data:
                    break
                rest += data
                n = len(rest) - len(rest) % 2
                data, rest = rest[:n], rest[n:]
                if self.muted or not data:
                    continue
                self.sink(np.frombuffer(data, dtype=np.int16))
            if self.stop.is_set():
                break
            err = p.stderr.read().decode(errors="replace").strip()
            self.log("microphone stream ended%s; reconnecting in 2 s" % (" (%s)" % err[-120:] if err else ""))
            time.sleep(2)

    # --- speakers ---
    def play(self, sr, pcm):
        """Send one chunk; returns the time by which it has reached the robot's microphone. Gates the microphone until it has
        all sounded (busy_until) plus the tail."""
        if self.play_proc is None or self.play_proc.poll() is not None:
            if self.play_proc is None or time.time() - self._player_t < 10:
                raise RuntimeError("the robot's player is not running")
            self.log("the player stopped; starting it again")
            self._start_player()
        pcm = resample(pcm, sr, PLAY_RATE)
        dur = len(pcm) / PLAY_RATE
        with self.lock:
            now = time.time()
            start = max(now + self.lag, self.busy_until)
            self.busy_until = start + dur
            self.quiet_until = self.busy_until + self.tail
            self.sent_s += dur
            self.play_proc.stdin.write(pcm.tobytes())
            self.play_proc.stdin.flush()
        return start

    def _player_log(self):
        for line in self.play_proc.stdout:
            line = line.decode(errors="replace").strip()
            if line and line != "idle":
                self.log("player: " + line[-200:])

    # --- the robot's REST API (volume) ---
    def rest(self, method, path, body=None):
        req = urllib.request.Request("http://%s:%d/v1/%s" % (self.host, REST_PORT, path), method=method,
                                     data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())

    def get_volume(self):
        return self.rest("GET", "devices/volume")["data"]["volume"]

    def set_volume(self, volume):
        """0-100 through the robot's REST API. The robot answers every change with a 0.6 s tip sound, so the microphone is
        gated meanwhile (a tip would otherwise start a phrase)."""
        self.quiet_until = max(self.quiet_until, time.time() + self.lag + 1.5)
        self.rest("PUT", "devices/volume", {"volume": max(0, min(100, int(volume)))})
        return self.get_volume()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="the robot's IP")
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--password", default=DEFAULT_PASSWORD, help="ssh password (default: YANSHEE_PASSWORD or the factory one)")
    ap.add_argument("--record", nargs=2, metavar=("SECONDS", "OUT.wav"), help="record from the robot's microphone")
    ap.add_argument("--play", metavar="FILE.wav", help="play a 16-bit WAV through the robot's speakers")
    ap.add_argument("--latency", action="store_true", help="play a beep and time it on the robot's own microphone")
    ap.add_argument("--volume", type=int, help="set the robot's volume (0-100) through its REST API")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tts_backends import read_wav, wav_bytes

    got = []
    robot = RobotAudio(a.host, sink=lambda item: got.append((time.time(), item)) if not isinstance(item, dict) else print("mic:", item["device"]),
                       user=a.user, password=a.password)
    if a.volume is not None:
        print("volume now %s" % robot.set_volume(a.volume))
    need_mic, need_player = bool(a.record or a.latency), bool(a.play or a.latency)
    if not (need_mic or need_player):
        print("volume %s; nothing else asked (see --help)" % robot.get_volume())
        return
    t = time.time()
    robot.connect(player=need_player, mic=need_mic)
    print("connected in %.1f s" % (time.time() - t))
    try:
        if a.record:
            secs, out = float(a.record[0]), a.record[1]
            time.sleep(secs + 0.5)
            audio = np.concatenate([x for _, x in got]) if got else np.zeros(0, dtype=np.int16)
            audio = audio[: int(secs * MIC_RATE)]
            open(out, "wb").write(wav_bytes(MIC_RATE, audio))
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
            print("saved %.1f s to %s  (rms %.0f = %.1f dBFS, peak %d)" % (len(audio) / MIC_RATE, out, rms, 20 * np.log10(max(rms, 1) / 32768), int(np.abs(audio).max()) if len(audio) else 0))
        if a.play:
            sr, pcm = read_wav(a.play)
            t = time.time()
            start = robot.play(sr, pcm)
            print("sent %.1f s of audio in %.2f s; expected to sound from +%.2f s; waiting for it to finish" % (len(pcm) / sr, time.time() - t, start - t))
            time.sleep(max(0.0, robot.busy_until - time.time()) + 0.3)
        if a.latency:
            robot.gate = False                                # listen while it sounds
            time.sleep(1.5)                                   # let the microphone settle
            n0 = len(got)
            beep = (np.sin(2 * np.pi * 1000 * np.arange(int(0.4 * PLAY_RATE)) / PLAY_RATE) * 12000).astype(np.int16)
            t0 = time.time()
            robot.play(PLAY_RATE, beep)
            time.sleep(2.5)
            before = [x.astype(np.float32) for _, x in got[max(0, n0 - 15):n0]]
            floor = np.median([np.sqrt(np.mean(b ** 2)) for b in before]) if before else 10.0
            onset = None
            for ts, x in got[n0:]:
                if np.sqrt(np.mean(x.astype(np.float32) ** 2)) > max(6 * floor, 300):
                    onset = ts
                    break
            if onset is None:
                print("no beep heard on the robot's microphone (floor rms %.0f); is the volume up?" % floor)
            else:
                lag = onset - t0 - MIC_BLOCK / 2 / MIC_RATE
                print("beep heard %.2f s after sending (room floor rms %.0f): send-to-microphone lag; ECHO_LAG_S is %.2f" % (lag, floor, ECHO_LAG_S))
    finally:
        robot.close()


if __name__ == "__main__":
    main()
