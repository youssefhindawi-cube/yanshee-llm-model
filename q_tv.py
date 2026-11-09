# -*- coding: utf-8 -*-
"""Q's TV layer: find a TV on the network by itself, and show a reply row's video or image on it while Q speaks.

Boot (TvCaster.start, in a background thread, so speech recognition and the voice load and run regardless):
  1. try the TV used last time (saved in ~/.cache/q_tv/last_tv.json): instant reconnect;
  2. otherwise scan every LAN interface (Wi-Fi/Ethernet; docker, VPN and loopback are skipped) three ways at once:
     - SSDP multicast search for DLNA MediaRenderers (the standard; works on most home networks),
     - SSDP unicast search to every host of the subnet (for networks that drop multicast between clients),
     - a probe of Samsung's renderer port 9197 on every host (Samsung TVs whose renderer does not answer SSDP at
       all, e.g. the Q80BA);
     then read each device's description and keep those with an AVTransport service (= something that can play a URL);
  3. no TV: print a WARNING once and keep scanning every --tv-rescan seconds; Q keeps running without the TV.
     A TV that stops answering is dropped and searched for again.

Casting is the method proven in yanshee_sim/tv_cast (rcast.py, tvcast): this machine serves the file over HTTP and
tells the TV's renderer to play that URL (UPnP SetAVTransportURI + Play). Standard library only, Python 3.5-compatible,
so it runs on the laptop and on the Yanshee alike.

TV actions (the "التلفزيون (TV)" column of the brain's quick-answer table):
    welcome.mp4          play the file once (files live in tv_media/ next to the brain .md, or --tv-media DIR)
    welcome.mp4 loop     play it and start it again whenever it ends, until another action replaces it
    stop                 stop whatever the TV is playing
    (empty or -)         leave the TV as it is

Choosing a TV: --tv auto (default: the last one used, else the first TV found), --tv off, --tv 192.168.1.6 (an IP),
or --tv "Q80" (part of the TV's name, when several are found).

Try it on its own:
    python3 q_tv.py --scan                               # list the TVs on the network
    python3 q_tv.py tv_media/welcome.mp4                 # find a TV and play the file once
    python3 q_tv.py --tv 192.168.1.6 stop
"""
from __future__ import print_function

import html
import ipaddress
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    from urllib.parse import urljoin, urlparse
except ImportError:                       # pragma: no cover
    from urlparse import urljoin, urlparse

AVT = "urn:schemas-upnp-org:service:AVTransport:1"
DLNA_FLAGS = "DLNA.ORG_OP=01;DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01700000000000000000000000000000"
MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
DEFAULT_PORT = 8809          # the port tvcast's CLI casts from (verified through the laptop's firewall)
SSDP = ("239.255.255.250", 1900)
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "q_tv", "last_tv.json")
KNOWN_RENDERERS = ((9197, "/dmr"),)      # description URLs of renderers that may not answer SSDP (Samsung)
SKIP_IF = ("lo", "docker", "br-", "veth", "virbr", "tailscale", "tun", "tap", "wg", "zt", "vmnet", "vboxnet")


def parse_action(cell):
    """Table cell -> None, {"stop": True} or {"file": name, "loop": bool}."""
    cell = (cell or "").strip().strip("`")
    if not cell or cell in ("-", "—", "–"):
        return None
    words = cell.split()
    if words[0].lower() == "stop":
        return {"stop": True}
    return {"file": words[0], "loop": any(w.lower() == "loop" for w in words[1:])}


# ================================================================ discovery
def lan_interfaces():
    """[(name, own_ip, network)] for the real LAN interfaces (Wi-Fi, Ethernet)."""
    found = []
    try:
        out = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], universal_newlines=True)
        for line in out.splitlines():
            parts = line.split()
            name, cidr = parts[1], parts[3]
            if name.startswith(SKIP_IF):
                continue
            iface = ipaddress.ip_interface(cidr)
            if iface.network.prefixlen >= 31 or not iface.ip.is_private:
                continue
            found.append((name, str(iface.ip), iface.network))
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        pass
    if not found:                          # no `ip` tool: the address of the default route, assume a /24
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            found.append(("default", ip, ipaddress.ip_interface(ip + "/24").network))
        except OSError:
            pass
    return found


def sweep_hosts(own_ip, network, limit=1022):
    """Hosts to probe: the whole subnet up to /22, else the /24 around this machine."""
    if network.num_addresses - 2 > limit:
        network = ipaddress.ip_interface(own_ip + "/24").network
    return [str(h) for h in network.hosts() if str(h) != own_ip]


def _msearch(st, mx=2):
    return ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: %d\r\nST: %s\r\n\r\n"
            % (mx, st)).encode("ascii")


def ssdp_locations(own_ip, hosts, wait=3.0):
    """Description URLs from SSDP: multicast searches out of this interface, plus a unicast search to each host."""
    locs = set()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        s.bind((own_ip, 0))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(own_ip))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.settimeout(0.25)
        for st in ("urn:schemas-upnp-org:device:MediaRenderer:1", AVT):
            for _ in range(2):
                s.sendto(_msearch(st), SSDP)
        uni = _msearch("ssdp:all", 1)
        for h in hosts:
            try:
                s.sendto(uni, (h, 1900))
            except OSError:
                pass
        end = time.time() + wait
        while time.time() < end:
            try:
                data = s.recvfrom(8192)[0].decode("utf-8", "ignore")
            except socket.timeout:
                continue
            except OSError:
                break
            m = re.search(r"^location:\s*(\S+)", data, re.I | re.M)
            if m and not data.startswith("M-SEARCH"):
                locs.add(m.group(1))
    except OSError:
        pass
    finally:
        s.close()
    return locs


def port_sweep(hosts, workers=64):
    """Description URLs of known renderers (Samsung :9197/dmr) on hosts whose port answers."""
    def probe(h):
        out = []
        for port, path in KNOWN_RENDERERS:
            try:
                socket.create_connection((h, port), 0.6).close()
                out.append("http://%s:%d%s" % (h, port, path))
            except OSError:
                pass
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return set(u for urls in ex.map(probe, hosts) for u in urls)


def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def describe(location, timeout=3):
    """Read a UPnP description: a renderer dict {name, ip, model, maker, control, location}, or None."""
    try:
        xml = urllib.request.urlopen(location, timeout=timeout).read()
        root = _strip_ns(ET.fromstring(xml))
    except Exception:
        return None
    base = (root.findtext("URLBase") or "").strip() or location
    for dev in root.iter("device"):
        for svc in dev.iter("service"):
            if "AVTransport" in (svc.findtext("serviceType") or ""):
                ctl = (svc.findtext("controlURL") or "").strip()
                if not ctl:
                    continue
                return {"name": (dev.findtext("friendlyName") or "TV").strip(),
                        "model": (dev.findtext("modelName") or "").strip(),
                        "maker": (dev.findtext("manufacturer") or "").strip(),
                        "category": (dev.findtext("X_deviceCategory") or dev.findtext("deviceType") or "").strip(),
                        "ip": urlparse(location).hostname, "location": location, "control": urljoin(base, ctl)}
    return None


def discover(log=None, hosts_limit=1022):
    """Every DLNA renderer on the LAN, TVs first: [{name, ip, model, maker, control, location}, ...]."""
    log = log or (lambda m: None)
    ifaces = lan_interfaces()
    if not ifaces:
        log("    tv scan: no LAN interface with an IPv4 address (not connected to a network?)")
        return []
    locs = set()
    for name, ip, net in ifaces:
        hosts = sweep_hosts(ip, net, hosts_limit)
        log("    tv scan: %s %s, %d hosts (SSDP multicast + unicast, renderer ports)" % (name, net, len(hosts)))
        res = {}
        t1 = threading.Thread(target=lambda: res.__setitem__("ssdp", ssdp_locations(ip, hosts)))
        t1.start()
        res["ports"] = port_sweep(hosts)
        t1.join()
        locs |= res.get("ssdp", set()) | res["ports"]
    with ThreadPoolExecutor(max_workers=16) as ex:
        devs = [d for d in ex.map(describe, sorted(locs)) if d]
    seen, out = set(), []
    for d in devs:                         # one entry per control URL (a TV can answer on several description URLs)
        if d["control"] not in seen:
            seen.add(d["control"])
            out.append(d)
    out.sort(key=lambda d: (0 if re.search(r"\bTV\b|Display\.TV", d["name"] + " " + d["category"]) else 1, d["ip"]))
    return out


def load_last():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return None


def save_last(dev):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as f:
            json.dump(dev, f, indent=1)
    except Exception:
        pass


def is_ip(text):
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def local_ip_towards(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 9))
        return s.getsockname()[0]
    finally:
        s.close()


# ================================================================ serving the files
class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler(media_dir):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self._serve(True)

        def do_GET(self):
            self._serve(False)

        def _serve(self, head):
            name = urllib.request.unquote(self.path.split("?")[0].lstrip("/"))
            path = os.path.join(media_dir, os.path.basename(name))      # only files directly inside media_dir
            mime = MIME.get(os.path.splitext(path)[1].lower())
            if not mime or not os.path.isfile(path):
                self.send_response(404)
                self.end_headers()
                return
            size = os.path.getsize(path)
            rng, a, b = self.headers.get("Range"), 0, size - 1
            m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
            if m:
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else size - 1
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", mime)
            self.send_header("contentFeatures.dlna.org", DLNA_FLAGS)
            self.send_header("transferMode.dlna.org", "Interactive" if mime.startswith("image/") else "Streaming")
            self.send_header("Accept-Ranges", "bytes")
            if m:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            if head:
                return
            with open(path, "rb") as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except Exception:
                        return
                    left -= len(chunk)
    return H


# ================================================================ the caster
class TvCaster(object):
    """Finds and keeps a TV in the background and casts the brain's TV actions to it. Nothing here blocks the caller:
    act() returns at once, and without a TV it only says so."""

    def __init__(self, tv="auto", media_dir="tv_media", port=DEFAULT_PORT, cooldown=20.0, rescan=30.0, log=print,
                 on_state=None):
        self.target = (tv or "auto").strip()
        self.media_dir = os.path.abspath(media_dir)
        self.port, self.cooldown, self.rescan, self.log = port, cooldown, rescan, log
        self.on_state = on_state or (lambda connected, text: None)
        self.dev, self.ip = None, None
        self.status = "searching for a TV..."
        self.srv = _Server(("0.0.0.0", port), _handler(self.media_dir))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.lock = threading.Lock()
        self.current, self.loop, self.pushed_at = None, False, 0.0
        self.closed, self.wake, self.found = threading.Event(), threading.Event(), threading.Event()

    def start(self):
        threading.Thread(target=self._connector, daemon=True).start()
        threading.Thread(target=self._looper, daemon=True).start()
        return self

    # ------------------------------------------------------------ finding the TV
    def _pick(self, devs):
        if not devs:
            return None
        if self.target not in ("auto", ""):
            for d in devs:
                if d["ip"] == self.target or self.target.lower() in d["name"].lower():
                    return d
            return None
        last = load_last() or {}
        for d in devs:
            if d["control"] == last.get("control"):
                return d
        return devs[0]

    def _find(self):
        """One boot/rescan attempt: the TV named/used before (quick), else a scan of the LAN."""
        quick = []
        if is_ip(self.target):
            quick = ["http://%s:%d%s" % (self.target, p, path) for p, path in KNOWN_RENDERERS]
        last = load_last()
        if last and (self.target in ("auto", "", last.get("ip")) or self.target.lower() in last.get("name", "").lower()):
            quick.insert(0, last["location"])
        for loc in quick:
            d = describe(loc, timeout=2)
            if d and self._pick([d]):
                return d, [d]
        devs = discover(self.log)
        if is_ip(self.target) and not any(d["ip"] == self.target for d in devs):   # an IP given but nothing described
            return None, devs
        return self._pick(devs), devs

    def _connector(self):
        warned = False
        while not self.closed.is_set():
            if self.dev is None:
                t = time.time()
                dev, devs = self._find()
                if dev:
                    self.ip = local_ip_towards(dev["ip"])
                    self.dev = dev
                    save_last(dev)
                    others = [d["name"] + " (" + d["ip"] + ")" for d in devs if d is not dev and d["control"] != dev["control"]]
                    self.status = "%s (%s)" % (dev["name"], dev["ip"])
                    self.log("TV connected: %s, %s %s at %s  (found in %.1f s; serving tv_media from %s:%d)%s"
                             % (dev["name"], dev["maker"], dev["model"], dev["ip"], time.time() - t, self.ip, self.port,
                                ("\n    other TVs found: " + ", ".join(others) + "  (choose with --tv NAME or IP)") if others else ""))
                    self.on_state(True, self.status)
                    self.found.set()
                    warned = False
                elif not warned:
                    what = "the TV '%s'" % self.target if self.target not in ("auto", "") else "a TV"
                    names = (" Renderers seen: " + ", ".join("%s (%s)" % (d["name"], d["ip"]) for d in devs) + ".") if devs else ""
                    self.status = "no TV found (still scanning every %.0f s)" % self.rescan
                    self.log("WARNING: could not find %s on the network.%s Q keeps running without the TV; "
                             "scanning again every %.0f s. (TV on? Same Wi-Fi as this machine?)" % (what, names, self.rescan))
                    self.on_state(False, self.status)
                    warned = True
            self.wake.wait(self.rescan)
            self.wake.clear()

    def _lost(self, err):
        if self.dev is not None:
            self.log("WARNING: lost the TV %s (%s); searching again. Q keeps running." % (self.status, err))
            self.dev, self.found = None, threading.Event()
            self.status = "TV lost, searching..."
            self.on_state(False, self.status)
            self.wake.set()

    # ------------------------------------------------------------ UPnP calls
    def _soap(self, action, body, dev=None):
        dev = dev or self.dev
        env = ('<?xml version="1.0" encoding="utf-8"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
               's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body><u:%s xmlns:u="%s">%s</u:%s>'
               '</s:Body></s:Envelope>' % (action, AVT, body, action)).encode("utf-8")
        req = urllib.request.Request(dev["control"], data=env, headers={
            "Content-Type": 'text/xml; charset="utf-8"', "SOAPACTION": '"%s#%s"' % (AVT, action)})
        return urllib.request.urlopen(req, timeout=6).read().decode("utf-8", "ignore")

    def state(self):
        if self.dev is None:
            return "?"
        try:
            m = re.search(r"<CurrentTransportState>([^<]*)", self._soap("GetTransportInfo", "<InstanceID>0</InstanceID>"))
            return m.group(1) if m else "?"
        except Exception:
            return "?"

    def _push(self, name):
        mime = MIME[os.path.splitext(name)[1].lower()]
        url = "http://%s:%d/%s" % (self.ip, self.port, urllib.request.quote(name))
        kind = "imageItem" if mime.startswith("image/") else "videoItem"
        meta = html.escape(
            '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/"><item id="0" parentID="-1" restricted="1">'
            '<dc:title>%s</dc:title><upnp:class>object.item.%s</upnp:class><res protocolInfo="http-get:*:%s:%s">%s</res>'
            '</item></DIDL-Lite>' % (html.escape(name), kind, mime, DLNA_FLAGS, html.escape(url)))
        body = ("<InstanceID>0</InstanceID><CurrentURI>%s</CurrentURI><CurrentURIMetaData>%s</CurrentURIMetaData>"
                % (html.escape(url), meta))
        try:
            self._soap("SetAVTransportURI", body)
        except urllib.request.HTTPError as e:
            if "705" not in e.read().decode("utf-8", "ignore"):
                raise
            # 705 "Transport is locked": the TV is still busy with the last item, or waiting for someone to allow this
            # device (Samsung asks once per new device). Stop, wait, try once more.
            try:
                self._soap("Stop", "<InstanceID>0</InstanceID>")
            except Exception:
                pass
            time.sleep(1.5)
            try:
                self._soap("SetAVTransportURI", body)
            except urllib.request.HTTPError:
                raise RuntimeError("the TV says its player is locked (UPnP 705). If the TV shows a prompt to allow this "
                                   "device, press Allow on the remote (asked once per new device), then greet Q again")
        try:
            self._soap("Play", "<InstanceID>0</InstanceID><Speed>1</Speed>")
        except Exception:
            if not mime.startswith("image/"):     # photos show on SetAVTransportURI; the TV answers Play with an error
                raise

    # ------------------------------------------------------------ what the brain calls
    def act(self, action):
        """Run a parsed TV action in the background. Returns a short line saying what was done (or skipped)."""
        if not action:
            return None
        if action.get("stop"):
            if self.dev is None:
                return "tv: no TV connected (%s)" % self.status
            with self.lock:
                self.current, self.loop = None, False
            threading.Thread(target=self._run, args=("stop", None), daemon=True).start()
            return "tv: stop"
        name = os.path.basename(action["file"])
        if os.path.splitext(name)[1].lower() not in MIME:
            return "tv: unsupported file type %s" % name
        if not os.path.isfile(os.path.join(self.media_dir, name)):
            return "tv: missing %s in %s" % (name, self.media_dir)
        if self.dev is None:
            return "tv: no TV connected, %s not shown (%s)" % (name, self.status)
        with self.lock:
            if name == self.current and time.time() - self.pushed_at < self.cooldown:
                return "tv: %s already on screen" % name        # "hi" twice in a row does not restart the video
            self.current, self.loop, self.pushed_at = name, action.get("loop", False), time.time()
        threading.Thread(target=self._run, args=("push", name), daemon=True).start()
        return "tv: %s%s on %s" % (name, " (loop)" if action.get("loop") else "", self.status)

    def _run(self, what, name):
        try:
            if what == "stop":
                self._soap("Stop", "<InstanceID>0</InstanceID>")
            else:
                self._push(name)
            self.log("    tv: %s %s" % (what, name or ""))
        except Exception as e:
            with self.lock:
                self.current = None
            if isinstance(e, (OSError, socket.timeout)) and not hasattr(e, "code"):   # unreachable, not an HTTP error
                self._lost(e)
            else:
                self.log("    tv %s failed: %s" % (what, e))

    def _looper(self):
        """For "loop" actions: start the file again whenever the TV reports it has finished."""
        while not self.closed.wait(4):
            with self.lock:
                name = self.current if self.loop and time.time() - self.pushed_at > 8 else None
            if name and self.dev is not None and self.state() in ("STOPPED", "NO_MEDIA_PRESENT"):
                with self.lock:
                    if self.current != name:
                        continue
                    self.pushed_at = time.time()
                self._run("push", name)

    def close(self):
        self.closed.set()
        self.wake.set()
        self.srv.shutdown()
        self.srv.server_close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Find TVs on the network, or cast one TV action the way Q does.")
    ap.add_argument("action", nargs="?", help="a file (path, or name inside --tv-media), optionally with 'loop' in quotes, or 'stop'")
    ap.add_argument("--scan", action="store_true", help="only list the TVs (DLNA renderers) on the network")
    ap.add_argument("--tv", default="auto", help="auto (default), an IP, or part of the TV's name")
    ap.add_argument("--tv-media", default=None, help="folder with the media (default: the file's own folder, else ./tv_media)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    if a.scan or not a.action:
        t = time.time()
        devs = discover(print)
        print("%d renderer(s) in %.1f s:" % (len(devs), time.time() - t))
        for d in devs:
            print("  %-28s %-15s %s %s   %s" % (d["name"], d["ip"], d["maker"], d["model"], d["control"]))
        return
    act = parse_action(a.action)
    media = a.tv_media
    if act and act.get("file"):
        if os.path.dirname(act["file"]) and not media:
            media = os.path.dirname(os.path.abspath(act["file"]))
        act["file"] = os.path.basename(act["file"])
    media = media or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv_media")
    c = TvCaster(a.tv, media, port=a.port, cooldown=0, rescan=1e9).start()
    if not c.found.wait(40):
        print("no TV found")
        return c.close()
    print(c.act(act))
    if act and act.get("file"):                  # keep serving until the TV has played it (or Ctrl-C)
        try:
            t0, began = time.time(), False
            while True:
                time.sleep(3)
                st = c.state()
                began = began or st in ("PLAYING", "TRANSITIONING", "PAUSED_PLAYBACK")
                if (began and st in ("STOPPED", "NO_MEDIA_PRESENT") and not act["loop"]) or (not began and time.time() - t0 > 30):
                    print("TV state %s; done" % st)
                    break
        except KeyboardInterrupt:
            pass
    else:
        time.sleep(2)
    c.close()


if __name__ == "__main__":
    main()
