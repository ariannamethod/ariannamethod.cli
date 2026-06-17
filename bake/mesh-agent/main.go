// mesh-agent — capability-broker daemon for the tailnet mesh.
//
// Each node runs one mesh-agent process. The agent:
//   - loads slot manifests from ~/.mesh/slots/*.toml
//   - exposes them over an HTTP API bound to the node's tailnet IP
//   - discovers peer agents and caches their slot registries
//   - provides an inbox for agent-to-agent messages
//
// Subcommands:
//
//	serve         start the HTTP server (long-running daemon)
//	slots         print local slots (debug)
//	peers         print discovered peers (debug)
//	register PATH copy a slot manifest into ~/.mesh/slots/
//
// Bind address defaults to the node's tailnet IPv4. Override with --bind.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const Version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "serve":
		cmdServe(args)
	case "slots":
		cmdSlots(args)
	case "peers":
		cmdPeers(args)
	case "register":
		cmdRegister(args)
	case "exec":
		cmdExec(args)
	case "send":
		cmdSend(args)
	case "broadcast":
		cmdBroadcast(args)
	case "listen":
		cmdListen(args)
	case "status":
		cmdStatus(args)
	case "version", "-v", "--version":
		fmt.Println("mesh-agent", Version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `mesh-agent %s — arianna mesh capability broker

usage: mesh-agent <command> [flags]

server:
  serve              start HTTP daemon
  slots              list local slot manifests
  peers              list discovered peers + their slots
  register PATH      install a slot manifest into ~/.mesh/slots/

client:
  status [PEER]      show /presence of local agent (or PEER)
  exec PEER SLOT [ARGS...]
                     POST /slot/SLOT/invoke on PEER, stream SSE log
  send PEER TEXT     POST /msg on PEER ({from: us, to: PEER, body: TEXT})
  broadcast TEXT     POST /msg to every peer in ~/.mesh/peers.json
                     except self (hostname-matched), parallel.
  listen [flags]     read ~/.mesh/inbox/ chronologically
                     flags: --since DURATION (e.g. 30m, 1h)
                            --watch (poll every 2s, tail-like)

other:
  version            print version
  help               show this message

defaults:
  state dir          ~/.mesh
  slots dir          ~/.mesh/slots
  bind addr          first tailnet IPv4 (100.x.x.x), port 4747
  client target      reads ~/.mesh/agent (host:port) or default localhost:4747
                     PEER may be host or host:port (port defaults to 4747)
`, Version)
}

// resolvePeer returns "host:port" string. If user passed "host" only, append :4747.
func resolvePeer(p string) string {
	if p == "" {
		// fallback: try ~/.mesh/agent file, else "127.0.0.1:4747"
		if data, err := os.ReadFile(filepath.Join(meshDir(), "agent")); err == nil {
			s := strings.TrimSpace(string(data))
			if s != "" {
				return s
			}
		}
		return "127.0.0.1:4747"
	}
	if strings.Contains(p, ":") {
		return p
	}
	return p + ":4747"
}

// writeAgentTarget records a locally-reachable address for this node's
// daemon in ~/.mesh/agent. Client commands invoked with no explicit PEER
// resolve through resolvePeer(""), which reads this file; without it they
// fall back to 127.0.0.1:4747 — wrong whenever resolveBind picked a
// tailnet IP (100.x) instead of the loopback, as it does on every node
// where tailscaled is up at serve time. A wildcard bind (0.0.0.0 / [::])
// is rewritten to loopback: a client cannot dial a wildcard address, and
// loopback is always reachable on the local host.
func writeAgentTarget(addr string, port int) {
	target := addr
	if strings.HasPrefix(addr, "0.0.0.0:") || strings.HasPrefix(addr, "[::]:") {
		target = fmt.Sprintf("127.0.0.1:%d", port)
	}
	if err := os.WriteFile(filepath.Join(meshDir(), "agent"),
		[]byte(target+"\n"), 0o644); err != nil {
		log.Printf("warn: write agent target: %v", err)
	}
}

// nodeName returns this node's mesh identity. It prefers an explicit
// MESH_NODE_NAME (set by the node's service file) over os.Hostname():
// Termux nodes report the Android default hostname "localhost", which
// would make every phone indistinguishable in presence and message
// `from` fields. Falls back to the hostname, then "unknown".
func nodeName() string {
	if n := strings.TrimSpace(os.Getenv("MESH_NODE_NAME")); n != "" {
		return n
	}
	if h, err := os.Hostname(); err == nil && h != "" {
		return h
	}
	return "unknown"
}

func meshDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("UserHomeDir: %v", err)
	}
	return filepath.Join(home, ".mesh")
}

func slotsDir() string  { return filepath.Join(meshDir(), "slots") }
func peersFile() string { return filepath.Join(meshDir(), "peers.json") }
func inboxDir() string  { return filepath.Join(meshDir(), "inbox") }
func outboxDir() string { return filepath.Join(meshDir(), "outbox") }

func ensureDirs() {
	for _, d := range []string{meshDir(), slotsDir(), inboxDir(), outboxDir()} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			log.Fatalf("mkdir %s: %v", d, err)
		}
	}
}

// cmdServe starts the HTTP daemon.
func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	bind := fs.String("bind", "", "bind address (default: first tailnet IPv4)")
	port := fs.Int("port", 4747, "TCP port")
	fs.Parse(args)
	ensureDirs()

	addr, err := resolveBind(*bind, *port)
	if err != nil {
		log.Fatalf("bind: %v", err)
	}
	writeAgentTarget(addr, *port)

	reg := newRegistry()
	if err := reg.loadFromDisk(slotsDir()); err != nil {
		log.Printf("warn: load slots: %v", err)
	}

	pr := newPeerCache(peersFile())
	pr.startDiscovery(reg)

	jm := newJobManager()
	srv := newServer(addr, reg, pr, jm)
	log.Printf("mesh-agent %s serving on http://%s", Version, addr)
	log.Printf("  slots: %d local", reg.count())
	if err := srv.run(); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

// cmdSlots prints local slot manifests.
func cmdSlots(args []string) {
	ensureDirs()
	reg := newRegistry()
	if err := reg.loadFromDisk(slotsDir()); err != nil {
		log.Fatalf("load slots: %v", err)
	}
	for _, s := range reg.list() {
		fmt.Printf("%s\t%s\n", s.ID, s.Description)
	}
}

// cmdPeers prints the cached peer registry.
func cmdPeers(args []string) {
	ensureDirs()
	pr := newPeerCache(peersFile())
	if err := pr.load(); err != nil {
		log.Fatalf("load peers: %v", err)
	}
	for _, p := range pr.list() {
		fmt.Printf("%s\t%s\tslots=%d\n", p.Name, p.Addr, len(p.Slots))
	}
}

// cmdRegister copies a slot manifest file into the slots dir.
func cmdRegister(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "register: missing manifest path")
		os.Exit(2)
	}
	ensureDirs()
	src := args[0]
	data, err := os.ReadFile(src)
	if err != nil {
		log.Fatalf("read %s: %v", src, err)
	}
	s, err := parseSlot(data)
	if err != nil {
		log.Fatalf("parse %s: %v", src, err)
	}
	dst := filepath.Join(slotsDir(), slotFilename(s.ID))
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		log.Fatalf("write %s: %v", dst, err)
	}
	fmt.Printf("registered %s -> %s\n", s.ID, dst)
}

// slotFilename converts a slot id like "train/llama3-bpe-15m" into a flat
// filename "train__llama3-bpe-15m.toml" suitable for the slots dir.
func slotFilename(id string) string {
	out := make([]rune, 0, len(id)+5)
	for _, r := range id {
		if r == '/' {
			out = append(out, '_', '_')
		} else {
			out = append(out, r)
		}
	}
	return string(out) + ".toml"
}
