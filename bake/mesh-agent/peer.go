package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Peer is a remote mesh-agent observed by the discovery loop.
type Peer struct {
	Name     string  `json:"name"`      // hostname or tailnet name
	Addr     string  `json:"addr"`      // host:port URL base, e.g. "host:4747"
	LastSeen string  `json:"last_seen"` // RFC3339
	Slots    []*Slot `json:"slots"`
}

// PeerCache persists discovered peers to peers.json.
type PeerCache struct {
	mu    sync.RWMutex
	path  string
	peers map[string]*Peer // keyed by Name
	stop  chan struct{}
}

func newPeerCache(path string) *PeerCache {
	return &PeerCache{
		path:  path,
		peers: make(map[string]*Peer),
		stop:  make(chan struct{}),
	}
}

func (p *PeerCache) load() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	data, err := os.ReadFile(p.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var peers map[string]*Peer
	if err := json.Unmarshal(data, &peers); err != nil {
		return err
	}
	if peers != nil {
		p.peers = peers
	}
	return nil
}

func (p *PeerCache) save() error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	data, err := json.MarshalIndent(p.peers, "", "  ")
	if err != nil {
		return err
	}
	tmp := p.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, p.path)
}

func (p *PeerCache) upsert(peer *Peer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	peer.LastSeen = time.Now().UTC().Format(time.RFC3339)
	p.peers[peer.Name] = peer
}

func (p *PeerCache) list() []*Peer {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]*Peer, 0, len(p.peers))
	for _, pr := range p.peers {
		out = append(out, pr)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// startDiscovery loads the existing cache and starts a background goroutine
// that polls peers listed in ~/.mesh/peers.txt every PollInterval.
//
// peers.txt format: one "host:port" per line. Lines starting with '#' are
// comments. Empty lines ignored.
func (p *PeerCache) startDiscovery(reg *Registry) {
	_ = p.load()
	go p.pollLoop()
}

// PollInterval is how often discovery hits each peer.
var PollInterval = 60 * time.Second

func (p *PeerCache) pollLoop() {
	tick := time.NewTicker(PollInterval)
	defer tick.Stop()
	// also fire once at startup so peers.json gets fresh data quickly
	p.pollOnce()
	for {
		select {
		case <-p.stop:
			return
		case <-tick.C:
			p.pollOnce()
		}
	}
}

func (p *PeerCache) pollOnce() {
	hosts, err := readPeersFile(filepath.Join(filepath.Dir(p.path), "peers.txt"))
	if err != nil || len(hosts) == 0 {
		return
	}
	changed := false
	for _, hp := range hosts {
		base := "http://" + hp
		slots, err := fetchSlots(base)
		if err != nil {
			// keep last-known data; don't drop on a single timeout
			continue
		}
		name := strings.SplitN(hp, ":", 2)[0]
		peer := &Peer{
			Name:  name,
			Addr:  hp,
			Slots: slots,
		}
		p.upsert(peer)
		changed = true
	}
	if changed {
		_ = p.save()
	}
}

// readPeersFile parses ~/.mesh/peers.txt, returning host:port entries.
func readPeersFile(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	return out, nil
}

// fetchSlots GET /slots from a remote agent and returns the parsed slots.
// Used by discovery loop in Phase 2.
func fetchSlots(baseURL string) ([]*Slot, error) {
	c := &http.Client{Timeout: 5 * time.Second}
	resp, err := c.Get(baseURL + "/slots")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	var out struct {
		Slots []*Slot `json:"slots"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Slots, nil
}
