package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/BurntSushi/toml"
)

// Slot is a network-addressable capability that a node exposes.
//
// Each slot maps to a runnable command (Exec) optionally with a JSON-shaped
// argument schema, plus declared resource requirements and metadata used by
// peers when they decide whether to dispatch a job here.
type Slot struct {
	ID          string `toml:"id"          json:"id"`
	Description string `toml:"description" json:"description,omitempty"`

	// Exec is the shell command. Argument substitution: $1, $2 (positional)
	// or {key} (named, when args_schema declares names).
	Exec string `toml:"exec" json:"exec"`

	// Async slots return immediately with a job-id; the agent runs the
	// command in the background and exposes /job/<id>/stream for SSE.
	Async bool `toml:"async" json:"async"`

	// ArgsSchema is a free-form description for human readers; the agent
	// does not enforce it. Example: "[steps:int, lr:float, corpus:path]"
	ArgsSchema string `toml:"args_schema" json:"args_schema,omitempty"`

	// Output describes what the slot produces.
	Output string `toml:"output" json:"output,omitempty"`

	// Requires declares resource requirements. All fields optional.
	Requires struct {
		RAM  string `toml:"ram"  json:"ram,omitempty"`
		Disk string `toml:"disk" json:"disk,omitempty"`
		GPU  bool   `toml:"gpu"  json:"gpu"`
	} `toml:"requires" json:"requires"`

	// Provides describes capabilities advertised by this slot.
	Provides struct {
		StreamingMetrics string `toml:"streaming_metrics" json:"streaming_metrics,omitempty"`
	} `toml:"provides" json:"provides"`

	// Ownership constrains where the slot is meaningful. The agent itself
	// does not refuse to register a manifest if the host doesn't match;
	// the fields are advisory for peers.
	Ownership struct {
		Arch []string `toml:"arch" json:"arch,omitempty"`
		OS   []string `toml:"os"   json:"os,omitempty"`
	} `toml:"ownership" json:"ownership"`

	// path is the manifest file on disk (set by loader, not in TOML, not in JSON).
	path string
}

// parseSlot decodes one TOML manifest.
func parseSlot(data []byte) (*Slot, error) {
	var wrap struct {
		Slot Slot `toml:"slot"`
	}
	// support both [slot] section style and flat-root style
	if _, err := toml.Decode(string(data), &wrap); err == nil && wrap.Slot.ID != "" {
		s := wrap.Slot
		return &s, nil
	}
	var s Slot
	if _, err := toml.Decode(string(data), &s); err != nil {
		return nil, err
	}
	if s.ID == "" {
		return nil, fmt.Errorf("manifest missing id")
	}
	return &s, nil
}

// Registry holds the loaded local slot manifests.
type Registry struct {
	mu    sync.RWMutex
	slots map[string]*Slot
}

func newRegistry() *Registry {
	return &Registry{slots: make(map[string]*Slot)}
}

func (r *Registry) loadFromDisk(dir string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.slots = make(map[string]*Slot)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".toml" {
			continue
		}
		full := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(full)
		if err != nil {
			return fmt.Errorf("%s: %w", e.Name(), err)
		}
		s, err := parseSlot(data)
		if err != nil {
			return fmt.Errorf("%s: %w", e.Name(), err)
		}
		s.path = full
		r.slots[s.ID] = s
	}
	return nil
}

func (r *Registry) get(id string) (*Slot, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.slots[id]
	return s, ok
}

func (r *Registry) list() []*Slot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Slot, 0, len(r.slots))
	for _, s := range r.slots {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (r *Registry) count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.slots)
}
