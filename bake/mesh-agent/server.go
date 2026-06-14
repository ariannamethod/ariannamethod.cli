package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// (imports clean — io, encoding/json, net, net/http, os, strings, time, fmt)

// Server is the HTTP API.
//
// Endpoints (Phase 1):
//
//	GET  /             health + version
//	GET  /slots        local registry
//	GET  /slot/{id}    one slot manifest
//	GET  /peers        cached peer registry
//	POST /msg          deposit a message into ~/.mesh/inbox/
//	GET  /presence     simple liveness ping
//
// Phase 3 will add: POST /slot/{id}/invoke, GET /job/{id}/stream.
type Server struct {
	addr string
	reg  *Registry
	peer *PeerCache
	jobs *JobManager
}

func newServer(addr string, reg *Registry, peer *PeerCache, jobs *JobManager) *Server {
	return &Server{addr: addr, reg: reg, peer: peer, jobs: jobs}
}

func (s *Server) run() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/slots", s.handleSlots)
	mux.HandleFunc("/slot/", s.handleSlot)
	mux.HandleFunc("/peers", s.handlePeers)
	mux.HandleFunc("/msg", s.handleMsg)
	mux.HandleFunc("/presence", s.handlePresence)
	mux.HandleFunc("/jobs", s.handleJobs)
	mux.HandleFunc("/job/", s.handleJob)
	srv := &http.Server{
		Addr:              s.addr,
		Handler:           logging(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}
	return srv.ListenAndServe()
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"agent":   "mesh-agent",
		"version": Version,
		"slots":   s.reg.count(),
	})
}

func (s *Server) handleSlots(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"slots": s.reg.list()})
}

func (s *Server) handleSlot(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/slot/")
	if rest == "" {
		http.Error(w, "missing slot id", 400)
		return
	}
	// Two shapes:
	//   /slot/<id>          GET manifest
	//   /slot/<id>/invoke   POST job invocation
	if strings.HasSuffix(rest, "/invoke") {
		id := strings.TrimSuffix(rest, "/invoke")
		s.handleInvoke(w, r, id)
		return
	}
	slot, ok := s.reg.get(rest)
	if !ok {
		http.Error(w, "not found", 404)
		return
	}
	writeJSON(w, 200, slot)
}

// handleInvoke spawns a new job for a slot.
//
// Request body (JSON): {"args": ["arg1", "arg2", ...]}
// Response: {"job_id": "...", "state": "running"}
func (s *Server) handleInvoke(w http.ResponseWriter, r *http.Request, slotID string) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	slot, ok := s.reg.get(slotID)
	if !ok {
		http.Error(w, "slot not found", 404)
		return
	}
	var req struct {
		Args []string `json:"args"`
	}
	body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "expected JSON {args:[...]}", 400)
			return
		}
	}
	job, err := s.jobs.invoke(slot, req.Args)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, 202, map[string]any{
		"job_id": job.ID,
		"state":  job.State,
	})
}

func (s *Server) handleJobs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"jobs": s.jobs.list()})
}

// handleJob serves both:
//
//	GET    /job/{id}         status JSON
//	GET    /job/{id}/stream  SSE stream of stdout/stderr lines (live + backfill)
//	DELETE /job/{id}         kill
func (s *Server) handleJob(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/job/")
	if rest == "" {
		http.Error(w, "missing job id", 400)
		return
	}
	stream := false
	id := rest
	if strings.HasSuffix(rest, "/stream") {
		stream = true
		id = strings.TrimSuffix(rest, "/stream")
	}
	job, ok := s.jobs.get(id)
	if !ok {
		http.Error(w, "job not found", 404)
		return
	}
	if r.Method == "DELETE" {
		if err := s.jobs.kill(id); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, map[string]any{"killed": id})
		return
	}
	if !stream {
		writeJSON(w, 200, job)
		return
	}
	// SSE stream
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Backfill ring buffer first
	for _, line := range job.snapshot() {
		writeSSE(w, "log", strings.TrimRight(line, "\n"))
	}
	flusher.Flush()
	// Live channel
	ch := job.subscribeChan()
	ctxDone := r.Context().Done()
	for {
		select {
		case <-ctxDone:
			return
		case line, more := <-ch:
			if !more {
				// job ended; send final state and close
				writeSSE(w, "state", string(job.State))
				flusher.Flush()
				return
			}
			writeSSE(w, "log", strings.TrimRight(line, "\n"))
			flusher.Flush()
		}
	}
}

func writeSSE(w http.ResponseWriter, event, data string) {
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
}

func (s *Server) handlePeers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"peers": s.peer.list()})
}

func (s *Server) handleMsg(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	// Minimal payload validation: must be valid JSON with from/to/body.
	var msg struct {
		From string `json:"from"`
		To   string `json:"to"`
		Body string `json:"body"`
	}
	if err := json.Unmarshal(body, &msg); err != nil {
		http.Error(w, "expected JSON {from,to,body}", 400)
		return
	}
	stamp := time.Now().UTC().Format("20060102T150405.000Z")
	fname := fmt.Sprintf("%s/%s_%s.json", inboxDir(), stamp, sanitize(msg.From))
	if err := os.WriteFile(fname, body, 0o644); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, 202, map[string]any{"queued": fname})
}

func (s *Server) handlePresence(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"name":    nodeName(),
		"now":     time.Now().UTC().Format(time.RFC3339),
		"slots":   s.reg.count(),
		"version": Version,
	})
}

// ── helpers ───────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(v)
}

func sanitize(s string) string {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '-' || r == '_' {
			out = append(out, r)
		} else {
			out = append(out, '_')
		}
	}
	if len(out) == 0 {
		return "anon"
	}
	return string(out)
}

// resolveBind picks an address to listen on.
// If bind is "" we try to pick the first tailnet IPv4 (100.x.x.x).
// If none found, we fall back to 0.0.0.0 (only safe inside a Tailscale-only ACL).
//
// Android note: Termux runs under an Android UID without netlink RIB
// permission, so net.InterfaceAddrs returns
// `route ip+net: netlinkrib: permission denied` on every phone in the mesh.
// Treat that the same as "no tailnet IP found" and fall through to the
// 0.0.0.0 catch-all rather than refusing to start. Tailscale binds all
// interfaces, so mesh reach via the tailnet still works; only
// writeAgentTarget records loopback for local client lookup.
func resolveBind(bind string, port int) (string, error) {
	if bind != "" {
		return fmt.Sprintf("%s:%d", bind, port), nil
	}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Printf("resolveBind: net.InterfaceAddrs failed (%v); falling back to 0.0.0.0:%d", err, port)
		return fmt.Sprintf("0.0.0.0:%d", port), nil
	}
	for _, a := range addrs {
		if ip, ok := a.(*net.IPNet); ok && ip.IP.To4() != nil {
			s := ip.IP.String()
			if strings.HasPrefix(s, "100.") {
				return fmt.Sprintf("%s:%d", s, port), nil
			}
		}
	}
	return fmt.Sprintf("0.0.0.0:%d", port), nil
}

// logging is a thin access-log middleware.
func logging(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		ww := &statusWriter{ResponseWriter: w, code: 200}
		h.ServeHTTP(ww, r)
		fmt.Printf("%s %s %d %s %s\n",
			r.Method, r.URL.Path, ww.code,
			time.Since(t0).Round(time.Millisecond), r.RemoteAddr)
	})
}

type statusWriter struct {
	http.ResponseWriter
	code int
}

func (s *statusWriter) WriteHeader(code int) {
	s.code = code
	s.ResponseWriter.WriteHeader(code)
}

// Flush exposes the underlying ResponseWriter's Flush so that SSE handlers
// can satisfy http.Flusher even when wrapped by our access-log middleware.
func (s *statusWriter) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
