package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// cmdStatus — `mesh-agent status [PEER]`
func cmdStatus(args []string) {
	target := ""
	if len(args) > 0 {
		target = args[0]
	}
	addr := resolvePeer(target)
	c := &http.Client{Timeout: 5 * time.Second}
	resp, err := c.Get("http://" + addr + "/presence")
	if err != nil {
		fmt.Fprintf(os.Stderr, "GET %s: %v\n", addr, err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	io.Copy(os.Stdout, resp.Body)
	fmt.Println()
}

// cmdExec — `mesh-agent exec PEER SLOT [ARGS...]`
//
// POSTs an invoke and then streams the SSE log to stdout. Exits with the
// remote job's exit_code (best-effort: 0 on done, 1 on failure).
func cmdExec(args []string) {
	if len(args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: mesh-agent exec PEER SLOT [ARGS...]")
		os.Exit(2)
	}
	addr := resolvePeer(args[0])
	slotID := args[1]
	jobArgs := args[2:]

	body, _ := json.Marshal(map[string]any{"args": jobArgs})
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Post("http://"+addr+"/slot/"+slotID+"/invoke",
		"application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "invoke: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		io.Copy(os.Stderr, resp.Body)
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}
	var jr struct {
		JobID string `json:"job_id"`
		State string `json:"state"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&jr); err != nil {
		fmt.Fprintf(os.Stderr, "decode invoke: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "[mesh] job %s on %s slot %s — streaming...\n",
		jr.JobID, addr, slotID)

	// Stream SSE
	streamClient := &http.Client{Timeout: 0}
	sresp, err := streamClient.Get(fmt.Sprintf("http://%s/job/%s/stream", addr, jr.JobID))
	if err != nil {
		fmt.Fprintf(os.Stderr, "stream: %v\n", err)
		os.Exit(1)
	}
	defer sresp.Body.Close()
	br := bufio.NewReader(sresp.Body)
	finalState := ""
	for {
		line, err := br.ReadString('\n')
		if line != "" {
			line = strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(line, "event: ") {
				event := strings.TrimPrefix(line, "event: ")
				// next line should be data:
				dataLine, _ := br.ReadString('\n')
				dataLine = strings.TrimRight(dataLine, "\r\n")
				dataLine = strings.TrimPrefix(dataLine, "data: ")
				switch event {
				case "log":
					fmt.Println(dataLine)
				case "state":
					finalState = dataLine
					fmt.Fprintf(os.Stderr, "[mesh] state: %s\n", dataLine)
				}
			}
		}
		if err != nil {
			break
		}
	}
	if finalState == string(JobFailed) || finalState == string(JobKilled) {
		os.Exit(1)
	}
}

// cmdSend — `mesh-agent send PEER TEXT...`
func cmdSend(args []string) {
	if len(args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: mesh-agent send PEER TEXT...")
		os.Exit(2)
	}
	addr := resolvePeer(args[0])
	text := strings.Join(args[1:], " ")
	from := nodeName()
	body, _ := json.Marshal(map[string]any{
		"from": from,
		"to":   args[0],
		"body": text,
	})
	c := &http.Client{Timeout: 5 * time.Second}
	resp, err := c.Post("http://"+addr+"/msg",
		"application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "POST: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	io.Copy(os.Stdout, resp.Body)
	fmt.Println()
}

// cmdBroadcast — `mesh-agent broadcast TEXT...`
//
// Fan out TEXT to every peer in ~/.mesh/peers.json except self
// (hostname-matched). Per-peer result printed; summary at end.
// Exit 1 only if every targeted peer failed.
func cmdBroadcast(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: mesh-agent broadcast TEXT...")
		os.Exit(2)
	}
	text := strings.Join(args, " ")
	from := nodeName()

	pr := newPeerCache(peersFile())
	if err := pr.load(); err != nil {
		fmt.Fprintf(os.Stderr, "load peers: %v\n", err)
		os.Exit(1)
	}
	peers := pr.list()
	if len(peers) == 0 {
		fmt.Fprintln(os.Stderr, "no peers known — check ~/.mesh/peers.txt + serve discovery loop")
		os.Exit(1)
	}

	type result struct {
		name string
		addr string
		err  error
	}
	ch := make(chan result, len(peers))
	sent := 0
	for _, peer := range peers {
		if peer.Name == from {
			continue
		}
		sent++
		go func(p *Peer) {
			body, _ := json.Marshal(map[string]any{
				"from": from,
				"to":   p.Name,
				"body": text,
			})
			c := &http.Client{Timeout: 5 * time.Second}
			resp, err := c.Post("http://"+p.Addr+"/msg",
				"application/json", bytes.NewReader(body))
			if err != nil {
				ch <- result{p.Name, p.Addr, err}
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode >= 300 {
				ch <- result{p.Name, p.Addr, fmt.Errorf("status %d", resp.StatusCode)}
				return
			}
			ch <- result{p.Name, p.Addr, nil}
		}(peer)
	}

	ok, fail := 0, 0
	for i := 0; i < sent; i++ {
		r := <-ch
		if r.err != nil {
			fmt.Fprintf(os.Stderr, "  err %-20s (%s): %v\n", r.name, r.addr, r.err)
			fail++
		} else {
			fmt.Printf("  ok  %-20s (%s)\n", r.name, r.addr)
			ok++
		}
	}
	fmt.Printf("broadcast from=%s: %d ok, %d fail (%d peers known, self excluded)\n",
		from, ok, fail, len(peers))
	if sent > 0 && ok == 0 {
		os.Exit(1)
	}
}

// cmdListen — `mesh-agent listen [--since DURATION] [--watch]`
//
// Read messages from ~/.mesh/inbox/ and print chronologically
// (oldest → newest). --since filters by file mtime (e.g. "30m",
// "1h", "5s") per time.ParseDuration. --watch polls every 2s, skips
// already-shown filenames via in-memory set.
//
// Message format (frozen by server.go handleMsg):
//
//	{"from": "<peer>", "to": "<peer>", "body": "<text>"}
//
// Inbox filenames are "<UTC_stamp>_<from>.json".
func cmdListen(args []string) {
	var sinceDur time.Duration
	watch := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--watch":
			watch = true
		case "--since":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--since requires DURATION (e.g. 30m, 1h)")
				os.Exit(2)
			}
			d, err := time.ParseDuration(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "bad duration %q: %v\n", args[i+1], err)
				os.Exit(2)
			}
			sinceDur = d
			i++
		case "-h", "--help":
			fmt.Fprintln(os.Stderr,
				"usage: mesh-agent listen [--since DURATION] [--watch]")
			return
		default:
			fmt.Fprintf(os.Stderr, "unknown flag: %s\n", args[i])
			os.Exit(2)
		}
	}

	shown := make(map[string]bool)
	first := true
	for {
		printed := listenScan(inboxDir(), sinceDur, shown)
		if first && printed == 0 && !watch {
			fmt.Fprintln(os.Stderr, "(inbox empty)")
		}
		first = false
		if !watch {
			return
		}
		time.Sleep(2 * time.Second)
	}
}

// listenScan reads inboxDir, prints new entries chronologically, and
// returns the count of newly-printed messages. shown[name]=true marks
// already-printed.
func listenScan(dir string, sinceDur time.Duration, shown map[string]bool) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read inbox: %v\n", err)
		return 0
	}

	type item struct {
		name  string
		mtime time.Time
		from  string
		to    string
		body  string
		raw   string
	}

	var items []item
	cutoff := time.Time{}
	if sinceDur > 0 {
		cutoff = time.Now().Add(-sinceDur)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		if shown[e.Name()] {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if !cutoff.IsZero() && info.ModTime().Before(cutoff) {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var m struct {
			From string `json:"from"`
			To   string `json:"to"`
			Body string `json:"body"`
		}
		it := item{name: e.Name(), mtime: info.ModTime()}
		if err := json.Unmarshal(data, &m); err != nil {
			it.raw = string(data)
		} else {
			it.from = m.From
			it.to = m.To
			it.body = m.Body
		}
		items = append(items, it)
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].mtime.Before(items[j].mtime)
	})

	for _, it := range items {
		ts := it.mtime.UTC().Format("2006-01-02 15:04:05Z")
		if it.raw != "" {
			fmt.Printf("%s  (unparsed %s)\n  %s\n\n", ts, it.name, it.raw)
		} else {
			fmt.Printf("%s  %s → %s\n  %s\n\n", ts, it.from, it.to, it.body)
		}
		shown[it.name] = true
	}
	return len(items)
}
