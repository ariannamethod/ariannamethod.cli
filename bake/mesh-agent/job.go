package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"sync"
	"syscall"
	"time"
)

// JobState is the lifecycle of an invoked slot.
type JobState string

const (
	JobPending JobState = "pending"
	JobRunning JobState = "running"
	JobDone    JobState = "done"
	JobFailed  JobState = "failed"
	JobKilled  JobState = "killed"
)

// Job represents one invocation of a slot.
type Job struct {
	ID         string    `json:"id"`
	SlotID     string    `json:"slot_id"`
	Args       []string  `json:"args"`
	State      JobState  `json:"state"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at,omitempty"`
	ExitCode   int       `json:"exit_code"`
	Error      string    `json:"error,omitempty"`

	// internal
	cmd       *exec.Cmd
	subscribe chan chan string // channels that want SSE chunks
	subMu     sync.Mutex
	buffer    []string // ring buffer of recent stdout lines
	bufMu     sync.Mutex
	done      chan struct{}
}

const ringBufferLines = 4096

// JobManager owns all jobs across the agent's lifetime.
type JobManager struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

func newJobManager() *JobManager {
	return &JobManager{jobs: make(map[string]*Job)}
}

// invoke starts a new job for the given slot.
//
// args are appended verbatim after the slot's Exec command, split shell-style
// elsewhere (caller pre-tokenizes). The agent does not parse a shell — Exec is
// a single program path; args are positional.
func (jm *JobManager) invoke(slot *Slot, args []string) (*Job, error) {
	if slot.Exec == "" {
		return nil, errors.New("slot has empty exec")
	}
	id := newJobID()
	j := &Job{
		ID:        id,
		SlotID:    slot.ID,
		Args:      args,
		State:     JobPending,
		StartedAt: time.Now().UTC(),
		subscribe: make(chan chan string, 4),
		done:      make(chan struct{}),
	}
	// Build command. Exec is run via `sh -c` so the manifest can use a
	// full shell command-line (pipes, env vars, glob); user args are
	// appended after the Exec line, separated by spaces (caller must
	// shell-quote args that contain whitespace).
	cmdline := slot.Exec
	if len(args) > 0 {
		cmdline = cmdline + " " + joinShellArgs(args)
	}
	c := exec.Command("sh", "-c", cmdline)
	stdoutR, err := c.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderrR, err := c.StderrPipe()
	if err != nil {
		return nil, err
	}
	j.cmd = c

	jm.mu.Lock()
	jm.jobs[id] = j
	jm.mu.Unlock()

	if err := c.Start(); err != nil {
		j.State = JobFailed
		j.Error = err.Error()
		j.FinishedAt = time.Now().UTC()
		close(j.done)
		return j, nil
	}
	j.State = JobRunning

	// stream stdout + stderr line-by-line into the ring buffer + subscribers
	go j.pumpReader(stdoutR, "")
	go j.pumpReader(stderrR, "stderr: ")

	// reaper
	go func() {
		err := c.Wait()
		j.FinishedAt = time.Now().UTC()
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
					j.ExitCode = status.ExitStatus()
				}
				j.State = JobFailed
				j.Error = err.Error()
			} else {
				j.State = JobFailed
				j.Error = err.Error()
			}
		} else {
			j.State = JobDone
			j.ExitCode = 0
		}
		// fan-out close to subscribers
		j.subMu.Lock()
		for _, ch := range j.subscriptions() {
			close(ch)
		}
		j.subMu.Unlock()
		close(j.done)
	}()
	return j, nil
}

func (j *Job) pumpReader(r io.Reader, prefix string) {
	br := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := br.ReadString('\n')
		if line != "" {
			full := prefix + line
			j.appendBuffer(full)
			j.fanOut(full)
		}
		if err != nil {
			return
		}
	}
}

func (j *Job) appendBuffer(line string) {
	j.bufMu.Lock()
	defer j.bufMu.Unlock()
	j.buffer = append(j.buffer, line)
	if len(j.buffer) > ringBufferLines {
		j.buffer = j.buffer[len(j.buffer)-ringBufferLines:]
	}
}

// snapshot returns the current ring buffer copy for backfill on subscribe.
func (j *Job) snapshot() []string {
	j.bufMu.Lock()
	defer j.bufMu.Unlock()
	out := make([]string, len(j.buffer))
	copy(out, j.buffer)
	return out
}

func (j *Job) subscriptions() []chan string {
	out := make([]chan string, 0)
	for {
		select {
		case ch := <-j.subscribe:
			out = append(out, ch)
		default:
			return out
		}
	}
}

func (j *Job) fanOut(line string) {
	j.subMu.Lock()
	defer j.subMu.Unlock()
	subs := j.subscriptions()
	for _, ch := range subs {
		select {
		case ch <- line:
			// re-queue subscriber so it can keep receiving
			j.subscribe <- ch
		default:
			// slow consumer: drop and re-queue (keeps stream sane)
			j.subscribe <- ch
		}
	}
}

// subscribeChan returns a fresh channel that receives all subsequent stdout
// lines until the job ends. The caller must drain promptly or risk drops.
func (j *Job) subscribeChan() chan string {
	ch := make(chan string, 64)
	j.subscribe <- ch
	return ch
}

func (jm *JobManager) get(id string) (*Job, bool) {
	jm.mu.RLock()
	defer jm.mu.RUnlock()
	j, ok := jm.jobs[id]
	return j, ok
}

func (jm *JobManager) list() []*Job {
	jm.mu.RLock()
	defer jm.mu.RUnlock()
	out := make([]*Job, 0, len(jm.jobs))
	for _, j := range jm.jobs {
		out = append(out, j)
	}
	sort.Slice(out, func(i, k int) bool {
		return out[i].StartedAt.After(out[k].StartedAt)
	})
	return out
}

// kill terminates a running job.
func (jm *JobManager) kill(id string) error {
	j, ok := jm.get(id)
	if !ok {
		return fmt.Errorf("no such job: %s", id)
	}
	if j.cmd == nil || j.cmd.Process == nil {
		return errors.New("job has no process")
	}
	if err := j.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		return err
	}
	j.State = JobKilled
	return nil
}

func newJobID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// joinShellArgs single-quotes each arg unless it's already safe (alnum + a few),
// then joins with spaces. Good enough for typical positional args.
func joinShellArgs(args []string) string {
	out := make([]string, len(args))
	for i, a := range args {
		out[i] = shellQuote(a)
	}
	return joinSpace(out)
}

func shellQuote(s string) string {
	if s == "" {
		return "''"
	}
	safe := true
	for _, r := range s {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '/' || r == '_' || r == '.' || r == '-' ||
			r == ':' || r == '=' || r == '+' || r == '@') {
			safe = false
			break
		}
	}
	if safe {
		return s
	}
	// single-quote, escape any single-quote in s as '\''
	out := "'"
	for _, r := range s {
		if r == '\'' {
			out += `'\''`
		} else {
			out += string(r)
		}
	}
	out += "'"
	return out
}

func joinSpace(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	out := ss[0]
	for _, s := range ss[1:] {
		out += " " + s
	}
	return out
}
