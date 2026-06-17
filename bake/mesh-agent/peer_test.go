package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestReadPeersFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "peers.txt")
	content := "# a comment\nhost-a:4747\n\n  host-b:4747  \n#another comment\nhost-c:4747\n"
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := readPeersFile(p)
	if err != nil {
		t.Fatalf("readPeersFile: %v", err)
	}
	want := []string{"host-a:4747", "host-b:4747", "host-c:4747"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("readPeersFile = %v, want %v (comments/blanks must be skipped, lines trimmed)", got, want)
	}
}

func TestReadPeersFileMissing(t *testing.T) {
	if _, err := readPeersFile(filepath.Join(t.TempDir(), "nope.txt")); err == nil {
		t.Error("expected error for missing peers.txt")
	}
}

func TestPeerCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "peers.json")

	pc := newPeerCache(path)
	pc.upsert(&Peer{Name: "host-a", Addr: "host-a:4747"})
	if err := pc.save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	pc2 := newPeerCache(path)
	if err := pc2.load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	list := pc2.list()
	if len(list) != 1 {
		t.Fatalf("len(list) = %d, want 1", len(list))
	}
	if list[0].Name != "host-a" || list[0].Addr != "host-a:4747" {
		t.Errorf("round-trip peer = %+v, want host-a/host-a:4747", list[0])
	}
	if list[0].LastSeen == "" {
		t.Error("LastSeen should be stamped by upsert")
	}
}

func TestPeerCacheLoadMissingIsNotError(t *testing.T) {
	pc := newPeerCache(filepath.Join(t.TempDir(), "nope.json"))
	if err := pc.load(); err != nil {
		t.Errorf("missing peers.json should be a no-op (nil err), got %v", err)
	}
	if len(pc.list()) != 0 {
		t.Error("expected empty cache after loading missing file")
	}
}
