package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseSlotSection(t *testing.T) {
	data := []byte("[slot]\nid = \"echo\"\ndescription = \"echo test\"\nexec = \"echo hi\"\n")
	s, err := parseSlot(data)
	if err != nil {
		t.Fatalf("parseSlot: %v", err)
	}
	if s.ID != "echo" {
		t.Errorf("ID = %q, want echo", s.ID)
	}
	if s.Exec != "echo hi" {
		t.Errorf("Exec = %q, want \"echo hi\"", s.Exec)
	}
}

func TestParseSlotFlat(t *testing.T) {
	data := []byte("id = \"train/x\"\nexec = \"run.sh\"\n")
	s, err := parseSlot(data)
	if err != nil {
		t.Fatalf("parseSlot flat: %v", err)
	}
	if s.ID != "train/x" {
		t.Errorf("ID = %q, want train/x", s.ID)
	}
}

func TestParseSlotMissingID(t *testing.T) {
	if _, err := parseSlot([]byte("exec = \"x\"\n")); err == nil {
		t.Fatal("expected error for manifest missing id")
	}
}

func TestSlotFilename(t *testing.T) {
	cases := map[string]string{
		"heart":                "heart.toml",
		"train/llama3-bpe-15m": "train__llama3-bpe-15m.toml",
		"a/b/c":                "a__b__c.toml",
	}
	for in, want := range cases {
		if got := slotFilename(in); got != want {
			t.Errorf("slotFilename(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRegistryLoadFromDisk(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "echo.toml"),
		[]byte("[slot]\nid = \"echo\"\nexec = \"echo hi\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// a non-.toml file must be ignored by the loader
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("ignore me"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := newRegistry()
	if err := r.loadFromDisk(dir); err != nil {
		t.Fatalf("loadFromDisk: %v", err)
	}
	if r.count() != 1 {
		t.Fatalf("count = %d, want 1 (note.txt must be ignored)", r.count())
	}
	s, ok := r.get("echo")
	if !ok {
		t.Fatal("get(\"echo\") not found")
	}
	if s.Exec != "echo hi" {
		t.Errorf("loaded Exec = %q, want \"echo hi\"", s.Exec)
	}
}

func TestRegistryMissingDirIsNotError(t *testing.T) {
	r := newRegistry()
	if err := r.loadFromDisk(filepath.Join(t.TempDir(), "does-not-exist")); err != nil {
		t.Errorf("missing slots dir should be a no-op (nil err), got %v", err)
	}
	if r.count() != 0 {
		t.Errorf("count = %d, want 0", r.count())
	}
}
