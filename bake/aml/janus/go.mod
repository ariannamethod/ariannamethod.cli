module github.com/ariannamethod/ariannamethod.ai/janus

go 1.21

require github.com/ariannamethod/yent v0.0.0

// janus links the external yent organism (github.com/ariannamethod/yent).
// The minimal CLI build (`make all`) does not build janus; for the optional
// `make janus` target, provide yent via a go.work file or a published module
// version — no local/sibling checkout path here.
