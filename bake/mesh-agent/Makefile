# mesh-agent — cross-compile + dev helpers
#
# Targets:
#   make            build native binary (current GOOS/GOARCH)
#   make all        build all release binaries into dist/
#   make test       go test ./...
#   make run        run native binary as daemon on this host
#   make clean      remove build artifacts

GO_BUILD_FLAGS = -trimpath -ldflags=-s\ -w
PKG     := github.com/ariannamethod/mesh-agent

.PHONY: all native test run clean dist-darwin-arm64 dist-darwin-amd64 dist-linux-amd64 dist-linux-arm64 dist-android-arm64

native:
	go build $(GO_BUILD_FLAGS) -o mesh-agent .

all: dist-darwin-arm64 dist-darwin-amd64 dist-linux-amd64 dist-linux-arm64 dist-android-arm64
	@ls -la dist/

dist:
	@mkdir -p dist

dist-darwin-arm64: dist
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 \
	  go build $(GO_BUILD_FLAGS) -o dist/mesh-agent-darwin-arm64 .

dist-darwin-amd64: dist
	GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 \
	  go build $(GO_BUILD_FLAGS) -o dist/mesh-agent-darwin-amd64 .

dist-linux-amd64: dist
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
	  go build $(GO_BUILD_FLAGS) -o dist/mesh-agent-linux-amd64 .

dist-linux-arm64: dist
	GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
	  go build $(GO_BUILD_FLAGS) -o dist/mesh-agent-linux-arm64 .

dist-android-arm64: dist
	GOOS=android GOARCH=arm64 CGO_ENABLED=0 \
	  go build $(GO_BUILD_FLAGS) -o dist/mesh-agent-android-arm64 .

test:
	go test ./...

run: native
	./mesh-agent serve

clean:
	rm -rf mesh-agent dist/
