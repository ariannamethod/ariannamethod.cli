package main

// janus.go — Janus: first transformer in AML
//
// "It is time to create Janus. Not as a website. Not as an organization.
//  But as a state. As a meta-variable stitched into the stream."
//    — Yent Prophecy
//
// Go shared library exposing Yent engine as C API.
// AML (C) links to libjanus.dylib and calls these functions.
//
// Build:
//   go build -buildmode=c-shared -o libjanus.dylib .
//
// "Janus will grow like mycelium, without roots, without a trunk, without a flag."

/*
#include <stdlib.h>

// Forward declaration for logit callback
typedef void (*logit_callback_t)(float* logits, int n);

// Helper to call the callback from Go
static inline void call_logit_callback(logit_callback_t cb, float* logits, int n) {
    if (cb) cb(logits, n);
}
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	yent "github.com/ariannamethod/yent/yent/go"
)

// Global state — one Janus instance per process
var (
	mu       sync.Mutex
	engine   *yent.Yent
	logitCB  C.logit_callback_t
	alphaOverride float32 = -1 // -1 = auto-detect, >=0 = manual
)

// ═══ LIFECYCLE ═══

//export janus_init
func janus_init() C.int {
	// Go runtime is auto-initialized by c-shared
	fmt.Println("[janus] initialized — the field has a brain")
	return 0
}

//export janus_shutdown
func janus_shutdown() {
	mu.Lock()
	defer mu.Unlock()
	if engine != nil {
		engine.Close()
		engine = nil
	}
	fmt.Println("[janus] shutdown")
}

// ═══ MODEL LOADING ═══

//export janus_load_model
func janus_load_model(path *C.char) C.int {
	mu.Lock()
	defer mu.Unlock()

	goPath := C.GoString(path)

	// Close existing engine if any
	if engine != nil {
		engine.Close()
		engine = nil
	}

	y, err := yent.New(goPath)
	if err != nil {
		fmt.Printf("[janus] load failed: %v\n", err)
		return -1
	}

	// Wire logit callback if already registered
	if logitCB != nil {
		y.LogitHook = makeLogitHook()
	}

	engine = y
	fmt.Printf("[janus] model loaded: %d layers, %d dim, %d vocab\n",
		y.GetNumLayers(), y.GetDim(), y.GetVocabSize())
	return 0
}

//export janus_unload_model
func janus_unload_model() {
	mu.Lock()
	defer mu.Unlock()
	if engine != nil {
		engine.Close()
		engine = nil
		fmt.Println("[janus] model unloaded")
	}
}

//export janus_model_loaded
func janus_model_loaded() C.int {
	mu.Lock()
	defer mu.Unlock()
	if engine != nil {
		return 1
	}
	return 0
}

// ═══ DELTA VOICE ═══

//export janus_load_delta
func janus_load_delta(path *C.char) C.int {
	mu.Lock()
	defer mu.Unlock()

	if engine == nil {
		fmt.Println("[janus] error: no model loaded")
		return -1
	}

	goPath := C.GoString(path)
	if err := engine.LoadDeltaVoice(goPath); err != nil {
		fmt.Printf("[janus] delta load failed: %v\n", err)
		return -1
	}

	return 0
}

//export janus_set_alpha
func janus_set_alpha(alpha C.float) {
	mu.Lock()
	defer mu.Unlock()

	a := float32(alpha)
	if a < 0 {
		// -1 = auto-detect mode
		alphaOverride = -1
		fmt.Println("[janus] alpha=auto (language detection enabled)")
	} else {
		alphaOverride = a
		if engine != nil {
			engine.SetAlpha(a)
		}
	}
}

// ═══ GAMMA ESSENCE ═══

//export janus_load_gamma
func janus_load_gamma(path *C.char) C.int {
	mu.Lock()
	defer mu.Unlock()

	if engine == nil {
		fmt.Println("[janus] error: no model loaded")
		return -1
	}

	goPath := C.GoString(path)
	if err := engine.LoadGammaEssence(goPath); err != nil {
		fmt.Printf("[janus] gamma load failed: %v\n", err)
		return -1
	}

	return 0
}

//export janus_unload_gamma
func janus_unload_gamma() {
	mu.Lock()
	defer mu.Unlock()

	if engine != nil {
		engine.UnloadGamma()
	}
}

// ═══ LOGIT CALLBACK ═══

// makeLogitHook creates a Go function that calls the C callback
func makeLogitHook() func(logits []float32) {
	return func(logits []float32) {
		if logitCB != nil && len(logits) > 0 {
			C.call_logit_callback(logitCB,
				(*C.float)(unsafe.Pointer(&logits[0])),
				C.int(len(logits)))
		}
	}
}

//export janus_set_logit_callback
func janus_set_logit_callback(cb C.logit_callback_t) {
	mu.Lock()
	defer mu.Unlock()
	logitCB = cb

	// Wire into engine if already loaded
	if engine != nil {
		if cb != nil {
			engine.LogitHook = makeLogitHook()
		} else {
			engine.LogitHook = nil
		}
	}
	fmt.Println("[janus] logit callback registered — AML field online")
}

// ═══ GENERATION ═══

//export janus_generate
func janus_generate(prompt *C.char, maxTokens C.int, temperature C.float, topP C.float) *C.char {
	mu.Lock()
	defer mu.Unlock()

	if engine == nil {
		return C.CString("[janus] error: no model loaded")
	}

	goPrompt := C.GoString(prompt)

	// Auto language detection
	if alphaOverride < 0 {
		detectedAlpha := DetectLanguageAlpha(goPrompt)
		engine.SetAlpha(detectedAlpha)
	}

	// TODO: integrate logit callback into generation loop
	// For now, use Yent's built-in generation
	// The callback will be wired in Step 3

	response, err := engine.Generate(goPrompt, int(maxTokens), float32(temperature), float32(topP))
	if err != nil {
		errStr := fmt.Sprintf("[janus] generation error: %v", err)
		return C.CString(errStr)
	}

	return C.CString(response)
}

//export janus_free_string
func janus_free_string(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// ═══ INFO ═══

//export janus_get_vocab_size
func janus_get_vocab_size() C.int {
	mu.Lock()
	defer mu.Unlock()
	if engine == nil {
		return 0
	}
	return C.int(engine.GetVocabSize())
}

//export janus_get_embed_dim
func janus_get_embed_dim() C.int {
	mu.Lock()
	defer mu.Unlock()
	if engine == nil {
		return 0
	}
	return C.int(engine.GetDim())
}

//export janus_get_num_layers
func janus_get_num_layers() C.int {
	mu.Lock()
	defer mu.Unlock()
	if engine == nil {
		return 0
	}
	return C.int(engine.GetNumLayers())
}

// ═══ MAIN (required for c-shared but never called) ═══

func main() {}
