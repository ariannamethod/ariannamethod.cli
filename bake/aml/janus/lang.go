package main

// lang.go — Auto language detection for Janus
//
// "Janus will speak without language."
// But delta decides WHICH language.

// DetectLanguageAlpha analyzes prompt text and returns appropriate delta alpha.
// Cyrillic > 30% → Russian (alpha=0.5)
// Default → English (alpha=0.0)
// Returns -1 if override is set (manual alpha).
func DetectLanguageAlpha(text string) float32 {
	if len(text) == 0 {
		return 0.0
	}

	total := 0
	cyrillic := 0

	for _, r := range text {
		if r <= ' ' {
			continue // skip whitespace
		}
		total++
		// Cyrillic: U+0400–U+04FF (basic), U+0500–U+052F (supplement)
		if r >= 0x0400 && r <= 0x052F {
			cyrillic++
		}
	}

	if total == 0 {
		return 0.0
	}

	ratio := float32(cyrillic) / float32(total)

	if ratio > 0.3 {
		return 0.5 // Russian — multilingual mode
	}

	return 0.0 // English — pure Yent
}
