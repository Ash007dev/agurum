package main

import (
	"bytes"
	"encoding/json"
)

func ValidateEvent(msg json.RawMessage) bool {
	dec := json.NewDecoder(bytes.NewReader(msg))

	t, err := dec.Token()
	if err != nil || t != json.Delim('{') {
		return false
	}

	hasTs := false
	hasKind := false

	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return false
		}

		keyStr, ok := t.(string)
		if !ok {
			return false
		}

		if keyStr == "ts" {
			hasTs = true
		} else if keyStr == "kind" {
			hasKind = true
		}

		if err := skipValue(dec); err != nil {
			return false
		}

		if hasTs && hasKind {
			return true
		}
	}
	return false
}

func skipValue(dec *json.Decoder) error {
	t, err := dec.Token()
	if err != nil {
		return err
	}
	if delim, ok := t.(json.Delim); ok {
		if delim == '{' || delim == '[' {
			depth := 1
			for {
				tok, err := dec.Token()
				if err != nil {
					return err
				}
				if d, ok := tok.(json.Delim); ok {
					if d == '{' || d == '[' {
						depth++
					} else if d == '}' || d == ']' {
						depth--
						if depth == 0 {
							return nil
						}
					}
				}
			}
		}
	}
	return nil
}
