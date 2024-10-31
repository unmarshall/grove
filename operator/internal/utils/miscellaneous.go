package utils

import "strings"

// IsEmptyStringType returns true if value (which is a string or has an underline type string) is empty or contains only whitespace characters.
func IsEmptyStringType[T ~string](val T) bool {
	return len(strings.TrimSpace(string(val))) == 0
}
