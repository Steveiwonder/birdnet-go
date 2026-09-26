//go:build windows

package cudabenchmark

import "time"

// processCPUTime is not measured on Windows; the benchmark reports CPU use as
// unavailable there.
func processCPUTime() (time.Duration, bool) {
	return 0, false
}
