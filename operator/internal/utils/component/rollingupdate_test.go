// Copyright 2026 The Grove Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package component

import (
	"testing"

	grovecorev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"

	"github.com/stretchr/testify/assert"
	"k8s.io/utils/ptr"
)

func TestEffectiveMaxUnavailable(t *testing.T) {
	testCases := []struct {
		description   string
		rollingUpdate *grovecorev1alpha1.RollingUpdateConfiguration
		want          int32
	}{
		{
			description:   "nil rollingUpdate returns the RollingRecreate default",
			rollingUpdate: nil,
			want:          DefaultRollingRecreateMaxUnavailable,
		},
		{
			description:   "nil MaxUnavailable returns the RollingRecreate default",
			rollingUpdate: &grovecorev1alpha1.RollingUpdateConfiguration{},
			want:          DefaultRollingRecreateMaxUnavailable,
		},
		{
			description:   "configured MaxUnavailable is returned",
			rollingUpdate: &grovecorev1alpha1.RollingUpdateConfiguration{MaxUnavailable: ptr.To[int32](3)},
			want:          3,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			assert.Equal(t, tc.want, EffectiveMaxUnavailable(tc.rollingUpdate))
		})
	}
}
