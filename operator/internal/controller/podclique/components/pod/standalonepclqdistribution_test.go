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

package pod

import (
	"context"
	"errors"
	"testing"

	apicommon "github.com/ai-dynamo/grove/operator/api/common"
	configv1alpha1 "github.com/ai-dynamo/grove/operator/api/config/v1alpha1"
	grovecorev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/grove/operator/internal/controller/common/component"
	groveerr "github.com/ai-dynamo/grove/operator/internal/errors"
	"github.com/ai-dynamo/grove/operator/internal/expect"
	testutils "github.com/ai-dynamo/grove/operator/test/utils"

	"github.com/go-logr/logr"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// TestBuildDesiredCountByPodGang verifies the desired per-anchor pod counts for a standalone
// PodClique are read from the PodGangMap anchor entries, skipping entries without this clique or with
// a zero count.
func TestBuildDesiredCountByPodGang(t *testing.T) {
	anchor0 := podGangNameForEpoch(testAnchor0Epoch)
	anchor1 := podGangNameForEpoch(testAnchor1Epoch)
	tests := []struct {
		name     string
		entries  []grovecorev1alpha1.PodGangEntry
		expected map[string]int32
	}{
		{
			name:     "single anchor carrying the clique",
			entries:  []grovecorev1alpha1.PodGangEntry{anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 3})},
			expected: map[string]int32{anchor0: 3},
		},
		{
			name: "multiple anchors carrying the clique",
			entries: []grovecorev1alpha1.PodGangEntry{
				anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 3}),
				anchorEntryWithCliques(testAnchor1Epoch, 1, map[string]int32{testCliqueName: 2}),
			},
			expected: map[string]int32{anchor0: 3, anchor1: 2},
		},
		{
			name:     "entry without the clique is skipped",
			entries:  []grovecorev1alpha1.PodGangEntry{anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{"other": 3})},
			expected: map[string]int32{},
		},
		{
			name:     "entry with a zero count is skipped",
			entries:  []grovecorev1alpha1.PodGangEntry{anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 0})},
			expected: map[string]int32{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ss := &syncSnapshot{pcs: pcs(), pcsReplicaIndex: testPCSReplicaIndex, cliqueName: testCliqueName, pgm: pgmWithEntries(tc.entries...)}
			actual := buildDesiredCountByPodGang(ss)
			assert.Equal(t, tc.expected, actual)
		})
	}

	t.Run("nil PodGangMap yields an empty map", func(t *testing.T) {
		ss := &syncSnapshot{pcs: pcs(), pcsReplicaIndex: testPCSReplicaIndex, cliqueName: testCliqueName, pgm: nil}
		assert.Empty(t, buildDesiredCountByPodGang(ss))
	})
}

// TestComputeCountDeltaByPodGang verifies the per-PodGang delta is desired minus live, positive for
// creation and negative for deletion, omitting matches and covering PodGangs present only in the live
// set.
func TestComputeCountDeltaByPodGang(t *testing.T) {
	tests := []struct {
		name     string
		desired  map[string]int32
		live     map[string]int32
		expected map[string]int32
	}{
		{"deficit needs creation", map[string]int32{"pg-a": 3}, map[string]int32{"pg-a": 1}, map[string]int32{"pg-a": 2}},
		{"excess needs deletion", map[string]int32{"pg-a": 1}, map[string]int32{"pg-a": 3}, map[string]int32{"pg-a": -2}},
		{"matching count is omitted", map[string]int32{"pg-a": 2}, map[string]int32{"pg-a": 2}, map[string]int32{}},
		{"live PodGang absent from desired is fully deleted", map[string]int32{}, map[string]int32{"pg-a": 2}, map[string]int32{"pg-a": -2}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := computeCountDeltaByPodGang(tc.desired, tc.live)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

// TestReconcileLabellessPods verifies labelless standalone pods are relabeled to the single anchor
// PodGang, or deleted when multiple anchors make the target ambiguous, and that a clean set is a
// no-op.
func TestReconcileLabellessPods(t *testing.T) {
	anchor0 := podGangNameForEpoch(testAnchor0Epoch)
	anchor1 := podGangNameForEpoch(testAnchor1Epoch)

	t.Run("single anchor relabels the labelless pod in place", func(t *testing.T) {
		pod := testutils.NewPodBuilder("pod-a", testNamespace).Build()
		cl := testutils.NewTestClientBuilder().WithObjects(pod).Build()
		r := &_resource{client: cl}

		repaired, err := r.reconcileLabellessPods(context.Background(), logr.Discard(), map[string]int32{anchor0: 3}, []*corev1.Pod{pod})
		require.NoError(t, err)
		assert.True(t, repaired)

		updated := &corev1.Pod{}
		require.NoError(t, cl.Get(context.Background(), client.ObjectKeyFromObject(pod), updated))
		assert.Equal(t, anchor0, updated.Labels[apicommon.LabelPodGang])
	})

	t.Run("multiple anchors delete the labelless pod for recreation", func(t *testing.T) {
		pod := testutils.NewPodBuilder("pod-a", testNamespace).Build()
		cl := testutils.NewTestClientBuilder().WithObjects(pod).Build()
		r := &_resource{client: cl}

		repaired, err := r.reconcileLabellessPods(context.Background(), logr.Discard(), map[string]int32{anchor0: 3, anchor1: 2}, []*corev1.Pod{pod})
		require.NoError(t, err)
		assert.True(t, repaired)

		err = cl.Get(context.Background(), client.ObjectKeyFromObject(pod), &corev1.Pod{})
		assert.True(t, apierrors.IsNotFound(err), "labelless pod should be deleted")
	})

	t.Run("no labelless pods is a no-op", func(t *testing.T) {
		pod := gatedPod("pod-a", anchor0)
		cl := testutils.NewTestClientBuilder().WithObjects(pod).Build()
		r := &_resource{client: cl}

		repaired, err := r.reconcileLabellessPods(context.Background(), logr.Discard(), map[string]int32{anchor0: 3}, []*corev1.Pod{pod})
		require.NoError(t, err)
		assert.False(t, repaired)
	})

	t.Run("single anchor propagates a relabel failure", func(t *testing.T) {
		pod := testutils.NewPodBuilder("pod-a", testNamespace).Build()
		cl := testutils.NewTestClientBuilder().WithObjects(pod).
			RecordErrorForObjects(testutils.ClientMethodPatch, apierrors.NewInternalError(errors.New("boom")), client.ObjectKeyFromObject(pod)).
			Build()
		r := &_resource{client: cl}

		repaired, err := r.reconcileLabellessPods(context.Background(), logr.Discard(), map[string]int32{anchor0: 3}, []*corev1.Pod{pod})
		require.Error(t, err)
		assert.False(t, repaired)
	})

	t.Run("multiple anchors propagate a delete failure", func(t *testing.T) {
		pod := testutils.NewPodBuilder("pod-a", testNamespace).Build()
		cl := testutils.NewTestClientBuilder().WithObjects(pod).
			RecordErrorForObjects(testutils.ClientMethodDelete, apierrors.NewInternalError(errors.New("boom")), client.ObjectKeyFromObject(pod)).
			Build()
		r := &_resource{client: cl}

		repaired, err := r.reconcileLabellessPods(context.Background(), logr.Discard(), map[string]int32{anchor0: 3, anchor1: 2}, []*corev1.Pod{pod})
		require.Error(t, err)
		assert.False(t, repaired)
	})
}

// TestReconcileStandalonePCLQDistributionEarlyReturn verifies the distribution requeues before
// reconciling pods when the PodGangMap is not ready, a labelless pod was repaired, or the PodGangMap
// has not yet absorbed a scale.
func TestReconcileStandalonePCLQDistributionEarlyReturn(t *testing.T) {
	labellessPod := testutils.NewPodBuilder("pod-a", testNamespace).Build()

	tests := []struct {
		name    string
		ss      *syncSnapshot
		objects []client.Object
	}{
		{
			name: "requeues when the PodGangMap has no anchor entry for the clique",
			ss:   &syncSnapshot{pcs: pcs(), pcsReplicaIndex: testPCSReplicaIndex, pclq: pclqWithReplicas(1), cliqueName: testCliqueName, pgm: pgmWithEntries()},
		},
		{
			name: "requeues after repairing a labelless pod",
			ss: &syncSnapshot{
				pcs: pcs(), pcsReplicaIndex: testPCSReplicaIndex, pclq: pclqWithReplicas(1), cliqueName: testCliqueName,
				pgm:              pgmWithEntries(anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 1})),
				existingPCLQPods: []*corev1.Pod{labellessPod},
			},
			objects: []client.Object{labellessPod},
		},
		{
			name: "requeues while waiting for the PodGangMap to absorb a scale",
			ss: &syncSnapshot{
				pcs: pcs(), pcsReplicaIndex: testPCSReplicaIndex, pclq: pclqWithReplicas(10), cliqueName: testCliqueName,
				pgm: pgmWithEntries(
					anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 3}),
					anchorEntryWithCliques(testAnchor1Epoch, 1, map[string]int32{testCliqueName: 3}),
				),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := testutils.NewTestClientBuilder()
			for _, obj := range tc.objects {
				builder = builder.WithObjects(obj)
			}
			r := &_resource{client: builder.Build()}

			err := r.reconcileStandalonePCLQDistribution(context.Background(), logr.Discard(), tc.ss)
			testutils.AssertGroveError(t, &groveerr.GroveError{Code: groveerr.ErrCodeRequeueAfter, Operation: component.OperationSync}, err)
		})
	}
}

// TestReconcileStandalonePCLQDistributionCreateAndDelete verifies the full non-early-return path
// creates the deficit pods for the anchor PodGang and deletes the excess pods, driving the real
// pod-creation build path through the default scheduler backend.
func TestReconcileStandalonePCLQDistributionCreateAndDelete(t *testing.T) {
	anchor0 := podGangNameForEpoch(testAnchor0Epoch)
	registry := testutils.NewDefaultFakeRegistry()
	podSpec := corev1.PodSpec{
		SchedulerName: string(configv1alpha1.SchedulerNameKube),
		Containers:    []corev1.Container{{Name: "worker", Image: "worker"}},
	}
	testPCS := testutils.NewPodCliqueSetBuilder(testPCSName, testNamespace, "uid").
		WithPodCliqueTemplateSpec(testutils.NewPodCliqueTemplateSpecBuilder(testCliqueName).WithPodSpec(podSpec).Build()).
		Build()
	// The PodClique name must be a real FQN so buildResource can resolve the PCS replica index from it.
	newPCLQ := func(replicas int32) *grovecorev1alpha1.PodClique {
		pclq := testutils.NewPodCliqueBuilder(testPCSName, "uid", testCliqueName, testNamespace, testPCSReplicaIndex).WithReplicas(replicas).Build()
		pclq.Spec.PodSpec = *podSpec.DeepCopy()
		return pclq
	}

	t.Run("creates the deficit pods for the anchor PodGang", func(t *testing.T) {
		pclq := newPCLQ(3)
		cl := testutils.NewTestClientBuilder().WithObjects(pclq).Build()
		r := _resource{client: cl, scheme: cl.Scheme(), schedRegistry: registry, eventRecorder: record.NewFakeRecorder(64), expectationsStore: expect.NewExpectationsStore()}
		ss := &syncSnapshot{
			pcs: testPCS, pclq: pclq, pcsReplicaIndex: testPCSReplicaIndex, cliqueName: testCliqueName,
			pgm:                      pgmWithEntries(anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 3})),
			pclqExpectationsStoreKey: testNamespace + "/" + pclq.Name,
		}

		err := r.reconcileStandalonePCLQDistribution(context.Background(), logr.Discard(), ss)
		require.NoError(t, err)

		pods := listPodsForPodGang(t, cl, anchor0)
		assert.Len(t, pods, 3)
	})

	t.Run("deletes the excess pods for the anchor PodGang", func(t *testing.T) {
		pclq := newPCLQ(1)
		excess := []*corev1.Pod{gatedPod("pod-a", anchor0), gatedPod("pod-b", anchor0), gatedPod("pod-c", anchor0)}
		cl := testutils.NewTestClientBuilder().WithObjects(pclq, excess[0], excess[1], excess[2]).Build()
		r := _resource{client: cl, scheme: cl.Scheme(), schedRegistry: registry, eventRecorder: record.NewFakeRecorder(64), expectationsStore: expect.NewExpectationsStore()}
		ss := &syncSnapshot{
			pcs: testPCS, pclq: pclq, pcsReplicaIndex: testPCSReplicaIndex, cliqueName: testCliqueName,
			pgm:                      pgmWithEntries(anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 1})),
			existingPCLQPods:         excess,
			pclqExpectationsStoreKey: testNamespace + "/" + pclq.Name,
		}

		err := r.reconcileStandalonePCLQDistribution(context.Background(), logr.Discard(), ss)
		require.NoError(t, err)

		pods := listPodsForPodGang(t, cl, anchor0)
		assert.Len(t, pods, 1)
	})

	t.Run("propagates a pod deletion failure", func(t *testing.T) {
		pclq := newPCLQ(1)
		excess := []*corev1.Pod{gatedPod("pod-a", anchor0), gatedPod("pod-b", anchor0)}
		cl := testutils.NewTestClientBuilder().WithObjects(pclq, excess[0], excess[1]).
			RecordErrorForObjects(testutils.ClientMethodDelete, apierrors.NewInternalError(errors.New("boom")), client.ObjectKeyFromObject(excess[0]), client.ObjectKeyFromObject(excess[1])).
			Build()
		r := _resource{client: cl, scheme: cl.Scheme(), schedRegistry: registry, eventRecorder: record.NewFakeRecorder(64), expectationsStore: expect.NewExpectationsStore()}
		ss := &syncSnapshot{
			pcs: testPCS, pclq: pclq, pcsReplicaIndex: testPCSReplicaIndex, cliqueName: testCliqueName,
			pgm:                      pgmWithEntries(anchorEntryWithCliques(testAnchor0Epoch, 0, map[string]int32{testCliqueName: 1})),
			existingPCLQPods:         excess,
			pclqExpectationsStoreKey: testNamespace + "/" + pclq.Name,
		}

		err := r.reconcileStandalonePCLQDistribution(context.Background(), logr.Discard(), ss)
		require.Error(t, err)
	})
}

// TestPartitionPodsByTermination verifies pods are split into the non-terminating slice and the
// terminating and non-terminating UID lists in a single pass.
func TestPartitionPodsByTermination(t *testing.T) {
	alive := newPod("alive", "uid-alive", false)
	dying := newPod("dying", "uid-dying", true)

	tests := []struct {
		name                    string
		pods                    []*corev1.Pod
		expectedNonTerminating  []types.UID
		expectedTerminatingUIDs []types.UID
	}{
		{"all pods non-terminating", []*corev1.Pod{alive}, []types.UID{"uid-alive"}, []types.UID{}},
		{"all pods terminating", []*corev1.Pod{dying}, []types.UID{}, []types.UID{"uid-dying"}},
		{"mixed pods split by termination", []*corev1.Pod{alive, dying}, []types.UID{"uid-alive"}, []types.UID{"uid-dying"}},
		{"no pods yields empty lists", nil, []types.UID{}, []types.UID{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nonTerminatingPods, terminatingPodUIDs, nonTerminatingPodUIDs := partitionPodsByTermination(tc.pods)
			assert.Equal(t, tc.expectedNonTerminating, nonTerminatingPodUIDs)
			assert.Equal(t, tc.expectedTerminatingUIDs, terminatingPodUIDs)
			assert.Equal(t, tc.expectedNonTerminating, uidsOf(nonTerminatingPods))
		})
	}
}

// TestGroupPodsByPodGang verifies pods are grouped by their grove.io/podgang label and labelless
// pods are skipped.
func TestGroupPodsByPodGang(t *testing.T) {
	t.Run("groups pods by their PodGang label", func(t *testing.T) {
		podsByPodGang := groupPodsByPodGang([]*corev1.Pod{gatedPod("pod-a", "pg-a"), gatedPod("pod-b", "pg-a"), gatedPod("pod-c", "pg-b")})
		assert.ElementsMatch(t, []string{"pod-a", "pod-b"}, podNamesOf(podsByPodGang["pg-a"]))
		assert.ElementsMatch(t, []string{"pod-c"}, podNamesOf(podsByPodGang["pg-b"]))
	})

	t.Run("skips a pod without the PodGang label", func(t *testing.T) {
		labelless := testutils.NewPodBuilder("pod-a", testNamespace).Build()
		podsByPodGang := groupPodsByPodGang([]*corev1.Pod{labelless})
		assert.Empty(t, podsByPodGang)
	})

	t.Run("returns an empty map for no pods", func(t *testing.T) {
		assert.Empty(t, groupPodsByPodGang(nil))
	})
}

// TestLiveCountByPodGang verifies grouped pods project to a per-PodGang live count.
func TestLiveCountByPodGang(t *testing.T) {
	t.Run("counts the pods in each PodGang group", func(t *testing.T) {
		podsByPodGang := map[string][]*corev1.Pod{
			"pg-a": {gatedPod("pod-a", "pg-a"), gatedPod("pod-b", "pg-a")},
			"pg-b": {gatedPod("pod-c", "pg-b")},
		}
		assert.Equal(t, map[string]int32{"pg-a": 2, "pg-b": 1}, liveCountByPodGang(podsByPodGang))
	})

	t.Run("returns an empty map for no groups", func(t *testing.T) {
		assert.Empty(t, liveCountByPodGang(nil))
	})
}

// TestSumCounts verifies the total across all PodGang counts.
func TestSumCounts(t *testing.T) {
	tests := []struct {
		name           string
		countByPodGang map[string]int32
		expected       int32
	}{
		{"sums multiple counts", map[string]int32{"pg-a": 3, "pg-b": 2}, 5},
		{"single count returns that count", map[string]int32{"pg-a": 4}, 4},
		{"no counts sum to zero", nil, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, sumCounts(tc.countByPodGang))
		})
	}
}

// TestBuildPerPodGangDeletionTasks verifies one deletion task is built per excess pod, that the
// DeletionSorter prefers the Pending pod, and that non-negative deltas and empty groups build no
// tasks.
func TestBuildPerPodGangDeletionTasks(t *testing.T) {
	r := &_resource{expectationsStore: expect.NewExpectationsStore()}
	ss := &syncSnapshot{pcs: pcs(), pclq: pclqWithReplicas(1), cliqueName: testCliqueName}

	t.Run("builds one task per excess pod picking the Pending pod first", func(t *testing.T) {
		pending := testutils.NewPodBuilder("pod-pending", testNamespace).WithPhase(corev1.PodPending).Build()
		running := testutils.NewPodBuilder("pod-running", testNamespace).WithPhase(corev1.PodRunning).Build()
		livePodsByPodGang := map[string][]*corev1.Pod{"pg-a": {running, pending}}

		tasks := r.buildPerPodGangDeletionTasks(logr.Discard(), ss, map[string]int32{"pg-a": -1}, []string{"pg-a"}, livePodsByPodGang)

		require.Len(t, tasks, 1)
		assert.Contains(t, tasks[0].Name, "pod-pending")
	})

	t.Run("skips a PodGang whose delta is non-negative", func(t *testing.T) {
		livePodsByPodGang := map[string][]*corev1.Pod{"pg-a": {gatedPod("pod-a", "pg-a")}}
		tasks := r.buildPerPodGangDeletionTasks(logr.Discard(), ss, map[string]int32{"pg-a": 0}, []string{"pg-a"}, livePodsByPodGang)
		assert.Empty(t, tasks)
	})

	t.Run("skips a PodGang with no live pods", func(t *testing.T) {
		tasks := r.buildPerPodGangDeletionTasks(logr.Discard(), ss, map[string]int32{"pg-a": -2}, []string{"pg-a"}, map[string][]*corev1.Pod{})
		assert.Empty(t, tasks)
	})
}

// TestBuildPerPodGangCreationTasks verifies one creation task is built per deficit pod across
// PodGangs, and no tasks when nothing is to be created.
func TestBuildPerPodGangCreationTasks(t *testing.T) {
	r := &_resource{expectationsStore: expect.NewExpectationsStore()}
	ss := &syncSnapshot{pcs: pcs(), pclq: pclqWithReplicas(1), cliqueName: testCliqueName}

	t.Run("builds one task per deficit pod across PodGangs", func(t *testing.T) {
		tasks, err := r.buildPerPodGangCreationTasks(logr.Discard(), ss, map[string]int32{"pg-a": 2, "pg-b": 1}, []string{"pg-a", "pg-b"})
		require.NoError(t, err)
		assert.Len(t, tasks, 3)
	})

	t.Run("builds no tasks when there is no deficit", func(t *testing.T) {
		tasks, err := r.buildPerPodGangCreationTasks(logr.Discard(), ss, map[string]int32{"pg-a": -1}, []string{"pg-a"})
		require.NoError(t, err)
		assert.Nil(t, tasks)
	})
}

// anchorEntryWithCliques builds an Anchor PodGangEntry carrying the given standalone PodClique counts.
func anchorEntryWithCliques(epoch string, anchorIndex int32, cliques map[string]int32) grovecorev1alpha1.PodGangEntry {
	return testutils.NewPodGangEntryBuilder("hash", epoch).
		WithRole(grovecorev1alpha1.PodGangEntryRoleAnchor).
		WithAnchorIndex(anchorIndex).
		WithPodCliques(cliques).
		Build()
}

// pclqWithReplicas returns a standalone PodClique named after testCliqueName with the given replicas.
func pclqWithReplicas(replicas int32) *grovecorev1alpha1.PodClique {
	return &grovecorev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: testCliqueName, Namespace: testNamespace},
		Spec:       grovecorev1alpha1.PodCliqueSpec{Replicas: replicas},
	}
}

// listPodsForPodGang lists pods in the test namespace carrying the given PodGang label.
func listPodsForPodGang(t *testing.T, cl client.Client, podGangName string) []corev1.Pod {
	t.Helper()
	podList := &corev1.PodList{}
	require.NoError(t, cl.List(context.Background(), podList, client.InNamespace(testNamespace)))
	matching := make([]corev1.Pod, 0, len(podList.Items))
	for i := range podList.Items {
		if podList.Items[i].Labels[apicommon.LabelPodGang] == podGangName {
			matching = append(matching, podList.Items[i])
		}
	}
	return matching
}

// newPod returns a pod carrying the given UID, optionally marked for termination.
func newPod(name string, uid types.UID, terminating bool) *corev1.Pod {
	builder := testutils.NewPodBuilder(name, testNamespace)
	if terminating {
		builder = builder.MarkForTermination()
	}
	pod := builder.Build()
	pod.UID = uid
	return pod
}

// uidsOf returns the UIDs of the given pods.
func uidsOf(pods []*corev1.Pod) []types.UID {
	return lo.Map(pods, func(pod *corev1.Pod, _ int) types.UID {
		return pod.GetUID()
	})
}

// podNamesOf returns the names of the given pods.
func podNamesOf(pods []*corev1.Pod) []string {
	return lo.Map(pods, func(pod *corev1.Pod, _ int) string {
		return pod.GetName()
	})
}
