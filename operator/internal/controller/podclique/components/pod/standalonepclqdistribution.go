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
	"fmt"
	"slices"
	"sort"

	apicommon "github.com/ai-dynamo/grove/operator/api/common"
	"github.com/ai-dynamo/grove/operator/internal/controller/common/component"
	groveerr "github.com/ai-dynamo/grove/operator/internal/errors"
	"github.com/ai-dynamo/grove/operator/internal/index"
	"github.com/ai-dynamo/grove/operator/internal/utils"
	k8sutils "github.com/ai-dynamo/grove/operator/internal/utils/kubernetes"

	"github.com/go-logr/logr"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// reconcileStandalonePCLQDistribution reconciles a standalone PodClique's pods across the anchor
// PodGangs the PodGangMap assigns them to. It first repairs any non-terminating pod missing the
// grove.io/podgang label, then reconciles per PodGang against the PodGangMap counts.
func (r _resource) reconcileStandalonePCLQDistribution(ctx context.Context, logger logr.Logger, ss *syncSnapshot) error {
	desiredCountByPodGang := buildDesiredCountByPodGang(ss)
	if len(desiredCountByPodGang) == 0 {
		return groveerr.New(groveerr.ErrCodeRequeueAfter, component.OperationSync,
			fmt.Sprintf("PodGangMap has no anchor entry for standalone PodClique %v yet, re-queueing", client.ObjectKeyFromObject(ss.pclq)))
	}

	// Partition the PodClique's pods once. All downstream steps consume these slices, so
	// IsResourceTerminating runs a single time per pod.
	nonTerminatingPods, terminatingPodUIDs, nonTerminatingPodUIDs := partitionPodsByTermination(ss.existingPCLQPods)

	// Repair labelless pods before computing deltas so the reconcile below sees a consistent label
	// state. Any repair requeues, and the next reconcile computes deltas on the settled labels.
	repaired, err := r.reconcileLabellessPods(ctx, logger, desiredCountByPodGang, nonTerminatingPods)
	if err != nil {
		return err
	}
	if repaired {
		return groveerr.New(groveerr.ErrCodeRequeueAfter, component.OperationSync,
			fmt.Sprintf("repaired pods missing the %s label for standalone PodClique %v, re-queueing", apicommon.LabelPodGang, client.ObjectKeyFromObject(ss.pclq)))
	}

	// With more than one anchor the PodGangMap decides which anchor a Spec.Replicas change lands on.
	// Wait until it has absorbed the change. A single anchor has nothing to decide.
	if len(desiredCountByPodGang) > 1 && sumCounts(desiredCountByPodGang) != ss.pclq.Spec.Replicas {
		return groveerr.New(groveerr.ErrCodeRequeueAfter, component.OperationSync,
			fmt.Sprintf("PodGangMap has not yet absorbed the replica change for standalone PodClique %v, re-queueing", client.ObjectKeyFromObject(ss.pclq)))
	}

	r.expectationsStore.SyncExpectations(ss.pclqExpectationsStoreKey, nonTerminatingPodUIDs, terminatingPodUIDs)

	livePodsByPodGang := groupPodsByPodGang(nonTerminatingPods)
	countDeltaByPodGang := computeCountDeltaByPodGang(desiredCountByPodGang, liveCountByPodGang(livePodsByPodGang))
	return r.applyCountDeltaByPodGang(ctx, logger, ss, countDeltaByPodGang, livePodsByPodGang)
}

// partitionPodsByTermination splits the pods into the non-terminating pods and the terminating and
// non-terminating UID lists in a single pass.
func partitionPodsByTermination(pods []*corev1.Pod) (nonTerminatingPods []*corev1.Pod, terminatingPodUIDs, nonTerminatingPodUIDs []types.UID) {
	nonTerminatingPods = make([]*corev1.Pod, 0, len(pods))
	terminatingPodUIDs = make([]types.UID, 0, len(pods))
	nonTerminatingPodUIDs = make([]types.UID, 0, len(pods))
	for _, pod := range pods {
		if k8sutils.IsResourceTerminating(pod.ObjectMeta) {
			terminatingPodUIDs = append(terminatingPodUIDs, pod.GetUID())
			continue
		}
		nonTerminatingPods = append(nonTerminatingPods, pod)
		nonTerminatingPodUIDs = append(nonTerminatingPodUIDs, pod.GetUID())
	}
	return
}

// reconcileLabellessPods repairs non-terminating pods that lack the grove.io/podgang label. Grove
// sets that label at pod creation, so a labelless pod is an inconsistent state. With a single anchor
// the pod can only belong to that anchor, so it is relabeled in place. With more than one anchor its
// target is ambiguous, and it may have been scheduled for a different gang, so it is deleted. It gets
// recreated with the correct label and scheduling gates on a later reconcile. It returns whether any
// pod was repaired.
func (r _resource) reconcileLabellessPods(ctx context.Context, logger logr.Logger, desiredCountByPodGang map[string]int32, nonTerminatingPods []*corev1.Pod) (bool, error) {
	labellessPods := lo.Filter(nonTerminatingPods, func(pod *corev1.Pod, _ int) bool {
		_, hasPodGangLabel := pod.Labels[apicommon.LabelPodGang]
		return !hasPodGangLabel
	})
	if len(labellessPods) == 0 {
		return false, nil
	}

	// There is only one anchor PodGang, in this case its safe to just re-label the Pod (in-place)
	// with the anchor PodGang.
	if len(desiredCountByPodGang) == 1 {
		targetPodGangName := lo.Keys(desiredCountByPodGang)[0]
		if err := r.relabelPodsToPodGang(ctx, logger, labellessPods, targetPodGangName); err != nil {
			return false, err
		}
		return true, nil
	}

	for _, pod := range labellessPods {
		if err := client.IgnoreNotFound(r.client.Delete(ctx, pod)); err != nil {
			return false, groveerr.WrapError(err, errCodeDeletePod, component.OperationSync,
				fmt.Sprintf("failed to delete pod %v missing the %s label", client.ObjectKeyFromObject(pod), apicommon.LabelPodGang))
		}
		logger.Info("Deleted pod missing the PodGang label under multiple anchors for recreation", "podObjectKey", client.ObjectKeyFromObject(pod))
	}
	return true, nil
}

// buildDesiredCountByPodGang returns the desired pod count per anchor PodGang for this standalone
// PodClique, read from the PodGangMap entries. Only anchor entries carry standalone PodClique counts.
// Entries without this PodClique, or with a zero count, are skipped.
func buildDesiredCountByPodGang(ss *syncSnapshot) map[string]int32 {
	desiredCountByPodGang := make(map[string]int32)
	if ss.pgm == nil {
		return desiredCountByPodGang
	}
	rnr := apicommon.ResourceNameReplica{Name: ss.pcs.Name, Replica: ss.pcsReplicaIndex}
	for _, entry := range ss.pgm.Spec.Entries {
		count, ok := entry.PodCliques[ss.cliqueName]
		if !ok || count == 0 {
			continue
		}
		desiredCountByPodGang[apicommon.GenerateAnchorPodGangName(rnr, entry.Epoch)] = count
	}
	return desiredCountByPodGang
}

// groupPodsByPodGang groups the given non-terminating pods by their grove.io/podgang label.
func groupPodsByPodGang(nonTerminatingPods []*corev1.Pod) map[string][]*corev1.Pod {
	podsByPodGang := make(map[string][]*corev1.Pod)
	for _, pod := range nonTerminatingPods {
		if podGangName, ok := pod.Labels[apicommon.LabelPodGang]; ok {
			podsByPodGang[podGangName] = append(podsByPodGang[podGangName], pod)
		}
	}
	return podsByPodGang
}

// liveCountByPodGang projects grouped pods to a per-PodGang live pod count.
func liveCountByPodGang(podsByPodGang map[string][]*corev1.Pod) map[string]int32 {
	countByPodGang := make(map[string]int32, len(podsByPodGang))
	for podGangName, pods := range podsByPodGang {
		countByPodGang[podGangName] = int32(len(pods))
	}
	return countByPodGang
}

// computeCountDeltaByPodGang returns desiredCount-liveCount for every PodGang in either map. A
// positive value is pods to create, a negative value is pods to delete. PodGangs that already match
// are omitted.
func computeCountDeltaByPodGang(desiredCountByPodGang, liveCountByPodGang map[string]int32) map[string]int32 {
	countDeltaByPodGang := make(map[string]int32)
	for podGangName, desiredCount := range desiredCountByPodGang {
		if delta := desiredCount - liveCountByPodGang[podGangName]; delta != 0 {
			countDeltaByPodGang[podGangName] = delta
		}
	}
	for podGangName, liveCount := range liveCountByPodGang {
		if _, ok := desiredCountByPodGang[podGangName]; !ok {
			countDeltaByPodGang[podGangName] = -liveCount
		}
	}
	return countDeltaByPodGang
}

// applyCountDeltaByPodGang creates the deficit and deletes the excess pods for each PodGang.
func (r _resource) applyCountDeltaByPodGang(ctx context.Context, logger logr.Logger, ss *syncSnapshot, countDeltaByPodGang map[string]int32, livePodsByPodGang map[string][]*corev1.Pod) error {
	if len(countDeltaByPodGang) == 0 {
		return nil
	}
	orderedPodGangNames := lo.Keys(countDeltaByPodGang)
	sort.Strings(orderedPodGangNames)

	createTasks, err := r.buildPerPodGangCreationTasks(logger, ss, countDeltaByPodGang, orderedPodGangNames)
	if err != nil {
		return err
	}
	deleteTasks := r.buildPerPodGangDeletionTasks(logger, ss, countDeltaByPodGang, orderedPodGangNames, livePodsByPodGang)

	if len(createTasks) > 0 {
		if runResult := utils.RunConcurrentlyWithSlowStart(ctx, logger, 1, createTasks); runResult.HasErrors() {
			err = runResult.GetAggregatedError()
			logger.Error(err, "failed to create pods for PCLQ", "runSummary", runResult.GetSummary())
			return err
		}
	}
	if len(deleteTasks) > 0 {
		if runResult := utils.RunConcurrentlyWithSlowStart(ctx, logger, 1, deleteTasks); runResult.HasErrors() {
			err = runResult.GetAggregatedError()
			logger.Error(err, "failed to delete pods for PCLQ", "runSummary", runResult.GetSummary())
			return groveerr.WrapError(err, errCodeDeletePod, component.OperationSync,
				fmt.Sprintf("failed to delete pods for PodClique %v", client.ObjectKeyFromObject(ss.pclq)))
		}
	}
	return nil
}

// buildPerPodGangCreationTasks builds creation tasks for each PodGang whose desired count exceeds the
// live count, assigning each new pod a host-name index.
func (r _resource) buildPerPodGangCreationTasks(logger logr.Logger, ss *syncSnapshot, countDeltaByPodGang map[string]int32, orderedPodGangNames []string) ([]utils.Task, error) {
	var totalToCreate int32
	for _, delta := range countDeltaByPodGang {
		if delta > 0 {
			totalToCreate += delta
		}
	}
	if totalToCreate == 0 {
		return nil, nil
	}
	availableIndices, err := index.GetAvailableIndices(logger, ss.existingPCLQPods, int(totalToCreate))
	if err != nil {
		return nil, groveerr.WrapError(err, errCodeGetAvailablePodHostNameIndices, component.OperationSync,
			fmt.Sprintf("error getting available indices for Pods in PodClique %v", client.ObjectKeyFromObject(ss.pclq)))
	}
	tasks := make([]utils.Task, 0, totalToCreate)
	taskIndex := 0
	for _, podGangName := range orderedPodGangNames {
		for created := int32(0); created < countDeltaByPodGang[podGangName]; created++ {
			tasks = append(tasks, r.createPodCreationTask(logger, ss.pcs, ss.pclq, podGangName, ss.pclqExpectationsStoreKey, taskIndex, availableIndices[taskIndex]))
			taskIndex++
		}
	}
	return tasks, nil
}

// buildPerPodGangDeletionTasks builds deletion tasks for each PodGang whose live count exceeds the
// desired count. Within each PodGang the DeletionSorter picks which pods to remove.
func (r _resource) buildPerPodGangDeletionTasks(logger logr.Logger, ss *syncSnapshot, countDeltaByPodGang map[string]int32, orderedPodGangNames []string, livePodsByPodGang map[string][]*corev1.Pod) []utils.Task {
	var tasks []utils.Task
	for _, podGangName := range orderedPodGangNames {
		if countDeltaByPodGang[podGangName] >= 0 {
			continue
		}
		numToDelete := int(-countDeltaByPodGang[podGangName])
		pods := livePodsByPodGang[podGangName]
		if len(pods) == 0 {
			continue
		}
		sorter := DeletionSorter{Pods: slices.Clone(pods), ExpectedPodTemplateHash: ss.expectedPodTemplateHash}
		sort.Sort(sorter)
		numToDelete = min(numToDelete, len(sorter.Pods))
		for _, pod := range sorter.Pods[:numToDelete] {
			tasks = append(tasks, r.createPodDeletionTask(logger, ss.pclq, pod, ss.pclqExpectationsStoreKey))
		}
	}
	return tasks
}

// sumCounts returns the total across all PodGang counts.
func sumCounts(countByPodGang map[string]int32) int32 {
	var total int32
	for _, count := range countByPodGang {
		total += count
	}
	return total
}
