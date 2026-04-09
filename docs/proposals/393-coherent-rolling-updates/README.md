<!-- toc -->
- [Summary](#summary)
- [Motivation](#motivation)
  - [Why version upgrades can be incompatible in Disaggregated Inference?](#why-version-upgrades-can-be-incompatible-in-disaggregated-inference)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Abbreviations](#abbreviations)
- [Proposal](#proposal)
  - [User Stories](#user-stories)
    - [Story 1](#story-1)
    - [Story 2](#story-2)
    - [Story 3](#story-3)
  - [Limitations/Risks &amp; Mitigations](#limitationsrisks--mitigations)
- [Design Details](#design-details)
  - [Gang scheduling during initial deployment of PCS](#gang-scheduling-during-initial-deployment-of-pcs)
  - [MVUs and gang-scheduling during update](#mvus-and-gang-scheduling-during-update)
    - [Rules of MVU gang-scheduling](#rules-of-mvu-gang-scheduling)
    - [MVU update flow](#mvu-update-flow)
    - [Illustration by example](#illustration-by-example)
  - [PCS Rollback and Roll-Forward](#pcs-rollback-and-roll-forward)
    - [PodCliqueRevision custom resource](#podcliquerevision-custom-resource)
    - [PCS-level revision tracking](#pcs-level-revision-tracking)
      - [Where to store this state: annotations, labels, or <code>PodCliqueSet.Status</code>?](#where-to-store-this-state-annotations-labels-or-podcliquesetstatus)
    - [Illustration](#illustration)
  - [Update concurrency](#update-concurrency)
  - [Handling scale-outs and scale-ins during update](#handling-scale-outs-and-scale-ins-during-update)
  - [Monitoring](#monitoring)
  - [Dependencies (<em>Optional</em>)](#dependencies-optional)
  - [Test Plan](#test-plan)
  - [Graduation Criteria](#graduation-criteria)
- [Implementation History (<em>Optional</em>)](#implementation-history-optional)
- [Alternatives (<em>Optional</em>)](#alternatives-optional)
- [Appendix (<em>Optional</em>)](#appendix-optional)
<!-- /toc -->

## Summary

Disaggregated inference architectures split LLM serving into distinct phases — most commonly (but not limited to) **prefill** (context generation) and **decode** (token generation) — running as separate, independently scalable components. While this can improve throughput and hardware utilisation, it introduces a hard operational constraint during version upgrades: prefill and decode instances that communicate must always run compatible software versions. This proposal introduces **Coherent Rolling Updates** for `PodCliqueSet`, enabling atomic, availability-preserving software upgrades at the granularity of **Minimal Viable Units (MVUs)** — the smallest sets of components that must be updated in lockstep to maintain compatibility and ensure availability.

## Motivation

Inference frameworks (e.g., vLLM, SGLang, TensorRT-LLM) support disaggregated LLM serving, where stages like prefill and decode run as separate, networked components. While this can improve throughput and resource efficiency, it complicates standard deployment practices. A standard Kubernetes rolling update inevitably creates a period where old and new version pods run at the same time and may communicate. In disaggregated systems, this cross-version communication is unsafe, so applications must prevent it. However, once cross-version communication is disabled, rolling updates introduce another issue: different components often update at different rates, which leads to mismatched pools of compatible instances. For example, you might still have many old-version prefill instances running while most old-version decode instances have already been replaced. Since old prefill can only talk to old decode, a portion of the prefill capacity becomes unusable due to the lack of matching decode capacity. This kind of mismatch reduces effective end-to-end serving capacity during the update. Our goal is to design a rolling update strategy that maintains balanced, compatible capacity across components, so version upgrades do not reduce serving capacity.

### Why cross-version communication is considered unsafe in Disaggregated Inference?

When there is a need to upgrade to a newer version of `Prefill` and `Decode` components in disaggregated inference, following are some of the areas where incompatibilities are usually seen (the list is only indicative and not comprehensive).

**KV-Cache transfer protocol** (*Frequency*: Very common)

Since there is no standard versioned protocol for the KV cache transfers between Prefill and Decode components, it can result in incompatibilities across wire-format versions. Some of the things that can/have changed across inference frameworks versions are as follows:

*dtype (data type)*

Defines the nature, precision and memory size of elements stored within a tensor. e.g., float16, bfloat16, float8_e4m3. A new version might default to FP8 for memory efficiency where the previous version used BF16. If the prefill sends FP8-packed bytes but the decode expects BF16, it will misinterpret every value, producing garbage tokens or NaNs.

*head-dim ordering*

KV-cache tensors have multiple logical dimensions: *[num_layers, num_heads, seq_len, head_dim]* which can be reordered across different versions of the same inference framework. This is typically done to optimize performance, memory layout optimization (e.g., switching to a more cache-friendly layout for Flash Attention 3) or to support newer attention mechanisms like `Grouped Query Attention` (GQA) or `Multi-Head Latent Attention` (MLA) among other reasons. If the decode reads a tensor using a different dimension order than the prefill used to write it, every attention computation is wrong silently.

*block size*

Paged attention divides the KV-cache into fixed-size blocks (pages), e.g., 16 or 32 tokens per block. The block size is baked into how memory is allocated and how block-table indices are communicated. If prefill was paged with block size 16 and decode expects block size 32, the block-table offsets the prefill sends point to wrong memory addresses on the decode side, causing memory corruption or out-of-bounds reads.

**RPC protocol** *(Frequency: Common)*

The serialisation format (*protobuf schema, msgpack frames*) of scheduler <-> worker (*inter-node communication between a Prefill/Decode node's local scheduler and its GPU workers*) messages and the disaggregation-specific handshake (*request metadata, sequence IDs, block tables - cross-node message the prefill node's scheduler sends to the decode node's scheduler*) can change between versions leading to either a silent error (*Decode node can misparse requests*) or a hard crash because either side (*Prefill/Decode*) does not have a version-negotiation step.

**Attention Backend & Kernel ABI ** *(Frequency: Occasional)*

Flash-Attention, Paged-Attention, and custom CUDA kernels expose internal data structures (block-tables, metadata buffers) that are shared across the disaggregation boundary. Kernel upgrades often change these layouts. In disaggregated inference, the prefill node writes KV blocks into its GPU memory, then transfers those raw bytes to the decode node via NCCL/NIXL/RDMA. The decode node's kernel then reads those bytes directly. There is no deserialisation step, no schema — just raw memory  copied from one GPU to another. The decode kernel must interpret those bytes using the exact same layout assumptions the prefill kernel used to write them. Raw GPU memory has no type tags, no field names, no length prefixes. It is just bytes at an address. A layout mismatch does not produce an error — the kernel runs to completion and produces numerically wrong results. The model generates plausible-looking but incorrect tokens, which is the worst failure mode: silent quality  degradation with no crash to alert the operator.

**Quantisation/compression format** *(Frequency: Occasional)*

Modern models are large. Storing weights and KV-cache in full precision (BF16, 32 bits) consumes enormous GPU memory.  Quantisation  reduces this by storing values in lower-bit formats (e.g., FP8, AWQ, GPTQ) which pack multiple values into single words using a bit layout defined by the framework implementation, not a standard. A version change that alters packing (different FP8 variant, different group size) means decode dequantises correctly-received bytes using the wrong codebook — every value is numerically wrong.

### Goals

* Enable rolling updates at the granularity of minimal sets of interdependent components that must be updated together in lockstep to maintain compatibility.
* It should be possible to gang-schedule each set of compatible interdependent components.
* Preserve `PodCliqueSet` availability during rolling updates to serve incoming traffic with sets of compatible interdependent components.
* Maintain a configurable revision history limit of `PodClique` versions to support rollback and roll-forward operations.
* Provide user-configurable concurrency control to limit the number of `PodCliqueSet` replicas that can be
  updated simultaneously.
* Provide user-configurable concurrency control to accelerate update of interdependent component sets within a `PodCliqueSet` replica.
* Support `scale-out` and `scale-in` of scale sub-resources (`PodClique`, `PodCliqueScalingGroup` and `PodCliqueSet`) during rolling update.

### Non-Goals

* Re-use of topology optimized resources during rolling update using resource reservations. Will be handled in future as a separate feature.
* Explicit support for `maxSurge` and `maxUnavailable` API. However, similar concurrency controls and functionality will be supported.

## Abbreviations

Throughout this proposal we will be using the Grove custom resource short forms for brevity:

| Long Form             | Short Form |
| --------------------- | ---------- |
| PodCliqueSet          | PCS        |
| PodCliqueScalingGroup | PCSG       |
| PodClique             | PCS        |
| PodGang               | PG         |
| BasePodGang           | BPG        |
| ScaledPodGang         | SPG        |
| MVUPodGang            | MPG        |

> *NOTE:*`BPG`, `SPG` and `MPG` are abbreviations introduced only to differentiate different types of `PodGang` resources and are not new custom resources.

## Proposal

The GREP introduces a new rolling update strategy, named **Coherent Rolling Updates**, based on the concept of a **Minimal Viable Unit** (a.k.a. MVU): the set of MinAvailable number of pods from each updated component (which defines the compatibility boundary as set by user) and is dynamically formed as single atomic update unit that needs to be gang-scheduled.

If pods in different PodCliques can’t communicate safely across disaggregation boundaries because their software versions are incompatible, updating all pods in an MVU as a unit (rather than individually) eliminates mixed-version imbalance.

For a typical disaggregated inference application where the compatibility boundary consists of prefill, decode and frontend components, a single MVU would contain the minimum number of version-compatible prefill, decode and frontend pods necessary to serve traffic.

This GREP also introduces PodClique revisions that can be used to maintain versioned sets of compatible interdependent components to support rollback and rollforward operations.

### User Stories

#### Story 1

As a platform engineer operating a disaggregated inference deployment (e.g., prefill and decode components) using modern inference frameworks (such as vLLM, SGLang, or TensorRT-LLM), I need to safely roll out new software versions where components are not backward compatible across versions. During an upgrade, prefills running the old version must not attempt to communicate with decodes running the new version (and vice versa), as this can lead to crashes, corrupted KV transfers, or undefined behavior.

The system must update prefill and decode pods together as a single atomic unit (MVU), ensuring that at no point does an old-version prefill hand off a KV-cache block to a new-version decode, or vice versa. While the update is in progress, replicas that have not yet been updated must continue serving traffic using only old-version components, and replicas that have already been updated must serve using only new-version components. The update should proceed replica-by-replica (or MVU-by-MVU within a replica) without requiring a full deployment restart, so that overall serving capacity is preserved throughout the rollout.

#### Story 2

As a platform engineer managing a large-scale disaggregated inference fleet with many `PodCliqueSet` replicas, I need to control the blast radius of a rolling update. If a new version turns out to be faulty, I want to limit the number of replicas upgraded simultaneously so that a bug is contained to a small fraction of live traffic. I also need the ability to pause an in-progress update and roll back all affected MVUs to the previously known-good version, restoring compatibility within each replica instantly without manual intervention.

#### Story 3

As an ML infrastructure team member deploying a disaggregated inference system where the prefill tier and decode tier are updated on different release cadences, I need to independently update only the decode `PodClique` (e.g., to pick up a memory-efficiency fix) without touching the prefill `PodClique`. The system should recognise that this is a backward compatible, single-component update, update decode pods incrementally (up to a configurable concurrency limit), and leave prefill pods untouched — all without requiring a full MVU replacement.

### Limitations/Risks & Mitigations

<!-- 
What are the current set of limitations or risks of this proposal? Think broadly by considering the impact of the changes proposed on kubernetes ecosystem. Optionally mention ways to mitigate these.
-->

## Design Details

### Gang scheduling during initial deployment of PCS

Grove’s scheduling API uses PodGangs to represent an application’s gang-scheduling constraints. The first PodGang created as part of a PCS’s initial deployment is called the `BasePodGang`, and any PodGangs created as a result for PCSG replicas that are above `MinAvailable`, are called `ScaledPodGang`s. In Grove’s current design, both BasePodGangs and ScaledPodGangs persist across update events: their PodReferences may be refreshed, but the PodGangs themselves—and their overall structure—remain the same. The new **Coherent Rolling Update** strategy changes this behavior: rolling updates will dynamically create new PodGangs, as described in the next section.

### MVUs and gang-scheduling during update

A PCS is composed of PCLQs and PCSGs.  Updates may target a subset of PCLQs or all of them.  A MVU consist of `MinAvailable` replicas of each of the standalone PCLQs and the PCSGs that are updated. Between two updates since one or more `Scale` subresources (`PodClique.Scale`, `PodCliqueScalingGroup.Scale`) may have been scaled in or out between two updates, MVUs must be recomputed before each update begins. Every identified MVU needs to be gang scheduled. Hence, Grove will now generate new PodGangs called `MVUPodGang`s (a.k.a MPG) out of the existing PodGangs, which will encode the gang-scheduling intent of the MVUs.

#### Rules of MVU gang-scheduling

**For standalone PodCliques**
If one or more standalone PCLQs get updated then following rules will be followed to determine if the newer version of Pods for these PCLQs will get new MPGs.

| Case# | Description                                                  | Gang Scheduling behavior                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1     | Only one standalone PCLQ with minAvailable == 1 is updated | No new MPG will be created. New versions of PCLQ replicas will be replaced in their original PG |
| 2     | Only one standalone PCLQ with minAvailable > 1 is updated | One or more new MPGs will be created with `minAvailable` replicas of the PCLQ in each MPG |
| 3     | More than one standalone PCLQs with minAvailable >= 1 are updated | One or more new MPGs will be created with `minAvailable` replicas of each standalone PCLQ in each PG |

**For PodCliques belonging to PodCliqueScalingGroups**

At the PCSG level, the MPG will contain `MinAvailable` replicas of the updated PCSGs. The entire set of PCLQs (containing all their replicas) of each PCSG are included in the MPG even if a subset of PCLQs are updated.

| Case# | Description | Gang Scheduling behavior |
| ----- | ----------- | ------------------------ |
| 1     | One or more PCLQs of a PCSG updated | MPG will contain the `MinAvailable` replicas of this PCSG. All the constituent PCLQs (with all their replicas) are included as `PodGroups` in the MPG. |
| 2     | One or more PCLQs from more than one PCSG are updated | One or more new MPG created with `minAvailable` replicas from each of the PCLQs of every PCSG and replicated over all the replicas of each PCSG |
| 3     | One or more PCLQs from one or more PCSG and/or one or more standalone PCLQs are updated | One or more new MPG created with `minAvailable` replicas from each of the PCLQs of every PCSG and replicated over all the replicas of each PCSG, along with `minAvailable` replicas of each of the standalone PCLQs |

#### MVU update flow

When new `MVUPodGang`s are generated on every update, this fundamentally changes the gang-scheduling structure of the application from the pre-update phase to the post-update phase. In many ways, an update event marks an epoch transition in the lifecycle of the PCS. The mental model can be summarized as: an epoch starts with an initial number of PodGangs in the system, and progressively, more PodGangs are added due to scale-outs and some deleted due to scale-ins. When the next update event comes in, it marks an epoch transition and a completely new set of MPGs are generated based on the state of the system at that time.

When a PCS is updated, Grove operator will react to that event by triggering a rollout of the changes to the appropriate PodClique resources. At the same time, Grove operator will create a plan to transition pods from existing PGs to the new MPGs based on the *MVU template* of that update event.

An **MVU template** is basis on which a MPG is created and is based on `MinAvailable` replicas of both standalone PCLQs and PCSGs which need to be updated. *Caveat:* When creating an MPG one can have additional replicas above the `MinAvailable` replicas for standalone PCLQs. At this time MVU update process might create additional PGs at the end which do not adhere to the MVU template. We call these additional PGs as **Tail-MPGs**.

The update flow will be handled as per the following steps:

* Schedule gate all `Pending` pods across all PCLQs. This ensures that pending pods at the older version do not get scheduled when we start to replace older version scheduled pods by first deleting them and then creating newer versioned pods.
* Start MVU update loop: *(continue until all the targeted older version pods have been updated)*
  * Based on MVU template, select the set of standalone PCLQ pods and PCSG replica pods to be taken down. If the remaining PCLQ pods and PCSG replicas together do not make up a full MVU template, then add the remaining pods to the *take-down set*.
    * Currently scheduled pods are selected for the take-down set ahead of pending pods.
  * Recreate all pods in the take-down set as schedule gated.
  * Create the MPG based on MVU template. In case there are remaining pods of a standalone PCLQ within the take-down set, add those pods to the MPG. Remove scheduling gate on the MPG.
  * Further update is blocked until this MPG gets scheduled and becomes available.
  * In case, there are remaining PCSG replicas in the take-down set, create Tail-MPGs out of each PCSG replica and remove their scheduling gate.

#### Illustration by example

To illustrate how MVUs are carved out from the child resources of a `PodCliqueSet`, consider a `PodCliqueSet` representing a typical disaggregated inference application, composed of the following PodCliques:

* `FrontEnd` - handles request ingestion, tokenization, KV cache routing, and load balancing.
* `Prefill Leader` - handles batch coordination, KV cache orchestration, sequence splitting, and completion signaling.
* `Prefill Worker` - handles KV cache population and tensor parallel compute.
* `Decode Leader` - handles step orchestration, sampling, and output streaming.
* `Decode Worker` - handles forward pass, KV cache updates, and activation sync.

There are two `PodCliqueScalingGroups` -

* `Prefill` - comprising of `Prefill Leader` and `Prefill Worker` PodCliques.
* `Decode` - comprising of `Decode Leader` and `Decode Worker` PodCliques.

```yaml
apiVersion: grove.io/v1alpha1
kind: PodCliqueSet
metadata:
  name: disagg-serving
spec:
  replicas: 1
  template:
    cliques:
      - name: frontend
        spec:
          replicas: 3
          minAvailable: 2
          podSpec:
            containers:
              - name: frontend
                image: <frontend-image>
                resources:
                  requests:
                    cpu: 10m
      - name: pleader
        spec:
          replicas: 1
          minAvailable: 1
          podSpec:
            containers:
              - name: prefill
                image: <prefill-image>
                resources:
                  requests:
                    cpu: 10m
      - name: pworker
        spec:
          replicas: 3
          minAvailable: 2
          podSpec:
            containers:
              - name: prefill
                image: <prefill-image>
                resources:
                  requests:
                    cpu: 10m
      - name: dleader
        spec:
          replicas: 1
          minAvailable: 1
          podSpec:
            containers:
              - name: decode
                image: <decode-image>
                resources:
                  requests:
                    cpu: 10m
      - name: dworker
        spec:
          replicas: 4
          minAvailable: 2
          podSpec:
            containers:
              - name: decode
                image: <decode-image>
                resources:
                  requests:
                    cpu: 10m
    podCliqueScalingGroups:
      - name: prefill
        minAvailable: 1
        replicas: 1
        cliqueNames:
          - pleader
          - pworker
      - name: decode
        minAvailable: 1
        replicas: 1
        cliqueNames:
          - dleader
          - dworker
```

Prior to update, replicas of each of the child resources of the `disagg-serving` PCS is as shown in the resource YAML above. The initial set of `PodGang`s that are created have the following composition:

*At time T1:*

```
PodGang-1: {  # this is the base PodGang that must be scheduled
  frontend (F): 3 Pod,
  prefill (P): { prefill-leader: 1 Pod, prefill-worker: 3 Pods (minAvailable=2) },
  decode (D): { decode-leader:  1 Pod, decode-worker: 4 Pods (minAvailable=2) },
}
# In short represented as {3F, 1P, 1D}
```

*At time T2 (T2> T1):*

`Prefill` PodCliqueScalingGroup scales out by 3, this results in the following additional PodGangs.

```
[PodGang-2, PodGang-3, PodGang-4] each will have: { prefill-leader: 1 Pod, prefill-worker: 3 Pods (minAvailable=2) } pods.
# In short represented as 3 * {P}
```

*At time T3 (T3> T2):*

`Decode` PodCliqueScalingGroup scales out by 2, this results in the following additional PodGangs:

```
[PodGang-5, PodGang-6] each will have: { decode-leader: 1 Pod, decode-worker: 4 Pods (minAvailable=2) } pods.
# In short represented as 2 * {D}
```

`Frontend` PodClique scales out by 2, this results in update of the first PodGang (a.k.a the base podgang):

```
PodGang-1: {  # this is the base PodGang that must be scheduled
  frontend: 5 Pod,
  prefill: { prefill-leader: 1 Pod, prefill-worker: 3 Pods (minAvailable=2) },
  decode: { decode-leader:  1 Pod, decode-worker: 4 Pods (minAvailable=2) }
}
# In short represented as {5F, 1P, 1D}
```

*At time T4 (T4 > T3)* - An update is triggered
Updates to a PCS can be done to a subset of PodCliques or all of the PodCliques. Lets evaluate how MVUs are computed and PodGangs are created in different cases.

Initial state prior to update:

```
BPG: {5F, 1P, 1D}, SPG: 3 * {P}, 2 * {D}
MinAvailable: {F: 2, P: 1, D: 1}
```

**Case #1: Only standalone PodClique(s) are updated**

In the above example, `Frontend` is the only standalone PodClique. Let us represent the new version of `Frontend` PodClique as Fv1 (where standalone `F` represents v0 or the initial version of the PodSpec). The `MinAvailable` replicas for `Frontend` PodClique is defined as 1. 
MVU template is {2F} as it is a function of `MinAvailable` replicas of all PodCliques that have been updated.

Following are the steps demonstrating the creation and update of MPGs during the update:

```
Step-1:
  Take-down set: {2F}
  Recreate order: {2Fv1}
  Expected state: PG: {3F, 1P, 1D}, 3 * {P}, 2 * {D}, MPG: {2Fv1}
Step-2 -> 
  Take-down set: {3F}
  Recreate order: {3Fv1}
  Expected state: PG: {1P, 1D}, 3 * {P}, 2 * {D}, MPG: {2Fv1}, {3Fv1}
```

**Case #2: Prefill and Decode are updated**

In this example the user has updated all PCLQs belonging to `Prefill` and `Decode` PCSGs. Updates to any constituent PCLQ of a PCSG is considered as an update of the entire PCSG. Let us represent the new version of Prefill and Decode as `Pv1` and `Dv1` respectively.
MVU template is {1P, 1D} as it is a function of `MinAvailable` replicas of all PCSGs that have been updated.

Following are the steps demonstrating the creation and update of MPGs during the update:

```
Step-1:
  Take-down set: {1P, 1D}
  Recreate order: {1Pv1, 1Dv1}
  Expected state: PG: {5F}, 3 * {P}, 2 * {D}, MPG: {1Pv1, 1Dv1}
Step-2 -> 
  Take-down set: 1 * {P}, 1 * {D}
  Recreate order: {1Pv1, 1Dv1}
  Expected state: PG: {5F}, 2 * {P}, 1 * {D}, MPG: {1Pv1, 1Dv1}, {1Pv1, 1Dv1}
Step-3 -> 
  Take-down set: 2 * {P}, 1 * {D}
  Recreate order: [{1Pv1, 1Dv1}] -> [{1Pv1}]
  Expected state: PG: {5F}, MPG: {1Pv1, 1Dv1}, {1Pv1, 1Dv1}, {1Pv1, 1Dv1}, Tail-MPG: {1Pv1}
```

**Case #3: All PCLQs are updated**

In this example the user has updated all PCLQs in a PCS.
MVU template is {2F, 1P, 1D} as it is a function of `MinAvailable` replicas of all PCSGs that have been updated.

Following are the steps demonstrating the creation and update of MPGs during the update:
```
Step-1:
  Take-down set: {2F, 1P, 1D}
  Recreate order: {2Fv1, 1Pv1, 1Dv1}
  Expected state: PG: {3F}, 3 * {P}, 2 * {D}, MPG: {2Fv1, 1Pv1, 1Dv1}
Step-2:
  Take-down set: {3F}, 3 * {P}, 2 * {D}
  Recreate order: [{3Fv1, 1Pv1, 1Dv1}] -> [{1Pv1}, {1Dv1}, {1Pv1}]
  Expected state: MPG: {2Fv1, 1Pv1, 1Dv1}, {3Fv1, 1Pv1, 1Dv1}, Tail-MPGs: {1Pv1}, {1Dv1}, {1Pv1}
```

### PCS Rollback and Roll-Forward

Version upgrades in disaggregated inference are high-risk operations. As described earlier, incompatibilities in KV-cache transfer protocols, RPC serialisation formats, attention kernel ABIs, or quantisation layouts can produce silent correctness failures — garbage tokens, memory corruption, or numerically wrong outputs — with no crash to alert the operator. A new version may pass initial validation and begin serving traffic before these failures surface under real load or specific model inputs.

When such a regression is detected, the ability to quickly revert PCS to a last known-good state is critical. All PCS constituents must be collectively roll-backed together (atomically), restoring all PCLQs to the exact set of Pod specs that were in service together at a prior revision. A partial rollback (reverting only subset of components) is unsafe and can end up with the same cross-version incompatibilities that the update was designed to avoid.

Roll-forward addresses the complementary case: after rolling back to investigate a regression, an operator may determine that the new version is actually safe and wish to re-apply it without triggering a fresh rollout.

To support rollback and roll-forward, two things are needed:

* **Revision tracking at the PCS level** — the PCS records a *monotonically increasing* revision counter and maintains a bounded history of revision tuples, where each tuple captures the set of `PodCliqueRevision`s that were active together across all PCLQs at a given PCS revision. This history is what makes it possible to reconstruct a prior compatible set of component specs.
* **A snapshot of the PodSpec at each revision** — captured as a `PodCliqueRevision` resource, one per PCLQ per revision in which its PodSpec changed.

#### PodCliqueRevision custom resource

```go
// PodCliqueRevision is an immutable snapshot of a PodClique's PodSpec at a specific revision.
// It is created whenever a PodClique's PodSpec changes as part of a PodCliqueSet update, and
// is never modified after creation.
type PodCliqueRevision struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              PodCliqueRevisionSpec `json:"spec"`
}

// PodCliqueRevisionSpec holds the PodSpec captured at the time this revision was created.
type PodCliqueRevisionSpec struct {
    // PodSpec is the exact PodSpec of the PodClique at this revision.
    PodSpec  corev1.PodSpec `json:"podSpec"`
}
```

**Naming convention:** `<pcs-name>-<pclq-name>-v<revision-number>`, where `pcs-name` is the name of the owning `PodCliqueSet`, `pclq-name` is the unqualified name of the `PodClique` whose spec is captured, and `revision-number` is a monotonically increasing integer assigned at the time of creation.

**Revision label:** `grove.io/revision` is set to the revision number on each `PodCliqueRevision`. This label exists solely to allow efficient lookup of the exact `PodCliqueRevision` resource targeted by a rollback or roll-forward operation.

`PodCliqueRevision` has no `scale-subresource` and plays no active role in pod lifecycle management — it is purely a snapshot for audit and recovery purposes. The PCLQ resource remains the single source of truth for all Pods and the target for any scaling operations on standalone PCLQs.

#### PCS-level revision tracking

Three pieces of revision state must be tracked at the `PodCliqueSet` level:

- **`maxRevision`** — the highest revision number assigned to any `PodCliqueRevision` across all PCLQs. This is the ceiling for roll-forward: an operator cannot roll forward past `maxRevision`.
- **`currentRevision`** — the revision at which the PCS currently sits. After a fresh update this equals `maxRevision`; after a rollback it is less than `maxRevision`, reflecting that the active specs belong to an earlier point in history.
- **`revisionHistory`** — an ordered list of revision tuples, one entry per PCS revision. Each tuple records the `PodCliqueRevision` revision number that was active for every PCLQ at that point. This is what allows the controller to reconstruct the exact compatible set of specs for any prior revision.

##### Where to store this state: annotations, labels, or `PodCliqueSet.Status`?

**Annotations** are an untyped, string-valued map on the object's metadata. Kubernetes itself uses the annotation `deployment.kubernetes.io/revision` to track the current rollout revision on a `Deployment`. Annotations are simple and require no API schema change, but they have no type safety, no defaulting, no validation, and are not intended for machine-readable structured data beyond simple scalars. Storing a structured revision history tuple as an annotation would require encoding it (e.g., as JSON) into a string value, making it fragile and hard to query.

**Labels** are also untyped strings and are intended for selection and filtering, not for storing operational state. Additionally, label values are limited to 63 characters, making them unsuitable for storing anything beyond simple scalars. They are the wrong tool here.

✅ **`PodCliqueSet.Status` fields** are the right choice. Status is the canonical location for controller-managed operational state in Kubernetes. It is strongly typed, versioned with the API, survives schema evolution, and is directly accessible to clients without parsing. The revision history in particular — a structured slice of tuples — is only representable cleanly as a typed status field. `currentRevision` and `maxRevision` are scalar integers that could technically live as annotations, but co-locating them in status alongside the history keeps all revision state in one place and makes it atomically observable.

The three fields are therefore introduced as part of `PodCliqueSetStatus`:

```go
type PodCliqueSetStatus struct {
    // CurrentRevision is the PCS revision at which all PCLQs are currently active.
    // After a fresh update this equals MaxRevision. After a rollback it is less than MaxRevision.
    CurrentRevision int32 `json:"currentRevision"`

    // MaxRevision is the highest revision number assigned across all PodCliqueRevision resources
    // owned by this PCS. It is the upper bound for roll-forward operations.
    MaxRevision int32 `json:"maxRevision"`

    // RevisionHistory is a bounded list of revision tuples recorded at each PCS revision.
    // Each tuple is an ordered slice of PodCliqueRevision revision numbers, one per PCLQ, in
    // the order PCLQs are defined in the PCS spec. Each tuple represents the set of PCLQ
    // revisions that were active together at a given PCS revision, enabling reconstruction of
    // any prior compatible set of specs. The number of retained tuples is controlled by
    // RevisionHistoryLimit on the PCS spec.
    //
    // Example: given PCLQs [a, b, c], the history [[1,1,1],[2,1,1],[2,3,1]] means:
    //   PCS revision 1: a@1, b@1, c@1
    //   PCS revision 2: a@2, b@1, c@1
    //   PCS revision 3: a@2, b@3, c@1
    RevisionHistory [][]int32 `json:"revisionHistory,omitempty"`

    // ... other existing status fields
}
```

The number of revision tuples retained in `RevisionHistory` is controlled by `RevisionHistoryLimit` on the `PodCliqueSet` spec. Once the limit is reached, the oldest entry is evicted and the `PodCliqueRevision` resources associated with that entry are deleted. This mirrors the same concept as `revisionHistoryLimit` on `Deployment`. Operators should set this high enough to cover the rollback depth they require; the default is 5.

```go
type PodCliqueSetSpec struct {
    // RevisionHistoryLimit is the maximum number of revision tuples to retain in
    // RevisionHistory. Once the limit is reached, the oldest entry is evicted.
    // Defaults to 5.
    // +optional
    RevisionHistoryLimit *int32 `json:"revisionHistoryLimit,omitempty"`

    // ... other existing spec fields
}
```

#### Illustration

The following illustration uses the `disagg-serving` PCS defined earlier. It has five PCLQs: `frontend`, `pleader`, `pworker`, `dleader`, `dworker`. The revision tuple in `RevisionHistory` follows that same order throughout: `[frontend, pleader, pworker, dleader, dworker]`.

For brevity, `PodCliqueRevision` resources are named `<pclq>-r<N>` (e.g. `frontend-r1`, `pworker-r3`).

---

**Initial state** — PCS deployed at revision 1, all PCLQs at their first revision.

```
currentRevision: 1  maxRevision: 1
revisionHistory: [[1, 1, 1, 1, 1]]

PodCliqueRevisions:
  frontend-r1  <- active
  pleader-r1   <- active
  pworker-r1   <- active
  dleader-r1   <- active
  dworker-r1   <- active
```

---

**Update 1** — `frontend` image is updated. A new `PodCliqueRevision` `frontend-r2` is created.

```
currentRevision: 2  maxRevision: 2
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2  <- active
  pleader-r1   <- active
  pworker-r1   <- active
  dleader-r1   <- active
  dworker-r1   <- active
```

---

**Update 2** —  `dleader and dworker` images are updated. A new `PodCliqueRevision` `dleader-r3 and dworker-r3` are created.

```
currentRevision: 3  maxRevision: 3
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2  <- active
  pleader-r1   <- active
  pworker-r1   <- active
  dleader-r1
  dleader-r3   <- active
  dworker-r1
  dworker-r3   <- active
```

---

**Update 3** — All PCLQs are updated together (e.g. a breaking protocol change requires all components to move in lockstep). New `PodCliqueRevision` resources are created for all five PCLQs.

```
currentRevision: 4  maxRevision: 4
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3], [4, 4, 4, 4, 4]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2
  frontend-r4  <- active
  pleader-r1
  pleader-r4   <- active
  pworker-r1
  pworker-r4   <- active
  dleader-r1
  dleader-r3
  dleader-r4   <- active
  dworker-r1
  dworker-r3
  dworker-r4   <- active
```

Silent quality degradation is detected after Update 3. The operator rolls back.

---

**Rollback** — `rollout undo --to-revision=3`. The controller looks up `revisionHistory[2]` = `[1, 1, 2, 1, 3]` and moves the active pointer on each PCLQ to the corresponding `PodCliqueRevision`. No new resources are created. `maxRevision` is preserved.

```
currentRevision: 3  maxRevision: 4
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3], [4, 4, 4, 4, 4]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2  <- active   (was frontend-r4)
  frontend-r4
  pleader-r1   <- active   (was pleader-r4)
  pleader-r4
  pworker-r1   <- active   (was pworker-r4)
  pworker-r4
  dleader-r1
  dleader-r3   <- active   (was dleader-r4)
  dleader-r4
  dworker-r1
  dworker-r3   <- active   (was dworker-r4)
  dworker-r4
```

---

**New update while rolled back** — while `currentRevision` is still at 3, the operator pushes a new `frontend` image. A new `PodCliqueRevision` `frontend-r5` is created. The revision counter always increments from `maxRevision+1` regardless of where `currentRevision` currently sits, so both `currentRevision` and `maxRevision` advance to 5. The existing history entries — including revision 4 — are retained. It remains a valid target for a future rollback since it represents a known compatible set of specs.

```
currentRevision: 5  maxRevision: 5
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3], [4, 4, 4, 4, 4], [5, 1, 1, 3, 3]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2
  frontend-r4
  frontend-r5  <- active
  pleader-r1   <- active
  pworker-r1   <- active
  dleader-r1
  dleader-r3   <- active
  dleader-r4
  dworker-r1
  dworker-r3   <- active
  dworker-r4
```

History entries are only evicted when `RevisionHistoryLimit` is reached, at which point the oldest entry is removed along with any `PodCliqueRevision` resources that are no longer referenced by any remaining history entry.

---

**Rollback again** — `rollout undo --to-revision=4`. The controller looks up `revisionHistory[3]` = `[4, 4, 4, 4, 4]` and moves the active pointers back to the revision-4 resources. `maxRevision` remains 5.

```
currentRevision: 4  maxRevision: 5
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3], [4, 4, 4, 4, 4], [5, 1, 1, 3, 3]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2
  frontend-r4  <- active   (was frontend-r5)
  frontend-r5
  pleader-r1
  pleader-r4   <- active   (was pleader-r1)
  pworker-r1
  pworker-r4   <- active   (was pworker-r1)
  dleader-r1
  dleader-r3
  dleader-r4   <- active   (was dleader-r3)
  dworker-r1
  dworker-r3
  dworker-r4   <- active   (was dworker-r3)
```

---

**Roll-forward** — `rollout redo --to-revision=5`. The controller looks up `revisionHistory[4]` = `[5, 1, 1, 3, 3]` and restores the active pointers. No new resources are created.

```
currentRevision: 5  maxRevision: 5
revisionHistory: [[1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 3, 3], [4, 4, 4, 4, 4], [5, 1, 1, 3, 3]]

PodCliqueRevisions:
  frontend-r1
  frontend-r2
  frontend-r4
  frontend-r5  <- active   (was frontend-r4)
  pleader-r1   <- active   (was pleader-r4)
  pleader-r4
  pworker-r1   <- active   (was pworker-r4)
  pworker-r4
  dleader-r1
  dleader-r3   <- active   (was dleader-r4)
  dleader-r4
  dworker-r1
  dworker-r3   <- active   (was dworker-r4)
  dworker-r4
```

### Update concurrency

<TBD>

### Handling scale-outs and scale-ins during update

<TBD>

### Monitoring

<TBD>

<!--
This section contains details of events, metrics, status conditions and other status fields that will aid in determining health of the feature, or help measure any service level objectives that might be optionally defined.
-->

### Dependencies

<!--
Are there any dependencies for this feature to work? If yes then those should be clearly listed with optional links on how to ensure that the dependencies are setup.
-->

### Test Plan

<!--
For the functionality an epic (issue) should be created. Along with a sub-issue for the GREP, there should be a dedicated issue created for integration and e2e tests. This issue should have details of all scenarios that needs to be tested. Provide a link to issue(s) in this section.
-->

### Graduation Criteria

<!-- 
In this section graduation milestones should be defined. The progression of the overall feature can be evaluated w.r.t API maturity, staged sub-feature implementation or some other criteria.

In general we try to use the same stages (alpha, beta, GA), regardless of how the
functionality is accessed. Refer to these for more details:"

* [Feature Gates](https://git.k8s.io/community/contributors/devel/sig-architecture/feature-gates.md)
* [Maturity levels](https://git.k8s.io/community/contributors/devel/sig-architecture/api_changes.md#alpha-beta-and-stable-versions)
* [Deprecation Policy](https://kubernetes.io/docs/reference/using-api/deprecation-policy/ ) 

**Note:** Generally we also wait at least two releases between beta and
GA/stable, because there's no opportunity for user feedback, or even bug reports,
in back-to-back releases. 
-->

## Implementation History (*Optional*)

<!--
Major milestones in the lifecycle of a GREP should be tracked in this section.
Major milestones might include:

- The date proposal was accepted and merged.
- The date implementation started.
- The date of Alpha release for the feature.
- The date the feature graduated to beta/GA

-->

## Alternatives (*Optional*)

<!--
What are the alternative approaches considered and reasons to rule those out. This section should have sufficient details (not too much) to express the alternative idea and why it was not accepted.
-->

## Appendix (*Optional*)

<!-- 
Use this section to put any prerequisite reading links or helpful information/data that supplements the proposal, thus providing additional context to the reviewer.
-->

