<!-- toc -->

- [Summary](#summary)
- [Motivation](#motivation)
    - [Goals](#goals)
    - [Non-Goals](#non-goals)
- [Proposal](#proposal)
    - [User Stories (<em>Optional</em>)](#user-stories-optional)
        - [Story 1 (<em>Optional</em>)](#story-1-optional)
        - [Story 2 (<em>Optional</em>)](#story-2-optional)
    - [Limitations/Risks &amp; Mitigations](#limitationsrisks--mitigations)
- [Design Details](#design-details)
    - [Monitoring](#monitoring)
    - [Dependencies (<em>Optional</em>)](#dependencies-optional)
    - [Test Plan](#test-plan)
    - [Graduation Criteria](#graduation-criteria)
- [Implementation History (<em>Optional</em>)](#implementation-history-optional)
- [Alternatives (<em>Optional</em>)](#alternatives-optional)
- [Appendix (<em>Optional</em>)](#appendix-optional)
  <!-- /toc -->

## Summary

Disaggregated inference architectures split LLM serving into distinct phases — most commonly (but not limited to) **prefill** (context generation) and **decode** (token generation) — running as separate, independently scalable components. While this improves throughput and hardware utilisation, it introduces a hard operational constraint during version upgrades: prefill and decode instances that communicate must always run compatible software versions. This proposal introduces **Coherent Rolling Updates** for `PodCliqueSet`, enabling atomic, availability-preserving software upgrades at the granularity of **Minimal Viable Units (MVUs)** — the smallest sets of components that must be updated in lockstep to maintain compatibility and ensure availability

## Motivation

Disaggregated inference frameworks (vLLM, SGLang, TensorRT-LLM, etc.) decompose LLM serving across multiple networked components. This decomposition makes version upgrades operationally risky: a standard Kubernetes rolling update replaces pods one at a time, inevitably producing a transient state where old-version and new-version pods are running simultaneously and may communicate with each other. For disaggregated systems, this cross-version communication is not safe.

### Why version upgrades can be incompatible in Disaggregated Inference?

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
* Provide user-configurable concurrency control to accelerate update of interdependents component sets within a `PodCliqueSet` replica.
* Support `scale-out` and `scale-in` of scale sub-resources (`PodClique`, `PodCliqueScalingGroup` and `PodCliqueSet`) during rolling update.

### Non-Goals

* Re-use of topology optimized resources during rolling update using resource reservations.
* Support for `maxSurge` and `maxUnavailable` like functionality.

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

## Proposal

The GREP introduces a new rolling update strategy, named **Coherent Rolling Updates**, based on the concept of a **Minimal Viable Unit** (a.k.a. MVU): the set of MinAvailable number of pods from each updated component (which defines the compatibility boundary as set by user) and is dynamically formed as single atomic update unit that needs to be gang-scheduled.

If pods in different PodCliques can’t communicate safely across disaggregation boundaries because their software versions are incompatible, updating all pods in an MVU as a unit (rather than individually) eliminates mixed-version communication. Therefore, each MVU must be gang-scheduled.

For a typical disaggregated inference application consisting of prefill, decode and frontend components, a single MVU would contain the minimum number of version-compatible prefill, decode and frontend pods necessary to serve traffic.

This GREP also introduces versioning of PodCliques that can be used to maintain versioned sets of compatible interdependent components to support rollback and rollforward operations.

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

Grove’s scheduling API uses PodGangs to represent an application’s gang-scheduling constraints. The first PodGang created as part of a PCS’s initial deployment is called the `BasePodGang`, and any PodGangs created as a result for PCSG replicas that are above `MinAvailable`, are called `ScaledPodGang`s. In Grove’s current design, both BasePodGangs and ScaledPodGangs persist across update events: their PodReferences may be refreshed, but the PodGangs themselves—and their overall structure—remain the same. With the new **Coherent Rolling Update** strategy, this will change: rolling updates will dynamically generate new PodGangs, as described in the next section.

### MVUs and gang-scheduling during update

A PCS is composed of PCLQs and PCSGs.  Updates may target a subset of PCLQs or all of them.  An MVU consist of `MinAvailable` replicas of each of the PCLQ's that are updated. Between two updates since one or more `Scale` subresources (`PodClique.Scale`, `PodCliqueScalingGroup.Scale`) may have been scaled in or out between two updates, MVUs must be recomputed before each update begins. Every identified MVU needs to be gang scheduled. Hence, Grove will now generate a new PodGangs called `MVUPodGang`s (a.k.a MPG) out of the existing PodGangs, which will encode the gang-scheduling intent of the MVUs.

#### Rules of MVU gang-scheduling

**For standalone PodCliques**
If one or more standalone PCLQs get updated then following rules will be followed to determine if the newer version of Pods for these PCLQs will get new MPGs.

| Case# | Description                                                  | Gang Scheduling behavior                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1     | There is only one standalone PCLQ with minAvailable == 1     | No new MPG will be created. New versions of PCLQ replicas will be replaced in their original PG |
| 2     | There is only one standalone PCLQ with minAvailable > 1      | One or more new MPGs will be created with `minAvailable` replicas of the PCLQ in each MPG |
| 3     | There are more than one standalone PCLQ  with minAvailable >= 1 | One or more new MPGs will be created with `minAvailable` replicas of each standalone PCLQ in each PG |

**For PodCliques belonging to PodCliqueScalingGroups**

| Case# | Description | Gang Scheduling behavior |
| ----- | ----------- | ------------------------ |
| 1     | One or more PCLQs of a PCSG updated | One or more new MPG created with `minAvailable` replicas from each of the PCLQs of that PCSG and replicated for all replicas of the PCSG |
| 2     | One or more PCLQs from more than one PCSG are updated | One or more new MPG created with `minAvailable` replicas from each of the PCLQs of every PCSG and replicated over all the replicas of each PCSG |
| 3     | One or more PCLQs from one or more PCSG and/or one or more standalone PCLQs are updated | One or more new MPG created with `minAvailable` replicas from each of the PCLQs of every PCSG and replicated over all the replicas of each PCSG, along with `minAvailable` replicas of each of the standalone PCLQs |

#### MVU update flow

When new `MVUPodGang`s are generated on every update, this fundamentally changes the gang-scheduling structure of the application from pre-update phase to the post-update phase. In other words, update events mark an epoch transition in the lifecycle of the PCS. The mental model of this is: an epoch will start with an initial number of PodGangs in the system. Over the epoch period, more PodGangs can be added due to scale-out of the application and some deleted due to scale-ins. When the next update event comes in, a completely new set of MPGs are generated based on the state of the system at that time.

When a PCS is updated, Grove operator will react to that event by triggering a rollout of the changes to the appropriate PodClique resources. At the same time, Grove operator will create a plan to transition pods from existing PGs to the new MPGs. This will be handled as per the following steps:
* First, all pending and unhealthy pods will be recreated and placed into their respective MPGs. These pending MPGs will be schedule-gated.
* Next, running pods from updated PodCliques that form one MPG will be deleted and recreated with their new specs.
* Block until that one MPG to get scheduled
* Once scheduled pickup create the next MPG from the running pods, and repeat until all running pods are updated.
* Remove the schedule-gates on the pending MPGs.

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
          replicas: 1
          minAvailable: 1
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
  frontend (F): 1 Pod,
  prefill (P): { prefill-leader: 1 Pod, prefill-worker: 3 Pods (minAvailable=2) },
  decode (D): { decode-leader:  1 Pod, decode-worker: 4 Pods (minAvailable=2) },
}
# In short represented as {1F, 1P, 1D}
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
  frontend: 3 Pod,
  prefill: { prefill-leader: 1 Pod, prefill-worker: 3 Pods (minAvailable=2) },
  decode: { decode-leader:  1 Pod, decode-worker: 4 Pods (minAvailable=2) }
}
# In short represented as {3F, 1P, 1D}
```

*At time T4 (T4 > T3)* - An update is triggered
Updates to a PCS can be done to a subset of PodCliques or all of the PodCliques. Lets evaluate how MVUs are computed and PodGangs are created in different cases.

**Case #1: Only standalone PodClique(s) are updated**

In the above example, `Frontend` is the only standalone PodClique. Let us represent the new version of `Frontend` PodClique as Fv1 (where standalone `F` represents v0 or the initial version of the PodSpec). The `MinAvailable` replicas for `Frontend` PodClique is defined as 1. Therefore the composition of the MVU will be {1F} as it is a function of `MinAvailable` replicas of all PodCliques that have been updated.


A rolling update is triggered at this stage.

Case #1: Only frontend is updated

A new `PodCliqueVersion` will be created for Frontend. Let's call this `Fv1`.
At the start of the update state of PodGangs is: `{1P, 1D, 5F}, 5 * {P} , 2 * {D}`
Since only the FrontEnd gets updated, it is assumed that this is a backward compatible update. The number of replicas in FrontEnd PCLQ to update is defined by the MinReplicas as defined in FrontEnd PCLQ.  That will form the minimum unit for update. In the above example it is 2.
PodGangs:
- Step1:  `{1P, 1D, 3F, 2 * Fv1}`, `5 * {P}` , `2 * {D}`
- Step2:  `{1P, 1D, 1F, 4 * Fv1}`, `5 * {P}` , `2 * {D}`
- Step3:  `{1P, 1D, 5Fv1}`, `5 * {P}` , `2 * {D}`

Since FrontEnd is a standalone PCLQ no new PodGangs are created which effectively means that the minimum viable unit to update (in this case 2 units of Frontend) are not gang scheduled but remain part of the original podgang.

>  Same principle applies to Prefill and Decode components if they are also standalone PodCliques.



Case #2: Only Prefill is updated

A new `PodCliqueVersion` will be created for Prefill worker and leader. Let's call these as:  `PLv1`. `PWv1` (collectively called `Pv1`). 
At the start of the update state of PodGangs is: `{1P, 1D, 5F}, 5 * {P} , 2 * {D}`
Since only the Prefill PCLQs gets updated, it is assumed that this is a backward compatible update.
PodGangs:
- Step1: `{1D, 5F}, 5 * {P} , 2 * {D}`, `1 * {Pv1}`
- Step2:  `{1D, 5F}, 4 * {P} , 2 * {D}`, `2 * {Pv1}`
- Step3:  `{1D, 5F}, 3 * {P} , 2 * {D}`, `3 * {Pv1}`
- Step4: `{1D, 5F}, 2 * {P} , 2 * {D}`, `4 * {Pv1}`
- Step5: `{1D, 5F}, 1 * {P}, 2 * {D}`, `5 * {Pv1}`
- Step6: `{1D, 5F}, 2 * {D}`, `6 * {Pv1}`



Case #3: Prefill and Decodes are updated

A new `PodCliqueVersion` will be created for Prefill and Decode. Let's call this `Pv1`, `Dv1`.
At the start of the update state of PodGangs is: `{1P, 1D, 5F}, 5 * {P} , 2 * {D}`

PodGangs:

- Initial: `{1P, 1D, 5F}`, `5 * {P}` , `2 * {D}`

- Step1: `{1P, 1D, 5F}`, `5 * {P}` , `2 * {D}`
- Step2: `{1P, 1D, 5F}`, `3 * {P}` , `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`
- Step3: `{1P, 5F}`, `3 * {P}` , `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, 
`{4F}`, `2 * {P}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1}`
`{4F}`, `1 * {P}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1}`, `{1Pv1}`
`{4F}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1, 1Dv1}`, `{1Pv1}`, `{1Pv1}`, `{1Pv1}`



Case #4: All PodCliques are updated

A new `PodCliqueVersion` will be created for Prefill and Decode. Let's call this `Pv1`, `Dv1` and `Fv1`.
At the start of the update state of PodGangs is: `{1P, 1D, 5F}, 5 * {P} , 2 * {D}`

- `{1P, 1D, 5F}`, `5 * {P}` , `2 * {D}`
- `{3F}`, `5 * {P}` , `2 * {D}`, `{1Pv1, 1Dv1, 2Fv1}`
- `{1F}`, `4 * {P}` , `1 * {D}`, `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 2Fv1}`
- `3 * {P}` , `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 1Fv1}`
- `2 * {P}` , `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 1Fv1}`, `{1Pv1}`
- `1 * {P}` , `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 1Fv1}`, `{1Pv1}`, `{1Pv1}`
- `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 2Fv1}`, `{1Pv1, 1Dv1, 1Fv1}`, `{1Pv1}`, `{1Pv1}`, `{1Pv1}`

### PCS rollback and rollforward

Currently `PodCliqueSet` does not maintain its current revision and history of revisions  and its child resources (specifically `PodClique`)



```go
// PodCliqueVersion captures the 
type PodCliqueVersion struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              PodCliqueVersionSpec `json:"spec"`
}

type PodCliqueVersionSpec struct {
    PodSpec  corev1.PodSpec `json:"podSpec"`
}
```

naming convention: `<pcs>-<pclq>-v<revision>`
revision label: `grove.io/revision`

#### Sample 2
```
PCS:
pclq-av1, pclq-bv1, pclq-cv1
pcsg-x:
pclq-b, pclq-c
revision: 1 (all pclqs are at revision 1)
maxRevision: 1

only pclq-a gets updated.

pcs revision: 2
pcs maxRevision: 2
revision-history: [1, 1, 1], [2, 1, 1]
pclq-av1: revision: 1
pclq-av2: revision: 2 <- active
pclq-bv1: revision: 1 <- active
pclq-cv1: revision: 1 <- active

now pclq-b gets updated (compatible update)

pcs revision: 3
pcs maxRevision: 3
revision-history: [1, 1, 1],[2, 1, 1], [2, 3, 1],
pclq-av1: revision: 1
pclq-av2: revision: 2 <- active
pclq-bv1: revision: 1
pclq-bv2: revision: 3  <- active
pclq-cv1: revision: 1

now pclq-c gets updated (compatible update)

pcs revision: 4
pcs maxRevision: 4
revision-history: [1, 1, 1],[2, 1, 1], [2, 3, 1], [2, 3, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2 <- active
pclq-bv1: revision: 1
pclq-bv2: revision: 3 <- active
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <-active

now pclq-b gets updated (compatible update)

pcs revision: 5
pcs maxRevision: 5
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2 <- active
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5  <- active
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <-active

only pclq-a gets updated.

pcs revision: 6
pcs maxRevision: 6
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2
pclq-av3: revision: 6 <- active
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5  <- active
pclq-cv1: revision: 1
pclq-cv2: revision: 4<-active

all pclqs get updated

pcs revision: 7
pcs maxRevision: 7
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7]
pclq-av1: revision: 1
pclq-av2: revision: 2
pclq-av3: revision: 6
pclq-av4: revision: 7 <- active
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5
pclq-bv3: revision: 7  <- active
pclq-cv1: revision: 1
pclq-cv2: revision: 4
pclq-cv2: revision: 7 <-active

rollout undo --to-revision=5

pcs revision: 5
pcs maxRevision: 7
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7]
pclq-av1: revision: 1
pclq-av2: revision: 2 <-active
pclq-av3: revision: 6
pclq-av4: revision: 7
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5 <-active
pclq-bv3: revision: 7
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <- active
pclq-cv2: revision: 7

rollout undo --to-revision=4

pcs revision: 4
pcs maxRevision: 7
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7]
pclq-av1: revision: 1
pclq-av2: revision: 2 <-active
pclq-av3: revision: 6
pclq-av4: revision: 7
pclq-bv1: revision: 1
pclq-bv2: revision: 3 <-active
pclq-bv3: revision: 5
pclq-bv3: revision: 7
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <- active
pclq-cv2: revision: 7

roll forward to revision 6

pcs revision: 6
pcs maxRevision: 7
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7]
pclq-av1: revision: 1
pclq-av2: revision: 2
pclq-av3: revision: 6 <-active
pclq-av4: revision: 7
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5 <-active
pclq-bv3: revision: 7
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <- active
pclq-cv2: revision: 7

update pclq-a

pcs revision: 8
pcs maxRevision: 8
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7], [8, 5, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2
pclq-av3: revision: 6
pclq-av4: revision: 7
pclq-av4: revision: 8 <-active
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5 <-active
pclq-bv3: revision: 7
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <- active
pclq-cv2: revision: 7

rollback to revision 3

pcs revision: 3
pcs maxRevision: 8
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7], [8, 5, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2 <-active
pclq-av3: revision: 6
pclq-av4: revision: 7
pclq-av4: revision: 8
pclq-bv1: revision: 1
pclq-bv2: revision: 3 <-active
pclq-bv3: revision: 5
pclq-bv3: revision: 7
pclq-cv1: revision: 1<-active
pclq-cv2: revision: 4
pclq-cv2: revision: 7
pclq-c: current-revision: 1, revision-history: 7, 4, 1

rollforward to revision 8

pcs revision: 8
pcs maxRevision: 8
revision-history: [1, 1, 1], [2, 1, 1],[2, 3, 1],[2, 3, 4], [2, 5, 4], [6, 5, 4], [7, 7, 7], [8, 5, 4]
pclq-av1: revision: 1
pclq-av2: revision: 2
pclq-av3: revision: 6
pclq-av4: revision: 7
pclq-av4: revision: 8 <-active
pclq-bv1: revision: 1
pclq-bv2: revision: 3
pclq-bv3: revision: 5 <-active
pclq-bv3: revision: 7
pclq-cv1: revision: 1
pclq-cv2: revision: 4 <-active
pclq-cv2: revision: 7
```

### Update concurrency

### Handling scale-outs and scale-ins during update

### Status and observability

```go
// PodCliqueSetStatus defines the status of a PodCliqueSet.
type PodCliqueSetStatus struct {
    // ObservedGeneration is the most recent generation observed by the controller.
    ObservedGeneration *int64 `json:"observedGeneration,omitempty"`
    ...
    RevisionHistory []PodCliqueSetRevision
}

type PodCliqueSetRevision struct {
    PodCliqueSetRevision int32
    PodCliquRevisions []int32
}
```

Add a new label(`grove.io/podcliqueset-revision`) on `PodCliqueSet` which represents the current revision.

```go
type PodCliqueVersion struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              PodCliqueVersionSpec `json:"spec"`
}

type PodCliqueVersionSpec struct {
    Replicas int32          `json:"replicas"`
    PodSpec  corev1.PodSpec `json:"podSpec"`
}

type PodCliqueVersionStatus struct {
    // ObservedGeneration is the most recent generation observed by the controller.
    ObservedGeneration *int64 `json:"observedGeneration,omitempty"`
    // Conditions represents the latest available observations of the clique by its controller.
    Conditions []metav1.Condition `json:"conditions,omitempty"`
    // Replicas is the total number of non-terminated Pods targeted by this PodClique.
    Replicas int32 `json:"replicas,omitempty"`
    // ReadyReplicas is the number of ready Pods targeted by this PodClique.
    // +kubebuilder:default=0
    ReadyReplicas int32 `json:"readyReplicas"`
    // UpdatedReplicas is the number of Pods that have been updated and are at the desired revision of the PodClique.
    // +kubebuilder:default=0
    UpdatedReplicas int32 `json:"updatedReplicas"`
    // ScheduleGatedReplicas is the number of Pods that have been created with one or more scheduling gate(s) set.
    // Sum of ReadyReplicas and ScheduleGatedReplicas will always be <= Replicas.
    // +kubebuilder:default=0
    ScheduleGatedReplicas int32 `json:"scheduleGatedReplicas"`
    // ScheduledReplicas is the number of Pods that have been scheduled by the kube-scheduler.
    // +kubebuilder:default=0
    ScheduledReplicas int32  `json:"scheduledReplicas"`
}
```

### Monitoring

<!--
This section contains details of events, metrics, status conditions and other status fields that will aid in determining health of the feature, or help measure any service level objectives that might be optionally defined.
-->

### Dependencies (*Optional*)

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

