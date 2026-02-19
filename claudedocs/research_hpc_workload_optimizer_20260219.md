# HPC Workload Optimizer: Deep Research Report

**Date:** 2026-02-19
**Confidence Level:** High (based on peer-reviewed research, production traces, and active development in the field)

---

## Executive Summary

This research validates the HPC Workload Optimizer concept as industrially viable. The gap between rule-based schedulers (Slurm, PBS, LSF) and intelligent, learning-based scheduling is well-documented in both academia and industry. Public datasets are mature and accessible. ML approaches for runtime prediction achieve 70-86% accuracy in production studies. Simulation frameworks like Batsim enable credible policy evaluation. Recent research (2024-2025) shows active development in RL-based scheduling, GPU fragmentation optimization, and uncertainty quantification.

**Key Findings:**
1. User-provided runtime estimates are wildly inaccurate: 35% of jobs use <10% of requested time
2. GPU cluster utilization hovers around 50% due to fragmentation and static allocation
3. ML-based backfilling improvements: up to 59% better scheduling performance vs EASY backfill
4. Slurm plugin integration architecture exists and has been demonstrated in production

---

## 1. The Industrial Problem: Documented Evidence

### 1.1 Scheduler Limitations

Current HPC schedulers have fundamental constraints that ML can address:

| Limitation | Evidence |
|------------|----------|
| **Resource Fragmentation** | Jobs reserve entire nodes/GPUs; a job needing 3 GPUs wastes 1 GPU on a 4-GPU node |
| **Static Scaling** | No concept of scaling to zero when idle; manual admin intervention for capacity changes |
| **No Native DL Support** | Distributed training requires glue code; no hyperparameter tuning or experiment management |
| **User Experience** | CLI and bash scripts; no Pythonic interface for ML engineers |
| **Inference Limitations** | Slurm is effective for training but not suited for inference workloads |

**Source:** [WhiteFiber - Understanding Slurm for AI/ML Workloads](https://www.whitefiber.com/blog/understanding-slurm-for-ai-ml-workloads), [Determined AI - Why Slurm Makes Deep Learning Engineers Squirm](https://www.determined.ai/blog/slurm-lacking-deep-learning)

### 1.2 User Estimate Inaccuracy

This is the core problem your optimizer solves:

> "For requested runtimes greater than one minute, **35% of jobs use less than 10% of their requested runtime** and another 15% complete within 10-30% of the requested runtime."

**Source:** [NREL - Mastering HPC Runtime Prediction](https://docs.nrel.gov/docs/fy23osti/86526.pdf)

### 1.3 GPU Cluster Utilization Crisis

> "GPU clusters have become essential for training and deploying modern AI systems, yet real deployments continue to report **average utilization near 50%**. This inefficiency is largely caused by fragmentation, heterogeneous workloads, and the limitations of static scheduling policies."

**Source:** [arXiv - Reducing Fragmentation and Starvation in GPU Clusters](https://arxiv.org/html/2512.10980v1)

### 1.4 Memory Underutilization

Analysis of NERSC's Perlmutter system:
- **64% of jobs** used 50% or less of available host memory
- **50% of GPU-enabled jobs** used only up to 25% of GPU memory
- CPUs commonly not fully utilized, especially for GPU-enabled jobs

**Source:** [Springer - Analyzing Resource Utilization in an HPC System: NERSC Perlmutter](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_16)

---

## 2. Available Datasets: Production-Grade Traces

### 2.1 Parallel Workloads Archive (PWA)

**URL:** https://www.cs.huji.ac.il/labs/parallel/workload/logs.html

The canonical resource for HPC scheduling research. Contains production logs from supercomputers worldwide.

**Standard Workload Format (SWF) Schema (18 fields):**

| Field | Description |
|-------|-------------|
| Job Number | Unique job identifier |
| Submit Time | Seconds from trace start |
| Wait Time | Time in queue before starting |
| Run Time | Actual execution time |
| Number of Allocated Processors | CPUs assigned |
| Average CPU Time Used | Per processor |
| Used Memory | Per processor |
| Requested Number of Processors | User estimate |
| Requested Time | User estimate (wallclock limit) |
| Requested Memory | User estimate |
| Status | Completion status (1=completed, 0=failed, 5=cancelled) |
| User ID | Anonymized user identifier |
| Group ID | Anonymized group identifier |
| Executable Number | Application type |
| Queue Number | Submission queue |
| Partition Number | Cluster partition |
| Preceding Job Number | Dependency chain |
| Think Time from Preceding Job | Time between dependent jobs |

**Data Quality:** Cleaned versions available that remove administrative activity and anomalous user behavior.

**Source:** [PWA - Standard Workload Format](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)

### 2.2 Alibaba Cluster Trace v2018

**URL:** https://github.com/alibaba/clusterdata

**Scale:** ~4,000 machines, 8 days, 270+ GB uncompressed

**Tables:**
1. `machine_meta.csv` - Machine metadata and events
2. `machine_usage.csv` - Resource consumption metrics
3. `container_meta.csv` - Container metadata
4. `container_usage.csv` - Container utilization
5. `batch_instance.csv` - Batch workload instances
6. `batch_task.csv` - Batch tasks with DAG structure

**Key Feature:** Task dependencies encoded in `task_name` field (e.g., M5_3_4 means task5 depends on tasks 3 and 4)

**GPU-Specific Traces Also Available:**
- **v2020:** 6,500+ GPUs on ~1,800 machines, 2 months, MLaaS workloads
- **v2023:** 6,200+ GPUs on ~1,200 machines, heterogeneous cluster
- **v2025:** 150+ DLRM inference services, 20k+ instances

**Source:** [Alibaba GitHub - Cluster Trace v2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)

### 2.3 Google Cluster Workload Traces 2019

**URL:** https://github.com/google/cluster-data

**Scale:** 8 clusters worldwide, 12,000 machines each, entire month of May 2019, 2.4 TiB total

**Access:** Available via Google BigQuery for sophisticated analyses

**Research Value:** Canonical benchmark for workload modeling, scheduling policy design, and infrastructure management research.

**Source:** [Google Research - Cluster Workload Traces 2019](https://research.google/resources/datasets/google-cluster-workload-traces-2019/)

### 2.4 Grid Workload Archive

**URL:** https://atlarge-research.com/gwa.html

Complementary traces for grid-style distributed computing workloads.

---

## 3. ML Approaches: State of the Art (2024-2025)

### 3.1 Runtime Prediction Methods

| Method | Description | Performance |
|--------|-------------|-------------|
| **ORA (2025)** | Online Retrieval-Augmented LM; encodes job metadata + scripts into vectors; similarity-based retrieval | Handles distribution shift without retraining |
| **ML + Genetic Algorithm (2024)** | KNN, SVR, XGBoost, DNN with GA optimization | Universal applicability across HPC systems |
| **PC-Transformer (2023)** | Transformer architecture for job sequences | Published in J. Supercomputing |
| **BOSER Plugin (2025)** | Stacking ensemble (LightGBM, XGBoost, CatBoost, AdaBoost, RF) with Bayesian-tuned ElasticNet | 86% accuracy in production |

**Critical Methodological Note:**
> "Time dependency necessitates a **time-aware train-test splitting process**. Most studies split the dataset based on submit time... However, five of the eleven most recent works use a time-agnostic, random train-test splitting process. The primary flaw of this approach is **using future jobs to make predictions for past jobs**."

**Source:** [NREL - Mastering HPC Runtime Prediction](https://docs.nrel.gov/docs/fy23osti/86526.pdf)

### 3.2 Uncertainty Quantification

**Current Gap:** Only 1 out of 14 existing studies on job queue time prediction investigates uncertainty aspects.

**Approach:** Variance estimation to provide upper/lower bounds at specified confidence levels.

**Your Opportunity:** Quantile regression for uncertainty (as you proposed) is underexplored in the literature.

**Source:** [ACM - Quantifying Uncertainty in HPC Job Queue Time Predictions](https://dl.acm.org/doi/10.1145/3626203.3670627)

### 3.3 Feature Engineering for User Behavior

Key features for runtime prediction:
- **User-based filtering:** ORA method uses `user_id` to retrieve similar historical jobs
- **Historical runtime per user**
- **Runtime variance per user/job type**
- **Queue congestion at submission time**
- **Time-of-day effects**
- **Job size class**

**Source:** [ORA Paper - ICS 2025](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-18.pdf)

---

## 4. Backfilling Optimization: Quantified Improvements

### 4.1 Baseline: EASY Backfilling

Standard algorithm that allows lower-priority jobs to run ahead if they don't delay queued jobs.

### 4.2 ML-Enhanced Backfilling Results

| Approach | Improvement over EASY Backfill |
|----------|-------------------------------|
| **Classical ML + new cost functions** | 28% better avg bounded slowdown |
| **Reinforcement Learning (RLBackfilling)** | 59% better with ML-predicted runtime; 30% better with ideal predicted runtime |
| **Regression-Classification Parallel** | 50.6% reduction in avg bounded slowdown; 6.63% reduction in avg wait time |
| **Parallel Backfill** | Increased scheduling throughput via parallel queue processing |

**Source:** [arXiv - A Reinforcement Learning Based Backfilling Strategy for HPC Batch Jobs](https://arxiv.org/html/2404.09264v1), [ResearchGate - Improving backfilling by using machine learning](https://www.researchgate.net/publication/310821188_Improving_backfilling_by_using_machine_learning_to_predict_running_times)

### 4.3 Reinforcement Learning Architecture

RLBackfilling uses:
- Deep neural networks with actor-critic model
- Proximal Policy Optimization (PPO) from OpenAI
- Policy network takes observation as input, outputs action for next job

**Source:** [RLBackfilling Paper](https://webpages.charlotte.edu/ddai/data/RLBackfilling.pdf)

---

## 5. GPU-Aware Scheduling: Cutting-Edge Research

### 5.1 Fragmentation Gradient Descent (FGD)

Schedules workloads "towards the direction of steepest descent of fragmentation."

**Results:** Implemented in Kubernetes; evaluated on 6,200+ GPU cluster; **reduces unallocated GPUs by up to 49%** (290 additional GPUs utilized).

**Source:** [USENIX ATC'23 - Beware of Fragmentation](https://www.usenix.org/conference/atc23/presentation/weng)

### 5.2 Dynamic Multi-Objective Schedulers

| Scheduler | Utilization | Throughput | Fairness |
|-----------|-------------|------------|----------|
| **HPS (Hybrid Priority)** | 78.2% | 25.8 jobs/hr | Lowest variance (457), 12 starved jobs |
| **PBS (Predictive Backfill)** | 76.1% | - | Improved fragmentation handling |
| **SBS (Smart Batch)** | 74.6% | - | Better for similar jobs |

**Source:** [arXiv - Reducing Fragmentation and Starvation in GPU Clusters](https://arxiv.org/html/2512.10980v1)

### 5.3 MIG Fragmentation Challenges

- **External fragmentation:** Scattered partial GPU resources insufficient for incoming jobs
- **Internal fragmentation:** MIG instance exceeds job requirements; surplus cannot be reallocated

**Source:** [arXiv - Power- and Fragmentation-aware Online Scheduling](https://arxiv.org/pdf/2412.17484)

### 5.4 Hybrid RL + Optimization (RLTune)

Couples RL-based dynamic prioritization with Mixed-Integer Linear Programming (MILP) for resource allocation.

**Features:** Job-level and system-level signals (user metadata, resource availability, queue state) as engineered features.

**Source:** [arXiv - Hybrid Learning and Optimization-Based Dynamic Scheduling](https://arxiv.org/html/2512.10271)

---

## 6. Simulation Frameworks for Policy Evaluation

### 6.1 Batsim

**URL:** https://batsim.org/

**Description:** Scientific simulator for batch schedulers built on SimGrid.

**Key Capabilities:**
- Language-independent via event-based communication interface
- Multiple realism levels (abstract rectangles to computation/communication/IO patterns)
- Reproducible experiments with detailed outputs
- Used to evaluate backfill strategies achieving **50.6% improvement** over EASY

**Source:** [ResearchGate - Batsim: A Realistic Language-Independent RJMS Simulator](https://www.researchgate.net/publication/318356924_Batsim_A_Realistic_Language-Independent_Resources_and_Jobs_Management_Systems_Simulator)

### 6.2 SimGrid

Foundation for Batsim. Models distributed systems including computation, communication, and I/O patterns.

### 6.3 Other Options

- **Alea** - Job scheduling simulator
- **AccaSim** - Customizable workload management simulator
- **Built-in simulators** - Torque/Maui, Moab Scheduler

---

## 7. Slurm Integration Architecture

### 7.1 Plugin System

Slurm supports dynamically linked plugins for customized implementations.

**Relevant Plugin Types:**
- Job submission plugins (intercept job submissions)
- Job completion plugins (capture outcomes)
- Scheduling plugins (influence scheduling decisions)

### 7.2 BOSER Implementation (2025)

**Architecture:**
1. **submission_plugin:** Intercepts job submissions, extracts parameters, triggers ML prediction
2. **completion_plugin:** Captures actual outcomes for model retraining
3. **ml_resource_handler:** Invokes ML models to predict runtime
4. **REST API:** Flask-based interface for predictions

**Implementation:** Rust for efficiency and concurrency

**Data Source:** Historical data from SlurmDB or .log files

**Source:** [PCT'2025 - A Machine Learning-Based Plugin for SLURM](https://xn--80ae1bo.xn--p1ai/2025/talks/Mahdi.pdf)

### 7.3 Production Results

> "Models achieved up to **86% accuracy** in predicting time and memory. Results show dramatic reductions in average waiting time (**from 380 to 4 hours** in RMACC Summit and **from 662 hours to 28 hours** in Beocat) and achieved up to **100% utilization**."

**Source:** [PMC - Ensemble Prediction of Job Resources](https://pmc.ncbi.nlm.nih.gov/articles/PMC8974354/)

---

## 8. Fair-Share Scheduling: Balancing Efficiency and Fairness

### 8.1 The Tradeoff

> "Fair sharing and high utilization are conflicting goals and aggressively using fair sharing has a negative effect on resource utilization."

**Source:** [arXiv - Job Scheduling in High Performance Computing](https://arxiv.org/pdf/2109.09269)

### 8.2 HeraSched: Hierarchical RL Scheduler

**Results:** Consistently outperforms SJF, Topology-Aware, and Best-Fit algorithms; effective in high-stress scenarios not part of training set.

**Key Innovation:** Addresses both job selection AND allocation (most RL schedulers only handle selection).

**Source:** [Springer - Optimizing HPC scheduling: a hierarchical reinforcement learning approach](https://link.springer.com/article/10.1007/s11227-025-07396-3)

### 8.3 ASA: RL-Based Co-Scheduler

**Results:** Improves cluster CPU utilization by up to **51%** while reducing response time and queue waiting times.

**Approach:** Embeds application performance, users' fair-share priorities, and cluster capacity directly into the model.

**Source:** [arXiv - A HPC Co-Scheduler with Reinforcement Learning](https://arxiv.org/html/2401.09706v1)

---

## 9. Recommended Architecture for Your Project

Based on this research, here's a validated technical approach:

### Layer 1: Data Pipeline
```
Trace Sources → Parser → Feature Store
├── PWA/SWF traces (standardized format)
├── Alibaba traces (DAG-aware)
├── Google traces (BigQuery)
└── Live Slurm logs (SlurmDB integration)
```

**Features to Extract:**
- User historical runtime and variance
- Job size class
- Queue congestion at submission
- Time-of-day effects
- DAG depth and parallelism (for Alibaba traces)

### Layer 2: ML Models

**Recommended Stack:**
1. **Runtime Prediction:** LightGBM with quantile regression for uncertainty
2. **Resource Over-Request Prediction:** XGBoost classifier
3. **Queue Wait Time:** Survival models or gradient boosting

**Critical:** Use time-aware train-test splits. Never use future jobs to predict past jobs.

### Layer 3: Simulation Engine

**Tool:** Batsim on SimGrid

**Approach:**
1. Replay historical trace
2. Replace user estimates with predicted values
3. Apply alternative scheduling policies (EASY, FGD, custom)
4. Measure: utilization %, avg wait time, throughput, fairness deviation

### Layer 4: Recommendation Engine

**Output Examples:**
- "Reduce GPU request for user group X by 18%"
- "Increase backfill window to Y"
- "Split queue into short/long job classes based on predicted runtime"
- "Adjust fair-share weights for Z"

---

## 10. Risk Factors and Mitigations

| Risk | Mitigation |
|------|------------|
| **Distribution Shift** | ORA-style online updating without retraining; drift detection |
| **User Behavior Variance** | Per-user/per-job-type models; user clustering |
| **Scheduler Politics** | Export recommendations to config preview, not auto-apply |
| **Privacy Concerns** | Anonymization, hashing, on-prem deployment options |
| **Adoption Resistance** | Simulation proves value before integration; incremental rollout |

---

## 11. Phased Execution Plan

### Phase 1: Foundation (30-day MVP)
**Deliverables:**
- PWA trace parser + feature pipeline
- Baseline runtime predictor (LightGBM)
- Replay simulator producing "utilization delta" under policy changes
- Minimal web dashboard

### Phase 2: Intelligence (90-day MVP)
**Deliverables:**
- Uncertainty-aware scheduling recommendations (quantile regression)
- Per-user/job-class models
- Fairness constraints integration
- Scheduler config preview export (YAML-like)

### Phase 3: Production Integration
**Deliverables:**
- Slurm plugin (submission/completion hooks)
- Real-time prediction API
- Model drift detection
- A/B testing framework for policy changes

### Phase 4: Advanced Features
**Deliverables:**
- GPU topology awareness
- Energy cost modeling
- RL-based scheduling agent
- Explainable scheduling decisions

---

## 12. Competitive Landscape

| Solution | Approach | Gap You Can Fill |
|----------|----------|------------------|
| **SchedMD Slurm** | Rule-based | No ML, no learning |
| **Run.ai** | Kubernetes-native | Not HPC-focused |
| **Determined AI** | DL training platform | Not a scheduler optimizer |
| **Academic Research** | Papers, not products | No productized solution |

**Your Positioning:** Intelligent decision layer that sits alongside existing schedulers, not replacing them.

---

## Sources

### Scheduler Limitations
- [WhiteFiber - Understanding Slurm for AI/ML Workloads](https://www.whitefiber.com/blog/understanding-slurm-for-ai-ml-workloads)
- [Determined AI - Why Slurm Makes Deep Learning Engineers Squirm](https://www.determined.ai/blog/slurm-lacking-deep-learning)
- [Nebius - Slurm Workload Manager](https://nebius.com/blog/posts/slurm-workload-manager)

### Datasets
- [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/)
- [Alibaba Cluster Data](https://github.com/alibaba/clusterdata)
- [Google Cluster Data](https://github.com/google/cluster-data)
- [Grid Workload Archive](https://atlarge-research.com/gwa.html)

### ML Approaches
- [NREL - Mastering HPC Runtime Prediction](https://docs.nrel.gov/docs/fy23osti/86526.pdf)
- [ORA Paper - ICS 2025](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-18.pdf)
- [ACM - Quantifying Uncertainty in HPC Job Queue Time Predictions](https://dl.acm.org/doi/10.1145/3626203.3670627)
- [ScienceDirect - ML + Genetic Algorithm for HPC Jobs](https://www.sciencedirect.com/science/article/abs/pii/S1568494624008275)

### Backfilling
- [arXiv - RL Based Backfilling Strategy](https://arxiv.org/html/2404.09264v1)
- [ACM - Parallel Backfill](https://dl.acm.org/doi/fullHtml/10.1145/3626203.3670610)
- [HAL - Improving Backfilling with ML](https://hal.science/hal-01221186v1/document)

### GPU Scheduling
- [arXiv - Reducing Fragmentation in GPU Clusters](https://arxiv.org/html/2512.10980v1)
- [USENIX ATC'23 - Fragmentation Gradient Descent](https://www.usenix.org/conference/atc23/presentation/weng)
- [arXiv - Power- and Fragmentation-aware Scheduling](https://arxiv.org/pdf/2412.17484)

### Simulation
- [Batsim](https://batsim.org/)
- [ResearchGate - Batsim Paper](https://www.researchgate.net/publication/318356924_Batsim_A_Realistic_Language-Independent_Resources_and_Jobs_Management_Systems_Simulator)

### Slurm Integration
- [PCT'2025 - ML-Based Plugin for SLURM](https://xn--80ae1bo.xn--p1ai/2025/talks/Mahdi.pdf)
- [PMC - Ensemble Prediction of Job Resources](https://pmc.ncbi.nlm.nih.gov/articles/PMC8974354/)
- [Slurm Plugin API](https://slurm.schedmd.com/plugins.html)

### Fair-Share and RL Scheduling
- [arXiv - Job Scheduling in HPC](https://arxiv.org/pdf/2109.09269)
- [Springer - HeraSched](https://link.springer.com/article/10.1007/s11227-025-07396-3)
- [arXiv - ASA Co-Scheduler](https://arxiv.org/html/2401.09706v1)

### Resource Utilization Analysis
- [Springer - NERSC Perlmutter Analysis](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_16)
- [arXiv - Perlmutter Analysis](https://arxiv.org/pdf/2301.05145)
