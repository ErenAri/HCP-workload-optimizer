use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Deterministic FIFO simulation runner (baseline contract scaffold)"
)]
struct Args {
    /// Input file containing a JSON array of canonical job records.
    #[arg(long)]
    input: PathBuf,

    /// Total CPU capacity of the simulated cluster.
    #[arg(long, default_value_t = 64)]
    capacity_cpus: u32,

    /// When enabled, invariant violations fail the run immediately.
    #[arg(long, default_value_t = false)]
    strict_invariants: bool,

    /// Optional output report path. If omitted, prints JSON report to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct Job {
    job_id: u64,
    submit_ts: i64,
    runtime_actual_sec: i64,
    requested_cpus: u32,
}

#[derive(Debug, Clone)]
struct RunningJob {
    job_id: u64,
    submit_ts: i64,
    start_ts: i64,
    end_ts: i64,
    requested_cpus: u32,
}

#[derive(Debug, Serialize)]
struct Metrics {
    policy: String,
    capacity_cpus: u32,
    jobs_total: usize,
    jobs_completed: usize,
    mean_wait_sec: f64,
    p95_wait_sec: f64,
    makespan_sec: i64,
    utilization_cpu: f64,
    invariant_violations: usize,
}

#[derive(Debug, Serialize)]
struct SimulationReport {
    metrics: Metrics,
    violations: Vec<String>,
}

fn percentile(sorted: &[i64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx] as f64
}

fn check_invariants(
    clock_ts: i64,
    capacity: u32,
    free_cpus: u32,
    running: &[RunningJob],
    queued: &VecDeque<Job>,
) -> Vec<String> {
    let mut violations = Vec::new();
    if free_cpus > capacity {
        violations.push("free_cpus_exceeds_capacity".to_string());
    }

    let running_cpus: u32 = running.iter().map(|job| job.requested_cpus).fold(0u32, |a, b| a.saturating_add(b));
    if running_cpus.saturating_add(free_cpus) != capacity {
        violations.push("cpu_conservation_broken".to_string());
    }

    for job in running {
        if job.start_ts < job.submit_ts {
            violations.push(format!("job_start_before_submit:{}", job.job_id));
        }
        if job.end_ts < job.start_ts {
            violations.push(format!("job_end_before_start:{}", job.job_id));
        }
        if job.requested_cpus == 0 {
            violations.push(format!("job_zero_cpu_request:{}", job.job_id));
        }
    }

    for job in queued {
        if job.submit_ts > clock_ts {
            violations.push(format!("queued_job_submit_in_future:{}", job.job_id));
        }
    }

    violations
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.capacity_cpus == 0 {
        anyhow::bail!("capacity_cpus must be > 0");
    }

    let input_raw = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let mut jobs: Vec<Job> = serde_json::from_str(&input_raw)
        .with_context(|| "failed to parse input JSON array of jobs")?;

    jobs.sort_by(|a, b| a.submit_ts.cmp(&b.submit_ts).then_with(|| a.job_id.cmp(&b.job_id)));

    let total_jobs = jobs.len();
    let mut submit_idx = 0usize;
    let mut queued: VecDeque<Job> = VecDeque::new();
    let mut running: Vec<RunningJob> = Vec::new();
    let mut free_cpus = args.capacity_cpus;
    let mut clock_ts = jobs.first().map(|job| job.submit_ts).unwrap_or(0);

    let min_submit_ts = clock_ts;
    let mut max_end_ts = min_submit_ts;
    let mut total_cpu_seconds: i128 = 0;
    let mut waits: Vec<i64> = Vec::new();
    let mut completed_jobs = 0usize;
    let mut all_violations: Vec<String> = Vec::new();

    while completed_jobs < total_jobs {
        let next_submit_ts = jobs
            .get(submit_idx)
            .map(|job| job.submit_ts)
            .unwrap_or(i64::MAX);
        let next_complete_ts = running.iter().map(|job| job.end_ts).min().unwrap_or(i64::MAX);

        if next_submit_ts == i64::MAX && next_complete_ts == i64::MAX {
            break;
        }

        clock_ts = next_submit_ts.min(next_complete_ts);

        // Deterministic tie order at same timestamp: complete before submit.
        let completed_now: Vec<RunningJob> = running
            .iter()
            .filter(|job| job.end_ts == clock_ts)
            .cloned()
            .collect();
        if !completed_now.is_empty() {
            let ids: HashSet<u64> = completed_now.iter().map(|job| job.job_id).collect();
            running.retain(|job| !ids.contains(&job.job_id));
            for job in completed_now {
                free_cpus = free_cpus.saturating_add(job.requested_cpus);
                completed_jobs += 1;
            }
        }

        while let Some(job) = jobs.get(submit_idx) {
            if job.submit_ts != clock_ts {
                break;
            }
            queued.push_back(job.clone());
            submit_idx += 1;
        }

        // FIFO dispatch: head-of-line only; no backfill in this baseline.
        loop {
            let can_dispatch = if let Some(head) = queued.front() {
                head.requested_cpus > 0 && head.requested_cpus <= free_cpus
            } else {
                false
            };
            if !can_dispatch {
                break;
            }

            let job = queued.pop_front().expect("queue head exists");
            let runtime = job.runtime_actual_sec.max(0);
            let start_ts = clock_ts;
            let end_ts = start_ts.saturating_add(runtime);
            let wait_sec = start_ts.saturating_sub(job.submit_ts);

            free_cpus -= job.requested_cpus;
            waits.push(wait_sec.max(0));
            max_end_ts = max_end_ts.max(end_ts);
            total_cpu_seconds += i128::from(job.requested_cpus) * i128::from(runtime);

            running.push(RunningJob {
                job_id: job.job_id,
                submit_ts: job.submit_ts,
                start_ts,
                end_ts,
                requested_cpus: job.requested_cpus,
            });
        }

        let violations = check_invariants(clock_ts, args.capacity_cpus, free_cpus, &running, &queued);
        if !violations.is_empty() {
            all_violations.extend(violations.iter().cloned());
            if args.strict_invariants {
                anyhow::bail!(
                    "strict invariants failed at ts={clock_ts}: {}",
                    violations.join(",")
                );
            }
        }
    }

    waits.sort_unstable();
    let mean_wait = if waits.is_empty() {
        0.0
    } else {
        let sum: i64 = waits.iter().copied().fold(0i64, |a, b| a.saturating_add(b));
        sum as f64 / waits.len() as f64
    };
    let p95_wait = percentile(&waits, 0.95);
    let makespan = max_end_ts.saturating_sub(min_submit_ts).max(0);
    let utilization_cpu = if makespan > 0 {
        let denom = (args.capacity_cpus as f64) * (makespan as f64);
        (total_cpu_seconds as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let report = SimulationReport {
        metrics: Metrics {
            policy: "FIFO_BASELINE".to_string(),
            capacity_cpus: args.capacity_cpus,
            jobs_total: total_jobs,
            jobs_completed: completed_jobs,
            mean_wait_sec: mean_wait,
            p95_wait_sec: p95_wait,
            makespan_sec: makespan,
            utilization_cpu,
            invariant_violations: all_violations.len(),
        },
        violations: all_violations,
    };

    let output = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        fs::write(&path, output)
            .with_context(|| format!("failed to write report {}", path.display()))?;
    } else {
        println!("{output}");
    }

    Ok(())
}
