use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Adapter contract decision check (FIFO/EASY/ML backfill)"
)]
struct Args {
    /// Input JSON state snapshot payload.
    #[arg(long)]
    input: PathBuf,

    /// Policy id: FIFO_STRICT | EASY_BACKFILL_BASELINE | ML_BACKFILL_P50
    #[arg(long)]
    policy: String,

    /// Strict uncertainty mode for ML policy (uses p90 gate).
    #[arg(long, default_value_t = false)]
    strict_uncertainty_mode: bool,

    /// Optional output path. If omitted, print to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct QueuedJob {
    job_id: u64,
    submit_ts: i64,
    requested_cpus: u32,
    runtime_estimate_sec: i64,
    runtime_p90_sec: Option<i64>,
    runtime_guard_sec: Option<i64>,
    estimate_source: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RunningJob {
    job_id: u64,
    end_ts: i64,
    allocated_cpus: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct Snapshot {
    clock_ts: i64,
    capacity_cpus: u32,
    free_cpus: u32,
    queued_jobs: Vec<QueuedJob>,
    running_jobs: Vec<RunningJob>,
}

#[derive(Debug, Clone, Serialize)]
struct DispatchDecision {
    job_id: u64,
    requested_cpus: u32,
    runtime_estimate_sec: i64,
    estimated_completion_ts: i64,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DecisionReport {
    policy_id: String,
    reservation_ts: Option<i64>,
    decisions: Vec<DispatchDecision>,
}

fn normalize_snapshot(snapshot: &mut Snapshot) -> Result<()> {
    if snapshot.capacity_cpus == 0 {
        anyhow::bail!("capacity_cpus must be > 0");
    }
    if snapshot.free_cpus > snapshot.capacity_cpus {
        anyhow::bail!("free_cpus cannot exceed capacity_cpus");
    }

    snapshot
        .queued_jobs
        .sort_by(|a, b| a.submit_ts.cmp(&b.submit_ts).then_with(|| a.job_id.cmp(&b.job_id)));
    snapshot
        .running_jobs
        .sort_by(|a, b| a.end_ts.cmp(&b.end_ts).then_with(|| a.job_id.cmp(&b.job_id)));
    Ok(())
}

fn reservation_ts_for_hol(snapshot: &Snapshot, hol: &QueuedJob) -> i64 {
    if hol.requested_cpus > snapshot.capacity_cpus {
        return i64::MAX / 2;
    }
    if hol.requested_cpus <= snapshot.free_cpus {
        return snapshot.clock_ts;
    }
    let mut free = snapshot.free_cpus;
    for running in &snapshot.running_jobs {
        free += running.allocated_cpus;
        if free >= hol.requested_cpus {
            return snapshot.clock_ts.max(running.end_ts);
        }
    }
    i64::MAX / 2
}

fn choose_fifo(snapshot: &Snapshot) -> DecisionReport {
    let mut available = snapshot.free_cpus;
    let mut decisions = Vec::new();

    for job in &snapshot.queued_jobs {
        if job.requested_cpus == 0 {
            continue;
        }
        if job.requested_cpus <= available {
            let runtime = job.runtime_estimate_sec.max(0);
            decisions.push(DispatchDecision {
                job_id: job.job_id,
                requested_cpus: job.requested_cpus,
                runtime_estimate_sec: runtime,
                estimated_completion_ts: snapshot.clock_ts + runtime,
                reason: "fifo_dispatch".to_string(),
            });
            available -= job.requested_cpus;
        } else {
            break;
        }
    }

    DecisionReport {
        policy_id: "FIFO_STRICT".to_string(),
        reservation_ts: None,
        decisions,
    }
}

fn choose_easy(snapshot: &Snapshot) -> DecisionReport {
    if snapshot.queued_jobs.is_empty() {
        return DecisionReport {
            policy_id: "EASY_BACKFILL_BASELINE".to_string(),
            reservation_ts: None,
            decisions: Vec::new(),
        };
    }

    let queue = snapshot.queued_jobs.clone();
    let hol = &queue[0];
    let reservation_ts = reservation_ts_for_hol(snapshot, hol);
    let mut available = snapshot.free_cpus;
    let mut decisions = Vec::new();

    if hol.requested_cpus > 0 && hol.requested_cpus <= available {
        let runtime = hol.runtime_estimate_sec.max(0);
        decisions.push(DispatchDecision {
            job_id: hol.job_id,
            requested_cpus: hol.requested_cpus,
            runtime_estimate_sec: runtime,
            estimated_completion_ts: snapshot.clock_ts + runtime,
            reason: "easy_head_dispatch".to_string(),
        });
        available -= hol.requested_cpus;

        for job in queue.iter().skip(1) {
            if job.requested_cpus == 0 {
                continue;
            }
            if job.requested_cpus <= available {
                let runtime = job.runtime_estimate_sec.max(0);
                decisions.push(DispatchDecision {
                    job_id: job.job_id,
                    requested_cpus: job.requested_cpus,
                    runtime_estimate_sec: runtime,
                    estimated_completion_ts: snapshot.clock_ts + runtime,
                    reason: "easy_follow_dispatch".to_string(),
                });
                available -= job.requested_cpus;
            }
        }
        return DecisionReport {
            policy_id: "EASY_BACKFILL_BASELINE".to_string(),
            reservation_ts: Some(reservation_ts),
            decisions,
        };
    }

    for job in queue.iter().skip(1) {
        if job.requested_cpus == 0 || job.requested_cpus > available {
            continue;
        }
        let runtime = job.runtime_estimate_sec.max(0);
        let completion = snapshot.clock_ts + runtime;
        if completion <= reservation_ts {
            decisions.push(DispatchDecision {
                job_id: job.job_id,
                requested_cpus: job.requested_cpus,
                runtime_estimate_sec: runtime,
                estimated_completion_ts: completion,
                reason: "easy_backfill".to_string(),
            });
            available -= job.requested_cpus;
        }
    }

    DecisionReport {
        policy_id: "EASY_BACKFILL_BASELINE".to_string(),
        reservation_ts: Some(reservation_ts),
        decisions,
    }
}

fn choose_ml(snapshot: &Snapshot, strict_uncertainty_mode: bool) -> DecisionReport {
    if snapshot.queued_jobs.is_empty() {
        return DecisionReport {
            policy_id: "ML_BACKFILL_P50".to_string(),
            reservation_ts: None,
            decisions: Vec::new(),
        };
    }

    let queue = snapshot.queued_jobs.clone();
    let hol = &queue[0];
    let reservation_ts = reservation_ts_for_hol(snapshot, hol);
    let mut available = snapshot.free_cpus;
    let mut decisions = Vec::new();

    if hol.requested_cpus > 0 && hol.requested_cpus <= available {
        let runtime = hol.runtime_estimate_sec.max(0);
        decisions.push(DispatchDecision {
            job_id: hol.job_id,
            requested_cpus: hol.requested_cpus,
            runtime_estimate_sec: runtime,
            estimated_completion_ts: snapshot.clock_ts + runtime,
            reason: "ml_head_dispatch".to_string(),
        });
        available -= hol.requested_cpus;

        for job in queue.iter().skip(1) {
            if job.requested_cpus == 0 {
                continue;
            }
            if job.requested_cpus <= available {
                let runtime = job.runtime_estimate_sec.max(0);
                let source = job
                    .estimate_source
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                decisions.push(DispatchDecision {
                    job_id: job.job_id,
                    requested_cpus: job.requested_cpus,
                    runtime_estimate_sec: runtime,
                    estimated_completion_ts: snapshot.clock_ts + runtime,
                    reason: format!("ml_follow_dispatch:{source}"),
                });
                available -= job.requested_cpus;
            }
        }
        return DecisionReport {
            policy_id: "ML_BACKFILL_P50".to_string(),
            reservation_ts: Some(reservation_ts),
            decisions,
        };
    }

    for job in queue.iter().skip(1) {
        if job.requested_cpus == 0 || job.requested_cpus > available {
            continue;
        }
        let runtime_for_gate = if strict_uncertainty_mode {
            job.runtime_p90_sec.unwrap_or(job.runtime_estimate_sec).max(0)
        } else {
            job.runtime_guard_sec
                .unwrap_or(job.runtime_estimate_sec)
                .max(0)
        };
        let completion = snapshot.clock_ts + runtime_for_gate;
        if completion <= reservation_ts {
            let source = job
                .estimate_source
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            decisions.push(DispatchDecision {
                job_id: job.job_id,
                requested_cpus: job.requested_cpus,
                runtime_estimate_sec: runtime_for_gate,
                estimated_completion_ts: completion,
                reason: format!("ml_backfill:{source}"),
            });
            available -= job.requested_cpus;
        }
    }

    DecisionReport {
        policy_id: "ML_BACKFILL_P50".to_string(),
        reservation_ts: Some(reservation_ts),
        decisions,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let raw = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let mut snapshot: Snapshot =
        serde_json::from_str(&raw).with_context(|| "failed to parse snapshot JSON")?;
    normalize_snapshot(&mut snapshot)?;

    let report = match args.policy.as_str() {
        "FIFO_STRICT" => choose_fifo(&snapshot),
        "EASY_BACKFILL_BASELINE" => choose_easy(&snapshot),
        "ML_BACKFILL_P50" => choose_ml(&snapshot, args.strict_uncertainty_mode),
        other => anyhow::bail!(
            "unsupported policy '{other}', expected FIFO_STRICT|EASY_BACKFILL_BASELINE|ML_BACKFILL_P50"
        ),
    };

    let output = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        fs::write(&path, output)
            .with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{output}");
    }
    Ok(())
}
