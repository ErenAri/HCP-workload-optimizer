use anyhow::{Context, Result};
use clap::Parser;
use flate2::read::GzDecoder;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

const SWF_FIELD_COUNT: usize = 18;

#[derive(Debug, Parser)]
#[command(author, version, about = "Fast SWF line parser/stats utility")]
struct Args {
    /// Input SWF or SWF.GZ file path.
    #[arg(long)]
    input: PathBuf,
}

#[derive(Debug, Serialize)]
struct ParseStats {
    input: String,
    total_lines: u64,
    comment_lines: u64,
    blank_lines: u64,
    malformed_lines: u64,
    parsed_rows: u64,
}

fn open_reader(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let is_gzip = path.extension().and_then(|ext| ext.to_str()) == Some("gz");

    if is_gzip {
        let decoder = GzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut stats = ParseStats {
        input: args.input.display().to_string(),
        total_lines: 0,
        comment_lines: 0,
        blank_lines: 0,
        malformed_lines: 0,
        parsed_rows: 0,
    };

    let reader = open_reader(&args.input)?;
    for line in reader.lines() {
        stats.total_lines += 1;
        let line = line.with_context(|| "failed to read line")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            stats.blank_lines += 1;
            continue;
        }
        if trimmed.starts_with(';') || trimmed.starts_with('#') {
            stats.comment_lines += 1;
            continue;
        }

        let token_count = trimmed.split_whitespace().count();
        if token_count != SWF_FIELD_COUNT {
            stats.malformed_lines += 1;
            continue;
        }
        stats.parsed_rows += 1;
    }

    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
