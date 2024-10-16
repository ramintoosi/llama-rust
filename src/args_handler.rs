use anyhow::{anyhow, Context};
use std::path::PathBuf;
use clap::{Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use std::str::FromStr;

#[derive(Subcommand, Debug, Clone)]
pub enum Model {
    /// Use an already downloaded model
    #[clap(name = "local")]
    Local {
        /// The path to the model. e.g. `../hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        /// or `./llama-3.2-1b-instruct-q8_0.gguf`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
        repo: String,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    pub fn get_or_load(self) -> anyhow::Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}


#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Mode {
    Chat,
    Completion,
}

#[derive(Parser, Debug, Clone)]
pub struct Args {
    /// The path to the model
    #[command(subcommand)]
    pub model: Model,

    /// The mode of the code: completion or chat
    #[clap(value_enum, short = 'm', long, default_value = "chat")]
    pub mode: Mode,

    // /// The prompt to use - valid only if the mode is `completion`
    // #[clap(short = 'p', long, required_if_eq("mode", "completion"))]
    // prompt: Option<String>,

    /// set the length of the prompt + output in tokens
    #[clap(long, default_value_t = 512)]
    pub max_token: u32,

    /// override some parameters of the model
    #[clap(short = 'o', value_parser = parse_key_val)]
    pub key_value_overrides: Vec<(String, ParamOverrideValue)>,

    /// how many layers to keep on the gpu - zero is cpu mode
    #[clap(
        short = 'g',
        long,
        help = "how many layers to keep on the gpu - zero is cpu mode (default: 0)"
    )]
    pub n_gpu_layers: u32,

    /// set the seed for the RNG
    #[clap(short = 's', long, default_value_t=561371)]
    pub seed: u32,

    /// number of threads to use during generation
    #[clap(
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    pub threads: Option<i32>,
    #[clap(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    pub threads_batch: Option<i32>,

    // /// size of the prompt context
    // #[clap(
    //     short = 'c',
    //     long,
    //     help = "size of the prompt context (default: loaded from the model)"
    // )]
    // pub ctx_size: Option<NonZeroU32>,
    
    /// show the token/s speed at the end of each turn
    #[clap(short = 'v', long, action)]
    pub verbose: bool,

}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> anyhow::Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}